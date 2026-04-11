import { Inject, Injectable, Logger, OnModuleDestroy } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { createHmac, randomUUID } from 'node:crypto';
import { OnEvent } from '@nestjs/event-emitter';
import { AI_TTS_STREAM_EVENT, type AiTtsStreamEvent } from '../common/stream-events';
import WebSocket from 'ws';

/** 浏览器 WS 与腾讯云流式 TTS 之间的会话状态（一对一双向中继） */
type ClientSession = {
    sessionId: string;
    clientWs: WebSocket;
    tencentWs?: WebSocket;
    /** 腾讯云返回 ready=1 之前不能发合成文本，先到 pendingChunks 排队 */
    ready: boolean;
    pendingChunks: string[];
    closed: boolean;
};

@Injectable()
export class TtsRelayService implements OnModuleDestroy {
    private readonly logger = new Logger(TtsRelayService.name);
    private readonly sessions = new Map<string, ClientSession>();
    private readonly secretId: string;
    private readonly secretKey: string;
    private readonly appId: number;
    private readonly voiceType: number;

    constructor(@Inject(ConfigService) configService: ConfigService) {
        this.secretId = configService.get<string>('SECRET_ID') ?? '';
        this.secretKey = configService.get<string>('SECRET_KEY') ?? '';
        this.appId = Number(configService.get<string>('APP_ID') ?? 0);
        this.voiceType = Number(configService.get<string>('TTS_VOICE_TYPE') ?? 101001);
    }

    onModuleDestroy(): void {
        for (const session of this.sessions.values()) {
            this.closeSession(session.sessionId, 'module destroy');
        }
    }

    /** 浏览器连接 /speech/tts/ws：建会话、下发 sessionId，供 SSE 侧 ttsSessionId 对齐 */
    registerClient(clientWs: WebSocket, wantedSessionId?: string): string {
        const sessionId = wantedSessionId?.trim() || randomUUID();
        const existing = this.sessions.get(sessionId);
        if (existing) {
            this.closeSession(sessionId, 'client reconnected');
        }

        this.sessions.set(sessionId, {
            sessionId,
            clientWs,
            ready: false,
            pendingChunks: [],
            closed: false,
        });
        this.sendClientJson(clientWs, { type: 'session', sessionId });
        this.logger.log(`TTS client connected: ${sessionId}`);
        return sessionId;
    }

    unregisterClient(sessionId: string): void {
        this.closeSession(sessionId, 'client disconnected');
    }

    /** 订阅 AI 流式输出事件：把 LLM 分片转成腾讯云 ACTION_SYNTHESIS，二进制音频原样推给前端 */
    @OnEvent(AI_TTS_STREAM_EVENT)
    handleAiStreamEvent(event: AiTtsStreamEvent): void {
        const session = this.sessions.get(event.sessionId);
        if (!session) return;

        switch (event.type) {
            case 'start': {
                // 首包：拉起腾讯云 WS，并通知前端可开始喂 MSE / 播放管线
                this.ensureTencentConnection(session);
                this.sendClientJson(session.clientWs, {
                    type: 'tts_started',
                    sessionId: session.sessionId,
                    query: event.query,
                });
                break;
            }
            case 'chunk': {
                const chunk = event.chunk?.trim();
                if (!chunk) return;
                // 上游未 ready 或 WS 未连通：缓存文本，避免丢字
                if (!session.ready || !session.tencentWs || session.tencentWs.readyState !== WebSocket.OPEN) {
                    session.pendingChunks.push(chunk);
                    return;
                }
                this.sendTencentChunk(session, chunk);
                break;
            }
            case 'end': {
                // 流结束：先清空队列，再通知腾讯云本轮合成完毕
                this.flushPendingChunks(session);
                if (session.tencentWs && session.tencentWs.readyState === WebSocket.OPEN) {
                    session.tencentWs.send(
                        JSON.stringify({
                            session_id: session.sessionId,
                            action: 'ACTION_COMPLETE',
                        }),
                    );
                }
                break;
            }
            case 'error': {
                this.sendClientJson(session.clientWs, {
                    type: 'tts_error',
                    message: event.error,
                });
                this.closeSession(session.sessionId, 'ai stream error');
                break;
            }
        }
    }

    /** 按 session 建立腾讯云 TextToStreamAudioWSv2，签名在 URL 查询参数中 */
    private ensureTencentConnection(session: ClientSession): void {
        if (session.tencentWs && session.tencentWs.readyState <= WebSocket.OPEN) {
            return;
        }
        if (!this.secretId || !this.secretKey || !this.appId) {
            this.sendClientJson(session.clientWs, {
                type: 'tts_error',
                message: 'TTS 凭证缺失，请检查 SECRET_ID/SECRET_KEY/APP_ID',
            });
            return;
        }

        const url = this.buildTencentTtsWsUrl(session.sessionId);
        const tencentWs = new WebSocket(url);
        session.tencentWs = tencentWs;
        session.ready = false;

        tencentWs.on('open', () => {
            this.logger.log(`Tencent TTS ws opened: ${session.sessionId}`);
        });

        tencentWs.on('message', (data, isBinary) => {
            if (session.closed) return;
            // 二进制：MP3 音频片，直接透传给浏览器（与前端 MediaSource audio/mpeg 一致）
            if (isBinary) {
                if (session.clientWs.readyState === WebSocket.OPEN) {
                    session.clientWs.send(data, { binary: true });
                }
                return;
            }

            const raw = data.toString();
            let msg: Record<string, unknown> | undefined;
            try {
                msg = JSON.parse(raw) as Record<string, unknown>;
            } catch {
                return;
            }

            // 控制面 JSON：ready=1 表示可开始发送待合成文本
            if (Number(msg.ready) === 1) {
                session.ready = true;
                this.flushPendingChunks(session);
            }

            if (Number(msg.code) && Number(msg.code) !== 0) {
                this.sendClientJson(session.clientWs, {
                    type: 'tts_error',
                    message: String(msg.message ?? 'Tencent TTS error'),
                    code: Number(msg.code),
                });
                this.closeSession(session.sessionId, 'tencent error');
                return;
            }

            // final=1：本轮流式合成结束，前端可 endOfStream() 等
            if (Number(msg.final) === 1) {
                this.sendClientJson(session.clientWs, { type: 'tts_final' });
            }
        });

        tencentWs.on('error', (error) => {
            this.sendClientJson(session.clientWs, {
                type: 'tts_error',
                message: `Tencent ws error: ${error.message}`,
            });
        });

        tencentWs.on('close', () => {
            session.tencentWs = undefined;
            session.ready = false;
        });
    }

    /** ready 之后把 AI 早到的文本片段按顺序补发给腾讯云 */
    private flushPendingChunks(session: ClientSession): void {
        if (!session.ready || !session.tencentWs || session.tencentWs.readyState !== WebSocket.OPEN) {
            return;
        }
        while (session.pendingChunks.length > 0) {
            const chunk = session.pendingChunks.shift();
            if (!chunk) continue;
            this.sendTencentChunk(session, chunk);
        }
    }

    /** 腾讯云流式协议：ACTION_SYNTHESIS 携带本段要合成的文本 */
    private sendTencentChunk(session: ClientSession, text: string): void {
        if (!session.tencentWs || session.tencentWs.readyState !== WebSocket.OPEN) {
            session.pendingChunks.push(text);
            return;
        }

        session.tencentWs.send(
            JSON.stringify({
                session_id: session.sessionId,
                message_id: `msg_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
                action: 'ACTION_SYNTHESIS',
                data: text,
            }),
        );
    }

    /** 双向关连接并清 Map，避免泄漏与重复 session */
    private closeSession(sessionId: string, reason: string): void {
        const session = this.sessions.get(sessionId);
        if (!session) return;
        session.closed = true;

        if (session.tencentWs && session.tencentWs.readyState < WebSocket.CLOSING) {
            session.tencentWs.close();
        }
        if (session.clientWs.readyState < WebSocket.CLOSING) {
            this.sendClientJson(session.clientWs, { type: 'tts_closed', reason });
            session.clientWs.close();
        }
        this.sessions.delete(sessionId);
        this.logger.log(`TTS session closed: ${sessionId}, reason: ${reason}`);
    }

    private sendClientJson(clientWs: WebSocket, payload: Record<string, unknown>): void {
        if (clientWs.readyState !== WebSocket.OPEN) return;
        clientWs.send(JSON.stringify(payload));
    }

    /** HMAC-SHA1 签 GET 请求串，与腾讯云文档 stream_wsv2 参数一致 */
    private buildTencentTtsWsUrl(sessionId: string): string {
        const now = Math.floor(Date.now() / 1000);
        const params: Record<string, string | number> = {
            Action: 'TextToStreamAudioWSv2',
            AppId: this.appId,
            Codec: 'mp3',
            Expired: now + 3600,
            SampleRate: 16000,
            SecretId: this.secretId,
            SessionId: sessionId,
            Speed: 0,
            Timestamp: now,
            VoiceType: this.voiceType,
            Volume: 5,
        };

        const signStr = Object.keys(params)
            .sort()
            .map((k) => `${k}=${params[k]}`)
            .join('&');
        const rawStr = `GETtts.cloud.tencent.com/stream_wsv2?${signStr}`;
        const signature = createHmac('sha1', this.secretKey).update(rawStr).digest('base64');
        const searchParams = new URLSearchParams({
            ...Object.fromEntries(Object.entries(params).map(([k, v]) => [k, String(v)])),
            Signature: signature,
        });

        return `wss://tts.cloud.tencent.com/stream_wsv2?${searchParams.toString()}`;
    }
}