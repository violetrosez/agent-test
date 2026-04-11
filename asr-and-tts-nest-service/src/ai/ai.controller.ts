import { Controller, Get, Query, Sse } from '@nestjs/common';
import { AiService } from './ai.service';
import { from, Observable } from 'rxjs';
import { map } from 'rxjs/operators';
import { EventEmitter2 } from '@nestjs/event-emitter';
import { AI_TTS_STREAM_EVENT } from '../common/stream-events';

@Controller('ai')
export class AiController {
    constructor(private readonly aiService: AiService, private readonly eventEmitter: EventEmitter2) { }

    @Sse('chat/stream')
    chatStream(@Query('query') query: string, @Query('ttsSessionId') ttsSessionId?: string): Observable<{ data: string }> {
        // 与浏览器 TTS WS 同一 sessionId：先发 start，中继层再建腾讯云连接
        if (ttsSessionId) {
            const event = {
                type: 'start',
                sessionId: ttsSessionId,
                query: query,
            };
            this.eventEmitter.emit(AI_TTS_STREAM_EVENT, event);
        }
        // 正文仍走 SSE；带 ttsSessionId 时 AiService 内会并行 emit chunk 给 TtsRelayService
        return from(this.aiService.streamChain(query, ttsSessionId)).pipe(
            map((chunk) => ({
                data: chunk,
            })),
        );
    }
}
