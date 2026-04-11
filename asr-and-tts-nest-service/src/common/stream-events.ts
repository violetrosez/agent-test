/** EventEmitter 事件名：AiController / AiService → TtsRelayService */
export const AI_TTS_STREAM_EVENT = 'ai.tts.stream';

/** 与 TtsRelayService.handleAiStreamEvent 的 switch 分支一一对应 */
export type AiTtsStreamEvent =
    | { type: 'start'; sessionId: string; query: string }
    | { type: 'chunk'; sessionId: string; chunk: string }
    | { type: 'end'; sessionId: string }
    | { type: 'error'; sessionId: string; error: string };