import { Inject, Injectable } from '@nestjs/common';
import { ChatOpenAI } from '@langchain/openai';
import { PromptTemplate } from '@langchain/core/prompts';
import type { Runnable } from '@langchain/core/runnables';
import { StringOutputParser } from '@langchain/core/output_parsers';
import { EventEmitter2 } from '@nestjs/event-emitter';
import { AI_TTS_STREAM_EVENT } from '../common/stream-events';

@Injectable()
export class AiService {
    private readonly chain: Runnable;

    constructor(
        @Inject('CHAT_MODEL') model: ChatOpenAI,
        private readonly eventEmitter: EventEmitter2
    ) {
        const prompt = PromptTemplate.fromTemplate(
            '请回答以下问题：\n\n{query}',
        );
        this.chain = prompt.pipe(model).pipe(new StringOutputParser());
    }

    async *streamChain(query: string, ttsSessionId?: string): AsyncGenerator<string> {
        const stream = await this.chain.stream({ query });
        for await (const chunk of stream) {
            // LLM 分片同步广播给 TTS 中继（与 SSE 并行，不阻塞 yield）
            if (ttsSessionId) {
                const event = {
                    type: 'chunk',
                    sessionId: ttsSessionId,
                    chunk: chunk,
                };
                this.eventEmitter.emit(AI_TTS_STREAM_EVENT, event);
            }
            yield chunk;
        }
    }
}