import { Inject, Injectable } from '@nestjs/common';
import { ChatOpenAI } from '@langchain/openai';
import { StringOutputParser } from '@langchain/core/output_parsers';
import type { Runnable } from '@langchain/core/runnables';
import { PromptTemplate } from '@langchain/core/prompts';

@Injectable()
export class AiService {

  private readonly chain: Runnable;

  constructor(@Inject('CHAT_MODEL') model: ChatOpenAI) {
    const prompt = PromptTemplate.fromTemplate(`
     你是AI助手，请根据用户的问题给出回答。
      {input}
    `);

    this.chain = prompt.pipe(model).pipe(new StringOutputParser());
  }

  runChain(input: string) {
    return this.chain.invoke({ input });
  }

  async *streamChain(input: string) {
    const stream = await this.chain.stream({ input });

    for await (const chunk of stream) {
      console.log(chunk);

      yield chunk;
    }
  }
}
