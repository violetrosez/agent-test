import { Inject, Injectable } from '@nestjs/common';
import { ChatOpenAI } from '@langchain/openai';
import { tool } from '@langchain/core/tools';
import {
  AIMessage,
  AIMessageChunk,
  BaseMessage,
  HumanMessage,
  SystemMessage,
  ToolMessage,
} from '@langchain/core/messages';
import { z } from 'zod';
import { Runnable } from '@langchain/core/runnables';


// 与 AiModule 里 QUERY_USER_TOOL 的 schema（userId）保持一致
const queryUserArgsSchema = z.object({
  userId: z.string().describe('用户 ID'),
});



@Injectable()
export class AiService {

  private readonly modelWithTools: Runnable<BaseMessage[], AIMessage>;

  constructor(
    @Inject('CHAT_MODEL') model: ChatOpenAI,
    @Inject('QUERY_USER_TOOL') private readonly queryUserTool: any,
    @Inject('SEND_MAIL_TOOL') private readonly sendEmailTool: any,
    @Inject('WEB_SEARCH_TOOL') private readonly webSearchTool: any,
  ) {
    this.modelWithTools = model.bindTools([this.queryUserTool, this.sendEmailTool, this.webSearchTool]);
  }

  async *runChain(input: string) {
    const messages: BaseMessage[] = [
      new SystemMessage(`你是一个AI助手，请根据用户的问题给出回答。
      `),
      new HumanMessage(input),
    ];
    const maxRounds = 16;
    for (let round = 0; round < maxRounds; round++) {
      const stream = await this.modelWithTools.stream(messages);

      let fullMessages: AIMessageChunk | null = null;
      for await (const chunk of stream) {
        // console.log(chunk);

        if (!AIMessageChunk.isInstance(chunk)) continue;
        fullMessages = fullMessages ? fullMessages.concat(chunk) : chunk;

        const hasToolCall =
          (fullMessages.tool_call_chunks?.length ?? 0) > 0 ||
          (fullMessages.tool_calls?.length ?? 0) > 0;
        if (!hasToolCall) {
          const c = chunk.content;
          if (typeof c === 'string' && c.length) yield c;
        }
      }
      // console.log(fullMessages);

      if (!fullMessages) return;
      messages.push(fullMessages);
      const toolCalls = fullMessages.tool_calls;
      if (!toolCalls?.length) return;

      for (const tc of toolCalls) {
        const { id, name } = tc;

        if (name === this.queryUserTool.name) {
          const args = queryUserArgsSchema.parse(tc.args);
          const toolResult = await this.queryUserTool.invoke(args);
          messages.push(
            new ToolMessage({
              content:
                typeof toolResult === 'string' ? toolResult : String(toolResult),
              name,
              tool_call_id: id ?? '',
            }),
          );
        } else if (name === this.sendEmailTool.name) {

          const toolResult = await this.sendEmailTool.invoke(tc.args);
          messages.push(
            new ToolMessage({
              content: toolResult,
              tool_call_id: id ?? '',
              name,
            }),
          );
        } else if (name === this.webSearchTool.name) {
          const toolResult = await this.webSearchTool.invoke(tc.args);
          messages.push(
            new ToolMessage({
              content: toolResult,
              tool_call_id: id ?? '',
              name,
            }),
          );
        } else {
          // 每个 tool_call 都要有对应 ToolMessage，否则下一轮请求会 400
          messages.push(
            new ToolMessage({
              tool_call_id: id ?? '',
              content: `未实现工具: ${name}`,
              name,
            }),
          );
        }
      }
    }
  }
}
