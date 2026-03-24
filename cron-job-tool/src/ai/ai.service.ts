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


const database = {
  users: {
    '001': { id: '001', name: '张三', email: 'zhangsan@example.com', role: 'admin' },
    '002': { id: '002', name: '李四', email: 'lisi@example.com', role: 'user' },
    '003': { id: '003', name: '王五', email: 'wangwu@example.com', role: 'user' },
  },
};

const queryUserArgsSchema = z.object({
  id: z.string().describe('用户ID'),
});

type QueryUserArgs = z.infer<typeof queryUserArgsSchema>;


const queryUserTool = tool(
  (args: QueryUserArgs): string => {
    const { id } = args;
    const user = database.users[id];
    if (!user) {
      return `用户ID ${id} 不存在`;
    }
    return `用户信息：\n- ID: ${user.id}\n- 姓名: ${user.name}\n- 邮箱: ${user.email}\n- 角色: ${user.role}`;

  },
  {
    name: 'query_user',
    description: '查询用户信息',
    schema: queryUserArgsSchema,
  }
);

@Injectable()
export class AiService {

  private readonly modelWithTools: Runnable<BaseMessage[], AIMessage>;

  constructor(@Inject('CHAT_MODEL') model: ChatOpenAI) {
    this.modelWithTools = model.bindTools([queryUserTool]);
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
        console.log(chunk);

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

        if (name === queryUserTool.name) {
          const args = queryUserArgsSchema.parse(tc.args);
          const toolResult = await queryUserTool.invoke(args);
          messages.push(
            new ToolMessage({
              content:
                typeof toolResult === 'string' ? toolResult : String(toolResult),
              name,
              tool_call_id: id ?? '',
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
