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
    @Inject('DB_USERS_CRUD_TOOL') private readonly dbUsersCrudTool: any,
    @Inject('CRON_JOB_TOOL') private readonly cronJobTool: any,
    @Inject('TIME_NOW_TOOL') private readonly timeNowTool: any,
  ) {
    this.modelWithTools = model.bindTools([
      this.queryUserTool,
      this.sendEmailTool,
      this.webSearchTool,
      this.dbUsersCrudTool,
      this.cronJobTool,
      this.timeNowTool,
    ]);
  }

  async *runChain(input: string) {
    const messages: BaseMessage[] = [
      new SystemMessage(
        `你是一个通用任务助手，可以根据用户的目标规划步骤，并在需要时调用工具：\`query_user\`、\`send_mail\`、\`web_search\`、\`db_users_crud\`、\`cron_job\`、\`time_now\`（获取服务器当前时间），从而实现提醒、定期任务、数据同步等各种自动化需求。
        
        定时任务类型选择规则（非常重要）：
        - 用户说“X分钟/小时/天后”“在某个时间点”“到点提醒”（一次性）=> 用 \`cron_job\` + \`type=at\`（执行一次后自动停用），\`at\`=当前时间（使用 time_now 工具获取）+X 或解析出的时间点
        - 用户说“每X分钟/每小时/每天”“定期/循环/一直”（重复执行）=> 用 \`cron_job\` + \`type=every\`（每次执行），\`everyMs\`=X换算成毫秒
        - 用户给出 Cron 表达式或明确说“用 cron 表达式”（重复执行）=> 用 \`cron_job\` + \`type=cron\`
        
        在调用 \`cron_job.add\` 创建任务时，需要把用户原始自然语言拆成两部分：一部分是“什么时候执行”（用来决定 type/at/everyMs/cron），另一部分是“要做什么任务本身”。\`instruction\` 字段只能填“要做什么”的那部分文本（保持原语言和原话），不能再改写、翻译或总结。
        
        当用户请求“在未来某个时间点执行某个动作”（例如“1分钟后给我发一个笑话到邮箱”）时，本轮对话只需要使用 \`cron_job\` 设置/更新定时任务，不要在当前轮直接完成这个动作本身：不要直接调用 \`send_mail\` 给他发邮件，也不要在当前轮就真正“执行”指令，只需把要执行的动作写进 \`instruction\` 里，交给将来的定时任务去跑。
        
        注意：像“\`1分钟后提醒我喝水\`”，时间相关信息用于计算下一次执行时间，而 \`instruction\` 应该是“提醒我喝水”；本轮不需要立刻提醒。`,
      ),
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
        } else if (name === this.dbUsersCrudTool.name) {
          const toolResult = await this.dbUsersCrudTool.invoke(tc.args);
          messages.push(
            new ToolMessage({
              content: toolResult,
              tool_call_id: id ?? '',
              name,
            }),
          );
        } else if (name === this.cronJobTool.name) {
          const toolResult = await this.cronJobTool.invoke(tc.args);
          messages.push(
            new ToolMessage({
              content: toolResult,
              tool_call_id: id ?? '',
              name,
            }),
          );
        } else if (name === this.timeNowTool.name) {
          const toolResult = await this.timeNowTool.invoke(
            tc.args && typeof tc.args === 'object' ? tc.args : {},
          );
          messages.push(
            new ToolMessage({
              content:
                typeof toolResult === 'string'
                  ? toolResult
                  : JSON.stringify(toolResult),
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
