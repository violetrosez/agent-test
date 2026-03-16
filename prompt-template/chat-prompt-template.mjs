import 'dotenv/config';
import { ChatOpenAI } from '@langchain/openai';
import { ChatPromptTemplate } from '@langchain/core/prompts';

const model = new ChatOpenAI({
    model: process.env.MODEL_NAME,
    apiKey: process.env.ALIYUN_API_KEY,
    temperature: 0,
    configuration: {
        baseURL: process.env.ALIYUN_BASE_URL,
    },
});

// 参数是一个二维数组，数组第一个元素是 message 类型：
const chatPrompt = ChatPromptTemplate.fromMessages([
    [
        'system',
        `你是一名资深工程团队负责人，擅长用结构化、易读的方式写技术周报。
        写作风格要求：{tone}。
        
        请根据后续用户提供的信息，帮他生成一份适合给老板和团队同时抄送的周报草稿。`,
    ],
    [
        'human',
        `本周信息如下：
        
        公司名称：{company_name}
        团队名称：{team_name}
        直接汇报对象：{manager_name}
        本周时间范围：{week_range}
        
        本周团队核心目标：
        {team_goal}
        
        本周开发数据（Git 提交 / Jira 任务等）：
        {dev_activities}
        
        请据此输出一份 Markdown 周报，结构建议包含：
        1. 本周概览（2-3 句话）
        2. 详细拆分（按项目或模块分段）
        3. 关键指标表格（字段示例：模块 / 亮点 / 风险 / 下周计划）
        
        语气专业但有人情味。`,
    ],


])

const chatMessage = await chatPrompt.formatMessages({
    tone: '专业、清晰、略带幽默',
    company_name: '星航科技',
    team_name: 'AI 平台组',
    manager_name: '王总',
    week_range: '2025-02-03 ~ 2025-02-09',
    team_goal: '完成智能周报 Agent 的 MVP 版本，并打通 Git / Jira 数据源。',
    dev_activities:
        '- Git: 58 次提交，3 个主要分支合并\n' +
        '- Jira: 完成 12 个 Story，关闭 7 个 Bug\n' +
        '- 关键任务：完成智能周报 Pipeline 设计、实现 Prompt 拆分、接入 ExampleSelector',
    company_values: '「极致、开放、靠谱」的价值观',
})

const chatResponse = await model.invoke(chatMessage);

console.log(chatResponse.content);