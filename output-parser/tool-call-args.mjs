/**
 * 工具调用方式获取结构化输出
 * 流程：定义 schema → 绑定为 tool → 模型返回 tool_calls → 从 args 中取结构化数据
 * 特点：利用模型原生的 function calling 能力，准确率更高
 */
import 'dotenv/config';
import { ChatOpenAI } from '@langchain/openai';
import { z } from 'zod';
import { tool } from "@langchain/core/tools";

// ---------- 大模型 ----------
const model = new ChatOpenAI({
    model: process.env.MODEL_NAME,
    apiKey: process.env.ALIYUN_API_KEY,
    temperature: 0,
    configuration: {
        baseURL: process.env.ALIYUN_BASE_URL,
    },
});

// ---------- 定义结构化输出的 schema ----------
const scientistSchema = z.object({
    name: z.string().describe("科学家的全名"),
    birth_year: z.number().describe("出生年份"),
    nationality: z.string().describe("国籍"),
    fields: z.array(z.string()).describe("研究领域列表"),
});

// ---------- 将 schema 包装为工具（tool 函数体为空，仅用于提取结构化数据） ----------
const format_scientist_tool = tool(
    () => { },
    {
        name: "format_scientist",
        description: "提取和结构化科学家的详细信息",
        schema: scientistSchema,
    }
);

const prompt = `请介绍一下爱因斯坦的信息`;

// ---------- 绑定工具到模型 ----------
const model_with_tools = model.bindTools([format_scientist_tool]);

try {
    console.log("🤔 正在调用大模型...\n");

    const output = await model_with_tools.invoke(prompt);

    console.log("✅ 收到响应:\n");
    console.log(output.tool_calls[0]);
    const result = output.tool_calls[0];
    console.log(`姓名: ${result.args.name}`);
    console.log(`出生年份: ${result.args.birth_year}`);
    console.log(`国籍: ${result.args.nationality}`);
    console.log(`研究领域: ${result.args.fields}`);



} catch (error) {
    console.error("❌ 错误:", error.message);
}