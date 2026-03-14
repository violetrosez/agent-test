/**
 * 原始工具调用流式输出（不使用解析器）
 * 流程：定义 schema → bindTools → stream() → 从 chunk.tool_call_chunks[0].args 获取 JSON 片段
 * 特点：直接访问原始流式数据，适合需要自定义处理的场景
 */
import 'dotenv/config';
import { ChatOpenAI } from '@langchain/openai';
import { z } from 'zod';

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
    death_year: z.number().optional().describe("去世年份，如果还在世则不填"),
    nationality: z.string().describe("国籍"),
    fields: z.array(z.string()).describe("研究领域列表"),
    achievements: z.array(z.string()).describe("主要成就"),
    biography: z.string().describe("简短传记")
});

// ---------- 绑定工具（用对象形式，也可以用 tool() 函数） ----------
const model_with_tools = model.bindTools([{
    name: "extract_scientist_info",
    description: "提取和结构化科学家的详细信息",
    schema: scientistSchema
}]);

const prompt = `请介绍一下牛顿的信息`;

// ---------- 流式调用：从 tool_call_chunks 中逐步获取 JSON 片段 ----------
const stream_result = await model_with_tools.stream(prompt);

for await (const chunk of stream_result) {
    // 每个 chunk 的 tool_call_chunks[0].args 是 JSON 字符串片段
    if (chunk.tool_call_chunks && chunk.tool_call_chunks.length > 0) {
        process.stdout.write(chunk.tool_call_chunks[0].args);
    }
}
console.log("\n✅ 流式输出完成");
