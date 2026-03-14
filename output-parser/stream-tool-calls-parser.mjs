/**
 * JsonOutputToolsParser 流式解析工具调用
 * 流程：定义 schema → bindTools → pipe(parser) → stream() 流式返回已解析的对象
 * 特点：解析器自动处理 tool_call_chunks，返回逐步构建的结构化对象
 */
import 'dotenv/config';
import { ChatOpenAI } from '@langchain/openai';
import { JsonOutputToolsParser } from '@langchain/core/output_parsers/openai_tools';
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

// ---------- 绑定工具到模型 ----------
const modelWithTool = model.bindTools([{
    name: "extract_scientist_info",
    description: "提取和结构化科学家的详细信息",
    schema: scientistSchema
}]);

// ---------- 挂载 JsonOutputToolsParser，自动解析 tool_calls ----------
const parser = new JsonOutputToolsParser();
const chain = modelWithTool.pipe(parser);

try {
    // ---------- 流式调用：每个 chunk 是逐步构建的对象数组 ----------
    const stream = await chain.stream("详细介绍牛顿的生平和成就");

    console.log("📡 实时输出流式内容:\n");

    for await (const chunk of stream) {
        // chunk 是数组，chunk[0].args 是逐步构建的结构化对象
        if (chunk.length > 0) {
            const toolCall = chunk[0];
            console.log(toolCall.args);
        }
    }

    console.log("\n✅ 流式输出完成");
} catch (error) {
    console.error("\n❌ 错误:", error.message);
}