/**
 * withStructuredOutput 流式结构化输出
 * 流程：定义 schema → model.withStructuredOutput(schema) → stream() 流式返回对象片段
 * 特点：每个 chunk 是逐步构建的对象，最后一个 chunk 是完整结果
 * 注意：需要模型支持 JSON 模式，且 prompt 中必须包含 "json" 关键词
 */
import 'dotenv/config';
import { ChatOpenAI } from '@langchain/openai';
import { z } from 'zod';

// ---------- 大模型（代码模型可能不支持 JSON 模式，建议用 qwen-plus） ----------
const model = new ChatOpenAI({
    model: process.env.MODEL_NAME,
    apiKey: process.env.ALIYUN_API_KEY,
    temperature: 0,
    configuration: {
        baseURL: process.env.ALIYUN_BASE_URL,
    },
});

// 使用 zod 定义结构化输出格式
const schema = z.object({
    name: z.string().describe("姓名"),
    birth_year: z.number().describe("出生年份"),
    death_year: z.number().describe("去世年份"),
    nationality: z.string().describe("国籍"),
    occupation: z.string().describe("职业"),
    famous_works: z.array(z.string()).describe("著名作品列表"),
    biography: z.string().describe("简短传记")
});

const prompt = `详细介绍莫扎特的信息，以 json 格式返回。`;
console.log("🌊 流式结构化输出演示（withStructuredOutput）\n");

const model_with_structured = model.withStructuredOutput(schema);
try {
    console.log("🤔 正在调用大模型...\n");

    let finalResult = null;
    const stream_result = await model_with_structured.stream(prompt);
    
    // 流式输出：每个 chunk 是逐步构建的对象，最后一个 chunk 是完整对象
    for await (const chunk of stream_result) {
        finalResult = chunk;
        console.log("📦 收到 chunk:", JSON.stringify(chunk, null, 2));
    }
    
    console.log("\n📝 最终结果:");
    console.log(JSON.stringify(finalResult, null, 2));
    console.log("\n✅ 流式输出完成\n");
} catch (error) {
    console.error("❌ 错误:", error.message);
}
