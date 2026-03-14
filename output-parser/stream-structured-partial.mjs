/**
 * 流式文本 + StructuredOutputParser 解析
 * 流程：注入格式指令 → stream() 流式输出文本 → 累积完整文本 → parser.parse() 解析为对象
 * 特点：不依赖 JSON 模式，兼容性最好，适合所有模型
 */
import 'dotenv/config';
import { ChatOpenAI } from '@langchain/openai';
import { z } from 'zod';
import { StructuredOutputParser } from '@langchain/core/output_parsers';

// ---------- 大模型（任意模型均可，不依赖 JSON 模式） ----------
const model = new ChatOpenAI({
    model: process.env.MODEL_NAME,
    apiKey: process.env.ALIYUN_API_KEY,
    temperature: 0,
    configuration: {
        baseURL: process.env.ALIYUN_BASE_URL,
    },
});

// ---------- 定义结构化输出的 schema ----------
const schema = z.object({
    name: z.string().describe("姓名"),
    birth_year: z.number().describe("出生年份"),
    death_year: z.number().describe("去世年份"),
    nationality: z.string().describe("国籍"),
    occupation: z.string().describe("职业"),
    famous_works: z.array(z.string()).describe("著名作品列表"),
    biography: z.string().describe("简短传记")
});



const parser = StructuredOutputParser.fromZodSchema(schema);

const prompt = `详细介绍莫扎特的信息
${parser.getFormatInstructions()}
`;
try {
    console.log("🤔 正在调用大模型...\n");

    let finalResult = '';
    const stream_result = await model.stream(prompt);

    // 流式输出：每个 chunk 是逐步构建的对象，最后一个 chunk 是完整对象
    for await (const chunk of stream_result) {
        finalResult += chunk.content;
        process.stdout.write(chunk.content); // 实时显示流式文本
    }
    const parsed_result = await parser.parse(finalResult);
    console.log("\n📝 最终结果:");
    // console.log(JSON.stringify(finalResult, null, 2));
    console.log("\n📝 格式化输出:");
    console.log(`姓名: ${parsed_result.name}`);
    console.log(`出生年份: ${parsed_result.birth_year}`);
    console.log(`去世年份: ${parsed_result.death_year}`);
    console.log(`国籍: ${parsed_result.nationality}`);
    console.log(`职业: ${parsed_result.occupation}`);
    console.log(`著名作品: ${parsed_result.famous_works.join(', ')}`);
    console.log(`传记: ${parsed_result.biography}`);
} catch (error) {
    console.error("❌ 错误:", error.message);
}
