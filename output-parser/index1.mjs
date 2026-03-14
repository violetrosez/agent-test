/**
 * StructuredOutputParser 结构化输出示例
 * 流程：定义 Zod schema → 生成格式指令注入 prompt → 大模型返回文本 → parser 解析为对象
 * 特点：不依赖模型的 JSON 模式，通过 prompt 工程让模型按格式返回
 */
import 'dotenv/config';
import { ChatOpenAI } from '@langchain/openai';
import { StructuredOutputParser } from '@langchain/core/output_parsers';
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

// ---------- 定义 Zod schema（支持嵌套对象、数组、可选字段） ----------
const scientistSchema = z.object({
    name: z.string().describe("科学家的全名"),
    birth_year: z.number().describe("出生年份"),
    death_year: z.number().optional().describe("去世年份，如果还在世则不填"),
    nationality: z.string().describe("国籍"),
    fields: z.array(z.string()).describe("研究领域列表"),
    awards: z.array(
        z.object({
            name: z.string().describe("奖项名称"),
            year: z.number().describe("获奖年份"),
            reason: z.string().optional().describe("获奖原因")
        })
    ).describe("获得的重要奖项列表"),
    major_achievements: z.array(z.string()).describe("主要成就列表"),
    famous_theories: z.array(
        z.object({
            name: z.string().describe("理论名称"),
            year: z.number().optional().describe("提出年份"),
            description: z.string().describe("理论简要描述")
        })
    ).describe("著名理论列表"),
    education: z.object({
        university: z.string().describe("主要毕业院校"),
        degree: z.string().describe("学位"),
        graduation_year: z.number().optional().describe("毕业年份")
    }).optional().describe("教育背景"),
    biography: z.string().describe("简短传记，100字以内")
});

// ---------- 创建解析器并获取格式指令 ----------
const parser = StructuredOutputParser.fromZodSchema(scientistSchema);

// prompt 中注入格式指令，告诉模型按什么格式返回
const prompt = `请介绍一下爱因斯坦的信息
${parser.getFormatInstructions()}
`;
console.log("📝 Prompt（含格式指令）:\n", prompt);


try {
    console.log("🤔 正在调用大模型...\n");

    const response = await model.invoke(prompt);

    console.log("✅ 收到响应:\n");
    console.log(response.content);



    const result = await parser.parse(response.content);
    console.log("\n📋 解析后的 JSON 对象:");
    console.log(result);
    console.log("✅ JsonOutputParser 自动解析的结果:\n");
    console.log(result);
    console.log(`姓名: ${result.name}`);
    console.log(`出生年份: ${result.birth_year}`);
    console.log(`国籍: ${result.nationality}`);
    console.log(`著名理论: ${result.famous_theory}`);
    console.log(`主要成就:`, result.major_achievements);


} catch (error) {
    console.error("❌ 错误:", error.message);
}