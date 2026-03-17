import 'dotenv/config';
import { StructuredOutputParser } from "@langchain/core/output_parsers";
import { PromptTemplate } from "@langchain/core/prompts";
import { ChatOpenAI } from "@langchain/openai";
import { RunnableSequence } from "@langchain/core/runnables";
import { z } from "zod";

const model = new ChatOpenAI({
    model: process.env.MODEL_NAME,
    apiKey: process.env.ALIYUN_API_KEY,
    temperature: 0,
    configuration: {
        baseURL: process.env.ALIYUN_BASE_URL,
    },
});


// 定义输出结构 schema
const schema = z.object({
    translation: z.string().describe("翻译后的英文文本"),
    keywords: z.array(z.string()).length(3).describe("3个关键词")
});

const outputParser = StructuredOutputParser.fromZodSchema(schema);

const template = PromptTemplate.fromTemplate(`
    请将以下中文文本翻译成英文，并提取出3个关键词：
    {input}
    {format_instructions}
`);

// const chain = RunnableSequence.from([template, model, outputParser]);

const chain = template.pipe(model).pipe(outputParser);

// 模板里有 {format_instructions}，invoke 时传入 outputParser.getFormatInstructions() 生成的说明文字
const result = await chain.invoke({
    input: "你好，世界！",
    format_instructions: outputParser.getFormatInstructions(),
});

console.log(result);