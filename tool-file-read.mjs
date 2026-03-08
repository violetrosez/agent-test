import { ChatOpenAI } from "@langchain/openai";
import dotenv from "dotenv";
import { tool } from "@langchain/core/tools";
import { HumanMessage, SystemMessage, ToolMessage, AIMessage } from "@langchain/core/messages";
import fs from "fs/promises";
import { z } from "zod";

dotenv.config();

const model = new ChatOpenAI({
    model: process.env.MODEL_NAME,
    apiKey: process.env.ALIYUN_API_KEY,
    temperature: 0.0,
    configuration: {
        baseURL: process.env.ALIYUN_BASE_URL,
    },
})


// 定义工具（LangChain 要求用 schema 且为 Zod，不能用 parameters）
const read_file_tool = tool(
    async ({ path }) => {
        try {
            const content = await fs.readFile(path, "utf8");
            console.log(`read_file_tool function call successfully: ${path}`);
            return content;
        } catch (err) {
            const msg = err instanceof Error ? err.message : String(err);
            return `读取失败: ${msg}`;
        }
    },
    {
        name: "read_file",
        description: "读取文件内容",
        schema: z.object({
            path: z.string().describe("文件路径"),
        }),
    }
);

const write_file_tool = tool(
    async ({ path, content }) => {
        try {
            await fs.writeFile(path, content);
            console.log(`write_file_tool function call successfully: ${path}`);
            return content;
        } catch (err) {
            const msg = err instanceof Error ? err.message : String(err);
            return `写入失败: ${msg}`;
        }
    },
    {
        name: "write_file",
        description: "写入文件内容",
        schema: z.object({
            path: z.string().describe("文件路径"),
            content: z.string().describe("文件内容"),
        }),
    }
);

/**
 * 绑定工具
 */
const model_with_tools = model.bindTools([read_file_tool, write_file_tool]);

let messages = [
    new SystemMessage(
        `你是一个代码助手，可以使用工具读取文件并解释代码。
        工作流程：
        1. 用户要求读取文件时，立即调用 read_file 工具
        2. 等待工具返回文件内容
        3. 基于文件内容进行分析和解释
        4. 用户要求写入文件时，立即调用 write_file 工具
        5. 等待工具返回写入结果

        工具列表：
        ${read_file_tool.name}: ${read_file_tool.description}
        ${write_file_tool.name}: ${write_file_tool.description}
        `
    ),
    new HumanMessage(
        "读取文件tool-file-read.mjs的内容并解释代码，解释结果写入文件tool-file-read.md"
    ),
];

const toolMap = { [read_file_tool.name]: read_file_tool, [write_file_tool.name]: write_file_tool };

let response;

while (true) {
    response = await model_with_tools.invoke(messages);
    messages.push(
        response instanceof AIMessage ? response : new AIMessage({ content: response.content, tool_calls: response.tool_calls ?? [] })
    );
    if (!response.tool_calls?.length) {
        break;
    } else {
        const toolMessages = await Promise.all(
            response.tool_calls.map(async (tc) => {
                const t = toolMap[tc.name];
                if (t) {
                    console.log(`调用工具: ${tc.name}`);
                    const out = await t.invoke(tc.args);
                    return new ToolMessage({ content: String(out), tool_call_id: tc.id });
                }
                return new ToolMessage({ content: `未知工具: ${tc.name}`, tool_call_id: tc.id });

            })
        );
        messages = [...messages, ...toolMessages];
    }
}

console.log(response.content);
