import { ChatOpenAI } from "@langchain/openai";
import dotenv from "dotenv";
import { HumanMessage, SystemMessage, ToolMessage, AIMessage } from "@langchain/core/messages";
import { read_file_tool, write_file_tool, execute_command_tool, list_files_tool } from "./tools.mjs";
import { case1 } from "./test-case.mjs";
import chalk from "chalk";
import { InMemoryChatMessageHistory } from "@langchain/core/chat_history";
import { JsonOutputToolsParser } from "@langchain/core/output_parsers/openai_tools";

dotenv.config();

const model = new ChatOpenAI({
    model: process.env.MODEL_NAME,
    apiKey: process.env.ALIYUN_API_KEY,
    temperature: 0.0,
    configuration: {
        baseURL: process.env.ALIYUN_BASE_URL,
    },
})




/**
 * 绑定工具
 */
const model_with_tools = model.bindTools([read_file_tool, write_file_tool, execute_command_tool, list_files_tool]);

const memory = new InMemoryChatMessageHistory();
await memory.addMessage(new SystemMessage(
    `你是一个项目管理助手，使用工具完成任务。

        当前工作目录: ${process.cwd()}

        工具：
        1. read_file: 读取文件
        2. write_file: 写入文件
        3. execute_command: 执行命令（参数: command, directory 为工作目录）
        4. list_files: 列出目录下的文件
       重要规则 - execute_command：
- directory 参数会自动切换到指定目录
- 当使用 directory 时，绝对不要在 command 中使用 cd
- 错误示例: { command: "cd react-todo-app && pnpm install", directory: "react-todo-app" }
这是错误的！因为 directory 已经在 react-todo-app 目录了，再 cd react-todo-app 会找不到目录
- 正确示例: { command: "pnpm install", directory: "react-todo-app" }
directory 已经切换到 react-todo-app，直接执行命令即可

回复要简洁，只说做了什么。

重要：当任务全部完成后，必须只回复一段总结文字，不要再次调用任何工具。`
));

await memory.addMessage(new HumanMessage(case1));


const toolMap = { [read_file_tool.name]: read_file_tool, [write_file_tool.name]: write_file_tool, [execute_command_tool.name]: execute_command_tool, [list_files_tool.name]: list_files_tool };

const MAX_TOOL_ROUNDS = 50;
let endedNormally = false;

for (let index = 0; index < MAX_TOOL_ROUNDS; index++) {
    try {
        // getMessages() 返回 Promise，必须 await 后再传入，否则 invoke 会收到 Promise 对象并报 toChatMessages is not a function
        const rawStreamResult = await model_with_tools.stream(await memory.getMessages());
        let fullAIMessage = null;

        console.log(chalk.bgBlue(`\n🚀 Agent 开始思考并生成流...\n`));
        const printedLengths = new Map();
        const parser = new JsonOutputToolsParser();

        for await (const chunk of rawStreamResult) {
            fullAIMessage = fullAIMessage ? fullAIMessage.concat(chunk) : chunk;
            let parsedTools = null;
            try {
                parsedTools = await parser.parseResult([{ message: fullAIMessage }]);
            } catch (e) {
                // 解析失败说明 JSON 还不完整，忽略错误继续累积
            }
            if (parsedTools && parsedTools?.length) {
                for (const toolCall of parsedTools) {
                    if (toolCall.type === 'write_file' && toolCall.args.content) {
                        const toolCallId = toolCall.id || toolCall.args.path || 'default';
                        const currentContent = String(toolCall.args.content);
                        const previousLength = printedLengths.get(toolCallId);
                        if (previousLength === undefined) {
                            printedLengths.set(toolCallId, 0);
                            console.log(chalk.bgBlue(`\n[工具调用] write_file("${toolCall.args.path}") - 开始写入（流式预览）\n`));
                        }
                        if (currentContent.length > previousLength) {
                            const newContent = currentContent.slice(previousLength);
                            process.stdout.write(newContent);
                            printedLengths.set(toolCallId, currentContent.length);
                        }
                    }
                }
            } else {
                if (chunk.content) process.stdout.write(chunk.content);
            }
        }

        // 流式合并结果是 AIMessageChunk，转为标准 AIMessage 再判断与入库，避免 tool_calls 形态不一致导致无法结束
        const content = typeof fullAIMessage?.content === "string" ? fullAIMessage.content : (Array.isArray(fullAIMessage?.content) ? "" : (fullAIMessage?.content ?? ""));
        const toolCalls = Array.isArray(fullAIMessage?.tool_calls) ? fullAIMessage.tool_calls : [];
        const normalizedMessage = new AIMessage({ content, tool_calls: toolCalls });
        await memory.addMessage(normalizedMessage);

        const hasToolCalls = normalizedMessage.tool_calls.length > 0;
        if (!hasToolCalls) {
            console.log(`\n✨ AI 最终回复:\n${normalizedMessage.content ?? ""}\n`);
            endedNormally = true;
            break;
        }

        const toolMessages = await Promise.all(
            normalizedMessage.tool_calls.map(async (tc) => {
                const t = toolMap[tc.name];
                const args = typeof tc.args === "string" ? (() => { try { return JSON.parse(tc.args); } catch { return {}; } })() : (tc.args ?? {});
                if (t) {
                    const out = await t.invoke(args);
                    return new ToolMessage({ content: String(out), tool_call_id: tc.id });
                }
                return new ToolMessage({ content: `未知工具: ${tc.name}`, tool_call_id: tc.id });
            })
        );
        await memory.addMessages(toolMessages);
    } catch (err) {
        console.error(chalk.red("执行出错:"), err);
        process.exitCode = 1;
        break;
    }
}

if (!endedNormally) {
    console.warn(chalk.yellow(`\n未收到 AI 最终回复（可能已达最大工具轮数 ${MAX_TOOL_ROUNDS} 或执行出错）。`));
}

