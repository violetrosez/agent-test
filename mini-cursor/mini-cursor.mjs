import { ChatOpenAI } from "@langchain/openai";
import dotenv from "dotenv";
import { HumanMessage, SystemMessage, ToolMessage, AIMessage } from "@langchain/core/messages";
import { read_file_tool, write_file_tool, execute_command_tool, list_files_tool } from "./tools.mjs";
import { case1 } from "./test-case.mjs";
import chalk from "chalk";

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

let messages = [
    new SystemMessage(
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

回复要简洁，只说做了什么`
    ),
    new HumanMessage(
        case1
    ),
];

const toolMap = { [read_file_tool.name]: read_file_tool, [write_file_tool.name]: write_file_tool, [execute_command_tool.name]: execute_command_tool, [list_files_tool.name]: list_files_tool };

const MAX_TOOL_ROUNDS = 25;
let response;
let rounds = 0;

while (true) {
    if (rounds >= MAX_TOOL_ROUNDS) {
        console.warn(chalk.yellow(`\n已达到最大工具调用轮数 ${MAX_TOOL_ROUNDS}，强制结束`));
        break;
    }
    rounds += 1;
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
                const args = typeof tc.args === "string" ? (() => { try { return JSON.parse(tc.args); } catch { return {}; } })() : (tc.args ?? {});
                if (t) {
                    console.log(chalk.green(`调用工具: ${tc.name}, 参数: ${JSON.stringify(args)}`));
                    const out = await t.invoke(args);
                    return new ToolMessage({ content: String(out), tool_call_id: tc.id });
                }
                return new ToolMessage({ content: `未知工具: ${tc.name}`, tool_call_id: tc.id });
            })
        );
        messages = [...messages, ...toolMessages];
    }
}

console.log(chalk.cyan("\n--- 助手回复 ---\n"), response?.content ?? "");

