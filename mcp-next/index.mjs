import dotenv from "dotenv";
import { MultiServerMCPClient } from '@langchain/mcp-adapters';
import { ChatOpenAI } from '@langchain/openai';
import chalk from 'chalk';
import { HumanMessage, ToolMessage } from '@langchain/core/messages';

dotenv.config();

const model = new ChatOpenAI({
    model: process.env.MODEL_NAME,
    apiKey: process.env.ALIYUN_API_KEY,
    temperature: 0.0,
    configuration: {
        baseURL: process.env.ALIYUN_BASE_URL,
    },
});

const client = new MultiServerMCPClient({
    mcpServers: {
        "amap-maps-streamableHTTP": {
            url: `https://mcp.amap.com/mcp?key=${process.env.AMAP_API_KEY}`,
        },
        // 文件系统 MCP 必须写在 mcpServers 里才会被加载
        "filesystem": {
            command: "npx",
            args: ["-y", "@modelcontextprotocol/server-filesystem", process.cwd()],
        },
        "chrome-devtools": {
            "command": "npx",
            "args": [
                "-y",
                "chrome-devtools-mcp@latest"
            ]
        }
    },
});

// 获取所有工具（getTools 返回数组）
const tools = await client.getTools();
const model_with_tools = model.bindTools(tools);
const toolMap = Object.fromEntries(tools.map((t) => [t.name, t]));

async function runAgentWithTools(query, maxRounds = 10) {
    let messages = [
        new HumanMessage(`${query}`),
    ];

    for (let i = 0; i < maxRounds; i++) {
        console.log(chalk.bgGreen(`⏳ 正在等待 AI 思考...`));
        const response = await model_with_tools.invoke(messages);
        if (!response.tool_calls?.length) {
            console.log(chalk.bgGreen(`🤖 AI 回复: ${response.content}`));
            return response.content;
        } else {
            messages.push(response);
            const toolMessages = await Promise.all(
                response.tool_calls.map(async (tc) => {
                    const tool = toolMap[tc.name];
                    console.log(chalk.bgBlue(`🔧 调用工具: ${tc.name}, 参数: ${JSON.stringify(tc.args)}`));
                    const result = await tool.invoke(tc.args);
                    console.log(chalk.bgYellow(`🔧 工具返回: ${JSON.stringify(result)}`));

                    return new ToolMessage({ content: String(result), tool_call_id: tc.id });
                })
            );
            messages.push(...toolMessages);
        }
    }


}


await runAgentWithTools("查询珠海金山软件园附件的公交站，以及去的路线, 路线规划生成文档保存为当前目录的一个md 文件, 选择最近的一个大学并拿到大学的图片，打开浏览器展示该图片");

// 关闭 MCP 客户端
client.close();