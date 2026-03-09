---
name: init-llm-project
description: Initializes a new folder for LLM (大模型) development with Node.js, LangChain, and MCP. Use when the user wants to scaffold or initialize a project for 大模型开发、LangChain、MCP、agent 开发.
---

# 大模型开发项目初始化

在**目标文件夹**（用户指定或当前工作目录）内按顺序执行以下步骤。

## 1. 初始化 package.json

```bash
npm init -y
```

## 2. 安装依赖

```bash
npm install @langchain/core @langchain/mcp-adapters @langchain/openai chalk dotenv
```

## 3. 新建 index.mjs

在项目根目录创建 `index.mjs`，内容为下方模板。包含引入、初始化及可选的 runAgentWithTools 示例循环，具体调用由用户编写。

```javascript
import dotenv from "dotenv";
import { MultiServerMCPClient } from "@langchain/mcp-adapters";
import { ChatOpenAI } from "@langchain/openai";
import chalk from "chalk";
import { HumanMessage, ToolMessage } from "@langchain/core/messages";

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
        // 按需添加，例如 "filesystem": { command: "npx", args: ["-y", "@modelcontextprotocol/server-filesystem", process.cwd()] },
    },
});

const tools = await client.getTools();
const model_with_tools = model.bindTools(tools);
const toolMap = Object.fromEntries(tools.map((t) => [t.name, t]));

async function runAgentWithTools(query, maxRounds = 10) {
    let messages = [new HumanMessage(`${query}`)];

    for (let i = 0; i < maxRounds; i++) {
        console.log(chalk.bgGreen(`⏳ 正在等待 AI 思考...`));
        const response = await model_with_tools.invoke(messages);
        if (!response.tool_calls?.length) {
            console.log(chalk.bgGreen(`🤖 AI 回复: ${response.content}`));
            return response.content;
        }
        messages.push(response);
        const toolMessages = await Promise.all(
            response.tool_calls.map(async (tc) => {
                const t = toolMap[tc.name];
                const args = typeof tc.args === "string" ? (() => { try { return JSON.parse(tc.args); } catch { return {}; } })() : (tc.args ?? {});
                console.log(chalk.bgBlue(`🔧 调用工具: ${tc.name}, 参数: ${JSON.stringify(args)}`));
                const result = t ? await t.invoke(args) : `未知工具: ${tc.name}`;
                console.log(chalk.bgYellow(`🔧 工具返回: ${JSON.stringify(result)}`));
                return new ToolMessage({ content: String(result), tool_call_id: tc.id });
            })
        );
        messages.push(...toolMessages);
    }
}

// 在此处调用，例如: await runAgentWithTools("你的问题");
```

## 4. 新建 .env

在项目根目录创建 `.env`，内容为下方模板。请提醒用户将占位符替换为真实配置，且不要提交真实密钥到仓库。

```env
# 阿里云 API 密钥
ALIYUN_API_KEY=your_aliyun_api_key
# 阿里云 API 地址
ALIYUN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
# 模型名称
MODEL_NAME=qwen3-coder-next
# 高德地图 API 密钥（若使用高德 MCP 再填）
AMAP_API_KEY=your_amap_api_key
```

## 可选：package.json 脚本

可在 `package.json` 的 `scripts` 中增加启动命令，例如：

```json
"scripts": {
  "start": "node index.mjs"
}
```

## 检查清单

- [ ] 已在目标目录执行 `npm init -y`
- [ ] 已安装上述 5 个依赖
- [ ] 已创建 `index.mjs`（含引入、初始化与可选 runAgentWithTools 示例）
- [ ] 已创建 `.env` 并提醒用户填写真实配置
