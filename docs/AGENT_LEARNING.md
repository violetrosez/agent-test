# Agent / LangChain 学习地图（本仓库）

面向「把散落示例串成体系、方便回忆与整合」的索引说明。各子目录多为独立 `package.json`，运行前进入对应目录配置 `.env` 并执行 `node <文件>.mjs`（以各包为准）。

---

## 1. 建议的心智模型

| 层次 | 你在学什么 | 本仓库典型落点 |
|------|------------|----------------|
| 统一模型接口 | 用 `ChatOpenAI` 等屏蔽厂商差异（国产常走 OpenAI 兼容 baseURL） | 几乎所有示例 |
| LCEL / Runnable | `pipe`、`RunnableSequence`、`RunnableLambda` 组合数据流 | `runnable-test/`、`rag-test/` |
| 状态与分支 | `RunnablePassthrough.assign`、`RunnableBranch` 做多步 agent 状态 | `runnable-test/mcp.mjs` |
| 工具调用 | `bindTools` + `AIMessage.tool_calls` + `ToolMessage` 闭环 | `mini-cursor/`、`mcp-next/`、`runnable-test/mcp.mjs` |
| 外部工具协议 | MCP：`MultiServerMCPClient` 把远端/子进程工具变成 LangChain Tool | `mcp-next/`、`runnable-test/mcp.mjs` |
| 检索增强 | 向量库 + topK 片段 + prompt 上下文 | `rag-test/`、`book-search/`、`milvus-test/`、`runnable-test/milvus.mjs` |
| 记忆 | 多轮消息如何进 `messages` / 会话隔离 | `memory-test/`、`RunnableWithMessageHistory.mjs` |
| 可靠性与可观测 | 重试、降级、`config` 注入、`callbacks` | `RunnableWithRetry.mjs`、`RunnableWithFallbacks.mjs`、`RunnableWithConfig.mjs`、`test-callback.mjs` |

---

## 2. 目录速查表

| 路径 | 主题 | 要点 |
|------|------|------|
| `runnable-test/` | Runnable 核心与 LCEL agent | 见下节「Runnable 专题」 |
| `mcp-next/` | 多 MCP + 手写 while 工具循环 | `MultiServerMCPClient`、`bindTools`、消息数组 |
| `runnable-test/mcp.mjs` | **同一思路的 LCEL 版** agent | `RunnablePassthrough.assign` + `RunnableBranch` + `ToolMessage` |
| `mcp-test/` | **自建 MCP Server（stdio）** | `@modelcontextprotocol/sdk` 注册 tool，供别的客户端连 |
| `mini-cursor/` | 本地文件/命令类 **LangChain Tool** agent | `DynamicStructuredTool`、while 循环、无 MCP |
| `rag-test/` | 内存向量库 RAG | `MemoryVectorStore`、`Document`、检索后拼 prompt |
| `book-search/` | Milvus 书籍 RAG（建库、导入、问答） | Milvus SDK + `TextLoader` + `RecursiveCharacterTextSplitter` |
| `milvus-test/` | Milvus 更小粒度示例 | 集合/连接等 |
| `runnable-test/milvus.mjs` | Milvus 检索 + LLM 回答链 | `RunnableSequence`、`StringOutputParser` 流式输出 |
| `output-parser/` | 流式/结构化/工具调用解析 | `StringOutputParser`、`withStructuredOutput`、tool call 流等 |
| `memory-test/` | 对话历史、摘要、截断、落盘等 | `InMemoryChatMessageHistory` 等 |
| `prompt-template/` | Prompt 模板与占位符 | `ChatPromptTemplate`、`MessagesPlaceholder`、few-shot 等 |
| `split-chunk/` | 文本切分 | 与 RAG 数据准备相关 |
| `test-main/` | 杂项入口、smart-import、工具演示 | 与 `mini-cursor` 部分重叠 |

---

## 3. Runnable 专题（`runnable-test/`）

| 文件 | 作用 |
|------|------|
| `index.mjs` | 最简链：`PromptTemplate.pipe(model).pipe(StructuredOutputParser)`，Zod 结构化输出 |
| `RunnableLambda.mjs` | 自定义函数包装成 `Runnable` |
| `RunnableWithMessageHistory.mjs` | `RunnableWithMessageHistory`：`getMessageHistory(sessionId)` 隔离会话，`inputMessagesKey` / `historyMessagesKey` |
| `RunnableWithConfig.mjs` | `RunnableLambda` 第二参数 `config`：`config.configurable` 传 userId、role、locale 等（与 LangGraph/agent 里 configurable 思路一致） |
| `RunnableWithRetry.mjs` | `runnable.withRetry({ stopAfterAttempt })` |
| `RunnableWithFallbacks.mjs` | `runnable.withFallbacks({ fallbacks: [...] })` 顺序兜底 |
| `test-callback.mjs` | `invoke(..., { callbacks: [...] })` 观测链步骤的 start/end/error |
| `mcp.mjs` | **Agent**：检索 MCP 工具 → `assign({ response: llmchain })` → 有/无 `tool_calls` 分支 → 执行工具 → 写回 `messages`，外循环直到 `done` |
| `milvus.mjs` | **RAG + 流式**：Milvus 检索 → 拼 context → LLM；流式末端用 `StringOutputParser`（勿对 stdout 直接 write `JsonOutputToolsParser` 的 chunk） |

**记忆口诀**：`assign` 保留 state 并挂新字段；`Branch` 按条件选子链；工具轮 = 模型消息进历史 + `ToolMessage` 再进历史。

---

## 4. Agent 的两种实现对照

| 维度 | `mcp-next/index.mjs`（及类似手写循环） | `runnable-test/mcp.mjs` |
|------|----------------------------------------|-------------------------|
| 结构 | `while` + `messages` 数组显式更新 | `RunnableSequence` + `RunnableBranch` 单步，外层 `for` |
| 工具执行 | `tool.invoke(args)` → `ToolMessage` | `executeToolChain` 内同样 `invoke` |
| 易扩展 | 逻辑全在循环里 | 新步骤多往 `RunnableSequence` 里塞节点 |

`mini-cursor/mini-cursor.mjs` 则是 **不用 MCP**、工具全部在 Node 里用 `bindTools` 注册，适合理解「纯工具循环」。

---

## 5. MCP 相关

- **消费方（Client）**：`@langchain/mcp-adapters` 的 `MultiServerMCPClient`，`mcpServers` 里可写 `url`（HTTP）或 `command` + `args`（stdio 子进程）。
- **提供方（Server）**：`mcp-test/index.mjs` 用官方 SDK 注册 `registerTool`，适合练「自己暴露工具」。

注意：filesystem 类 MCP 工具通常要求**本地路径**；把 URL 当路径会触发服务端错误（你之前在终端里见过的 `read_media_file` 类问题）。

---

## 6. RAG 链路（本仓库的三档）

1. **教学向、零外部向量库**：`rag-test/index.mjs` — `MemoryVectorStore`。
2. **生产向、Milvus 整书**：`book-search/index.mjs` — 切分、写入、检索、生成。
3. **与 Runnable 组合**：`runnable-test/milvus.mjs` — 检索与 prompt 拼进 `RunnableSequence`，并演示 `stream`。

---

## 7. Output Parser / 流式

目录 `output-parser/` 建议按文件名记忆：

- `stream-normal.mjs` — `withStructuredOutput` + `stream`，chunk 为逐步完整的对象。
- `stream-structured-partial.mjs`、`stream-tool-calls-raw.mjs`、`stream-tool-calls-parser.mjs`、`tool-call-args.mjs` — 流式场景下工具调用与结构化片段的差异。
- **规则**：要往终端打印 token，链末端应能产出**字符串增量**（如 `StringOutputParser`）；`JsonOutputToolsParser` 等面向 tool/JSON，chunk 可能是数组或对象。

---

## 8. Memory（`memory-test/`）

- `index.mjs`：手动维护 `InMemoryChatMessageHistory`，理解 `getMessages()` 如何喂给 `model.invoke`。
- 同目录还有摘要、截断、文件持久化等脚本（见各文件头注释）。

与 `RunnableWithMessageHistory.mjs` 对比：后者把「按 sessionId 取 history」封装进链，invoke 时带 `configurable: { sessionId }`。

---

## 9. Prompt（`prompt-template/`）

- `ChatPromptTemplate` + `MessagesPlaceholder`：多轮对话占位与 `runnable-test` / agent 一致。
- `fewshot`、`example-selector`、`pipeline-prompt-template`：演示动态示例与模板组合。

---

## 10. 环境与运行提示

- 各子项目依赖见各自 `package.json`；API Key、base URL、Milvus 地址等多来自 `.env`。
- 模型能力差异：结构化输出、工具调用、流式 JSON 等需选用支持相应能力的模型名（示例里已有注释的以注释为准）。

---

## 11. 你可自行补充的一块（原消息未写完）

若你希望文档里固定强调某类内容，可把下面占位补全，或直接在本文追加一节：

> **我主要想强化记忆的是：** ___（例如：只记 MCP、只记 LCEL、只记 RAG 数据流、只记 stream 与 parser 区别等）___

---

## 12. 延伸阅读（与仓库主题一致）

- LangChain 文档中 **Runnable**、**LCEL**、**Tool calling**、**MCP** 章节。
- 多厂商 Chat API 差异（OpenAI / Anthropic / Gemini）可用统一 `BaseChatModel` 抽象对接——与你用 `ChatOpenAI` + 兼容网关的学习路径一致。

本文档随仓库示例更新；新增目录时只要在第 2 节表与第 3 节补充一行即可保持「地图」可用。
