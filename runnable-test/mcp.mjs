/**
 * 基于 LCEL 的多轮 Agent：连接多个 MCP 服务，让模型 bindTools 后循环「思考 → 工具 → 再思考」直到产出最终文本。
 */
import dotenv from "dotenv";
import { MultiServerMCPClient } from '@langchain/mcp-adapters';
import { ChatOpenAI } from '@langchain/openai';
import chalk from 'chalk';
import { HumanMessage, ToolMessage } from '@langchain/core/messages';
import { RunnableLambda, RunnablePassthrough, RunnableSequence, RunnableBranch } from '@langchain/core/runnables';
import { ChatPromptTemplate, MessagesPlaceholder } from '@langchain/core/prompts';

dotenv.config();

// 阿里云兼容 OpenAI 接口的聊天模型
const model = new ChatOpenAI({
  model: process.env.MODEL_NAME,
  apiKey: process.env.ALIYUN_API_KEY,
  temperature: 0.0,
  configuration: {
    baseURL: process.env.ALIYUN_BASE_URL,
  },
});

// 声明并启动多个 MCP 子进程 / HTTP 端点；getTools() 会把各 server 的工具合并成 LangChain DynamicStructuredTool 列表
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

const tools = await client.getTools();
// 将 MCP 工具 schema 绑定到模型，输出里可能出现 tool_calls
const model_with_tools = model.bindTools(tools);

// messages 占位符与 state.messages 对齐，由 invoke 时传入
const chatPrompt = ChatPromptTemplate.fromMessages([
  ["system", "你是一个可以调用MCP工具的AI助手, 请结合MCP工具和用户的问题, 给出最终的答案"],
  new MessagesPlaceholder('messages'),
])

// 单轮「模板填槽 → 带工具的 LLM」；输入需含 messages，输出为 AIMessage（可能含 tool_calls）
const llmchain = RunnableSequence.from([chatPrompt, model_with_tools]);

/**
 * 根据本轮 AIMessage 的 tool_calls 并行调用 MCP 工具，封装为 ToolMessage 列表。
 * 注意：filesystem 的 read_media_file 等需要本地路径；若模型传入 URL 会报错。
 */
const executeToolChain = RunnableLambda.from(async (state) => {
  const { response, tools } = state;
  const tool_calls = response.tool_calls;
  const toolMessages = await Promise.all(
    tool_calls.map(async (tc) => {
      const tool = tools.find((t) => t.name === tc.name);
      console.log(chalk.bgBlue(`🔧 调用工具: ${tc.name}, 参数: ${JSON.stringify(tc.args)}`));
      const result = await tool.invoke(tc.args);
      return new ToolMessage({ content: String(result), tool_call_id: tc.id });
    })
  );
  return toolMessages
})

/**
 * 单步 agent：
 * 1) assign({ response: llmchain }) 在保留 state 的同时，把 LLM 输出写入 response（llmchain 会消费 state.messages）
 * 2) Branch：无 tool_calls → 标记 done/final；有 tool_calls → 追加 AIMessage、执行工具、再追加 ToolMessage
 */
const agentStepChain = RunnableSequence.from([
  RunnablePassthrough.assign({
    response: llmchain
  }),
  RunnableBranch.from([
    // 模型直接回复（无工具调用）→ 结束本轮循环
    [
      ({ response }) => {
        return response.tool_calls?.length === 0;
      },
      RunnableLambda.from((state) => {
        return {
          ...state,
          final: state.response.content,
          done: true,
        }
      })
    ],
    RunnableSequence.from([
      // 将带 tool_calls 的 AIMessage 并入历史，供下一轮 llmchain 使用
      (state) => {
        const { response, messages } = state;
        const tool_calls = response.tool_calls;
        const newMessages = [...messages, response];
        console.log(chalk.bgBlue(`🔧 工具调用个数: ${tool_calls.length}`));

        return {
          ...state,
          messages: newMessages,
        }
      },
      // 在「已含本轮 AIMessage」的 state 上跑 executeToolChain（读 response.tool_calls）
      RunnablePassthrough.assign({
        toolMessages: executeToolChain,
      }),
      // 工具结果写回 messages，下一轮 invoke 时模型能看到工具输出
      RunnableLambda.from((state) => {
        const { toolMessages } = state;
        return {
          ...state,
          messages: [...state.messages, ...toolMessages],
        }
      })

    ])
  ])
]);

/**
 * 外循环：每轮跑一遍 agentStepChain，直到 done 或达到 maxIterations。
 * state.tools 固定传入，供 executeToolChain 查找并 invoke 对应 MCP 工具。
 */
async function runAgentWithTools(query, maxIterations = 30) {
  let state = {
    messages: [new HumanMessage(query)],
    done: false,
    final: null,
    tools,
  };

  for (let i = 0; i < maxIterations; i++) {
    console.log(chalk.bgGreen(`⏳ 正在等待 AI 思考...`));

    // 每一轮都通过一个完整的 Runnable chain（LLM + 工具调用处理）
    state = await agentStepChain.invoke(state);

    if (state.done) {
      console.log(`\n✨ AI 最终回复:\n${state.final}\n`);
      return state.final;
    }
  }

  // 未在轮次内结束则退回最后一条消息内容（兜底）
  return state.messages[state.messages.length - 1].content;
}


await runAgentWithTools("北京南站附近的酒店，最近的 3 个酒店，拿到酒店图片，打开浏览器，展示每个酒店的图片，每个 tab 一个 url 展示，并且在把那个页面标题改为酒店名");

// 释放 MCP 子进程 / 连接
client.close();