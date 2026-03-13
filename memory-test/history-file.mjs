/**
 * 文件系统持久化记忆：对话历史保存到本地 JSON 文件，下次启动可恢复
 * 使用 FileSystemChatMessageHistory 自动读写 history.json
 */
import dotenv from "dotenv";
import { OpenAIEmbeddings, ChatOpenAI } from "@langchain/openai";
import { InMemoryChatMessageHistory } from "@langchain/core/chat_history";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { FileSystemChatMessageHistory } from "@langchain/community/stores/message/file_system";
import path from "path";

dotenv.config();

// ---------- 大模型 ----------
const model = new ChatOpenAI({
    model: process.env.MODEL_NAME,
    apiKey: process.env.ALIYUN_API_KEY,
    temperature: 0.2,
    configuration: {
        baseURL: process.env.ALIYUN_BASE_URL,
    },
});

/**
 * 文件持久化演示：两轮对话，每轮自动保存到 history.json
 */
const historyFileDemo = async () => {
    const sessionId = "1234567890";
    const filePath = path.join(process.cwd(), "history.json");

    // FileSystemChatMessageHistory 会自动创建/读取 JSON 文件
    const history = new FileSystemChatMessageHistory({
        filePath: filePath,
        sessionId: sessionId,
    });

    const system_message = new SystemMessage(`你是一个友好、幽默的做菜助手，喜欢分享美食和烹饪技巧。`);

    // ---------- 第一轮对话 ----------
    const human_message1 = new HumanMessage("红烧肉怎么做？");
    await history.addMessage(human_message1);
    const response1 = await model.invoke([system_message, ...await history.getMessages()]);
    console.log('助手返回:', response1.content);
    await history.addMessage(response1);
    console.log('对话已更新到文件中');
    console.log('--------------------------------');

    // ---------- 第二轮对话 ----------
    const human_message2 = new HumanMessage("好吃吗？");
    await history.addMessage(human_message2);
    const response2 = await model.invoke([system_message, ...await history.getMessages()]);
    console.log('助手返回:', response2.content);
    await history.addMessage(response2);
    console.log('对话已更新到文件中');
};

historyFileDemo();



