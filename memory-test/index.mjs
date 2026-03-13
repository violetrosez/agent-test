/**
 * 内存对话历史演示：使用 InMemoryChatMessageHistory 管理多轮对话
 * 对话只在内存中保留，程序重启后丢失
 */
import dotenv from "dotenv";
import { OpenAIEmbeddings, ChatOpenAI } from "@langchain/openai";
import { InMemoryChatMessageHistory } from "@langchain/core/chat_history";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";

dotenv.config();

// ---------- 大模型 ----------
const model = new ChatOpenAI({
    model: process.env.MODEL_NAME,
    apiKey: process.env.ALIYUN_API_KEY,
    temperature: 0.7,
    configuration: {
        baseURL: process.env.ALIYUN_BASE_URL,
    },
});

// ---------- 内存历史（程序结束后丢失） ----------
const history = new InMemoryChatMessageHistory();

const system_message = new SystemMessage(`你是一个友好、幽默的做菜助手，喜欢分享美食和烹饪技巧。`);
history.addMessage(system_message);

// ---------- 第一轮对话 ----------
const human_message1 = new HumanMessage("你今天吃了什么？");
history.addMessage(human_message1);



const response = await model.invoke(await history.getMessages());
console.log(response.content);

history.addMessage(response);

// ---------- 第二轮对话 ----------
const human_message2 = new HumanMessage("好不好吃啊？");
history.addMessage(human_message2);

const response2 = await model.invoke(await history.getMessages());

console.log(response2.content);
history.addMessage(response2);

// ---------- 打印所有历史消息 ----------
const allMessages = await history.getMessages();
console.log(`共保存了 ${allMessages.length} 条消息：`);
allMessages.forEach((msg, index) => {
    const type = msg.type;
    const prefix = type === 'human' ? '用户' : '助手';
    console.log(`  ${index + 1}. [${prefix}]: ${msg.content.substring(0, 50)}...`);
});


