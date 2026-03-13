/**
 * 截断策略记忆：当消息过多时，丢弃最旧的消息，只保留最近 N 条或 N token
 * 包含两种策略：按消息数量截断、按 token 数量截断
 */
import { InMemoryChatMessageHistory } from "@langchain/core/chat_history";
import { HumanMessage, AIMessage, trimMessages } from "@langchain/core/messages";
import dotenv from "dotenv";
import { getEncoding } from "js-tiktoken";
import { ChatOpenAI } from "@langchain/openai";

dotenv.config();

// ---------- 大模型（本例仅用于演示，实际可不用） ----------
const model = new ChatOpenAI({
    model: process.env.MODEL_NAME,
    apiKey: process.env.ALIYUN_API_KEY,
    temperature: 0.7,
    configuration: {
        baseURL: process.env.ALIYUN_BASE_URL,
    },
});

/**
 * 策略一：按消息数量截断，保留最近 maxMessages 条
 */
async function messageCountTruncation() {
    const history = new InMemoryChatMessageHistory();
    const maxMessageCount = 4; // 只保留最近 4 条消息

    // ---------- 模拟 10 条对话 ----------
    const messages = [
        { type: 'human', content: '我叫张三' },
        { type: 'ai', content: '你好张三，很高兴认识你！' },
        { type: 'human', content: '我今年25岁' },
        { type: 'ai', content: '25岁正是青春年华，有什么我可以帮助你的吗？' },
        { type: 'human', content: '我喜欢编程' },
        { type: 'ai', content: '编程很有趣！你主要用什么语言？' },
        { type: 'human', content: '我住在北京' },
        { type: 'ai', content: '北京是个很棒的城市！' },
        { type: 'human', content: '我的职业是软件工程师' },
        { type: 'ai', content: '软件工程师是个很有前景的职业！' },
    ];

    for (const message of messages) {
        if (message.type === 'human') {
            await history.addMessage(new HumanMessage(message.content));
        } else {
            await history.addMessage(new AIMessage(message.content));
        }
    }

    const allMessages = await history.getMessages();
    console.log('所有消息:', allMessages.map(msg => msg.content));

    // 截断：只保留最后 maxMessageCount 条
    const truncatedMessages = allMessages.slice(-maxMessageCount);

    console.log('保留的消息条数:', truncatedMessages.length);
    console.log('截断后的消息:\n', truncatedMessages.map(msg => msg.content).join('\n'));
    console.log('--------------------------------');


}

/**
 * 计算消息数组的 token 总数
 */
async function countTokens(messages, encoder) {
    let totalTokens = 0;
    for (const message of messages) {
        const tokens = encoder.encode(message.content);
        totalTokens += tokens.length;
    }
    return totalTokens;
}

/**
 * 策略二：按 token 数量截断，保留最近 maxTokenCount token 以内的消息
 */
async function tokenCountTruncation() {
    const history = new InMemoryChatMessageHistory();
    const maxTokenCount = 100; // 只保留最近 100 token 以内的消息

    const encoder = getEncoding("cl100k_base");

    // ---------- 模拟 8 条对话 ----------
    const messages = [
        { type: 'human', content: '我叫李四' },
        { type: 'ai', content: '你好李四，很高兴认识你！' },
        { type: 'human', content: '我是一名设计师' },
        { type: 'ai', content: '设计师是个很有创造力的职业！你主要做什么类型的设计？' },
        { type: 'human', content: '我喜欢艺术和音乐' },
        { type: 'ai', content: '艺术和音乐都是很好的爱好，它们能激发创作灵感。' },
        { type: 'human', content: '我擅长 UI/UX 设计' },
        { type: 'ai', content: 'UI/UX 设计非常重要，好的用户体验能让产品更成功！' },
    ];
    for (const message of messages) {
        if (message.type === 'human') {
            await history.addMessage(new HumanMessage(message.content));
        } else {
            await history.addMessage(new AIMessage(message.content));
        }
    }
    const allMessages = await history.getMessages();

    // 从后往前累加 token，找到截断点
    let totalTokens = 0;
    let _index = 0;
    for (let index = allMessages.length - 1; index >= 0; index--) {
        const element = allMessages[index];
        const tokens = encoder.encode(element.content);
        // console.log('tokens:', tokens.length, totalTokens);
        if (totalTokens + tokens.length > maxTokenCount) {
            _index = index + 1;
            break;
        }
        totalTokens += tokens.length;
    }


    const remainingMessages = allMessages.slice(_index);
    console.log('保留的消息条数:', remainingMessages.length);
    console.log('总Token数:', totalTokens);
    console.log('截断后的消息:\n', remainingMessages.map(msg => msg.content).join('\n'));
    // console.log('--------------------------------');

    // // 使用 trimMessages API：使用 js-tiktoken 计算 token 数量
    // const trimmedMessages = await trimMessages(allMessages, {
    //     maxTokens: maxTokenCount,
    //     tokenCounter: async (msgs) => countTokens(msgs, encoder),
    //     strategy: "last"
    // });

    // // 计算实际 token 数用于显示
    // const totalTokens = await countTokens(trimmedMessages, encoder);

    // console.log(`总 token 数: ${totalTokens}/${maxTokenCount}`);
    // console.log(`保留消息数量: ${trimmedMessages.length}`);
    // console.log("保留的消息:", trimmedMessages.map(m => {
    //     const content = typeof m.content === 'string' ? m.content : JSON.stringify(m.content);
    //     const tokens = encoder.encode(content).length;
    //     return `${m.constructor.name} (${tokens} tokens): ${content}`;
    // }).join('\n  '));

}

// messageCountTruncation();
tokenCountTruncation();

