/**
 * 从文件恢复对话历史：读取已有的 history.json，接续进行第三轮对话
 * 演示 FileSystemChatMessageHistory 的持久化能力
 */
import 'dotenv/config';
import { ChatOpenAI } from '@langchain/openai';
import { FileSystemChatMessageHistory } from "@langchain/community/stores/message/file_system";
import { HumanMessage, AIMessage, SystemMessage } from "@langchain/core/messages";
import path from "node:path";

// ---------- 大模型 ----------
const model = new ChatOpenAI({
    model: process.env.MODEL_NAME,
    apiKey: process.env.ALIYUN_API_KEY,
    temperature: 0,
    configuration: {
        baseURL: process.env.ALIYUN_BASE_URL,
    },
});

/**
 * 从文件恢复历史，进行第三轮对话
 */
async function fileHistoryDemo() {
    const filePath = path.join(process.cwd(), "history.json");
    const sessionId = "1234567890";

    const systemMessage = new SystemMessage(
        "你是一个友好、幽默的做菜助手，喜欢分享美食和烹饪技巧。"
    );

    // ---------- 从文件恢复历史消息 ----------
    const restoredHistory = new FileSystemChatMessageHistory({
        filePath: filePath,
        sessionId: sessionId,
    });

    // 打印已恢复的历史
    const restoredMessages = await restoredHistory.getMessages();
    console.log(`从文件恢复了 ${restoredMessages.length} 条历史消息：`);
    restoredMessages.forEach((msg, index) => {
        const type = msg.type;
        const prefix = type === 'human' ? '用户' : '助手';
        console.log(`  ${index + 1}. [${prefix}]: ${msg.content.substring(0, 50)}...`);
    });

    // ---------- 第三轮对话（接续已有历史） ----------
    console.log("[第三轮对话]");
    const userMessage3 = new HumanMessage("需要哪些食材？");
    await restoredHistory.addMessage(userMessage3);

    // 拼接系统消息 + 历史消息，调用大模型
    const messages3 = [systemMessage, ...(await restoredHistory.getMessages())];
    const response3 = await model.invoke(messages3);
    await restoredHistory.addMessage(response3);

    console.log(`用户: ${userMessage3.content}`);
    console.log(`助手: ${response3.content}`);
    console.log(`✓ 对话已保存到文件`);
}

fileHistoryDemo().catch(console.error);
