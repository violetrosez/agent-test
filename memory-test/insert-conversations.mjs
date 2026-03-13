/**
 * 基于 Milvus 的对话记忆：对话向量化存储 + 语义检索 + 大模型回答
 * 流程：历史对话写入 Milvus → 新问题向量检索相关对话 → 拼上下文 → 大模型生成回答 → 新对话再存入 Milvus
 */
import "dotenv/config";
import { MilvusClient, DataType, MetricType, IndexType } from '@zilliz/milvus2-sdk-node';
import { OpenAIEmbeddings, ChatOpenAI } from "@langchain/openai";
import { InMemoryChatMessageHistory } from "@langchain/core/chat_history";
import { HumanMessage } from "@langchain/core/messages";

const COLLECTION_NAME = 'conversations';
const VECTOR_DIM = 1024;

// ---------- Milvus 客户端 ----------
const client = new MilvusClient({
    address: process.env.MILVUS_ADDRESS
});

// ---------- 大模型（根据检索到的历史对话生成回答） ----------
const model = new ChatOpenAI({
    model: process.env.MODEL_NAME,
    apiKey: process.env.ALIYUN_API_KEY,
    temperature: 0,
    configuration: {
        baseURL: process.env.ALIYUN_BASE_URL,
    },
});

// ---------- 嵌入模型（对话文本向量化，用于存储和检索） ----------
const embeddings = new OpenAIEmbeddings({
    apiKey: process.env.ALIYUN_API_KEY,
    model: process.env.EMBEDDING_MODEL_NAME,
    configuration: {
        baseURL: process.env.ALIYUN_BASE_URL,
    },
    dimensions: VECTOR_DIM,
});

async function main() {
    // ---------- 连接 Milvus ----------
    await client.connectPromise;
    console.log("连接成功");

    const hasCollection = await client.hasCollection({
        collection_name: COLLECTION_NAME
    });

    // ---------- 若集合不存在则创建（id / vector / content / round / timestamp）并建索引 ----------
    if (!hasCollection.value) {
        console.log("创建集合");
        await client.createCollection({
            collection_name: COLLECTION_NAME,
            fields: [
                { name: 'id', data_type: DataType.VarChar, max_length: 50, is_primary_key: true },
                { name: 'vector', data_type: DataType.FloatVector, dim: VECTOR_DIM },
                { name: 'content', data_type: DataType.VarChar, max_length: 5000 },
                { name: 'round', data_type: DataType.Int64 },
                { name: 'timestamp', data_type: DataType.VarChar, max_length: 100 }
            ],
        });
        console.log("创建集合成功");

        await client.createIndex({
            collection_name: COLLECTION_NAME,
            field_name: "vector",
            index_type: IndexType.IVF_FLAT,
            metric_type: MetricType.COSINE,
            params: {
                nlist: 1024,
            },
        });
        console.log("创建索引成功");

        // ---------- 加载集合到内存 ----------
        await client.loadCollection({
            collection_name: COLLECTION_NAME
        });
        console.log("加载集合成功");

        // ---------- 插入示例对话数据（首次运行时初始化） ----------
        console.log('插入对话数据...');
        const conversations = [
            {
                id: 'conv_001',
                content: '用户: 我叫赵六，是一名数据科学家\n助手: 很高兴认识你，赵六！数据科学是一个很有趣的领域。',
                round: 1,
                timestamp: new Date().toISOString()
            },
            {
                id: 'conv_002',
                content: '用户: 我最近在研究机器学习算法\n助手: 机器学习确实很有意思，你在研究哪些算法呢？',
                round: 2,
                timestamp: new Date().toISOString()
            },
            {
                id: 'conv_003',
                content: '用户: 我喜欢打篮球和看电影\n助手: 运动和文化娱乐都是很好的爱好！',
                round: 3,
                timestamp: new Date().toISOString()
            },
            {
                id: 'conv_004',
                content: '用户: 我周末经常去电影院\n助手: 看电影是很好的放松方式。',
                round: 4,
                timestamp: new Date().toISOString()
            },
            {
                id: 'conv_005',
                content: '用户: 我的职业是软件工程师\n助手: 软件工程师是个很有前景的职业！',
                round: 5,
                timestamp: new Date().toISOString()
            }
        ];

        const conversationData = await Promise.all(
            conversations.map(async (conversation) => {
                const vector = await embeddings.embedQuery(conversation.content);
                return {
                    ...conversation,
                    vector: vector
                };
            })
        );

        const insertResult = await client.insert({
            collection_name: COLLECTION_NAME,
            data: conversationData
        });
        console.log("插入数据成功~ 插入数据条数：", insertResult.insert_cnt);
    }

    // ---------- 测试：用新问题检索历史对话，拼上下文后大模型回答 ----------
    const history = new InMemoryChatMessageHistory();

    const conversations = [
        { input: "我之前提到的机器学习项目进展如何？" },
        { input: "我周末经常做什么？" },
        { input: "我的职业是什么？" },
    ];

    for (let i = 0; i < conversations.length; i++) {
        const conversation = conversations[i];

        console.log(`用户: ${conversation.input}\n`);
        const query = conversation.input;
        const userMessage = new HumanMessage(query);
        // 从 Milvus 检索语义相关的历史对话
        const retrievedConversations = await retrieveRelevantConversations(query);
        let relevantHistory = "";

        if (retrievedConversations.length > 0) {
            retrievedConversations.forEach((conv, idx) => {
                console.log(`\n[历史对话 ${idx + 1}] 相似度: ${conv.score.toFixed(4)}`);
                console.log(`轮次: ${conv.round}`);
                console.log(`内容: ${conv.content}`);
            });
        }

        relevantHistory = retrievedConversations.map((conv, idx) => `[历史对话 ${idx + 1}] 内容: ${conv.content}`).join('\n');

        const prompt = `请根据以下历史对话，回答用户的问题：
        ${relevantHistory}
        用户问题: ${conversation.input}
        `;
        // 调用大模型
        const response = await model.invoke(prompt);
        console.log('助手返回:\n', response.content);

        // 保存到内存历史
        await history.addMessage(userMessage);
        await history.addMessage(response);

        // 新对话也向量化并存入 Milvus，供下次检索
        const conversationText = `用户: ${userMessage.content}\n助手: ${response.content}`;
        const vector = await embeddings.embedQuery(conversationText);
        await client.insert({
            collection_name: COLLECTION_NAME,
            data: [{
                id: `conv_${Date.now()}_${i+1}`,
                vector: vector,
                content: conversationText,
                round: i + 1,
                timestamp: new Date().toISOString()
            }]
        });
        console.log(`已保存对话到 Milvus，插入数据条数：1`);
        
        console.log('--------------------------------');
    }

}

/**
 * 从 Milvus 检索与 query 语义相关的历史对话
 * @param {string} query - 用户当前问题
 * @returns {Promise<Array>} 相关对话数组（含 score、content、round、timestamp）
 */
async function retrieveRelevantConversations(query) {
    const queryVector = await embeddings.embedQuery(query);

    const searchResult = await client.search({
        collection_name: COLLECTION_NAME,
        limit: 3,
        output_fields: ["id", "content", "round", "timestamp"],
        metric_type: MetricType.COSINE,
        vector: queryVector,
        params: { nprobe: 256 },
    });

    return searchResult.results;
}

main();