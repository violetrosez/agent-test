/**
 * 基于 Milvus 日记集合的 RAG 问答
 * 流程：用户问题 → 向量检索 topK 日记 → 拼成上下文 → 大模型生成回答
 */
import { MilvusClient, DataType, IndexType, MetricType } from "@zilliz/milvus2-sdk-node";
import dotenv from "dotenv";
import { OpenAIEmbeddings, ChatOpenAI } from "@langchain/openai";

dotenv.config();

const collection_name = "ai_diary";
const vector_dim = 1024;

// ---------- 大模型（根据检索到的日记内容生成回答） ----------
const model = new ChatOpenAI({
    model: process.env.MODEL_NAME,
    apiKey: process.env.ALIYUN_API_KEY,
    temperature: 0.0,
    configuration: {
        baseURL: process.env.ALIYUN_BASE_URL,
    },
});

// ---------- 嵌入模型（问题与日记 content 的向量化，维度与集合一致） ----------
const embeddings = new OpenAIEmbeddings({
    apiKey: process.env.ALIYUN_API_KEY,
    model: process.env.EMBEDDING_MODEL_NAME,
    configuration: {
        baseURL: process.env.ALIYUN_BASE_URL,
    },
    dimensions: vector_dim,
})

const get_embeddings = async (text) => embeddings.embedQuery(text);

// ---------- Milvus 客户端 ----------
const client = new MilvusClient({
    address: process.env.MILVUS_ADDRESS
})

async function main() {
    await client.connectPromise;
    console.log("连接成功");

    // ---------- RAG：问题向量检索 → 拼上下文 → 大模型回答 ----------
    const answerQuestionByRAG = async (question) => {
        const queryVector = await get_embeddings(question);
        const queryResult = await client.search({
            collection_name: collection_name,
            limit: 2,
            output_fields: ["id", "content", "date", "mood", "tags"],
            metric_type: MetricType.COSINE,
            vector: queryVector,
            params: { nprobe: 256 },
        });
        const raw = queryResult.results ?? [];
        const hits = raw.length > 0 && Array.isArray(raw[0]) ? raw[0] : raw;

        const context = hits.map((item) => {
            return `[日记${item.id}] ${item.content}
            日期：${item.date}
            心情：${item.mood}
            标签：${Array.isArray(item.tags) ? item.tags.join(",") : item.tags ?? ""}
            内容：${item.content}
            --------------------------------
            `;
        }).join("\n");

        const prompt = `你是一个日记分析专家，根据以下日记内容，回答用户的问题。
        日记内容：${context}
        用户问题：${question}
        回答要求：
        1. 如果日记中有相关信息，请结合日记内容给出详细、温暖的回答
        2. 可以总结多篇日记的内容，找出共同点或趋势
        3. 如果日记中没有相关信息，请温和地告知用户
        4. 用第一人称"你"来称呼日记的作者
        5. 回答要有同理心，让用户感到被理解和关心
        `;

        const response = await model.invoke(prompt);
        console.log(response.content);
    };

    await answerQuestionByRAG("我什么时候学习了Milvus 向量数据库");

}

main();