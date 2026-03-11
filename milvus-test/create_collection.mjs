/**
 * 创建日记集合并写入示例数据
 * 流程：连接 Milvus → 创建集合 → 建向量索引 → 加载集合 → 插入日记（含向量）→ 简单检索验证
 */
import { MilvusClient, DataType, IndexType, MetricType } from "@zilliz/milvus2-sdk-node";
import dotenv from "dotenv";
import { OpenAIEmbeddings, ChatOpenAI } from "@langchain/openai";

dotenv.config();

const collection_name = "ai_diary";
const vector_dim = 1024;

// ---------- 大模型（本脚本仅用于验证，可选） ----------
const model = new ChatOpenAI({
    model: process.env.MODEL_NAME,
    apiKey: process.env.ALIYUN_API_KEY,
    temperature: 0.0,
    configuration: {
        baseURL: process.env.ALIYUN_BASE_URL,
    },
});

// ---------- 嵌入模型（文本 → 1024 维向量，与集合 vector 维度一致） ----------
const embeddings = new OpenAIEmbeddings({
    apiKey: process.env.ALIYUN_API_KEY,
    model: process.env.EMBEDDING_MODEL_NAME,
    configuration: {
        baseURL: process.env.ALIYUN_BASE_URL,
    },
    dimensions: vector_dim,
})

const get_embeddings = async (text) => embeddings.embedQuery(text);

// ---------- Milvus 客户端（地址从 .env 的 MILVUS_ADDRESS 读取） ----------
const client = new MilvusClient({
    address: process.env.MILVUS_ADDRESS
})

async function main() {
    // ---------- 连接 Milvus ----------
    await client.connectPromise;
    console.log("连接成功");

    // ---------- 创建集合（表结构：id / vector / content / date / mood / tags） ----------
    await client.createCollection({
        collection_name: collection_name,
        fields: [
            { name: 'id', data_type: DataType.VarChar, max_length: 50, is_primary_key: true },
            { name: 'vector', data_type: DataType.FloatVector, dim: vector_dim },
            { name: 'content', data_type: DataType.VarChar, max_length: 5000 },
            { name: 'date', data_type: DataType.VarChar, max_length: 50 },
            { name: 'mood', data_type: DataType.VarChar, max_length: 50 },
            { name: 'tags', data_type: DataType.Array, element_type: DataType.VarChar, max_capacity: 10, max_length: 50 }
        ],
    });
    console.log("创建集合成功");

    // ---------- 为 vector 字段创建 IVF_FLAT 索引（余弦相似度，nlist 聚类数） ----------
    await client.createIndex({
        collection_name: collection_name,
        field_name: "vector",
        index_type: IndexType.IVF_FLAT,
        metric_type: MetricType.COSINE,
        params: {
            nlist: 1024,
        },
    });
    console.log("创建索引成功");

    // ---------- 加载集合到内存（建索引后必须 load 才能检索） ----------
    console.log("\nLoading collection...");
    await client.loadCollection({ collection_name: collection_name });
    console.log("Collection loaded");

    // ---------- 插入日记数据：先对 content 做向量化，再写入 Milvus ----------
    console.log("\nInserting diary entries...");
    const diaryContents = [
        {
            id: 'diary_001',
            content: '今天天气很好，去野外钓鱼了，心情愉快。看到了很多花开了，春天真美好。',
            date: '2026-01-10',
            mood: 'happy',
            tags: ['生活', '户外']
        },
        {
            id: 'diary_002',
            content: '今天工作很忙，完成了一个重要的项目里程碑。团队合作很愉快，感觉很有成就感。',
            date: '2026-01-11',
            mood: 'excited',
            tags: ['工作', '成就']
        },
        {
            id: 'diary_003',
            content: '周末和朋友去爬山，天气很好，心情也很放松。享受大自然的感觉真好。',
            date: '2026-01-12',
            mood: 'relaxed',
            tags: ['户外', '朋友']
        },
        {
            id: 'diary_004',
            content: '今天学习了 Milvus 向量数据库，感觉很有意思。向量搜索技术真的很强大。',
            date: '2026-01-12',
            mood: 'curious',
            tags: ['学习', '技术']
        },
        {
            id: 'diary_005',
            content: '晚上做了一顿丰盛的晚餐，尝试了新菜谱。家人都说很好吃，很有成就感。',
            date: '2026-01-13',
            mood: 'proud',
            tags: ['美食', '家庭']
        }
    ];

    console.log('Generating embeddings...');
    const diaryData = await Promise.all(
        diaryContents.map(async (diary) => ({
            ...diary,
            vector: await get_embeddings(diary.content)
        }))
    );

    const insertResult = await client.insert({
        collection_name: collection_name,
        data: diaryData,
    });
    console.log("插入数据成功~ 插入数据条数：", insertResult.insert_cnt);

    // ---------- 简单检索验证：按“户外活动”语义查 top2（IVF 需传 nprobe 否则可能只返回 1 条） ----------
    const query = "我想看看关于户外活动的日记";
    const queryVector = await get_embeddings(query);
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

    hits.forEach((item) => {
        console.log("id：", item.id);
        console.log("content：", item.content);
        console.log("date：", item.date);
        console.log("mood：", item.mood);
        console.log("tags：", item.tags);
        console.log("--------------------------------");
    });
}

main();