/**
 * 书籍 RAG：本地 txt 按章节+长度切片 → 向量写入 Milvus → 用户问题检索 topK → 大模型回答
 * 能力：建集合/索引、导入书籍（readBookData + embedAndInsertInBatches）、问答（search + LLM）
 */
import { MilvusClient, DataType, IndexType, MetricType } from "@zilliz/milvus2-sdk-node";
import dotenv from "dotenv";
import { TextLoader } from "@langchain/classic/document_loaders/fs/text";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { OpenAIEmbeddings, ChatOpenAI } from "@langchain/openai";

dotenv.config();

// ---------- Milvus 客户端（地址见 .env MILVUS_ADDRESS） ----------
const client = new MilvusClient({
    address: process.env.MILVUS_ADDRESS
});

const collection_name = "xiaoaojianghu_book";
const vector_dim = 1024;

// ---------- 大模型（用于最终回答） ----------
const model = new ChatOpenAI({
    model: process.env.MODEL_NAME,
    apiKey: process.env.ALIYUN_API_KEY,
    temperature: 0.0,
    configuration: {
        baseURL: process.env.ALIYUN_BASE_URL,
    },
});

async function main() {
    try {
        // ---------- 连接 Milvus ----------
        await client.connectPromise;
        console.log("连接成功");

        const hasCollection = await client.hasCollection({
            collection_name: collection_name
        });

        // ---------- 若集合不存在则创建（id / book_id / book_name / chapter_num / vector / content）并建 IVF 索引 ----------
        if (!hasCollection.value) {
            await client.createCollection({
                collection_name: collection_name,
                fields: [
                    { name: 'id', data_type: DataType.VarChar, max_length: 50, is_primary_key: true },
                    { name: 'book_id', data_type: DataType.VarChar, max_length: 50, },
                    { name: 'book_name', data_type: DataType.VarChar, max_length: 50, },
                    { name: 'chapter_num', data_type: DataType.Int32 },
                    { name: 'vector', data_type: DataType.FloatVector, dim: vector_dim },
                    { name: 'content', data_type: DataType.VarChar, max_length: 5000 },
                ],
            });
            console.log("创建集合成功");
            await client.createIndex({
                collection_name: collection_name,
                field_name: "vector",
                index_type: IndexType.IVF_FLAT,
                metric_type: MetricType.COSINE,
                params: {
                    nlist: 1024,
                },
            });
        }

        // ---------- 加载集合到内存（建索引后需 load 才能检索） ----------
        await client.loadCollection({
            collection_name: collection_name
        });

        // ---------- 可选：导入书籍（读 txt → 按章+chunkSize 切片 → 分批向量化并插入） ----------
        // const chunks = await readBookData("1", "笑傲江湖", "./笑傲江湖.txt");
        // console.log("切片总数:", chunks.length);

        // await embedAndInsertInBatches({
        //     client,
        //     collection_name,
        //     chunks,
        //     bookId: "1",
        //     bookName: "笑傲江湖",
        // });
        // console.log("插入数据成功");

        // ---------- RAG 检索：问题向量化 → Milvus 查 topK 片段（需传 params.nprobe 时可在下方加） ----------
        const query = "令狐冲在第几章学会了孤独九剑";
        const queryVector = await embeddings.embedQuery(query);
        const queryResult = await client.search({
            collection_name,
            limit: 10,
            output_fields: ["id", "content", "book_id", "book_name", "chapter_num"],
            metric_type: MetricType.COSINE,
            vector: queryVector,
            params: { nprobe: 256 },
        });

        const raw = queryResult.results ?? [];
        const hits = raw.length > 0 && Array.isArray(raw[0]) ? raw[0] : raw;

        // ---------- 拼检索结果为上下文，再交给大模型生成回答 ----------
        // queryResult.results.forEach((item) => {
        //     console.log("id：", item.id);
        //     console.log("content：", item.content);
        //     console.log("book_id：", item.book_id);
        //     console.log("book_name：", item.book_name);
        //     console.log("chapter_num：", item.chapter_num);
        //     console.log("chapter_title：", item.chapter_title);
        //     console.log("--------------------------------");
        // });

        const context = hits.map((item) => `[${item.book_name} 第${item.chapter_num}章] ${item.content}`).join("\n");

        const prompt = `你是一个小说分析专家，根据以下小说内容，回答用户的问题。
        ${context}
        ${query}
        回答要求：
        1. 如果片段中有相关信息，请结合小说内容给出详细、准确的回答
        2. 可以综合多个片段的内容，提供完整的答案
        3. 如果片段中没有相关信息，请如实告知用户
        4. 回答要准确，符合小说的情节和人物设定
        5. 可以引用原文内容来支持你的回答
        `;
        const response = await model.invoke(prompt);
        console.log(response.content);

    } catch (error) {
        console.error(error);
    }

}

// ---------- 嵌入模型（问题与书籍切片的向量化，维度与集合 vector 一致） ----------
const embeddings = new OpenAIEmbeddings({
    apiKey: process.env.ALIYUN_API_KEY,
    model: process.env.EMBEDDING_MODEL_NAME,
    configuration: {
        baseURL: process.env.ALIYUN_BASE_URL,
    },
    dimensions: vector_dim,
});

/**
 * 先按章节拆分，再按 chunkSize 切片
 * 章节识别：第01章、第02章、第1章 等（正则 第\d+章）
 * @param {string} bookId - 书籍 ID
 * @param {string} bookName - 书籍名称
 * @param {string} filePath - 本地 txt 路径
 * @param {{ chunkSize?: number, chunkOverlap?: number }} options - 切片参数
 * @returns {Promise<import("@langchain/core/documents").Document[]>}
 */
const readBookData = async (bookId, bookName = "笑傲江湖", filePath = "./笑傲江湖.txt", options = {}) => {
    const { chunkSize = 400, chunkOverlap = 80 } = options;

    const loader = new TextLoader(filePath);
    const rawDocs = await loader.load();
    const fullText = rawDocs.map((d) => d.pageContent).join("\n");

    // 1. 按章节拆：匹配 "第01章 标题" 或 "第1章"（带捕获组时 split 会保留分隔符）
    const chapterRegex = /(第\d+章\s*[^\n]*)/g;
    const parts = fullText.split(chapterRegex).map((s) => s.trim()).filter(Boolean);

    const chapterDocs = [];
    for (let i = 0; i < parts.length; i++) {
        const part = parts[i];
        const chapterMatch = part.match(/^第(\d+)章\s*(.*)$/);
        if (chapterMatch) {
            const chapterNum = parseInt(chapterMatch[1], 10);
            const chapterTitle = chapterMatch[2].trim();
            const content = i + 1 < parts.length ? part + "\n" + parts[i + 1] : part;
            chapterDocs.push({
                pageContent: content,
                metadata: { book_id: bookId, book_name: bookName, chapter_num: chapterNum, chapter_title: chapterTitle },
            });
            i++; // 跳过下一段（本章正文）
        }
    }

    // 2. 每章再按 chunkSize 切
    const splitter = new RecursiveCharacterTextSplitter({
        chunkSize,
        chunkOverlap,
        separators: ["\n\n", "\n", "。", "？", "！", "；", " ", ""],
        keepSeparator: false,
    });

    const allChunks = [];
    for (const doc of chapterDocs) {
        const chunks = await splitter.splitDocuments([doc]);
        chunks.forEach((c, i) => {
            c.metadata = {
                ...c.metadata,
                book_id: bookId,
                book_name: doc.metadata.book_name,
                chapter_num: doc.metadata.chapter_num,
                chapter_title: doc.metadata.chapter_title ?? "",
                chunk_index: allChunks.length + i + 1,
            };
        });
        allChunks.push(...chunks);
    }
    return allChunks;
};

/** 每批条数，避免一次性大量请求嵌入接口导致卡住或限流 */
const EMBED_BATCH_SIZE = 10;

/**
 * 对切片逐批做向量化并插入 Milvus（一段一段插入，不一次性全量）
 * @param {object} params
 * @param {import("@zilliz/milvus2-sdk-node").MilvusClient} params.client - Milvus 客户端
 * @param {string} params.collection_name - 集合名
 * @param {Array<{ pageContent: string, metadata: object }>} params.chunks - readBookData 返回的切片
 * @param {string} params.bookId
 * @param {string} params.bookName
 */
const embedAndInsertInBatches = async ({ client, collection_name, chunks, bookId, bookName }) => {
    const total = chunks.length;
    for (let i = 0; i < total; i += EMBED_BATCH_SIZE) {
        const batch = chunks.slice(i, i + EMBED_BATCH_SIZE);
        const rows = await Promise.all(
            batch.map(async (chunk) => {
                const vector = await embeddings.embedQuery(chunk.pageContent);
                return {
                    id: `${bookId}_${chunk.metadata.chapter_num}_${chunk.metadata.chunk_index}`,
                    book_id: bookId,
                    book_name: bookName,
                    chapter_num: chunk.metadata.chapter_num,
                    content: chunk.pageContent.slice(0, 5000),
                    vector,
                };
            })
        );
        await client.insert({ collection_name, data: rows });
        // 每批插入后打进度，避免长任务无输出
        console.log(`已插入 ${Math.min(i + EMBED_BATCH_SIZE, total)} / ${total} 条`);
    }
};

main();