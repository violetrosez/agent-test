/**
 * RAG 示例：基于本地文档的检索增强生成
 * 流程：文档向量化 → 按问题检索相关片段 → 拼成上下文 → 大模型生成回答
 */
import dotenv from "dotenv";
import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { Document } from "@langchain/core/documents";
import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";

dotenv.config();

// ---------- 大模型（用于最终回答） ----------
const model = new ChatOpenAI({
    model: process.env.MODEL_NAME,
    apiKey: process.env.ALIYUN_API_KEY,
    temperature: 0.0,
    configuration: {
        baseURL: process.env.ALIYUN_BASE_URL,
    },
});

// ---------- 嵌入模型（用于把文本转成向量，供检索） ----------
const embeddings = new OpenAIEmbeddings({
    apiKey: process.env.ALIYUN_API_KEY,
    model: process.env.EMBEDDING_MODEL_NAME,
    configuration: {
        baseURL: process.env.ALIYUN_BASE_URL,
    },
});

// ---------- 加载 PDF：先整篇加载，再按自定义分隔符分块 ----------
const loader = new PDFLoader("./test.pdf", { splitPages: false });
const rawDocs = await loader.load();

// 自定义分隔符（按优先级尝试）：段落 → 换行 → 中文句号/问号/叹号 → 空格 → 单字
const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 500,
    chunkOverlap: 50,
    separators: ["\n\n", "\n", "。", "？", "！", "；", " ", ""],
    keepSeparator: false,
});

const docs = await splitter.splitDocuments(rawDocs);
// console.log(`共 ${docs.length} 个块`);
// docs.forEach((d, i) => console.log(`[${i + 1}] ${d.pageContent.slice(0, 80)}...`));

const vectorStore = await MemoryVectorStore.fromDocuments(docs, embeddings);

const retriever = vectorStore.asRetriever({ k: 3 });

const question = "建模软件和算量软件有什么区别";
const indexedDocs = await retriever.invoke(question);

// for (const doc of docs) {
//     const scoredResult = indexedDocs.find(d => d.pageContent === doc.pageContent);
//     console.log(scoredResult);
//     const similarity = scoredResult ? (1 - scoredResult.score).toFixed(4) : "N/A";
//     console.log(`相似度: ${similarity}`);
//     console.log(`内容: ${doc.pageContent}`);
//     console.log(`元数据: ${JSON.stringify(doc.metadata)}`);
//     console.log("--------------------------------");
// }

const context = indexedDocs.map((doc, i) => `[片段${i + 1}] ${doc.pageContent}`).join("\n");

const prompt = `根据提供的BIM资料内容回答我的问题"。

BIM资料内容:
${context}

问题: ${question}

回答:
`;

const response = await model.invoke(prompt);
console.log(response.content);