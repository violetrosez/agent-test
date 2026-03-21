/**
 * 书籍 RAG：本地 txt 按章节+长度切片 → 向量写入 Milvus → 用户问题检索 topK → 大模型回答
 * 能力：建集合/索引、导入书籍（readBookData + embedAndInsertInBatches）、问答（search + LLM）
 */
import { MilvusClient, MetricType } from "@zilliz/milvus2-sdk-node";
import dotenv from "dotenv";
import { OpenAIEmbeddings, ChatOpenAI } from "@langchain/openai";
import { PromptTemplate } from "@langchain/core/prompts";
import { RunnableLambda, RunnableSequence } from "@langchain/core/runnables";
import { StringOutputParser } from "@langchain/core/output_parsers";

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

// ---------- 嵌入模型（问题与书籍切片的向量化，维度与集合 vector 一致） ----------
const embeddings = new OpenAIEmbeddings({
    apiKey: process.env.ALIYUN_API_KEY,
    model: process.env.EMBEDDING_MODEL_NAME,
    configuration: {
        baseURL: process.env.ALIYUN_BASE_URL,
    },
    dimensions: vector_dim,
});

const milvusChain = RunnableLambda.from(
    async (input) => {
        const { query } = input;
        const queryVector = await embeddings.embedQuery(query);
        const queryResult = await client.search({
            collection_name,
            limit: 5,
            output_fields: ["id", "content", "book_id", "book_name", "chapter_num"],
            metric_type: MetricType.COSINE,
            vector: queryVector,
            params: { nprobe: 256 },
        });
        const results = queryResult.results ?? [];


        const retrievedContent = results.map((item, idx) => ({
            id: item.id,
            book_id: item.book_id,
            chapter_num: item.chapter_num,
            index: item.index ?? idx,
            content: item.content,
            score: item.score,
        }));

        return {
            retrievedContent,
            query
        };
    }
)


const buildPromptInput = RunnableLambda.from(
    async (response) => {
        const { retrievedContent, query } = response;
        if (!retrievedContent.length) {
            return {
                hasContext: false,
                query,
                context: "",
                retrievedContent,
            };
        }
        // 打印检索结果
        console.log("=".repeat(80));
        console.log(`问题: ${query}`);
        console.log("=".repeat(80));
        console.log("\n【检索相关内容】");
        retrievedContent.forEach((item, i) => {
            console.log(`\n[片段 ${i + 1}] 相似度: ${item.score ?? "N/A"}`);
            console.log(`书籍: ${item.book_id}`);
            console.log(`章节: 第 ${item.chapter_num} 章`);
            console.log(`片段索引: ${item.index}`);
            const content = item.content ?? "";
            console.log(
                `内容: ${content.substring(0, 200)}${content.length > 200 ? "..." : ""
                }`
            );
        });
        return {
            hasContext: true,
            context: retrievedContent.map((item) => `[${item.book_name} 第${item.chapter_num}章] ${item.content}`).join("\n"),
            query
        };
    }
)
async function main() {
    try {
        // ---------- 连接 Milvus ----------
        await client.connectPromise;


        // ---------- 加载集合到内存（建索引后需 load 才能检索） ----------
        await client.loadCollection({
            collection_name: collection_name
        });

        const promptTemplate = PromptTemplate.fromTemplate(
            `你是一个小说分析专家，根据以下小说内容，回答用户的问题。
        {context}
        {query}
        回答要求：
        1. 如果片段中有相关信息，请结合小说内容给出详细、准确的回答
        2. 可以综合多个片段的内容，提供完整的答案
        3. 如果片段中没有相关信息，请如实告知用户
        4. 回答要准确，符合小说的情节和人物设定
        5. 可以引用原文内容来支持你的回答
        `);

        const llmChain = RunnableSequence.from([milvusChain, buildPromptInput,
            RunnableLambda.from(async (input) => {
                const { hasContext, context, query } = input;
                if (!hasContext) {
                    console.log("没有检索到相关内容");
                    return {
                        noContext: true,
                        query: query,
                    };
                }
                return {
                    noContext: false,
                    context: context,
                    query: query,
                };
            }),
            promptTemplate,
            model,
            // 流式打印需要字符串增量；JsonOutputToolsParser 面向 tool_calls，chunk 常为 Array，不能 write 到 stdout
            new StringOutputParser(),
        ]);
        const streamResult = await llmChain.stream({ query: "令狐冲到结局掌握了多少门武功" });
        for await (const chunk of streamResult) {
            if (typeof chunk === "string" && chunk.length) process.stdout.write(chunk);
        }
        process.stdout.write("\n");

    } catch (error) {
        console.error(error);
    }
}

main();