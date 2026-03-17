import 'dotenv/config';
import { RunnableLambda, RunnableSequence, RunnableMap, RunnablePassthrough, RunnableEach } from "@langchain/core/runnables";

// const model = new ChatOpenAI({
//     model: process.env.MODEL_NAME,
//     apiKey: process.env.ALIYUN_API_KEY,
//     temperature: 0,
//     configuration: {
//         baseURL: process.env.ALIYUN_BASE_URL,
//     },
// });


// const add1 = RunnableLambda.from(x => x + 1);

// const multiplyBy2 = RunnableLambda.from(x => x * 2);

// const chain = RunnableSequence.from([add1, multiplyBy2]);

// const result = await chain.invoke(5);

// console.log(result);

// const map = RunnableMap.from({
//     add1: add1,
//     multiplyBy2: multiplyBy2,
// });

// const result2 = await map.invoke(3);

// console.log(result2);


// const chain = RunnableSequence.from([
//     RunnableLambda.from((input) => ({ concept: input })),
//     RunnableMap.from({
//         original: new RunnablePassthrough(),
//         processed: RunnableLambda.from((obj) => ({
//             concept: input,
//             upper: obj.concept.toUpperCase(),
//             length: obj.concept.length,
//         }))
//     })
// ]);

// const input = "神说要有光";
// const result = await chain.invoke(input);
// console.log(result);


const toUpperCase = RunnableLambda.from((input) => input.toUpperCase());
const addGreeting = RunnableLambda.from((input) => `你好，${input}！`);



// 使用 RunnableEach 对数组中的每个元素应用这个链
const chain = new RunnableEach({
    bound: toUpperCase.pipe(addGreeting),
});

const input = ["alice", "bob", "carol"];
const result = await chain.invoke(input);

console.log('✅ RunnableEach - 数组元素处理:');
console.log('输入:', input);
console.log('输出:', result);