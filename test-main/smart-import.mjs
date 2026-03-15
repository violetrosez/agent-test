import 'dotenv/config';
import { ChatOpenAI } from '@langchain/openai';
import { z } from 'zod';
import mysql from 'mysql2/promise';

const model = new ChatOpenAI({
    model: process.env.MODEL_NAME,
    apiKey: process.env.ALIYUN_API_KEY,
    temperature: 0,
    configuration: {
        baseURL: process.env.ALIYUN_BASE_URL,
    },
})

// 定义单个好友信息的 zod schema，匹配 friends 表结构
const friendSchema = z.object({
    name: z.string().describe('姓名'),
    gender: z.string().describe('性别（男/女）'),
    birth_date: z.string().describe('出生日期，格式：YYYY-MM-DD，如果无法确定具体日期，根据年龄估算'),
    company: z.string().nullable().describe('公司名称，如果没有则返回 null'),
    title: z.string().nullable().describe('职位/头衔，如果没有则返回 null'),
    phone: z.string().nullable().describe('手机号，如果没有则返回 null'),
    wechat: z.string().nullable().describe('微信号，如果没有则返回 null'),
});

const friendListSchema = z.array(friendSchema).describe('好友列表');

const modelWithOutput = model.withStructuredOutput(friendListSchema);


// 数据库连接配置
const connectionConfig = {
    host: 'localhost',
    port: 3307,
    user: 'root',
    password: 'admin',
    multipleStatements: true,
};

async function getFriendList(text) {
    const connection = await mysql.createConnection(connectionConfig);
    try {
        await connection.query('USE hello;');



    const prompt = `请从以下文本中提取所有好友信息，文本中可能包含一个或多个人的信息。请将每个人的信息分别提取出来，返回一个数组。
${text}
要求：
1. 如果文本中包含多个人，请为每个人创建一个对象
2. 每个对象包含以下字段：
- 姓名：提取文本中的人名
- 性别：提取性别信息（男 / 女）
- 出生日期：如果能找到具体日期最好，否则根据年龄描述估算（格式：YYYY - MM - DD）
- 公司：提取公司名称
- 职位：提取职位 / 头衔信息
- 手机号：提取手机号码
- 微信号：提取微信号
3. 如果某个字段在文本中找不到，请返回 null
4. 返回格式必须是一个数组，即使只有一个人也要放在数组中`;

    const response = await modelWithOutput.invoke(prompt);

    console.log(response);
    // 转为行数组：不要用 async map，否则 values 是 Promise[]，mysql2 无法识别
    const values = response.map((item) => [
        item.name,
        item.gender,
        item.birth_date,
        item.company,
        item.title,
        item.phone,
        item.wechat,
    ]);
    const [insertResult] = await connection.query('INSERT INTO friends (name, gender, birth_date, company, title, phone, wechat) VALUES ?', [values]);

    console.log(`✅ 成功批量插入 ${insertResult.affectedRows} 条数据`);
    console.log(`   插入的ID范围：${insertResult.insertId} - ${insertResult.insertId + insertResult.affectedRows - 1}`);

        return {
            count: insertResult.affectedRows,
            insertIds: Array.from({ length: insertResult.affectedRows }, (_, i) => insertResult.insertId + i),
        };
    } finally {
        await connection.end();
    }
}
const sampleText = `我最近认识了几个新朋友。第一个是张总，女的，看起来30出头，在腾讯做技术总监，手机13800138000，微信是zhangzong2024。第二个是李工，男，大概28岁，在阿里云做架构师，电话15900159000，微信号lee_arch。还有一个是陈经理，女，35岁左右，在美团做产品经理，手机号是18800188000，微信chenpm2024。`;
getFriendList(sampleText).catch((err) => {
    console.error('执行失败:', err);
    process.exitCode = 1;
});