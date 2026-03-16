/**
 * example-selector1.mjs
 *
 * 功能：演示「按长度自动选择 Few-Shot 示例」的周报生成场景。
 * 核心：用 LengthBasedExampleSelector 在示例池里按字符（或 token）总长度，
 *       自动挑选若干条示例塞进 prompt，既保证有示例可学，又不超过 maxLength，
 *       避免 prompt 过长导致超限或浪费。
 */

import 'dotenv/config';
import { ChatOpenAI } from '@langchain/openai';
import {
    FewShotPromptTemplate,
    PromptTemplate,
} from '@langchain/core/prompts';
import { LengthBasedExampleSelector } from '@langchain/core/example_selectors';

// ========== 1. 初始化 Chat 模型（本示例仅用其配置，实际调用在底部注释中） ==========
const model = new ChatOpenAI({
    model: process.env.MODEL_NAME,
    apiKey: process.env.ALIYUN_API_KEY,
    temperature: 0,
    configuration: {
        baseURL: process.env.ALIYUN_BASE_URL,
    },
});

// ========== 2. 单条示例的展示模板 ==========
// 每条 few-shot 示例会按这个模板渲染成一段文字（含「用户需求」+「周报片段」）
const examplePrompt = PromptTemplate.fromTemplate(
    `用户需求：{user_requirement}
周报片段示例：
{report_snippet}
---`
);

// ========== 3. 示例池：多条「需求 → 周报片段」示例，长度差异故意拉大 ==========
// 短示例（如第 3 条）占字符少，长示例（如第 4 条）占字符多；
// Selector 会按「当前已选总长度 + 下一条长度 ≤ maxLength」的规则逐条加入，直到放不下为止
const examples = [
    {
        user_requirement: '本周主要在做基础设施稳定性治理，想突出风险控制。',
        report_snippet:
            `- 核心链路共处理 P1 级别故障 1 起，P2 故障 2 起，均在 SLA 内完成处置；\n` +
            `- 对 5 个高风险接口补充了限流与熔断策略，覆盖 80% 高峰流量；\n` +
            `- 新增 6 条针对延迟抖动的告警规则，减少漏报风险。`,
    },
    {
        user_requirement: '偏向对外展示成果，多写一些亮点和业务价值。',
        report_snippet:
            `- 上线「实时订单看板」，支持业务实时查看转化漏斗；\n` +
            `- 打通埋点 → 数据仓库 → 实时服务的闭环，支撑后续精细化运营；\n` +
            `- 完成 2 场内部分享，会后收到 15 条正向反馈。`,
    },
    {
        user_requirement:
            '只是想要一个非常简短的周报，两三句话就够了，主要告诉老板「一切稳定」即可。',
        report_snippet: `本周整体运行平稳，未发生重大事故，核心指标均在预期范围内。`,
    },
    {
        user_requirement:
            '需要一份比较详细的技术周报，涵盖研发、测试、上线、监控等各个环节，篇幅可以略长。',
        report_snippet:
            `- 研发：完成结算服务重构第一阶段，拆分出 3 个独立子服务，接口延迟较旧架构下降约 35%；\n` +
            `- 测试：补齐 20+ 条关键路径自动化用例，整体用例数量提升到 180 条，回归时间从 2 天缩短到 0.5 天；\n` +
            `- 上线：采用灰度 + Canary 策略，期间监控到 2 次轻微指标抖动，均在 5 分钟内回滚处理；\n` +
            `- 监控：新增 8 条核心告警和 3 个 SLO 指标，后续会结合值班反馈继续收敛噪音告警。`,
    },
];

// ========== 4. 按长度选择器：从示例池中「能塞多少塞多少」，总长度不超过 maxLength ==========
// 内部逻辑：用 examplePrompt 把每条示例渲染成字符串，按 getTextLength 算长度，
// 从第一条开始累加，直到再加一条会超过 maxLength 就停止，返回已选中的示例列表
const exampleSelector = await LengthBasedExampleSelector.fromExamples(examples, {
    examplePrompt,
    maxLength: 700,                                    // 所有被选中的示例渲染后的总长度上限（字符数）
    getTextLength: (text) => text.length,               // 用字符数近似；生产环境可改为 token 计数
});

// ========== 5. Few-Shot 大模板：前缀 + 若干条「由 selector 选出的示例」+ 后缀 ==========
// 实际 format 时，selector 会先根据「当前输入」选出示例（本例未用输入参与选择，仅按长度），
// 再把选中的示例用 examplePrompt 填好，插在 prefix 和 suffix 之间
const fewShotPrompt = new FewShotPromptTemplate({
    examplePrompt,
    exampleSelector,                                     // 用选择器替代固定 example 列表，动态决定放几条
    prefix:                                             // 放在所有示例前面的说明
        '下面是一些不同风格和长度的周报片段示例，你可以从中学习语气和结构：\n',
    suffix:                                             // 放在所有示例后面，含本次要填的「当前需求」占位符
        '\n\n现在请根据上面的示例风格，为下面这个场景写一份新的周报：\n' +
        '场景描述：{current_requirement}\n' +
        '请输出一份适合发给老板和团队同步的 Markdown 周报草稿。',
    inputVariables: ['current_requirement'],             // 最终 prompt 里只有这一个变量（由调用方传入）
});

// ========== 6. 本次要生成周报的「当前需求」 ==========
const currentRequirement =
    '我们本周在做「内部 AI 助手」项目，既有稳定性保障（处理线上问题），' +
    '也有新功能上线（接入知识库、日志检索）。希望周报既能体现「把坑都兜住了」，' +
    '又能展示一部分业务侧能感知到的亮点。';


// 生成最终发给模型的完整 prompt：内部会先调用 selector 选示例，再拼成 prefix + 示例们 + suffix
const finalPrompt = await fewShotPrompt.format({
    current_requirement: currentRequirement,
});

console.log(finalPrompt);

// const stream = await model.stream(finalPrompt);
// console.log('\n=== AI 输出 ===');
// for await (const chunk of stream) {
//   process.stdout.write(chunk.content);
// }