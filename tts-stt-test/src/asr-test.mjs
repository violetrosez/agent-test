import dotenv from "dotenv";
import tencentcloud from "tencentcloud-sdk-nodejs";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const rootDir = path.join(__dirname, "..");
dotenv.config({ path: path.join(rootDir, ".env") });

const secretId =
    process.env.TENCENTCLOUD_SECRET_ID ?? process.env.SECRET_ID ?? "";
const secretKey =
    process.env.TENCENTCLOUD_SECRET_KEY ?? process.env.SECRET_KEY ?? "";

if (!secretId || !secretKey) {
    console.error(
        "缺少密钥：请在项目根目录 .env 中设置 TENCENTCLOUD_SECRET_ID 与 TENCENTCLOUD_SECRET_KEY（或 SECRET_ID / SECRET_KEY）。",
    );
    process.exit(1);
}
if (/^\d+$/.test(secretId.trim())) {
    console.error(
        "当前 SecretId 是纯数字，像是 AppId。请到 https://console.cloud.tencent.com/cam/capi 使用正确的密钥对。",
    );
    process.exit(1);
}

const AsrClient = tencentcloud.asr.v20190614.Client;
const AUDIO_FILE = path.join(rootDir, "output.mp3");

const client = new AsrClient({
    credential: {
        secretId,
        secretKey,
    },
    region: "ap-shanghai",
    profile: {
        httpProfile: {
            reqMethod: "POST",
            reqTimeout: 30,
        },
    },
});

async function run() {
    if (!fs.existsSync(AUDIO_FILE)) {
        console.error(`找不到音频文件：${AUDIO_FILE}，请先合成或指定路径。`);
        process.exit(1);
    }

    const audioBase64 = fs.readFileSync(AUDIO_FILE).toString("base64");

    const params = {
        EngSerViceType: "16k_zh",
        SourceType: 1,
        Data: audioBase64,
        DataLen: Buffer.byteLength(audioBase64),
        VoiceFormat: "mp3",
    };

    try {
        const data = await client.SentenceRecognition(params);
        console.log("识别结果：", data.Result);
    } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        console.error("识别失败：", err);
        if (/unopened|ASR_OneSentence/i.test(msg)) {
            console.error(`
【说明】User is unopened / ASR_OneSentence：当前账号尚未开通「语音识别」里的「一句话识别」能力，与代码、SecretId 是否正确无关。
请到腾讯云控制台开通语音识别产品并完成实名/计费相关流程，例如：
https://console.cloud.tencent.com/asr
在控制台搜索「一句话识别」或「录音文件识别极速版」等产品说明，开通后再调用 SentenceRecognition。`);
        }
    }
}

run();
