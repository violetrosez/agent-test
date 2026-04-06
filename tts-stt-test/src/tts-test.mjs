import dotenv from "dotenv";
import tencentcloud from "tencentcloud-sdk-nodejs-tts";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
// 无论从哪一级目录执行 node，都从项目根目录读 .env
dotenv.config({ path: path.join(__dirname, "..", ".env") });

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
// SecretId 多为 AKID 开头；纯数字一般是「主账号/AppId」，不能当 SecretId 用
if (/^\d+$/.test(secretId.trim())) {
    console.error(
        "当前 SECRET_ID 是纯数字，像是 AppId 而非 API 密钥。请到 https://console.cloud.tencent.com/cam/capi 创建或查看「访问密钥」，将「SecretId」与「SecretKey」成对填入 .env（不要把二者填反）。",
    );
    process.exit(1);
}

const TtsClient = tencentcloud.tts.v20190823.Client;

const client = new TtsClient({
    credential: {
        secretId,
        secretKey,
    },
    region: "ap-beijing",
    profile: {
        httpProfile: {
            endpoint: "tts.tencentcloudapi.com",
        },
    },
});

const params = {
    Text: "下班路上，我还在为晚霞开心。突然电话响起：系统崩了。我的心一下揪紧，冲进办公室时几乎要绝望。可当大家一起排查、重启，屏幕终于恢复正常，我长长松了口气，笑着说：还好，我们没放弃。",  // 要合成的文本
    SessionId: "session-001",
    VoiceType: 502006, // 101007：智瑜（女声）
    Codec: "mp3", // 指定输出格式为 mp3
};

client.TextToVoice(params).then(
    (data) => {
        // 返回的 Audio 字段是 Base64 编码的音频数据
        const audioBuffer = Buffer.from(data.Audio, "base64");
        const outputPath = "./output.mp3";

        fs.writeFile(outputPath, audioBuffer, (err) => {
            if (err) {
                console.error("保存文件失败：", err);
            } else {
                console.log("MP3 已保存至：", outputPath);
            }
        });
    },
    (err) => {
        console.error("合成失败：", err);
    }
);