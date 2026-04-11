import { BadRequestException, Inject, Injectable } from '@nestjs/common';
import type * as tencentcloud from 'tencentcloud-sdk-nodejs';
import { webmTo16kMonoWav } from './audio-transcode.util';

type UploadedAudio = {
    buffer: Buffer;
    originalname: string;
    mimetype: string;
    size: number;
};

type AsrClient = InstanceType<typeof tencentcloud.asr.v20190614.Client>;

@Injectable()
export class SpeechService {
    constructor(@Inject('ASR_CLIENT') private readonly asrClient: AsrClient) { }

    private isWebm(file: UploadedAudio): boolean {
        const mime = (file.mimetype ?? '').toLowerCase();
        const name = (file.originalname ?? '').toLowerCase();
        return mime.includes('webm') || name.endsWith('.webm');
    }

    private voiceFormatNonWebm(file: UploadedAudio): string {
        const mime = (file.mimetype ?? '').toLowerCase();
        const name = (file.originalname ?? '').toLowerCase();
        if (mime.includes('ogg') || name.endsWith('.ogg')) return 'ogg-opus';
        if (mime.includes('mpeg') || name.endsWith('.mp3')) return 'mp3';
        if (mime.includes('wav') || name.endsWith('.wav')) return 'wav';
        if (
            mime.includes('mp4') ||
            mime.includes('m4a') ||
            name.endsWith('.m4a')
        ) {
            return 'm4a';
        }
        return 'ogg-opus';
    }

    async recognizeBySentence(file: UploadedAudio): Promise<string> {
        let pcmBuffer = file.buffer;
        let voiceFormat = this.voiceFormatNonWebm(file);

        if (this.isWebm(file)) {
            try {
                pcmBuffer = await webmTo16kMonoWav(file.buffer);
                voiceFormat = 'wav';
            } catch (e) {
                const hint =
                    e instanceof Error ? e.message : String(e);
                throw new BadRequestException(
                    `WebM 转 WAV 失败（需本机或依赖包内提供 ffmpeg）。${hint}。` +
                        `可将 FFMPEG_BIN 指向 ffmpeg 可执行文件；若使用 pnpm 且跳过了依赖脚本，请执行：pnpm rebuild ffmpeg-static 或 node node_modules/ffmpeg-static/install.js。`,
                );
            }
        }

        const audioBase64 = pcmBuffer.toString('base64');

        const result = await this.asrClient.SentenceRecognition({
            EngSerViceType: '16k_zh',
            SourceType: 1,
            Data: audioBase64,
            DataLen: pcmBuffer.length,
            VoiceFormat: voiceFormat,
        });

        return result.Result ?? '';
    }
}
