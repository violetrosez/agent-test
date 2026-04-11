import { execFile } from 'node:child_process';
import { randomUUID } from 'node:crypto';
import { existsSync } from 'node:fs';
import { readFile, unlink, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { promisify } from 'node:util';
import ffmpegStatic from 'ffmpeg-static';

const execFileAsync = promisify(execFile);

function resolveFfmpegExecutable(): string {
    const fromEnv = process.env.FFMPEG_BIN?.trim();
    if (fromEnv && existsSync(fromEnv)) {
        return fromEnv;
    }
    const bundled =
        typeof ffmpegStatic === 'string' && ffmpegStatic.length > 0
            ? ffmpegStatic
            : '';
    if (bundled && existsSync(bundled)) {
        return bundled;
    }
    return 'ffmpeg';
}

/**
 * 将浏览器 MediaRecorder 产出的 WebM（常见为 Opus）转为 16kHz 单声道 PCM WAV，
 * 以匹配腾讯云一句话识别 EngSerViceType=16k_zh + VoiceFormat=wav。
 */
export async function webmTo16kMonoWav(input: Buffer): Promise<Buffer> {
    const ffmpeg = resolveFfmpegExecutable();
    const id = randomUUID();
    const inFile = join(tmpdir(), `asr-in-${id}.webm`);
    const outFile = join(tmpdir(), `asr-out-${id}.wav`);

    await writeFile(inFile, input);
    try {
        await execFileAsync(
            ffmpeg,
            [
                '-y',
                '-i',
                inFile,
                '-ar',
                '16000',
                '-ac',
                '1',
                '-c:a',
                'pcm_s16le',
                '-f',
                'wav',
                outFile,
            ],
            { maxBuffer: 32 * 1024 * 1024 },
        );
        return await readFile(outFile);
    } finally {
        await unlink(inFile).catch(() => undefined);
        await unlink(outFile).catch(() => undefined);
    }
}
