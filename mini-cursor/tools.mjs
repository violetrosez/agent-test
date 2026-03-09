import fs from "fs/promises";
import { exec } from "child_process";
import { promisify } from "util";
import { z } from "zod";
import { tool } from "@langchain/core/tools";

const execAsync = promisify(exec);

const MAX_OUTPUT_LEN = 3000;
const EXEC_TIMEOUT_MS = 5 * 60 * 1000;

function truncateOutput(str) {
    if (typeof str !== "string" || str.length <= MAX_OUTPUT_LEN) return str;
    return str.slice(0, MAX_OUTPUT_LEN - 50) + "\n\n...(输出已截断，仅保留前 " + (MAX_OUTPUT_LEN - 50) + " 字符)";
}


/**
 * 读取文件工具
 * @param {Object} params 
 * @param {string} params.path 文件路径
 * @returns {Promise<string>} 读取结果
 */
const read_file_tool = tool(
    async ({ path }) => {
        try {
            const content = await fs.readFile(path, "utf8");
            console.log(`read_file_tool function call successfully: ${path}`);
            return content;
        } catch (err) {
            const msg = err instanceof Error ? err.message : String(err);
            return `读取失败: ${msg}`;
        }
    },
    {
        name: "read_file",
        description: "读取文件内容",
        schema: z.object({
            path: z.string().describe("文件路径"),
        }),
    }
);

/**
 * 写入文件工具
 * @param {Object} params 
 * @param {string} params.path 文件路径
 * @param {string} params.content 文件内容
 * @returns {Promise<string>} 写入结果
 */
const write_file_tool = tool(
    async ({ path, content }) => {
        try {
            await fs.writeFile(path, content);
            console.log(`write_file_tool function call successfully: ${path}`);
            return content;
        } catch (err) {
            const msg = err instanceof Error ? err.message : String(err);
            return `写入失败: ${msg}`;
        }
    },
    {
        name: "write_file",
        description: "写入文件内容",
        schema: z.object({
            path: z.string().describe("文件路径"),
            content: z.string().describe("文件内容"),
        }),
    }
);

/**
 * 执行命令工具
 * @param {Object} params 
 * @param {string} params.command 命令
 * @param {string} params.directory 目录
 * @returns {Promise<string>} 执行结果
 */
const execute_command_tool = tool(
    async ({ command, directory }) => {
        try {
            const cwd = directory ?? process.cwd();
            const { stdout, stderr } = await execAsync(command, {
                cwd,
                shell: true,
                stdio: "inherit",
                maxBuffer: 2 * 1024 * 1024,
                timeout: EXEC_TIMEOUT_MS,
            });
            console.log(`execute_command_tool OK: ${command} in ${cwd}`);
            const raw = [stdout, stderr].filter(Boolean).join("\n").trim() || "(无输出)";
            return truncateOutput(raw);
        } catch (err) {
            const msg = err instanceof Error ? err.message : String(err);
            const isTimeout = err.code === "ETIMEDOUT" || msg.includes("timeout");
            return isTimeout
                ? `执行超时(${EXEC_TIMEOUT_MS / 1000}秒)，可能是常驻进程(如 dev server)。${msg}`
                : `执行失败: ${msg}`;
        }
    },
    {
        name: "execute_command",
        description: "执行命令",
        schema: z.object({
            command: z.string().describe("命令"),
            directory: z.string().describe("目录"),
        }),
    }
);

/**
 * 列出目录下的文件工具
 * @param {Object} params 
 * @param {string} params.directory 目录
 * @returns {Promise<string[]>} 文件列表
 */
const list_files_tool = tool(
    async ({ directory }) => {
        try {
            const files = await fs.readdir(directory);
            console.log(`list_files_tool function call successfully: ${directory}`);
            return files.join("\n");
        } catch (err) {
            const msg = err instanceof Error ? err.message : String(err);
            return `列出失败: ${msg}`;
        }
    },
    {
        name: "list_files",
        description: "列出目录下的文件",
        schema: z.object({
            directory: z.string().describe("目录"),
        }),
    }
);


export { read_file_tool, write_file_tool, execute_command_tool, list_files_tool };
