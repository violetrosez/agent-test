import { Module } from '@nestjs/common';
import { AiController } from './ai.controller';
import { AiService } from './ai.service';
import { ConfigService } from '@nestjs/config';
import { ChatOpenAI } from '@langchain/openai';
import { tool } from '@langchain/core/tools';
import { z } from 'zod';

@Module({
  controllers: [AiController],
  providers: [
    AiService,
    {
      provide: 'CHAT_MODEL',
      useFactory: (configService: ConfigService) => {
        return new ChatOpenAI({
          model: configService.get('MODEL_NAME'),
          apiKey: configService.get('MINIMAX_API_KEY'),
          configuration: {
            baseURL: configService.get('MINIMAX_BASE_URL'),
          },
        });
      },
      inject: [ConfigService],
    },
    {
      provide: 'WEB_SEARCH_TOOL',
      useFactory: (configService: ConfigService) => {
        const webSearchArgsSchema = z.object({
          query: z
            .string()
            .min(1)
            .describe('搜索关键词，例如：公司年报、某个事件等'),
          count: z
            .number()
            .int()
            .min(1)
            .max(20)
            .optional()
            .describe('返回的搜索结果数量，默认 10 条'),
        });

        return tool(
          async ({ query, count }: { query: string; count?: number }) => {
            const apiKey = configService.get<string>('TAVILY_API_KEY')?.trim();
            if (!apiKey) {
              return 'Tavily Web Search 的 API Key 未配置（环境变量 TAVILY_API_KEY），请先在服务端配置后再重试。';
            }

            // Tavily: https://api.tavily.com/search — Body 用 max_results，不是 count；响应为 { results: [{ title, url, content, ... }] }，不是 Bing 的 data.webPages
            const maxResults = Math.min(Math.max(count ?? 10, 1), 20);
            const body = {
              query,
              max_results: maxResults,
              search_depth: 'basic' as const,
              topic: 'general' as const,
              include_answer: true,
            };

            const response = await fetch('https://api.tavily.com/search', {
              method: 'POST',
              headers: {
                Authorization: `Bearer ${apiKey}`,
                'Content-Type': 'application/json',
              },
              body: JSON.stringify(body),
            });

            const rawText = await response.text();
            let json: Record<string, unknown>;
            try {
              json = JSON.parse(rawText) as Record<string, unknown>;
            } catch {
              return `搜索 API 返回非 JSON（${response.status}）：${rawText.slice(0, 500)}`;
            }

            if (!response.ok) {
              const detail = json as { detail?: { error?: string } };
              const msg =
                detail.detail?.error ??
                (typeof json === 'object' && json !== null && 'message' in json
                  ? String((json as { message: unknown }).message)
                  : rawText);
              return `搜索 API 请求失败（${response.status}）：${msg}`;
            }

            const results = (json.results as Array<{
              title?: string;
              url?: string;
              content?: string;
              score?: number;
            }>) ?? [];

            if (!results.length) {
              return '未找到相关结果。';
            }

            const answer =
              typeof json.answer === 'string' && json.answer.length > 0
                ? `【简要归纳】\n${json.answer}\n\n---\n\n`
                : '';

            const formatted = results
              .map((r, idx) => {
                const title = r.title ?? '（无标题）';
                const url = r.url ?? '';
                const snippet = r.content ?? '';
                const score =
                  typeof r.score === 'number' ? `相关度: ${r.score.toFixed(3)}` : '';
                return `[${idx + 1}] ${title}\nURL: ${url}\n${score ? `${score}\n` : ''}摘要: ${snippet}`;
              })
              .join('\n\n');

            return answer + formatted;
          },
          {
            name: 'web_search',
            description:
              '使用 Tavily 搜索互联网。参数：query 关键词；可选 count（1–20，默认 10）。返回简要归纳（若有）及多条结果的标题、URL、摘要。',
            schema: webSearchArgsSchema,
          },
        );
      },
      inject: [ConfigService],
    },
  ],
})
export class AiModule { }
