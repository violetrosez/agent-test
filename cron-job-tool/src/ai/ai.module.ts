import { Module } from '@nestjs/common';
import { AiService } from './ai.service';
import { AiController } from './ai.controller';
import { ConfigService } from '@nestjs/config';
import { ChatOpenAI } from '@langchain/openai';
import { tool } from '@langchain/core/tools';
import { UserService } from './user.service';
import { z } from 'zod';
import { MailerService } from '@nestjs-modules/mailer';

@Module({
  // MailerModule.forRootAsync 在 AppModule 里已注册，且 MailerCoreModule 为 global，此处不要再把 MailerService 放进 providers
  controllers: [AiController],
  providers: [
    UserService,
    AiService,
    {
      provide: 'CHAT_MODEL',
      useFactory: (configService: ConfigService) => {
        return new ChatOpenAI({
          model: configService.get('MODEL_NAME'),
          apiKey: configService.get('ALIYUN_API_KEY'),
          temperature: 0,
          configuration: {
            baseURL: configService.get('ALIYUN_BASE_URL'),
          },
        });
      },
      inject: [ConfigService],
    }, {
      provide: 'QUERY_USER_TOOL',
      useFactory: (userService: UserService) => {
        const queryUserArgsSchema = z.object({
          userId: z.string().describe('用户 ID，例如: 001, 002, 003'),
        });
        return tool(
          ({ userId }: { userId: string }): string => {

            const user = userService.findOne(userId);
            if (!user) {
              const availableIds = userService
                .findAll()
                .map((u) => u.id)
                .join(', ');

              return `用户 ID ${userId} 不存在。可用的 ID: ${availableIds}`;
            }

            return `用户信息：\n- ID: ${user.id}\n- 姓名: ${user.name}\n- 邮箱: ${user.email}\n- 角色: ${user.role}`;
          }, {
          name: 'query_user',
          description: '查询数据库中的用户信息。输入用户 ID，返回该用户的详细信息（姓名、邮箱、角色）',
          schema: queryUserArgsSchema
        }
        )
      },
      inject: [UserService],
    }, {
      provide: 'SEND_MAIL_TOOL',
      useFactory: (mailerService: MailerService, configService: ConfigService) => {
        const sendMailArgsSchema = z.object({
          to: z
            .email()
            .describe('收件人邮箱地址，例如：someone@example.com'),
          subject: z.string().describe('邮件主题'),
          text: z.string().optional().describe('纯文本内容，可选'),
          html: z.string().optional().describe('HTML 内容，可选'),
        });

        return tool(
          async ({ to, subject, text, html }: {
            to: string;
            subject: string;
            text?: string;
            html?: string;
          }) => {
            const fallbackFrom =
              configService.get<string>('MAIL_FROM')

            await mailerService.sendMail({
              to,
              subject,
              text: text ?? '（无文本内容）',
              html: html ?? `<p>${text ?? '（无 HTML 内容）'}</p>`,
              from: fallbackFrom,
            });

            return `邮件已发送到 ${to}，主题为「${subject}」`;
          },
          {
            name: 'send_mail',
            description:
              '发送电子邮件。需要提供收件人邮箱、主题，可选文本内容和 HTML 内容。',
            schema: sendMailArgsSchema,
          },
        );
      },
      inject: [MailerService, ConfigService],
    }, {
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
    },],
})
export class AiModule { }
