import { Inject, Injectable } from '@nestjs/common';
import { tool } from '@langchain/core/tools';
import { z } from 'zod';
import { MailerService } from '@nestjs-modules/mailer';
import { ConfigService } from '@nestjs/config';

@Injectable()
export class SendMailToolService {
    readonly tool;
    constructor(
        @Inject(MailerService) private readonly mailerService: MailerService,
        @Inject(ConfigService) private readonly configService: ConfigService,
    ) {
        const sendMailArgsSchema = z.object({
            to: z
                .email()
                .describe('收件人邮箱地址，例如：someone@example.com'),
            subject: z.string().describe('邮件主题'),
            text: z.string().optional().describe('纯文本内容，可选'),
            html: z.string().optional().describe('HTML 内容，可选'),
        });

        this.tool = tool(
            async ({ to, subject, text, html }: {
                to: string;
                subject: string;
                text?: string;
                html?: string;
            }) => {
                const fallbackFrom =
                    this.configService.get<string>('MAIL_FROM')

                try {
                    await this.mailerService.sendMail({
                        to,
                        subject,
                        text: text ?? '（无文本内容）',
                        html: html ?? `<p>${text ?? '（无 HTML 内容）'}</p>`,
                        from: fallbackFrom,
                    });
                } catch (e: unknown) {
                    const msg =
                        e instanceof Error ? e.message : String(e);
                    return `邮件发送失败（SMTP 已拒绝）：${msg}。请核对收件人地址是否存在、是否拼写正确；测试环境可改用自己真实邮箱。`;
                }

                return `邮件已发送到 ${to}，主题为「${subject}」`;
            },
            {
                name: 'send_mail',
                description:
                    '发送电子邮件。需要提供收件人邮箱、主题，可选文本内容和 HTML 内容。',
                schema: sendMailArgsSchema,
            },
        );
    }
}