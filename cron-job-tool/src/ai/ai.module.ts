import { Module } from '@nestjs/common';
import { AiService } from './ai.service';
import { AiController } from './ai.controller';
import { ConfigService } from '@nestjs/config';
import { ChatOpenAI } from '@langchain/openai';
import { tool } from '@langchain/core/tools';
import { UserService } from './user.service';
import { z } from 'zod';
import { MailerService } from '@nestjs-modules/mailer';
import { UsersService } from 'src/users/users.service';
import { UsersModule } from 'src/users/users.module';
import { JobModule } from 'src/job/job.module';
import { ToolModule } from 'src/tool/tool.module';
@Module({
  imports: [UsersModule, JobModule, ToolModule],
  // MailerModule.forRootAsync 在 AppModule 里已注册，且 MailerCoreModule 为 global，此处不要再把 MailerService 放进 providers
  controllers: [AiController],
  providers: [
    UserService,
    AiService,
    {
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
    },],
})
export class AiModule { }
