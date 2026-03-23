import { Module } from '@nestjs/common';
import { AiService } from './ai.service';
import { AiController } from './ai.controller';
import { ChatOpenAI } from '@langchain/openai';
import { ConfigService } from '@nestjs/config';

@Module({
  controllers: [AiController],
  providers: [AiService, {
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
  },
],
})
export class AiModule {}
