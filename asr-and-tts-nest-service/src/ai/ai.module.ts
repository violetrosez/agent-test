import { Module } from '@nestjs/common';
import { AiService } from './ai.service';
import { AiController } from './ai.controller';
import { ConfigService } from '@nestjs/config';
import { ChatOpenAI } from '@langchain/openai';

@Module({
    controllers: [AiController],
    providers: [AiService,
        {
            provide: 'CHAT_MODEL',
            useFactory: (configService: ConfigService) => {
                const apiKey =
                    configService.get<string>('MINIMAX_API_KEY') ??
                    configService.get<string>('ALIYUN_API_KEY');
                const baseURL =
                    configService.get<string>('MINIMAX_BASE_URL') ??
                    configService.get<string>('ALIYUN_BASE_URL');
                return new ChatOpenAI({
                    model: configService.get('MODEL_NAME'),
                    apiKey,
                    configuration: {
                        baseURL,
                    },
                });
            },
            inject: [ConfigService],
        }
    ],
})
export class AiModule { }