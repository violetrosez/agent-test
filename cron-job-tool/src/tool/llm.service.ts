import { Injectable } from '@nestjs/common';
import { tool } from '@langchain/core/tools';
import { ChatOpenAI } from '@langchain/openai';

@Injectable()
export class LlmService {
    getModel() {
        return new ChatOpenAI({
            model: process.env.MODEL_NAME,
            apiKey: process.env.ALIYUN_API_KEY,
            temperature: 0,
            configuration: {
                baseURL: process.env.ALIYUN_BASE_URL,
            },
        });
    }
}