import { ChatOpenAI } from '@langchain/openai';
export declare class AiService {
    private readonly chain;
    constructor(model: ChatOpenAI);
    runChain(input: string): Promise<any>;
    streamChain(input: string): AsyncGenerator<any, void, unknown>;
}
