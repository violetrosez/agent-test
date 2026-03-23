"use strict";
var __decorate = (this && this.__decorate) || function (decorators, target, key, desc) {
    var c = arguments.length, r = c < 3 ? target : desc === null ? desc = Object.getOwnPropertyDescriptor(target, key) : desc, d;
    if (typeof Reflect === "object" && typeof Reflect.decorate === "function") r = Reflect.decorate(decorators, target, key, desc);
    else for (var i = decorators.length - 1; i >= 0; i--) if (d = decorators[i]) r = (c < 3 ? d(r) : c > 3 ? d(target, key, r) : d(target, key)) || r;
    return c > 3 && r && Object.defineProperty(target, key, r), r;
};
var __metadata = (this && this.__metadata) || function (k, v) {
    if (typeof Reflect === "object" && typeof Reflect.metadata === "function") return Reflect.metadata(k, v);
};
var __param = (this && this.__param) || function (paramIndex, decorator) {
    return function (target, key) { decorator(target, key, paramIndex); }
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.AiService = void 0;
const common_1 = require("@nestjs/common");
const openai_1 = require("@langchain/openai");
const output_parsers_1 = require("@langchain/core/output_parsers");
const prompts_1 = require("@langchain/core/prompts");
let AiService = class AiService {
    chain;
    constructor(model) {
        const prompt = prompts_1.PromptTemplate.fromTemplate(`
     你是AI助手，请根据用户的问题给出回答。
      {input}
    `);
        this.chain = prompt.pipe(model).pipe(new output_parsers_1.StringOutputParser());
    }
    runChain(input) {
        return this.chain.invoke({ input });
    }
    async *streamChain(input) {
        const stream = await this.chain.stream({ input });
        for await (const chunk of stream) {
            console.log(chunk);
            yield chunk;
        }
    }
};
exports.AiService = AiService;
exports.AiService = AiService = __decorate([
    (0, common_1.Injectable)(),
    __param(0, (0, common_1.Inject)('CHAT_MODEL')),
    __metadata("design:paramtypes", [openai_1.ChatOpenAI])
], AiService);
//# sourceMappingURL=ai.service.js.map