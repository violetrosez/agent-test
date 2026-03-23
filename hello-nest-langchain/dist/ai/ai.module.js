"use strict";
var __decorate = (this && this.__decorate) || function (decorators, target, key, desc) {
    var c = arguments.length, r = c < 3 ? target : desc === null ? desc = Object.getOwnPropertyDescriptor(target, key) : desc, d;
    if (typeof Reflect === "object" && typeof Reflect.decorate === "function") r = Reflect.decorate(decorators, target, key, desc);
    else for (var i = decorators.length - 1; i >= 0; i--) if (d = decorators[i]) r = (c < 3 ? d(r) : c > 3 ? d(target, key, r) : d(target, key)) || r;
    return c > 3 && r && Object.defineProperty(target, key, r), r;
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.AiModule = void 0;
const common_1 = require("@nestjs/common");
const ai_service_1 = require("./ai.service");
const ai_controller_1 = require("./ai.controller");
const openai_1 = require("@langchain/openai");
const config_1 = require("@nestjs/config");
let AiModule = class AiModule {
};
exports.AiModule = AiModule;
exports.AiModule = AiModule = __decorate([
    (0, common_1.Module)({
        controllers: [ai_controller_1.AiController],
        providers: [ai_service_1.AiService, {
                provide: 'CHAT_MODEL',
                useFactory: (configService) => {
                    return new openai_1.ChatOpenAI({
                        model: configService.get('MODEL_NAME'),
                        apiKey: configService.get('ALIYUN_API_KEY'),
                        temperature: 0,
                        configuration: {
                            baseURL: configService.get('ALIYUN_BASE_URL'),
                        },
                    });
                },
                inject: [config_1.ConfigService],
            },
        ],
    })
], AiModule);
//# sourceMappingURL=ai.module.js.map