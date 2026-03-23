"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.UpdateAiDto = void 0;
const mapped_types_1 = require("@nestjs/mapped-types");
const create_ai_dto_1 = require("./create-ai.dto");
class UpdateAiDto extends (0, mapped_types_1.PartialType)(create_ai_dto_1.CreateAiDto) {
}
exports.UpdateAiDto = UpdateAiDto;
//# sourceMappingURL=update-ai.dto.js.map