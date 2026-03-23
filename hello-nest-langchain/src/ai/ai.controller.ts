import {
  Controller,
  Get,
  Query,
  Sse,
  Header,
  type MessageEvent,
} from '@nestjs/common';
import { AiService } from './ai.service';
import { from, Observable } from 'rxjs';
import { map } from 'rxjs/operators';

@Controller('ai')
export class AiController {
  constructor(private readonly aiService: AiService) {}

  @Get('chat')
  @Header('Content-Type', 'application/json; charset=utf-8')
  chat(@Query('query') query: string) {
    return this.aiService.runChain(query);
  }

  /** SSE 未带 charset 时，部分客户端会把 UTF-8 字节按 Latin-1 解码导致中文乱码 */
  @Sse('stream')
  @Header('Content-Type', 'text/event-stream; charset=utf-8')
  @Header('Cache-Control', 'no-cache')
  streamChat(@Query('query') query: string): Observable<{ data: string }> {
    return from(this.aiService.streamChain(query)).pipe(
      map((chunk) => ({
        data: chunk,
      })),
    );
  }
}
