import { Controller, Get, Query, Sse } from '@nestjs/common';
import { AiService } from './ai.service';
import { from, Observable } from 'rxjs';
import { map } from 'rxjs/operators';


@Controller('ai')
export class AiController {
  constructor(private readonly aiService: AiService) { }


  @Sse('chat')
  chat(@Query('query') query: string): Observable<string> {
    return from(this.aiService.runChain(query)).pipe(
      map((chunk) => chunk as string),
    );
  }

}
