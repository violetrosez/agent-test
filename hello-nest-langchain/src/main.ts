import { NestFactory } from '@nestjs/core';
import { AppModule } from './app.module';

async function bootstrap() {
  const app = await NestFactory.create(AppModule);
  // Node 进程 stdout 默认 UTF-8；终端乱码时检查控制台代码页（Windows: chcp 65001）
  await app.listen(process.env.PORT ?? 3000);
}
bootstrap();
