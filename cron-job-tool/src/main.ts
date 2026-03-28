import { NestFactory } from '@nestjs/core';
import { AppModule } from './app.module';

async function bootstrap() {
  const app = await NestFactory.create(AppModule);
  // 允许前端（如 public 静态页、其他端口）跨域访问；含 @Sse() 的 EventSource
  app.enableCors({
    origin: true,
    methods: ['GET', 'HEAD', 'PUT', 'PATCH', 'POST', 'DELETE', 'OPTIONS'],
  });
  await app.listen(process.env.PORT ?? 3000);
}
bootstrap();
