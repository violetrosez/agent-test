import { Module } from '@nestjs/common';
import { JobService } from './job.service';
import { JobAgentService } from '../ai/job-agent.service';
import { ToolModule } from 'src/tool/tool.module';

@Module({
  imports: [ToolModule],
  providers: [JobService, JobAgentService],
  exports: [JobService],
})
export class JobModule {}
