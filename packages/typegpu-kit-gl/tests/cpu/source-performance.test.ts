import { execFileSync } from 'node:child_process';
import { it } from 'vitest';

it.runIf(process.env.CPU_SOURCE_BENCHMARK === '1')(
  'compares generated kernels and source through native Node imports',
  () => {
    // Vitest's module-runner import indirections distort TypeGPU's JS math cost.
    // Both paths therefore run together in one native Node process, outside the runner.
    const fixture = new URL('./source-performance.ts', import.meta.url).href;
    execFileSync(
      process.execPath,
      [
        '--input-type=module',
        '--eval',
        `
      import install from 'unplugin-typegpu/node';
      install();
      await (await import(${JSON.stringify(fixture)})).runSourcePerformance();
    `,
      ],
      { stdio: 'pipe', timeout: 300_000 },
    );
  },
  310_000,
);
