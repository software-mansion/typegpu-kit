import typegpu from 'unplugin-typegpu/vite';
import { defineConfig } from 'vitest/config';

export default defineConfig({
  plugins: [typegpu()],
  test: {
    setupFiles: ['@typegpu/kit/testing/warnings'],
    testTimeout: 120_000,
    hookTimeout: 120_000,
    projects: (
      [
        ['packages/typegpu-kit', true],
        ['packages/typegpu-gl', false],
        ['packages/typegpu-kit-gl', true],
        ['packages/typegpu-kit-examples', true],
        ['packages/typegpu-kit-pull', true],
        ['packages/typegpu-create-kit', true],
        ['apps/typegpu-kit-docs', true],
        ['apps/typegpu-kit-native', true],
      ] as const
    ).flatMap(([root, gpus]) =>
      (gpus ? [false, true] : [false]).map((gpu) => ({
        extends: true as const,
        test: {
          root,
          name: `${root.split('/').pop()} (${gpu ? 'gpu' : 'unit'})`,
          include: [gpu ? 'tests/**/*.test.gpu.ts' : 'tests/**/*.test.ts'],
          ...(gpu ? { pool: 'forks' as const, fileParallelism: false } : {}),
        },
      })),
    ),
  },
});
