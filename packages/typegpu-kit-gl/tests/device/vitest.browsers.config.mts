// Runs the device's browser tests in WebKit or Firefox, which have WebGL 2 but no WebGPU:
// `BROWSER=webkit npx vitest run -c packages/typegpu-kit-gl/tests/device/vitest.browsers.config.mts`.
// Tests that compare with WebGPU need Chromium and skip themselves where `navigator.gpu` is missing.

import { playwright } from '@vitest/browser-playwright';
import typegpu from 'unplugin-typegpu/vite';
import { defineConfig } from 'vitest/config';

const browser = (process.env.BROWSER ?? 'webkit') as 'webkit' | 'firefox';

export default defineConfig({
  plugins: [typegpu()],
  root: new URL('../..', import.meta.url).pathname,
  test: {
    include: ['tests/device/**/*.test.gl.ts'],
    testTimeout: 120_000,
    hookTimeout: 120_000,
    fileParallelism: false,
    browser: {
      enabled: true,
      headless: true,
      screenshotFailures: false,
      provider: playwright(),
      instances: [{ browser }],
    },
  },
});
