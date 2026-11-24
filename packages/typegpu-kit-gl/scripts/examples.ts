// Opens every example page of a running docs server on the WebGL fallback and on WebGPU, and
// reports which ones draw: `node packages/typegpu-kit-gl/scripts/examples.ts [--base url]
// [--out dir] [--backend webgl|webgpu|both] [--timeout seconds] [example ...]`

import { mkdirSync, readdirSync, existsSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { parseArgs } from 'node:util';
import { chromium, type Page } from 'playwright';

const { values, positionals } = parseArgs({
  allowPositionals: true,
  options: {
    base: { type: 'string', default: 'http://localhost:4321' },
    out: { type: 'string', default: 'packages/typegpu-kit-gl/.examples' },
    backend: { type: 'string', default: 'webgl' },
    timeout: { type: 'string', default: '90' },
  },
});

const examplesDir = join(import.meta.dirname, '../../typegpu-kit-examples/src');
const all = readdirSync(examplesDir).filter((name) =>
  existsSync(join(examplesDir, name, 'meta.json')),
);
const keys = positionals.length > 0 ? positionals : all;
const backends = values.backend === 'both' ? ['webgpu', 'webgl'] : [values.backend];
const timeout = Number(values.timeout) * 1000;

interface Result {
  readonly example: string;
  readonly backend: string;
  /** `ready` once the cover lifted, `failed` when the page showed an error, else `timeout`. */
  readonly state: 'ready' | 'failed' | 'timeout';
  /** The page's notice, or the first console error. */
  readonly error?: string;
  /** Animation frames per second over two seconds once ready. */
  readonly fps?: number;
  /** The share of canvas pixels that differ from the first one, so a blank canvas reads 0. */
  readonly ink?: number;
  readonly errors: readonly string[];
  readonly seconds: number;
}

/** Waits for the cover to lift or the notice to fill, whichever comes first. */
async function settle(page: Page): Promise<'ready' | 'failed' | 'timeout'> {
  try {
    const handle = await page.waitForFunction(
      () => {
        if (document.querySelector('#notice')?.textContent) return 'failed';
        if (!document.querySelector('#cover')) return 'ready';
        return false;
      },
      undefined,
      { timeout, polling: 250 },
    );
    return (await handle.jsonValue()) as 'ready' | 'failed';
  } catch {
    return 'timeout';
  }
}

const fpsOf = (page: Page) =>
  page.evaluate(
    () =>
      new Promise<number>((resolve) => {
        let frames = 0;
        const start = performance.now();
        const tick = (now: number) => {
          frames++;
          if (now - start >= 2000) resolve((frames * 1000) / (now - start));
          else requestAnimationFrame(tick);
        };
        requestAnimationFrame(tick);
      }),
  );

mkdirSync(values.out, { recursive: true });
const browser = await chromium.launch({
  channel: 'chromium',
  args: ['--enable-unsafe-webgpu', '--enable-gpu', '--ignore-gpu-blocklist'],
});
const results: Result[] = [];

for (const example of keys) {
  for (const backend of backends) {
    const context = await browser.newContext({
      viewport: { width: 800, height: 600 },
      deviceScaleFactor: 1,
    });
    const page = await context.newPage();
    const errors: string[] = [];
    page.on('console', (message) => {
      if (message.type() === 'error') errors.push(message.text().slice(0, 4000));
    });
    page.on('pageerror', (error) => errors.push(String(error.stack ?? error).slice(0, 4000)));

    const started = Date.now();
    const query = backend === 'webgl' ? '?backend=webgl' : '';
    let state: Result['state'] = 'timeout';
    let notice = '';
    let fps: number | undefined;
    let ink: number | undefined;
    try {
      await page.goto(`${values.base}/examples/${example}/${query}`, { timeout });
      state = await settle(page);
      notice = (await page.locator('#notice').textContent()) ?? '';
      if (state === 'ready') {
        fps = await fpsOf(page);
        await page.waitForTimeout(500);
      }
      const shot = await page.locator('#example canvas').screenshot({
        path: join(values.out, `${example}.${backend}.png`),
        timeout: 20_000,
      });
      // A flat frame compresses to almost nothing; the byte count stands in for how much it shows
      ink = shot.byteLength;
    } catch (error) {
      errors.push(String(error).slice(0, 2000));
    }

    const result: Result = {
      example,
      backend,
      state,
      ...(notice || errors[0] ? { error: (notice || errors[0]!).slice(0, 600) } : {}),
      ...(fps === undefined ? {} : { fps: Math.round(fps * 10) / 10 }),
      ...(ink === undefined ? {} : { ink }),
      errors,
      seconds: Math.round((Date.now() - started) / 100) / 10,
    };
    results.push(result);
    process.stdout.write(
      `${example.padEnd(16)} ${backend.padEnd(7)} ${state.padEnd(8)} ${String(result.fps ?? '-').padStart(6)} fps ${String(ink ?? '-').padStart(8)} B  ${(result.error ?? '').split('\n')[0]!.slice(0, 140)}\n`,
    );
    await context.close();
  }
}

await browser.close();
writeFileSync(join(values.out, 'report.json'), JSON.stringify(results, null, 2));
const ready = results.filter((result) => result.state === 'ready').length;
process.stdout.write(
  `\n${ready}/${results.length} ready. Report and screenshots in ${values.out}\n`,
);
