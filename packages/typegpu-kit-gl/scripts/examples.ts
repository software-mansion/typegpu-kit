// Opens every example page of a running docs server on the WebGL fallback and on WebGPU, and
// reports which ones draw: `node packages/typegpu-kit-gl/scripts/examples.ts [--base url]
// [--out dir] [--backend webgl|webgpu|both|nogpu] [--browser chromium|firefox|webkit]
// [--timeout seconds] [--reference dir] [example ...]`. `webgl` forces the fallback with `?backend=webgl`; `nogpu` removes WebGPU from
// the page first, as a browser without it, so the page must fall back by itself.
// With `--reference`, it also scores each screenshot against the one of the same name there.
// Screenshots outside Chromium carry the browser in their name, `forest.nogpu.webkit.png`.

import { existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { parseArgs } from 'node:util';
import { chromium, firefox, webkit, type Browser, type Page } from 'playwright';

const { values, positionals } = parseArgs({
  allowPositionals: true,
  options: {
    base: { type: 'string', default: 'http://localhost:4321' },
    out: { type: 'string', default: 'packages/typegpu-kit-gl/.examples' },
    backend: { type: 'string', default: 'webgl' },
    browser: { type: 'string', default: 'chromium' },
    timeout: { type: 'string', default: '90' },
    reference: { type: 'string' },
  },
});

const examplesDir = join(import.meta.dirname, '../../typegpu-kit-examples/src');
const all = readdirSync(examplesDir).filter((name) =>
  existsSync(join(examplesDir, name, 'meta.json')),
);
const keys = positionals.length > 0 ? positionals : all;
const backends = values.backend === 'both' ? ['webgpu', 'webgl'] : [values.backend];
const timeout = Number(values.timeout) * 1000;

const engines = { chromium, firefox, webkit } as const;
const engineName = values.browser as keyof typeof engines;
if (!(engineName in engines)) throw new Error(`Unknown browser '${values.browser}'`);
const engine = engines[engineName];
const suffix = engineName === 'chromium' ? '' : `.${engineName}`;

interface Result {
  readonly example: string;
  readonly backend: string;
  readonly browser: string;
  /** Whether the page had `navigator.gpu` when it loaded. */
  readonly webgpu?: boolean;
  /**
   * `ready` once the cover lifted, `failed` when the page showed an error, `timeout` when neither
   * happened, and `hung` when the page stopped answering, as in a frame that never returns.
   */
  readonly state: 'ready' | 'failed' | 'timeout' | 'hung';
  /** The page's notice, or the first console error. */
  readonly error?: string;
  /** Animation frames per second over two seconds once ready. */
  readonly fps?: number;
  /** The share of canvas pixels that differ from the first one, so a blank canvas reads 0. */
  readonly ink?: number;
  /**
   * The mean difference from the reference screenshot, 0 to 255, after shrinking both to 64 by 48,
   * so it measures composition and colour and ignores noise and animation phase.
   */
  readonly difference?: number;
  readonly errors: readonly string[];
  readonly seconds: number;
}

/** Scores two PNGs in a page, which decodes them. */
const differenceOf = (page: Page, a: string, b: string) =>
  page.evaluate(
    async ([first, second]) => {
      const pixels = async (source: string) => {
        const image = new Image();
        image.src = `data:image/png;base64,${source}`;
        await image.decode();
        const canvas = new OffscreenCanvas(64, 48);
        const context = canvas.getContext('2d')!;
        context.drawImage(image, 0, 0, 64, 48);
        return context.getImageData(0, 0, 64, 48).data;
      };
      const [left, right] = await Promise.all([pixels(first!), pixels(second!)]);
      let sum = 0;
      for (let i = 0; i < left.length; i += 4) {
        for (let c = 0; c < 3; c++) sum += Math.abs(left[i + c]! - right[i + c]!);
      }
      return sum / ((left.length / 4) * 3);
    },
    [a, b],
  );

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

/** One line of an error: a resolution error's last line names the cause, others their first. */
function summary(error: string): string {
  const lines = error.split('\n').filter((line) => line.trim() !== '');
  const line = error.startsWith('Resolution of the following tree failed')
    ? lines.at(-1)
    : lines[0];
  return (line ?? '').trim().slice(0, 200);
}

mkdirSync(values.out, { recursive: true });
const results: Result[] = [];
const args = ['--enable-unsafe-webgpu', '--enable-gpu', '--ignore-gpu-blocklist'];
// Headless Firefox disables WebGL unless forced
const launchOptions =
  engineName === 'chromium'
    ? { channel: 'chromium', args }
    : engineName === 'firefox'
      ? { firefoxUserPrefs: { 'webgl.force-enabled': true, 'webgl.disabled': false } }
      : {};

/** Opens one example on one backend in `browser` and reports what it did. */
async function visit(browser: Browser, example: string, backend: string): Promise<Result> {
  const context = await browser.newContext({
    viewport: { width: 800, height: 600 },
    deviceScaleFactor: 1,
  });
  if (backend === 'nogpu') {
    await context.addInitScript(() => {
      delete (Navigator.prototype as { gpu?: unknown }).gpu;
      for (const name of Object.getOwnPropertyNames(globalThis)) {
        if (name.startsWith('GPU')) delete (globalThis as Record<string, unknown>)[name];
      }
    });
  }
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
  let webgpu: boolean | undefined;
  // The dev server reloads a page when it optimises a dependency it had not seen: try again
  for (let attempt = 0; attempt < 3; attempt++) {
    try {
      await page.goto(`${values.base}/examples/${example}/${query}`, { timeout });
      webgpu = await page.evaluate(() => 'gpu' in navigator && !!navigator.gpu);
      state = await settle(page);
      notice = (await page.locator('#notice').textContent()) ?? '';
      if (state === 'ready') {
        fps = await fpsOf(page);
        await page.waitForTimeout(500);
      }
      const shot = await page.locator('#example canvas').screenshot({
        path: join(values.out, `${example}.${backend}${suffix}.png`),
        timeout: 20_000,
      });
      // A flat frame compresses to almost nothing; the byte count stands in for how much it shows
      ink = shot.byteLength;
      break;
    } catch (error) {
      if (attempt < 2 && String(error).includes('navigation')) {
        errors.length = 0;
        continue;
      }
      errors.push(String(error).slice(0, 2000));
      break;
    }
  }

  let difference: number | undefined;
  const reference = values.reference && join(values.reference, `${example}.webgpu.png`);
  const own = join(values.out, `${example}.${backend}${suffix}.png`);
  if (reference && existsSync(reference) && existsSync(own) && state === 'ready') {
    try {
      const blank = await context.newPage();
      difference = await differenceOf(
        blank,
        readFileSync(own).toString('base64'),
        readFileSync(reference).toString('base64'),
      );
    } catch (error) {
      errors.push(String(error).slice(0, 2000));
    }
  }

  return {
    example,
    backend,
    browser: engineName,
    ...(webgpu === undefined ? {} : { webgpu }),
    state,
    ...(difference === undefined ? {} : { difference: Math.round(difference * 10) / 10 }),
    ...(notice || errors[0] ? { error: (notice || errors[0]!).slice(0, 3000) } : {}),
    ...(fps === undefined ? {} : { fps: Math.round(fps * 10) / 10 }),
    ...(ink === undefined ? {} : { ink }),
    errors,
    seconds: Math.round((Date.now() - started) / 100) / 10,
  };
}

for (const example of keys) {
  for (const backend of backends) {
    // A page whose frame never returns answers nothing, not even a close: give each visit a
    // browser of its own and kill it at the deadline
    const server = await engine.launchServer(launchOptions);
    const browser = await engine.connect(server.wsEndpoint());
    const started = Date.now();
    let alarm: NodeJS.Timeout | undefined;
    const deadline = new Promise<Result>((resolve) => {
      alarm = setTimeout(
        () =>
          resolve({
            example,
            backend,
            browser: engineName,
            state: 'hung',
            error: 'The page stopped answering.',
            errors: [],
            seconds: Math.round((Date.now() - started) / 100) / 10,
          }),
        timeout + 30_000,
      );
    });
    const result = await Promise.race([
      visit(browser, example, backend).catch((error): Result => ({
        example,
        backend,
        browser: engineName,
        state: 'hung',
        error: String(error).slice(0, 2000),
        errors: [String(error)],
        seconds: Math.round((Date.now() - started) / 100) / 10,
      })),
      deadline,
    ]);
    clearTimeout(alarm);
    await server.kill();

    results.push(result);
    process.stdout.write(
      `${example.padEnd(16)} ${backend.padEnd(7)} ${engineName.padEnd(8)} ${result.webgpu ? 'gpu' : '   '} ${result.state.padEnd(8)} ${String(result.fps ?? '-').padStart(6)} fps ${String(result.ink ?? '-').padStart(8)} B ${String(result.difference ?? '-').padStart(6)} diff  ${summary(result.error ?? '')}\n`,
    );
    writeFileSync(join(values.out, `report${suffix}.json`), JSON.stringify(results, null, 2));
  }
}

const ready = results.filter((result) => result.state === 'ready').length;
process.stdout.write(
  `\n${ready}/${results.length} ready. Report and screenshots in ${values.out}\n`,
);
