// Opens every example page of a running docs server on the WebGL fallback and on WebGPU, and
// reports which ones draw: `node packages/typegpu-kit-gl/scripts/examples.ts [--base url]
// [--out dir] [--backend webgl|webgpu|both|nogpu] [--browser chromium|firefox|webkit]
// [--timeout seconds] [--reference dir] [--review] [--minimum] [--texture-size n] [--hide ext,…|all]
// [--flag chromium-switch …] [--stats] [--stats-frames n]
// [example ...]`. `webgl` forces the fallback with `?backend=webgl`; `nogpu` removes WebGPU from
// the page first, as a browser without it, so the page must fall back by itself.
// With `--reference`, it also scores each screenshot against the one of the same name there.
// Screenshots outside Chromium carry the browser in their name, `forest.nogpu.webkit.png`.
// `--review` also takes three shots a second apart, `forest.webgl.1.png` to `.3.png`, resizes the
// page to a portrait viewport for `forest.webgl.resized.png`, and reports the canvas size before
// and after, the frame times and the live WebGL objects the page holds.
// `--stats` lets each example run three seconds once ready, then reads the WebGL device's
// statistics (`webgl.statsOf`, through the page's `?stats` hook) over the next `--stats-frames`
// frames, default 120, and reports per frame what it read back, uploaded and ran on the CPU, with
// a table sorted by readback time at the end. Those are milliseconds and counts the device
// measured, which depend much less on the machine's load than frame rates do.

import { existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { parseArgs } from 'node:util';
import { chromium, firefox, webkit, type Browser, type Page } from 'playwright';
import { EXTENSIONS, installScript } from '../tests/device/minimum.ts';

const { values, positionals } = parseArgs({
  allowPositionals: true,
  options: {
    base: { type: 'string', default: 'http://localhost:4321' },
    out: { type: 'string', default: 'packages/typegpu-kit-gl/.examples' },
    backend: { type: 'string', default: 'webgl' },
    browser: { type: 'string', default: 'chromium' },
    timeout: { type: 'string', default: '90' },
    reference: { type: 'string' },
    review: { type: 'boolean', default: false },
    minimum: { type: 'boolean', default: false },
    'texture-size': { type: 'string' },
    hide: { type: 'string' },
    flag: { type: 'string', multiple: true, default: [] },
    stats: { type: 'boolean', default: false },
    'stats-frames': { type: 'string', default: '120' },
  },
});

// `--minimum` reports WebGL 2's minimum limits, `--hide` refuses extensions (`all` hides every one
// the device asks for), as a weaker GPU than this one would
const hidden = values.hide === 'all' ? [...EXTENSIONS] : (values.hide?.split(',') ?? []);
const restriction =
  values.minimum || hidden.length > 0
    ? {
        minimum: values.minimum,
        hide: hidden,
        ...(values['texture-size'] ? { textureSize: Number(values['texture-size']) } : {}),
      }
    : undefined;

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
  /** Seconds from navigation until the cover lifted. */
  readonly ready?: number;
  /** With `--review`: the differences between consecutive shots a second apart. */
  readonly shots?: readonly number[];
  /** With `--review`: the canvas's CSS and backing size, before and after the resize. */
  readonly canvas?: { readonly before: CanvasSize; readonly after: CanvasSize };
  /** With `--review`: median and worst frame time in milliseconds over the two seconds. */
  readonly frameMs?: { readonly median: number; readonly worst: number };
  /** With `--review`: WebGL objects created and not deleted, by kind. */
  readonly live?: Record<string, number>;
  /** With `--stats`: what the WebGL device did per frame over a window of steady frames. */
  readonly stats?: FrameStats;
}

/** A label and its amount per frame, with how many times a frame it happened. */
interface Top {
  readonly label: string;
  readonly perFrame: number;
  readonly count: number;
}

interface FrameStats {
  /** Scene frames in the window, and the wall-clock milliseconds it took. */
  readonly frames: number;
  readonly windowMs: number;
  /**
   * Milliseconds a frame in reads from GL to the CPU, buffers and textures together. A read is a
   * synchronous `readPixels`, so this is the wait for the GPU to finish what it was given plus the
   * transfer; the device does not time them apart.
   */
  readonly readbackMs: number;
  readonly readbacks: number;
  /** The three costliest reads, milliseconds a frame, by the label the device gives them. */
  readonly readbackTop: readonly Top[];
  /** Bytes a frame uploaded into buffers, with the count of uploads. */
  readonly uploadBytes: number;
  readonly uploads: number;
  readonly uploadTop: readonly Top[];
  /** Milliseconds a frame in buffer uploads. */
  readonly uploadMs: number;
  /**
   * Textures a frame uploaded from CPU copies kernels wrote, and the milliseconds; the device
   * counts no bytes for them.
   */
  readonly textureUploads: number;
  readonly textureUploadMs: number;
  /**
   * Milliseconds a frame in kernels run on the CPU. A CPU kernel reads back the buffers and
   * textures it binds while it runs, so this includes the readbacks those kernels caused.
   */
  readonly cpuKernelMs: number;
  readonly cpuKernelTop: readonly Top[];
  /** Pipelines that dispatched on the CPU in the window, with the device's reason. */
  readonly cpu: readonly { readonly label: string; readonly reason: string }[];
  /** GPU-form dispatches a frame, and the CPU milliseconds a frame recording them took. */
  readonly gpuDispatches: number;
  readonly gpuKernelMs: number;
  /** Draw calls a frame. */
  readonly draws: number;
  /**
   * Every kernel, read and buffer upload in the window, per frame, in the order the device first
   * saw them. A read waits for all the GPU work before it, so the first read after GPU work pays
   * for that work whatever its size.
   */
  readonly kernels: readonly Top[];
  readonly reads: readonly Top[];
  readonly bufferUploads: readonly Top[];
}

/** The device's statistics at one moment, as plain objects. */
interface Snapshot {
  readonly frames: number;
  readonly at: number;
  readonly kernels: Record<string, { ms: number; dispatches: number }>;
  readonly textures: Record<string, { ms: number; levels: number }>;
  readonly buffers: Record<
    string,
    { readMs: number; reads: number; uploadMs: number; uploads: number; uploadBytes: number }
  >;
  readonly upload: { ms: number; textures: number };
  readonly draws: { count: number; instances: number };
  readonly fallbacks: Record<string, string>;
  readonly placement: Record<string, string>;
}

/** Reads the device's statistics from the page's `?stats` hook, or undefined before it is there. */
const snapshotOf = (page: Page) =>
  page.evaluate((): Snapshot | undefined => {
    const hook = (
      globalThis as {
        kitGlStats?: () => {
          frames: number;
          stats?: {
            kernels: Map<string, object>;
            readbacks: Map<string, object>;
            buffers: Map<string, object>;
            upload: object;
            draws: object;
            fallbacks: Map<string, string>;
            placement: Map<string, string>;
          };
        };
      }
    ).kitGlStats;
    const taken = hook?.();
    if (!taken?.stats) return undefined;
    const { stats } = taken;
    // oxlint-disable-next-line unicorn/consistent-function-scoping -- the page sees only this callback
    const plain = <T>(map: Map<string, T>) =>
      Object.fromEntries([...map].map(([key, value]) => [key, structuredClone(value)]));
    return {
      frames: taken.frames,
      at: performance.now(),
      kernels: plain(stats.kernels),
      textures: plain(stats.readbacks),
      buffers: plain(stats.buffers),
      upload: structuredClone(stats.upload),
      draws: structuredClone(stats.draws),
      fallbacks: plain(stats.fallbacks),
      placement: plain(stats.placement),
    } as Snapshot;
  });

const round = (value: number, places = 2) => Math.round(value * 10 ** places) / 10 ** places;

/** The three largest of `amounts`, per frame. */
const top = (amounts: Iterable<[string, number, number]>, frames: number): Top[] =>
  every(
    [...amounts].toSorted((a, b) => b[1] - a[1]).filter(([, amount]) => amount > 0),
    frames,
  ).slice(0, 3);

/** Each of `amounts` that happened in the window, per frame. */
function every(amounts: Iterable<[string, number, number]>, frames: number): Top[] {
  return [...amounts]
    .filter(([, , count]) => count > 0)
    .map(([label, amount, count]) => ({
      label,
      perFrame: round(amount / frames, 3),
      count: round(count / frames, 2),
    }));
}

const total = <T>(items: readonly T[], of: (item: T) => number) =>
  items.reduce((amount, item) => amount + of(item), 0);

/** What the device did per frame between two snapshots. */
function frameStats(before: Snapshot, after: Snapshot): FrameStats {
  const frames = Math.max(1, after.frames - before.frames);
  const kernels = Object.entries(after.kernels).map(([label, now]) => {
    const was = before.kernels[label];
    return {
      label,
      ms: now.ms - (was?.ms ?? 0),
      dispatches: now.dispatches - (was?.dispatches ?? 0),
    };
  });
  const textures = Object.entries(after.textures).map(([label, now]) => {
    const was = before.textures[label];
    return { label, ms: now.ms - (was?.ms ?? 0), count: now.levels - (was?.levels ?? 0) };
  });
  const buffers = Object.entries(after.buffers).map(([label, now]) => {
    const was = before.buffers[label];
    return {
      label,
      readMs: now.readMs - (was?.readMs ?? 0),
      reads: now.reads - (was?.reads ?? 0),
      uploadMs: now.uploadMs - (was?.uploadMs ?? 0),
      uploads: now.uploads - (was?.uploads ?? 0),
      uploadBytes: now.uploadBytes - (was?.uploadBytes ?? 0),
    };
  });
  const reads: [string, number, number][] = [
    ...textures.map((t): [string, number, number] => [`texture ${t.label}`, t.ms, t.count]),
    ...buffers.map((b): [string, number, number] => [`buffer ${b.label}`, b.readMs, b.reads]),
  ];
  const cpu = kernels.filter((k) => !k.label.endsWith(' (gpu)') && k.dispatches > 0);
  const gpu = kernels.filter((k) => k.label.endsWith(' (gpu)') && k.dispatches > 0);
  return {
    frames,
    windowMs: Math.round(after.at - before.at),
    readbackMs: round(total(reads, ([, ms]) => ms) / frames, 3),
    readbacks: round(total(reads, ([, , count]) => count) / frames),
    readbackTop: top(reads, frames),
    uploadBytes: Math.round(total(buffers, (b) => b.uploadBytes) / frames),
    uploads: round(total(buffers, (b) => b.uploads) / frames),
    uploadTop: top(
      buffers.map((b): [string, number, number] => [b.label, b.uploadBytes, b.uploads]),
      frames,
    ),
    uploadMs: round(total(buffers, (b) => b.uploadMs) / frames, 3),
    textureUploads: round((after.upload.textures - before.upload.textures) / frames),
    textureUploadMs: round((after.upload.ms - before.upload.ms) / frames, 3),
    cpuKernelMs: round(total(cpu, (k) => k.ms) / frames, 3),
    cpuKernelTop: top(
      cpu.map((k): [string, number, number] => [k.label, k.ms, k.dispatches]),
      frames,
    ),
    cpu: cpu.map(({ label }) => ({
      label,
      reason: after.fallbacks[label] ?? after.placement[label] ?? 'no GPU form',
    })),
    gpuDispatches: round(total(gpu, (k) => k.dispatches) / frames),
    gpuKernelMs: round(total(gpu, (k) => k.ms) / frames, 3),
    draws: round((after.draws.count - before.draws.count) / frames),
    kernels: every(
      kernels.map((k): [string, number, number] => [k.label, k.ms, k.dispatches]),
      frames,
    ),
    reads: every(reads, frames),
    bufferUploads: every(
      buffers.map((b): [string, number, number] => [b.label, b.uploadBytes, b.uploads]),
      frames,
    ),
  };
}

const statsFrames = Number(values['stats-frames']);

/** Lets the scene run, then measures `statsFrames` frames of it. */
async function measure(page: Page): Promise<FrameStats | undefined> {
  await page.waitForTimeout(3000);
  const before = await snapshotOf(page);
  if (!before) return undefined;
  await page.waitForFunction(
    (frames) =>
      ((globalThis as { kitGlStats?: () => { frames: number } }).kitGlStats?.().frames ?? 0) >=
      frames,
    before.frames + statsFrames,
    { timeout: 60_000, polling: 50 },
  );
  const after = await snapshotOf(page);
  return after && frameStats(before, after);
}

interface CanvasSize {
  readonly css: readonly [number, number];
  readonly backing: readonly [number, number];
  /** The viewport's CSS size, so stretching shows as a CSS aspect unlike the backing one. */
  readonly viewport: readonly [number, number];
}

const RESIZED = { width: 600, height: 760 };

const canvasOf = (page: Page) =>
  page.evaluate((): CanvasSize => {
    const canvas = document.querySelector<HTMLCanvasElement>('#example canvas')!;
    const box = canvas.getBoundingClientRect();
    return {
      css: [Math.round(box.width), Math.round(box.height)],
      backing: [canvas.width, canvas.height],
      viewport: [innerWidth, innerHeight],
    };
  });

/** Counts WebGL objects as the page creates and deletes them, under `globalThis.kitGlLive`. */
const countObjects = () => {
  const live: Record<string, number> = {};
  (globalThis as { kitGlLive?: Record<string, number> }).kitGlLive = live;
  const proto = WebGL2RenderingContext.prototype as unknown as Record<
    string,
    (...args: unknown[]) => unknown
  >;
  for (const kind of [
    'Buffer',
    'Texture',
    'Framebuffer',
    'Renderbuffer',
    'Program',
    'Shader',
    'Sampler',
    'VertexArray',
    'Query',
    'Sync',
  ]) {
    const create = proto[kind === 'Sync' ? 'fenceSync' : `create${kind}`];
    const remove = proto[`delete${kind}`];
    if (!create || !remove) continue;
    proto[kind === 'Sync' ? 'fenceSync' : `create${kind}`] = function (this: unknown, ...args) {
      const made = create.apply(this, args);
      if (made) live[kind] = (live[kind] ?? 0) + 1;
      return made;
    };
    proto[`delete${kind}`] = function (this: unknown, ...args) {
      if (args[0]) live[kind] = (live[kind] ?? 0) - 1;
      return remove.apply(this, args);
    };
  }
};

/** Scores two PNGs in a page, which decodes them. */
const differenceOf = (page: Page, a: string, b: string) =>
  page.evaluate(
    async ([first, second]) => {
      // oxlint-disable-next-line unicorn/consistent-function-scoping -- the page sees only this callback
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
      new Promise<{ fps: number; median: number; worst: number }>((resolve) => {
        const times: number[] = [];
        const start = performance.now();
        let last = start;
        const tick = (now: number) => {
          times.push(now - last);
          last = now;
          if (now - start < 2000) {
            requestAnimationFrame(tick);
            return;
          }
          const sorted = times.toSorted((a, b) => a - b);
          resolve({
            fps: (times.length * 1000) / (now - start),
            median: sorted[sorted.length >> 1]!,
            worst: sorted.at(-1)!,
          });
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
// `--flag` adds Chromium switches, such as `--use-angle=gl` for another GL backend
const args = ['--enable-unsafe-webgpu', '--enable-gpu', '--ignore-gpu-blocklist', ...values.flag];
// Headless Firefox disables WebGL unless forced
const launchOptions =
  engineName === 'chromium'
    ? { channel: 'chromium', args }
    : engineName === 'firefox'
      ? {
          firefoxUserPrefs: {
            'webgl.force-enabled': true,
            'webgl.disabled': false,
            'security.sandbox.content.level': 0,
          },
          // Firefox's child processes cannot take sandbox extensions inside another sandbox
          env: {
            ...process.env,
            ...Object.fromEntries(
              ['CONTENT', 'GPU', 'RDD', 'SOCKET_PROCESS', 'UTILITY', 'GMP'].map((kind) => [
                `MOZ_DISABLE_${kind}_SANDBOX`,
                '1',
              ]),
            ),
          },
        }
      : {};

/** Opens one example on one backend in `browser` and reports what it did. */
async function visit(browser: Browser, example: string, backend: string): Promise<Result> {
  const context = await browser.newContext({
    viewport: { width: 800, height: 600 },
    deviceScaleFactor: 1,
  });
  if (values.review) await context.addInitScript(countObjects);
  if (restriction) await context.addInitScript({ content: installScript(restriction) });
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
  const query = [
    ...(backend === 'webgl' ? ['backend=webgl'] : []),
    ...(values.stats ? ['stats'] : []),
  ].join('&');
  let state: Result['state'] = 'timeout';
  let notice = '';
  let fps: number | undefined;
  let frameMs: Result['frameMs'];
  let ready: number | undefined;
  let shots: number[] | undefined;
  let canvas: Result['canvas'];
  let live: Record<string, number> | undefined;
  let ink: number | undefined;
  let webgpu: boolean | undefined;
  let stats: FrameStats | undefined;
  // The dev server reloads a page when it optimises a dependency it had not seen: try again
  for (let attempt = 0; attempt < 3; attempt++) {
    try {
      await page.goto(`${values.base}/examples/${example}/${query ? `?${query}` : ''}`, {
        timeout,
      });
      webgpu = await page.evaluate(() => 'gpu' in navigator && !!navigator.gpu);
      state = await settle(page);
      ready = Math.round((Date.now() - started) / 100) / 10;
      notice = (await page.locator('#notice').textContent()) ?? '';
      if (state === 'ready') {
        const timing = await fpsOf(page);
        fps = timing.fps;
        frameMs = { median: Math.round(timing.median * 10) / 10, worst: Math.round(timing.worst) };
        await page.waitForTimeout(500);
      }
      if (state === 'ready' && values.stats) stats = await measure(page);
      if (state === 'ready' && values.review) {
        const taken: string[] = [];
        for (let i = 1; i <= 3; i++) {
          if (i > 1) await page.waitForTimeout(1000);
          const shot = await page.locator('#example canvas').screenshot({
            path: join(values.out, `${example}.${backend}${suffix}.${i}.png`),
            timeout: 20_000,
          });
          taken.push(shot.toString('base64'));
        }
        shots = [];
        for (let i = 1; i < taken.length; i++) {
          const score = await differenceOf(page, taken[i - 1]!, taken[i]!);
          shots.push(Math.round(score * 10) / 10);
        }
        const before = await canvasOf(page);
        await page.setViewportSize(RESIZED);
        await page.waitForTimeout(1500);
        await page.screenshot({
          path: join(values.out, `${example}.${backend}${suffix}.resized.png`),
          timeout: 20_000,
        });
        canvas = { before, after: await canvasOf(page) };
        live = await page.evaluate(
          () => (globalThis as { kitGlLive?: Record<string, number> }).kitGlLive ?? {},
        );
        await page.setViewportSize({ width: 800, height: 600 });
        await page.waitForTimeout(1000);
      }
      const shot = await page.locator('#example canvas').screenshot({
        path: join(values.out, `${example}.${backend}${suffix}.png`),
        timeout: 20_000,
      });
      // A flat frame compresses to almost nothing; the byte count stands in for how much it shows
      ink = shot.byteLength;
      // A dev server that reloaded the page after it was ready leaves the cover in the shot
      if (state === 'ready' && (await page.locator('#cover').count()) > 0) {
        throw new Error('navigation: the page reloaded after it was ready');
      }
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
    ...(frameMs ? { frameMs } : {}),
    ...(ready === undefined ? {} : { ready }),
    ...(shots ? { shots } : {}),
    ...(canvas ? { canvas } : {}),
    ...(live ? { live } : {}),
    ...(stats ? { stats } : {}),
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

if (values.stats) {
  const measured = results
    .filter((result) => result.stats)
    .toSorted((a, b) => b.stats!.readbackMs - a.stats!.readbackMs);
  const kb = (bytes: number) => round(bytes / 1024, 1);
  process.stdout.write(
    `\n${'example'.padEnd(16)} ${'frames'.padStart(6)} ${'read ms'.padStart(8)} ${'reads'.padStart(6)} ${'up KB'.padStart(8)} ${'ups'.padStart(5)} ${'tex up'.padStart(6)} ${'cpu ms'.padStart(7)} ${'gpu disp'.padStart(8)}  read most / cpu kernels\n`,
  );
  for (const { example, stats } of measured) {
    const s = stats!;
    const read = s.readbackTop[0] ? `${s.readbackTop[0].label} ${s.readbackTop[0].perFrame}` : '';
    const cpu = s.cpu.map((k) => k.label).join(', ');
    process.stdout.write(
      `${example.padEnd(16)} ${String(s.frames).padStart(6)} ${s.readbackMs.toFixed(2).padStart(8)} ${String(s.readbacks).padStart(6)} ${String(kb(s.uploadBytes)).padStart(8)} ${String(s.uploads).padStart(5)} ${String(s.textureUploads).padStart(6)} ${s.cpuKernelMs.toFixed(2).padStart(7)} ${String(s.gpuDispatches).padStart(8)}  ${read}${read && cpu ? ' / ' : ''}${cpu}\n`,
    );
  }
}

const ready = results.filter((result) => result.state === 'ready').length;
process.stdout.write(
  `\n${ready}/${results.length} ready. Report and screenshots in ${values.out}\n`,
);
