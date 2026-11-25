// Packs the kit and its WebGL fallback, installs both tarballs in a temporary consumer project with
// the TypeGPU patch registered as the fallback's README says, then type-checks and bundles a program
// that hosts a scene with `webgl.fallback()`, and checks what bundlers keep of the fallback. With
// `--browser`, it also runs the program in headless Chromium without WebGPU and reads a pixel.

import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { cp, mkdir, mkdtemp, readdir, readFile, rename, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { build, preview } from 'vite';

const repo = path.resolve(import.meta.dirname, '..');
const browser = process.argv.includes('--browser');
const temporary = await mkdtemp(path.join(tmpdir(), 'kit-gl-consumer-'));
const run = (command: string, args: string[], cwd = temporary) =>
  execFileSync(command, args, { cwd, stdio: 'inherit' });
const PATCH = 'typegpu@0.12.11.patch';
/** The fallback's own error messages, which only its code holds. */
const MARKER = '@typegpu/kit-gl CPU';

const scene = `import { backends, type Frame, type Surface } from '@typegpu/kit';
import { common, d, tgpu, type TgpuRoot } from 'typegpu';

export async function createScene(root: TgpuRoot, surface: Surface) {
  const tint = root.createUniform(d.vec4f, d.vec4f(1, 0.5, 0.25, 1));
  const fill = tgpu.fragmentFn({ out: d.vec4f })(() => {
    'use gpu';
    return tint.$;
  });
  const drawing = root.createRenderPipeline({
    vertex: common.fullScreenTriangle,
    fragment: fill,
    targets: { format: surface.format },
  });
  Object.assign(globalThis, { backend: backends.of(root) });
  return {
    frame(frame: Frame) {
      const pass = frame.encoder.beginRenderPass({ colorAttachments: { view: surface.view(frame) } });
      pass.setPipeline(drawing);
      pass.draw(3);
      pass.end();
      Object.assign(globalThis, { drawn: true });
    },
  };
}
`;
const index = `import * as web from '@typegpu/kit/web';
import { webgl } from '@typegpu/kit-gl';
import { createScene } from './scene.ts';

web.host(createScene, { canvas: document.querySelector('canvas')!, fallback: webgl.fallback() });
`;

try {
  for (const name of ['@typegpu/kit', '@typegpu/kit-gl'])
    run('pnpm', ['--filter', name, 'pack', '--pack-destination', temporary], repo);
  const tarballs = (await readdir(temporary)).filter((file) => file.endsWith('.tgz'));
  const kit = tarballs.find((file) => file.startsWith('typegpu-kit-0'));
  const gl = tarballs.find((file) => file.startsWith('typegpu-kit-gl-'));
  assert(kit && gl, `pack produced both tarballs: ${tarballs.join(', ')}`);

  // As the README says: copy the patch the package ships and register it
  run('tar', ['-xzf', gl, 'package/patches']);
  await mkdir(path.join(temporary, 'patches'));
  await rename(
    path.join(temporary, 'package/patches', PATCH),
    path.join(temporary, 'patches', PATCH),
  );
  const starter = path.join(repo, 'packages/typegpu-create-kit/template');
  const { dependencies, devDependencies } = JSON.parse(
    await readFile(path.join(starter, 'package.json'), 'utf8'),
  );
  await writeFile(
    path.join(temporary, 'package.json'),
    JSON.stringify({
      private: true,
      type: 'module',
      dependencies: {
        ...dependencies,
        ...devDependencies,
        '@typegpu/kit': `file:./${kit}`,
        '@typegpu/kit-gl': `file:./${gl}`,
      },
    }),
  );
  await writeFile(
    path.join(temporary, 'pnpm-workspace.yaml'),
    `allowBuilds:\n  esbuild: false\npatchedDependencies:\n  typegpu@0.12.11: patches/${PATCH}\n`,
  );
  run('pnpm', ['install', '--config.auto-install-peers=false']);
  const manifest = JSON.parse(
    await readFile(path.join(temporary, 'node_modules/@typegpu/kit-gl/package.json'), 'utf8'),
  );
  assert.equal(manifest.peerDependencies.typegpu, '0.12.11');
  assert(!('@typegpu/gl' in (manifest.dependencies ?? {})), 'the vendored generator ships inside');

  // The program, under the starter's settings
  const app = path.join(temporary, 'app');
  await mkdir(path.join(app, 'src'), { recursive: true });
  for (const file of ['index.html', 'tsconfig.json', 'vite.config.ts'])
    await writeFile(path.join(app, file), await readFile(path.join(starter, file), 'utf8'));
  await writeFile(path.join(app, 'src/scene.ts'), scene);
  await writeFile(path.join(app, 'src/index.ts'), index);
  run(path.join(temporary, 'node_modules/.bin/tsc'), ['-p', app]);
  // The program is one chunk on purpose, which Vite warns about
  run(path.join(temporary, 'node_modules/.bin/vite'), ['build', '--logLevel', 'error'], app);

  /** Vite's chunks for an entry module, with TypeGPU external. */
  const chunks = async (name: string, code: string) => {
    const entry = path.join(temporary, `${name}.js`);
    await writeFile(entry, code);
    const result = await build({
      configFile: false,
      root: temporary,
      logLevel: 'error',
      build: {
        write: false,
        minify: false,
        lib: { entry, formats: ['es'] },
        rolldownOptions: { external: (id) => id === 'typegpu' || id.startsWith('typegpu/') },
      },
    });
    const output = Array.isArray(result) ? result[0] : result;
    assert(output && 'output' in output);
    return output.output.flatMap((item) => (item.type === 'chunk' ? [item] : []));
  };
  // A program that imports the fallback and never hands it to the host keeps none of it
  const unused = await chunks(
    'unused',
    "import * as web from '@typegpu/kit/web';\nimport { webgl } from '@typegpu/kit-gl';\nexport const host = web.host;\n",
  );
  assert(
    !unused.some((chunk) => chunk.code.includes(MARKER)),
    'an unused fallback stays in the bundle',
  );
  // One that imports it where it needs it gets it as a chunk of its own
  const lazy = await chunks(
    'lazy',
    "export const load = () => import('@typegpu/kit-gl').then(({ webgl }) => webgl.fallback());\n",
  );
  const entry = lazy.find((chunk) => chunk.isEntry);
  assert(entry && !entry.code.includes(MARKER), 'a dynamic import inlines the fallback');
  assert(
    lazy.some((chunk) => !chunk.isEntry && chunk.code.includes(MARKER)),
    'a dynamic import splits the fallback into its own chunk',
  );

  if (browser) {
    const { chromium } = await import('playwright');
    const launched = await chromium.launch({ args: ['--enable-gpu', '--ignore-gpu-blocklist'] });
    /** Opens the built program in a browser without WebGPU and returns the page and its errors. */
    const open = async (root: string) => {
      const server = await preview({
        root,
        logLevel: 'warn',
        preview: { port: 0, host: '127.0.0.1' },
      });
      const page = await launched.newPage({ viewport: { width: 64, height: 64 } });
      // As the example runner's `nogpu` backend makes one
      await page.addInitScript(() => {
        delete (Navigator.prototype as { gpu?: unknown }).gpu;
        for (const name of Object.getOwnPropertyNames(globalThis)) {
          if (name.startsWith('GPU')) delete (globalThis as Record<string, unknown>)[name];
        }
      });
      const errors: string[] = [];
      page.on('pageerror', (error) => errors.push(String(error)));
      page.on('console', (message) => {
        if (message.type() === 'error') errors.push(message.text());
      });
      await page.goto(server.resolvedUrls!.local[0]!);
      const close = () => new Promise<void>((done) => server.httpServer.close(() => done()));
      return { page, errors, close };
    };
    try {
      const { page, errors, close } = await open(app);
      try {
        await page.waitForFunction(() => 'drawn' in globalThis, undefined, { timeout: 30_000 });
        assert.deepEqual(errors, []);
        const backend = await page.evaluate(() => {
          const { kind, compute } = (globalThis as { backend?: Record<string, unknown> }).backend!;
          return { kind, compute };
        });
        assert.deepEqual(backend, { kind: 'webgl', compute: 'cpu' });
        const shot = await page.locator('canvas').screenshot();
        const pixel = await page.evaluate(async (png) => {
          const image = new Image();
          image.src = `data:image/png;base64,${png}`;
          await image.decode();
          const canvas = new OffscreenCanvas(1, 1);
          const context = canvas.getContext('2d')!;
          context.drawImage(image, -image.width / 2, -image.height / 2);
          return [...context.getImageData(0, 0, 1, 1).data];
        }, shot.toString('base64'));
        const expected = [255, 128, 64, 255];
        assert(
          pixel.every((value, i) => Math.abs(value - expected[i]!) <= 3),
          `the canvas shows ${pixel}, not ${expected}`,
        );
      } finally {
        await close();
      }

      // Without the patch, the host reports the one error that names it
      const bare = path.join(temporary, 'unpatched');
      await mkdir(path.join(bare, 'app'), { recursive: true });
      const project = JSON.parse(await readFile(path.join(temporary, 'package.json'), 'utf8'));
      for (const name of ['@typegpu/kit', '@typegpu/kit-gl'])
        project.dependencies[name] = project.dependencies[name].replace('file:./', 'file:../');
      await writeFile(path.join(bare, 'package.json'), JSON.stringify(project));
      await writeFile(path.join(bare, 'pnpm-workspace.yaml'), 'allowBuilds:\n  esbuild: false\n');
      run('pnpm', ['install', '--config.auto-install-peers=false'], bare);
      await cp(path.join(app, 'src'), path.join(bare, 'app/src'), { recursive: true });
      for (const file of ['index.html', 'vite.config.ts'])
        await cp(path.join(app, file), path.join(bare, 'app', file));
      run(
        path.join(bare, 'node_modules/.bin/vite'),
        ['build', '--logLevel', 'error'],
        path.join(bare, 'app'),
      );
      const unpatched = await open(path.join(bare, 'app'));
      try {
        await unpatched.page.waitForFunction(() => document.readyState === 'complete');
        const deadline = Date.now() + 30_000;
        while (!unpatched.errors.some((error) => error.includes(PATCH)) && Date.now() < deadline)
          await unpatched.page.waitForTimeout(100);
        assert(
          unpatched.errors.some((error) => error.includes(PATCH)),
          `an unpatched TypeGPU reports: ${unpatched.errors.join('\n') || 'nothing'}`,
        );
      } finally {
        await unpatched.close();
      }
    } finally {
      await launched.close();
    }
  }
  console.log(
    `test:pack (gl) passed: tarballs, patch, types, bundles, tree shaking${browser ? ', Chromium without WebGPU, the unpatched error' : ''}`,
  );
} finally {
  await rm(temporary, { recursive: true, force: true });
}
