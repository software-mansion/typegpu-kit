import { execFileSync } from 'node:child_process';
import { mkdirSync, mkdtempSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { frames } from '@typegpu/kit';
import { offscreenSurface, recordingRoot } from '@typegpu/kit/testing';
import { expect, it } from 'vitest';

const here = import.meta.dirname;
const json = (file: string) => JSON.parse(readFileSync(file, 'utf8'));

const scenes = readdirSync(path.join(here, '../scenes'));

/** Runs the initializer in a new temporary directory, without a terminal, as pnpm would. */
function create(...args: string[]) {
  const cwd = mkdtempSync(path.join(tmpdir(), 'create-kit-'));
  const bin = path.join(here, '../index.js');
  const agent = 'pnpm/11.18.0 npm/? node/v24.2.0 darwin arm64';
  const run = () =>
    execFileSync('node', [bin, ...args], {
      cwd,
      encoding: 'utf8',
      stdio: 'pipe',
      env: { ...process.env, npm_config_user_agent: agent },
    });
  return { cwd, run };
}

it('writes the shader scene, named for its directory, and says what to run next', () => {
  const { cwd, run } = create('My Scene');
  const printed = run();
  const project = path.join(cwd, 'My Scene');

  expect(readdirSync(project, { recursive: true }).toSorted()).toEqual([
    '.gitignore',
    'AGENTS.md',
    'index.html',
    'package.json',
    'src',
    'src/index.ts',
    'src/scene.ts',
    'tsconfig.json',
    'vite.config.ts',
  ]);
  expect(json(path.join(project, 'package.json')).name).toBe('my-scene');
  expect(printed).toContain("cd 'My Scene'\n  pnpm install\n  pnpm dev\n");
  expect(readFileSync(path.join(project, 'src/scene.ts'), 'utf8')).toBe(
    readFileSync(path.join(here, '../scenes/shader/scene.ts'), 'utf8'),
  );
});

it('starts from the scene --scene names, and refuses one it lacks', () => {
  const { cwd, run } = create('lit', '--scene', 'lit');
  run();
  for (const file of ['index.ts', 'scene.ts']) {
    expect(readFileSync(path.join(cwd, 'lit/src', file), 'utf8')).toBe(
      readFileSync(path.join(here, '../scenes/lit', file), 'utf8'),
    );
  }

  const missing = create('none', '--scene', 'none');
  expect(missing.run).toThrow(/There is no none scene\. Choose from shader, /);
  expect(readdirSync(missing.cwd)).toEqual([]);
});

it('refuses a directory that is not empty', () => {
  const { cwd, run } = create('taken');
  mkdirSync(path.join(cwd, 'taken'));
  writeFileSync(path.join(cwd, 'taken/notes.md'), '');

  expect(run).toThrow(/taken exists and is not empty/);
  expect(readdirSync(path.join(cwd, 'taken'))).toEqual(['notes.md']);
});

it('depends on the versions the workspace builds the kit with', () => {
  const repo = path.join(here, '../../..');
  const workspace = readFileSync(path.join(repo, 'pnpm-workspace.yaml'), 'utf8');
  const specs = new Map<string, string>();
  for (const [, name, spec] of workspace.matchAll(/^ {2,4}'?([@\w/.-]+)'?: (.+)$/gm))
    if (!specs.has(name!)) specs.set(name!, spec!);
  specs.set('vite', json(path.join(repo, 'package.json')).devDependencies.vite);
  specs.set('@typegpu/kit', `^${json(path.join(here, '../../typegpu-kit/package.json')).version}`);

  const { dependencies, devDependencies } = json(path.join(here, '../template/package.json'));
  const pinned = { ...dependencies, ...devDependencies };
  expect(pinned).toEqual(
    Object.fromEntries(Object.keys(pinned).map((name) => [name, specs.get(name)])),
  );
  expect(json(path.join(here, '../package.json')).version).toBe(
    specs.get('@typegpu/kit')!.slice(1),
  );
});

it('prints the logo the site shows', () => {
  const site = path.join(here, '../../../apps/typegpu-kit-docs/src/assets/logo.svg');
  expect(readFileSync(path.join(here, '../logo.svg'), 'utf8')).toBe(readFileSync(site, 'utf8'));
});

it.each(scenes)(
  'creates nothing in a steady frame of the %s scene, and submits once',
  async (name) => {
    const { root, log } = recordingRoot();
    const surface = offscreenSurface(root, { width: 64, height: 36, format: 'bgra8unorm' });
    const { createScene } = await import(`../scenes/${name}/scene.ts`);
    const scene = await createScene(root, surface);
    const loop = frames.create(root, scene.frame);

    loop.run(0);
    loop.run(0.04);
    const counts = log.counts();
    const { submissions } = log;
    loop.run(0.08);
    loop.run(0.12);

    expect(log.counts()).toEqual(counts);
    expect(log.submissions - submissions).toBe(2);
  },
);

it('draws the shader scene in one pass', async () => {
  const { root, log } = recordingRoot();
  const surface = offscreenSurface(root, { width: 64, height: 36, format: 'bgra8unorm' });
  const { createScene } = await import('../scenes/shader/scene.ts');
  const scene = await createScene(root, surface);
  frames.create(root, scene.frame).run(0);

  expect(log.events).toEqual(['begin-render', 'draw', 'end-render', 'submit']);
  expect(log.pipelines).toBe(1);
});

it('steps the simulation scene in one compute pass, then draws it', async () => {
  const { root, log } = recordingRoot({ features: ['indirect-first-instance'] });
  const surface = offscreenSurface(root, { width: 64, height: 36, format: 'bgra8unorm' });
  const { createScene } = await import('../scenes/simulation/scene.ts');
  const scene = await createScene(root, surface);
  const loop = frames.create(root, scene.frame);
  loop.run(0);
  loop.run(0.04);
  log.events.length = 0;
  loop.run(0.08);

  // Two fixed steps, then the two dispatches that list the live motes for the indirect draw
  const compute = ['begin-compute', 'dispatch', 'dispatch', 'dispatch', 'dispatch', 'end-compute'];
  expect(log.events).toEqual([...compute, 'begin-render', 'draw-indirect', 'end-render', 'submit']);
  expect(log.pipelines).toBe(5);
});
