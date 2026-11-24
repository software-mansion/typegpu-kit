import { mkdirSync, writeFileSync } from 'node:fs';
import { frames } from '@typegpu/kit';
import { offscreenSurface } from '@typegpu/kit/testing';
import { afterAll, beforeAll, expect, it } from 'vitest';
import { examples } from '../../../typegpu-kit-examples/src/index.ts';
import { create } from '../../../typegpu-kit-examples/src/hosts.ts';
import { fromDisk } from '../../../typegpu-kit-examples/tests/utils/examples.ts';
import { parse } from '../../src/glsl/renderModule.ts';
import { glslRoot } from './utils.ts';
import { problems, validator, type Validator } from './validator.ts';

/** Set to a directory to write each failing module's stages there. */
const dump = process.env.GLSL_DUMP;

let gl: Validator;
beforeAll(async () => {
  gl = await validator();
});
afterAll(() => gl.close());

const settle = () => new Promise((resolve) => setTimeout(resolve, 0));

/**
 * Every example's render pipelines, resolved to GLSL by drawing its frames on a recording device,
 * compile and link as GLSL ES 3.00 in WebGL 2.
 */
it.each(examples.map((example) => example.key))('%s compiles and links on WebGL 2', async (key) => {
  const { root, codes } = glslRoot();
  const surface = offscreenSurface(root, { width: 64, height: 36, format: 'bgra8unorm' });
  const scene = await create(key, root, surface, fromDisk, {});
  const loop = frames.create(root, (frame) => scene.frame(frame));
  const state: { ready: boolean; error?: unknown } = { ready: scene.ready === undefined };
  scene.ready?.then(
    () => (state.ready = true),
    (error: unknown) => (state.error = error),
  );
  let seconds = 0;
  for (let i = 0; i < 100 && !state.ready && !state.error; i++, seconds += 0.04) {
    loop.run(seconds);
    await settle();
  }
  if (state.error) throw state.error;
  for (let i = 0; i < 4; i++, seconds += 0.04) loop.run(seconds);
  loop.destroy();
  scene.destroy?.();

  const unique = [...new Set(codes())];
  const modules = unique.map((code) => parse(code));
  const reports = await gl.check(modules);
  const failures = reports.flatMap((report, i) => {
    const found = problems(report);
    if (found.length > 0 && dump) {
      mkdirSync(dump, { recursive: true });
      writeFileSync(`${dump}/${key}-${i}.vert`, modules[i]?.vertex ?? '');
      writeFileSync(`${dump}/${key}-${i}.frag`, modules[i]?.fragment ?? '');
      writeFileSync(`${dump}/${key}-${i}.module`, unique[i] ?? '');
    }
    return found.map((line) => `#${i}: ${line.split('\n').slice(0, 4).join(' | ')}`);
  });
  expect(modules.length, 'render modules').toBeGreaterThan(0);
  expect(failures).toEqual([]);
});
