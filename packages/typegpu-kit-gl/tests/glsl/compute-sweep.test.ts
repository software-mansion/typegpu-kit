import { writeFileSync } from 'node:fs';
import { frames } from '@typegpu/kit';
import { offscreenSurface } from '@typegpu/kit/testing';
import { afterAll, beforeAll, expect, it } from 'vitest';
import { examples } from '../../../typegpu-kit-examples/src/index.ts';
import { create } from '../../../typegpu-kit-examples/src/hosts.ts';
import { fromDisk } from '../../../typegpu-kit-examples/tests/utils/examples.ts';
import { PASS_VERTEX } from '../../src/glsl/computeModule.ts';
import { glslRoot, type ComputeVerdict } from './utils.ts';
import { problems, validator, type Validator } from './validator.ts';

/** Set to a file to write every example's verdicts there as JSON. */
const report = process.env.COMPUTE_REPORT;

let gl: Validator;
beforeAll(async () => {
  gl = await validator();
});
const verdicts: Record<string, { name: string; grid?: number; reason?: string }[]> = {};
afterAll(async () => {
  await gl.close();
  if (report) writeFileSync(report, `${JSON.stringify(verdicts, null, 2)}\n`);
});

/**
 * The GPU form's bindings that the pipeline's WGSL declares differently: missing, of another kind,
 * or written by the form but not writable in WGSL.
 */
function misnumbered(verdict: ComputeVerdict): string[] {
  if (!('module' in verdict)) return [];
  const declared = new Map<string, string>();
  for (const [, group, binding, rest] of verdict.wgsl.matchAll(
    /@group\((\d+)\)\s*@binding\((\d+)\)\s*var(.*?);/g,
  )) {
    declared.set(`${group}:${binding}`, rest ?? '');
  }
  return verdict.module.bindings.flatMap(({ group, binding, kind, writes }) => {
    const wgsl = declared.get(`${group}:${binding}`);
    const right =
      wgsl !== undefined &&
      (kind === 'buffer'
        ? /^<(uniform|storage)/.test(wgsl) && (!writes || /read_write/.test(wgsl))
        : kind === 'texture'
          ? /:\s*texture_/.test(wgsl) &&
            (!writes || /texture_storage_.*(write|read_write)>/.test(wgsl))
          : /:\s*sampler/.test(wgsl));
    return right ? [] : [`${verdict.name} @group(${group}) @binding(${binding}) ${kind}: ${wgsl}`];
  });
}

const settle = () => new Promise((resolve) => setTimeout(resolve, 0));

/**
 * Every example's compute pipelines get a GPU form or a reason they cannot, and every GPU form's
 * passes compile and link in WebGL 2.
 */
it.each(examples.map((example) => example.key))('%s', async (key) => {
  const { root, computes } = glslRoot();
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

  const seen = new Map<string, ComputeVerdict>();
  for (const verdict of computes()) {
    seen.set('code' in verdict ? verdict.code : `${verdict.name}: ${verdict.reason}`, verdict);
  }
  const unique = [...seen.values()];
  verdicts[key] = unique.map((verdict) =>
    'module' in verdict
      ? { name: verdict.name, grid: verdict.module.grid }
      : { name: verdict.name, reason: verdict.reason },
  );

  expect(unique.flatMap(misnumbered)).toEqual([]);

  const passes = unique.flatMap((verdict) =>
    'module' in verdict
      ? verdict.module.passes.map((pass) => ({ name: verdict.name, fragment: pass.fragment }))
      : [],
  );
  const reports = await gl.check(
    passes.map(({ fragment }) => ({
      kind: 'render' as const,
      vertex: PASS_VERTEX,
      fragment,
      buffers: [],
      textures: [],
      levels: [],
    })),
  );
  const failures = reports.flatMap((each, i) =>
    problems(each).map((line) => `${passes[i]?.name}: ${line.split('\n').slice(0, 3).join(' | ')}`),
  );
  expect(failures).toEqual([]);
});
