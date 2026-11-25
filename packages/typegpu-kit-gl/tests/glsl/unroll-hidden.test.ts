import { d, std, tgpu } from 'typegpu';
import { expect, it, vi } from 'vitest';
import { glslRoot } from './utils.ts';

// `tgpu.unroll` as a TypeGPU whose internals moved might look to the generator: it still unrolls
// when TypeGPU's code generation calls it with a context, but gives the generator's probe, which
// passes none, nothing to recover `UnrollableIterable` from
vi.mock('typegpu', async (importOriginal) => {
  const actual = await importOriginal<typeof import('typegpu')>();
  const real = actual.tgpu.unroll as unknown as ((value: unknown) => unknown) &
    Record<symbol, unknown>;
  const hidden = ((value: unknown) => real(value)) as typeof real;
  for (const key of Object.getOwnPropertySymbols(real)) {
    const value = real[key];
    hidden[key] =
      typeof value === 'function'
        ? (ctx: unknown, args: unknown) => {
            if (ctx === undefined) throw new Error('No context');
            return (value as (c: unknown, a: unknown) => unknown)(ctx, args);
          }
        : value;
  }
  return { ...actual, tgpu: { ...actual.tgpu, unroll: hidden } };
});

it('counts every loop as a runtime loop when the unrolled class cannot be recovered', () => {
  const { root, computes, recording } = glslRoot({ compute: true });
  const cells = root.createMutable(d.arrayOf(d.atomic(d.u32), 4));
  const kernel = tgpu.computeFn({ workgroupSize: [4] })(() => {
    'use gpu';
    for (const v of tgpu.unroll([0, 1, 2, 3])) std.atomicOr(cells.$[v], 1);
  });
  root.unwrap(root.createComputePipeline({ compute: kernel }));
  const [verdict] = computes();
  if (!verdict) throw new Error('No compute module');
  expect('reason' in verdict ? verdict.reason : 'form').toContain('inside a loop');
  // TypeGPU still unrolled it: the kernel's entry is four iterations, with no loop left
  const cpu = recording.log.shaders.find((code) => code.includes('//#kit-gl compute')) ?? '';
  const { name } = JSON.parse(/\/\/#entry (.*)/.exec(cpu)?.[1] ?? '{}') as { name?: string };
  const from = cpu.indexOf(`function ${name}(`);
  const next = cpu.indexOf('\nfunction', from + 1);
  const body = cpu.slice(from, next === -1 ? undefined : next);
  expect(from).toBeGreaterThanOrEqual(0);
  expect(body).not.toMatch(/\b(for|while)\s*\(/);
  expect(body.match(/unrolled iteration/g)?.length).toBe(4);
});
