import { d, std, tgpu as real, type TgpuMutable } from 'typegpu';
import { expect, it } from 'vitest';
import { glslRoot } from './utils.ts';

// Spelled `tgpu.unroll` in the kernels below, but a runtime range: only what TypeGPU resolves a
// loop's iterable to may make the loop unrolled, never how the call is written
const tgpu = { unroll: std.range };
// The real one under other names
const unroll = real.unroll;
const kit = { unroll: real.unroll };

const gid = { gid: d.builtin.globalInvocationId };

type Cells = TgpuMutable<d.WgslArray<d.Atomic<d.U32>>>;

/** The form, or the reason, a kernel that `make` builds over four atomic words gets. */
function verdictOf(make: (cells: Cells) => unknown): string {
  const { root, computes } = glslRoot({ compute: true });
  const cells = root.createMutable(d.arrayOf(d.atomic(d.u32), 4));
  root.unwrap(root.createComputePipeline({ compute: make(cells) as never }));
  const [verdict] = computes();
  if (!verdict) throw new Error('No compute module');
  return 'reason' in verdict ? verdict.reason : 'form';
}

it('counts a loop spelled tgpu.unroll over a runtime range as a runtime loop', () => {
  const verdict = verdictOf((cells) =>
    real.computeFn({ workgroupSize: [4], in: gid })(() => {
      'use gpu';
      for (const v of tgpu.unroll(4)) {
        std.atomicOr(cells.$[v], 1);
      }
    }),
  );
  expect(verdict).toContain('inside a loop');
});

it('expands a loop over the real tgpu.unroll, under its own name or another', () => {
  const made = {
    real: (cells: Cells) =>
      real.computeFn({ workgroupSize: [4], in: gid })(() => {
        'use gpu';
        for (const v of real.unroll([0, 1, 2, 3])) std.atomicOr(cells.$[v], 1);
      }),
    alias: (cells: Cells) =>
      real.computeFn({ workgroupSize: [4], in: gid })(() => {
        'use gpu';
        for (const v of unroll([0, 1, 2, 3])) std.atomicOr(cells.$[v], 1);
      }),
    member: (cells: Cells) =>
      real.computeFn({ workgroupSize: [4], in: gid })(() => {
        'use gpu';
        for (const v of kit.unroll([0, 1, 2, 3])) std.atomicOr(cells.$[v], 1);
      }),
  };
  for (const [name, make] of Object.entries(made)) expect(verdictOf(make), name).toBe('form');
});
