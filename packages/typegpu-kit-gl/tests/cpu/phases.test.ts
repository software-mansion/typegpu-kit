import { workgroup } from '@typegpu/kit';
import { d, std, tgpu } from 'typegpu';
import { expect, it } from 'vitest';
import { cpuRoot } from './device.ts';

it('reuses plain barrier phases across workgroups and preserves private state and locals', async () => {
  let code = '';
  const root = cpuRoot({
    onModule: (source) => {
      code = source;
    },
  });
  const shared = tgpu.workgroupVar(d.arrayOf(d.u32, 4));
  const state = tgpu.privateVar(d.vec2u);
  const output = root.createMutable(d.arrayOf(d.vec2u, 12));
  const kernel = tgpu.computeFn({
    workgroupSize: [4],
    in: { gid: d.builtin.globalInvocationId, lid: d.builtin.localInvocationIndex },
  })(({ gid, lid }) => {
    'use gpu';
    let first = gid.x * d.u32(7);
    state.$ = d.vec2u(first, gid.x);
    shared.$[lid] = gid.x;
    std.workgroupBarrier();
    let next = shared.$[(lid + 1) % 4];
    first += next;
    state.$.y += d.u32(100);
    std.workgroupBarrier();
    output.$[gid.x] = d.vec2u(first + state.$[0], state.$[1]);
  });
  try {
    const pipeline = root.createComputePipeline({ compute: kernel });
    for (let run = 0; run < 2; run++) {
      pipeline.dispatchWorkgroups(3);
      expect((await output.read()).flatMap((v) => [v.x, v.y])).toEqual(
        Array.from({ length: 12 }, (_, i) => [
          i * 14 + Math.floor(i / 4) * 4 + ((i + 1) % 4),
          i + 100,
        ]).flat(),
      );
    }
    expect(code).toContain('"phases":3');
  } finally {
    root.destroy();
  }
});

it('keeps repeated scan helper locals separate across unrolled barrier phases', async () => {
  let code = '';
  const root = cpuRoot({
    onModule: (source) => {
      code = source;
    },
  });
  const scan = workgroup.scan(workgroup.sum(d.u32), 4);
  const output = root.createMutable(d.arrayOf(d.vec2u, 12));
  const kernel = tgpu.computeFn({
    workgroupSize: [4],
    in: { gid: d.builtin.globalInvocationId, lid: d.builtin.localInvocationIndex },
  })(({ gid, lid }) => {
    'use gpu';
    let first = scan(gid.x + d.u32(1), lid);
    let second = scan(d.u32(1), lid);
    if (gid.x === 11) return;
    output.$[gid.x] = d.vec2u(first, second);
  });
  try {
    root.createComputePipeline({ compute: kernel }).dispatchWorkgroups(3);
    expect((await output.read()).flatMap((v) => [v.x, v.y])).toEqual(
      Array.from({ length: 12 }, (_, i) =>
        i === 11
          ? [0, 0]
          : [
              Array.from({ length: i % 4 }, (_, j) => Math.floor(i / 4) * 4 + j + 1).reduce(
                (a, b) => a + b,
                0,
              ),
              i % 4,
            ],
      ).flat(),
    );
    expect(code).toContain('"phases":13');
  } finally {
    root.destroy();
  }
});
