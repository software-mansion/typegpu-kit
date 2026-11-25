import { splitPhases } from '../../src/cpu/phases.ts';
import { workgroup } from '@typegpu/kit';
import { d, std, tgpu } from 'typegpu';
import { expect, it } from 'vitest';
import { cpuRoot } from './device.ts';
import { workgroupReturns } from './calls.ts';

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
              Array.from({ length: i % 4 }, (_item, j) => Math.floor(i / 4) * 4 + j + 1).reduce(
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

it('skips whole workgroups that return before or between barrier phases', async () => {
  let code = '';
  const root = cpuRoot({
    onModule: (source) => {
      code = source;
    },
  });
  try {
    expect(await workgroupReturns(root)).toEqual(
      Array.from({ length: 16 }, (_, i) => {
        const group = Math.floor(i / 4);
        return group === 1
          ? [0, 0]
          : [group * 4 + ((i + 1) % 4) + 1, group === 2 ? 0 : group * 4 + ((i + 2) % 4) + 1];
      }).flat(),
    );
    expect(code).toContain('"phases":3');
  } finally {
    root.destroy();
  }
});

it('reports lanes returning at different barrier phases as divergent', () => {
  const root = cpuRoot();
  try {
    for (const afterFirst of [false, true]) {
      const kernel = tgpu.computeFn({
        workgroupSize: [4],
        in: { lid: d.builtin.localInvocationIndex },
      })(({ lid }) => {
        'use gpu';
        if (afterFirst) std.workgroupBarrier();
        if (lid === 0) return;
        std.workgroupBarrier();
      });
      expect(() => root.createComputePipeline({ compute: kernel }).dispatchWorkgroups(1)).toThrow(
        'divergent workgroup barrier',
      );
    }
  } finally {
    root.destroy();
  }
});

it('zeros reused workgroup integer arrays between groups and dispatches', async () => {
  const root = cpuRoot();
  const shared = tgpu.workgroupVar(d.arrayOf(d.atomic(d.u32), 4));
  const enabled = root.createUniform(d.u32, 1);
  const output = root.createMutable(d.arrayOf(d.u32, 8));
  const kernel = tgpu.computeFn({
    workgroupSize: [4],
    in: {
      gid: d.builtin.globalInvocationId,
      wid: d.builtin.workgroupId,
      lid: d.builtin.localInvocationIndex,
    },
  })(({ gid, wid, lid }) => {
    'use gpu';
    if (wid.x === 0 && enabled.$ !== 0) std.atomicAdd(shared.$[lid], gid.x + d.u32(1));
    std.workgroupBarrier();
    output.$[gid.x] = std.atomicLoad(shared.$[lid]);
  });
  try {
    const pipeline = root.createComputePipeline({ compute: kernel });
    pipeline.dispatchWorkgroups(2);
    expect(await output.read()).toEqual([1, 2, 3, 4, 0, 0, 0, 0]);
    enabled.write(0);
    pipeline.dispatchWorkgroups(2);
    expect(await output.read()).toEqual([0, 0, 0, 0, 0, 0, 0, 0]);
  } finally {
    root.destroy();
  }
});

it('hoists uninitialized temporaries into each lane closure', () => {
  const phases = splitPhases(
    '{let cpuIndex,cpuBase;(yield);cpuIndex=id;cpuBase=values;(yield);cpuBase[cpuIndex]+=1;}',
    ['id', 'values'],
  )!;
  const factory = new Function(`'use strict';return (${phases.source})();`);
  const values = [10, 20, 30, 40];
  const before = new Set(Reflect.ownKeys(globalThis));
  const lanes = values.map((_, id) => {
    const lane = factory();
    lane[0](id, values);
    return lane;
  });
  for (const lane of lanes) lane[1]();
  for (const lane of lanes) lane[2]();
  expect(values).toEqual([11, 21, 31, 41]);
  expect(Reflect.ownKeys(globalThis).filter((key) => !before.has(key))).toEqual([]);
});

it('keeps compiled compound-store temporaries private across plain phases', async () => {
  let code = '';
  const root = cpuRoot({
    onModule: (source) => {
      code = source;
    },
  });
  const values = root.createMutable(d.arrayOf(d.u32, 5), [10, 20, 30, 40, 50]);
  const before = new Set(Reflect.ownKeys(globalThis));
  root
    .createComputePipeline({
      compute: tgpu.computeFn({ workgroupSize: [4], in: { lid: d.builtin.localInvocationIndex } })(
        ({ lid }) => {
          'use gpu';
          std.workgroupBarrier();
          values.$[lid + 1] += 1;
        },
      ),
    })
    .dispatchWorkgroups(1);
  expect(code).toContain('"phases":2');
  expect(await values.read()).toEqual([10, 21, 31, 41, 51]);
  expect(Reflect.ownKeys(globalThis).filter((key) => !before.has(key))).toEqual([]);
  root.destroy();
});
