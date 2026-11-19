import { d, tgpu } from 'typegpu';
import { expect, it } from 'vitest';
import { census } from '../../src/debug/index.ts';
import { recordingRoot } from '../../src/testing.ts';

it('counts modules, layout entries per stage and compiles that warm-up missed', async () => {
  const { root } = recordingRoot();
  const counted = census.create(root, { limits: { maxStorageBuffersPerShaderStage: 1 } });
  const data = root.createMutable(d.arrayOf(d.u32, 4));
  const other = root.createReadonly(d.arrayOf(d.u32, 4));
  // A shared layout binds both entries, though the shader reads one
  const shared = tgpu.bindGroupLayout({
    read: { storage: d.arrayOf(d.u32, 4) },
    unread: { uniform: d.u32 },
  });

  const empty = tgpu.computeFn({ workgroupSize: [1] })(() => {
    'use gpu';
  });
  const busy = tgpu.computeFn({ workgroupSize: [1] })(() => {
    'use gpu';
    data.$[0] = data.$[1];
  });
  const heavy = tgpu.computeFn({ workgroupSize: [1] })(() => {
    'use gpu';
    data.$[0] = other.$[1] + shared.$.read[0];
  });

  await root.createComputePipeline({ compute: empty }).$name('early').initAsync();
  counted.warm();
  root.createComputePipeline({ compute: busy }).$name('late').initSync();
  root.createComputePipeline({ compute: heavy }).$name('heavy').initSync();

  expect(counted.modules).toBe(3);
  expect(
    counted.pipelines.map((p) => [p.label, p.async, p.groups, p.bindings.map((b) => b.kind)]),
  ).toEqual([
    ['early', true, 0, []],
    ['late', false, 1, ['storage']],
    ['heavy', false, 2, ['storage', 'storage', 'storage', 'uniform']],
  ]);
  expect(counted.late.map((p) => p.label)).toEqual(['late', 'heavy']);
  expect(counted.overBudget.map((p) => [p.label, p.over])).toEqual([
    ['heavy', [{ stage: 'compute', kind: 'storage', count: 3, limit: 1 }]],
  ]);
});

it('restores the device on destroy', () => {
  const { root } = recordingRoot();
  const before = root.device.createShaderModule;
  const counted = census.create(root);

  expect(root.device.createShaderModule).not.toBe(before);

  counted.destroy();
  expect(root.device.createShaderModule).toBe(before);
});
