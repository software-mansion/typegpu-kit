import { d, std, tgpu, type TgpuRoot } from 'typegpu';
import { afterEach, expect, test } from 'vitest';
import type { GLDevice } from '../../src/device/index.ts';
import { glRoot } from './compare.ts';

let root: TgpuRoot | undefined;

afterEach(() => {
  root?.destroy();
  root = undefined;
});

const SIZE = 64;

test('unlabelled compute pipelines take their entry functions’ names in the stats', async () => {
  root = glRoot();
  const made = root;
  const cells = made.createMutable(d.arrayOf(d.u32, SIZE));
  const counter = made.createMutable(d.atomic(d.u32));
  const fillCells = tgpu.computeFn({
    workgroupSize: [SIZE],
    in: { id: d.builtin.globalInvocationId },
  })(({ id }) => {
    'use gpu';
    cells.$[id.x] = id.x;
  });
  // A barrier keeps this one on the CPU, so it has a fallback reason
  const countCells = tgpu.computeFn({
    workgroupSize: [SIZE],
    in: { id: d.builtin.globalInvocationId },
  })(({ id }) => {
    'use gpu';
    std.workgroupBarrier();
    std.atomicAdd(counter.$, cells.$[id.x]!);
  });
  // Made in a list, so TypeGPU names neither pipeline
  const pipelines = [fillCells, countCells].map((compute) =>
    made.createComputePipeline({ compute }),
  );
  for (const pipeline of pipelines) pipeline.dispatchWorkgroups(1);
  await counter.read();
  const { stats } = made.device as GLDevice;
  expect([...stats.kernels.keys()].toSorted()).toEqual(['countCells', 'fillCells (gpu)']);
  expect(stats.fallbacks.get('countCells')).toMatch(/^CPU only/);
  expect([...stats.fallbacks.keys()]).not.toContain('<unnamed>');
});

test('an explicit name still labels a compute pipeline', async () => {
  root = glRoot();
  const made = root;
  const cells = made.createMutable(d.arrayOf(d.u32, SIZE));
  const fillCells = tgpu.computeFn({
    workgroupSize: [SIZE],
    in: { id: d.builtin.globalInvocationId },
  })(({ id }) => {
    'use gpu';
    cells.$[id.x] = id.x;
  });
  const [pipeline] = [fillCells].map((compute) =>
    made.createComputePipeline({ compute }).$name('filling'),
  );
  pipeline!.dispatchWorkgroups(1);
  await cells.read();
  expect([...(made.device as GLDevice).stats.kernels.keys()]).toEqual(['filling (gpu)']);
});

test('entry functions no variable holds, which TypeGPU names after their stage, get numbers', async () => {
  root = glRoot();
  const made = root;
  const cells = made.createMutable(d.arrayOf(d.u32, SIZE));
  const pipelines = [0, 1].map((offset) =>
    made.createComputePipeline({
      compute: tgpu.computeFn({ workgroupSize: [SIZE], in: { id: d.builtin.globalInvocationId } })(
        ({ id }) => {
          'use gpu';
          cells.$[id.x] = id.x + offset;
        },
      ),
    }),
  );
  for (const pipeline of pipelines) pipeline.dispatchWorkgroups(1);
  await cells.read();
  expect([...(made.device as GLDevice).stats.kernels.keys()].toSorted()).toEqual([
    'compute (gpu)',
    'compute 2 (gpu)',
  ]);
});
