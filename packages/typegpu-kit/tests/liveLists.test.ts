import { d, tgpu } from 'typegpu';
import { expect, it } from 'vitest';
import { liveLists, table } from '../src/index.ts';
import { recordingRoot } from '../src/testing.ts';
import { recorded } from './recorded.ts';

const Block = table.define('Block', { host: { kind: d.u32, pos: d.vec3f } });
const CUBE = 1;

const vars = (wgsl: string) =>
  (wgsl.match(/var<[^>]+> \w+: [^;]+/g) ?? []).map((v) => v.replace(/> \w+:/, '>:')).toSorted();

function setup() {
  const recording = recordingRoot();
  const blocks = table.create(recording.root, Block, { capacity: 1000 });

  const live = liveLists.create(recording.root, {
    ...blocks.live(),
    group: Block.kind.pipe(blocks.inject()),
    draws: [{ vertexCount: 6 }, { indexCount: 36, firstIndex: 12, baseVertex: 4 }],
  });

  return { ...recording, blocks, live };
}

it('counts live rows per group and writes the slots of each group', () => {
  const { log, live } = setup();
  live.initAsync();
  const kernel = log.shaders.find((wgsl) => wgsl.includes('workgroupBarrier'))!;

  expect(vars(kernel)).toEqual([
    'var<storage, read>: array<BlockHost, 1000>',
    'var<storage, read>: array<u32, 32>',
    'var<storage, read_write>: array<atomic<u32>, 2>',
    'var<storage, read_write>: array<u32, 2000>',
    'var<uniform>: index',
    'var<workgroup>: array<atomic<u32>, 2>',
    'var<workgroup>: array<u32, 2>',
  ]);
  expect(kernel).toContain('@workgroup_size(256)');
  expect(kernel).toContain('if (Block_alive(i)) {');
});

it('dispatches up to the table span, then writes the instance counts', () => {
  const { root, log, blocks, live } = setup();
  const record = () => {
    const events = recorded(log, () => {
      const pass = root.createCommandEncoder().beginComputePass();
      live.compact(pass);
      pass.end();
    });
    return events.filter((e) => e === 'dispatch').length;
  };

  expect(record()).toBe(1); // an empty table leaves only the counts to write
  for (let i = 0; i < 300; i++) blocks.add({ kind: i % 2 });
  expect(record()).toBe(2);
});

it('writes each group’s draw and keeps firstInstance at zero', () => {
  const { root, bytes, live } = setup();
  const args = new Uint32Array(bytes(root.unwrap(live.args)).buffer);

  expect([...args]).toEqual([6, 0, 0, 0, 0, 36, 0, 12, 4, 0]);

  live.setDraw(0, { indexCount: 3 });
  expect([...args.subarray(0, 5)]).toEqual([3, 0, 0, 0, 0]);
});

it('draws a group indexed or not, at its record, once something listed it', () => {
  const { root, live } = setup();
  const calls: string[] = [];
  const drawer = {
    drawIndirect: (_: unknown, offset: number) => calls.push(`draw ${offset}`),
    drawIndexedIndirect: (_: unknown, offset: number) => calls.push(`indexed ${offset}`),
  };

  expect(() => live.draw(drawer, 0)).toThrow('Record `compact(pass)`');
  const pass = root.createCommandEncoder().beginComputePass();
  live.compact(pass);
  pass.end();
  live.draw(drawer, 0);
  live.draw(drawer, CUBE);

  expect(calls).toEqual(['draw 0', 'indexed 20']);
});

it('maps instances to rows in a render pipeline, with the group offset baked in', () => {
  const { root, blocks, live } = setup();

  const vertex = tgpu.vertexFn({
    in: { iid: d.builtin.instanceIndex },
    out: { position: d.builtin.position },
  })(({ iid }) => {
    'use gpu';
    const i = live.slot(CUBE, iid);
    return { position: d.vec4f(Block.pos(i), 1) };
  });

  const fragment = tgpu.fragmentFn({ out: d.vec4f })(() => {
    'use gpu';
    return d.vec4f(1);
  });

  const pipeline = root
    .pipe(blocks.inject())
    .pipe(live.inject())
    .createRenderPipeline({ vertex, fragment, targets: { format: 'rgba8unorm' } });

  const wgsl = tgpu.resolve([pipeline]);

  expect(vars(wgsl)).toEqual([
    'var<storage, read>: array<BlockHost, 1000>',
    'var<storage, read>: array<u32, 2000>',
  ]);
  expect(wgsl).toMatch(/1000u/);
});

it('lets one kernel read two compactions, each through its own slots', () => {
  const { root, blocks, live } = setup();
  const cubes = liveLists.create(root, {
    ...blocks.live(),
    draws: [{ vertexCount: 36 }],
  });
  const out = root.createMutable(d.arrayOf(d.u32, 2));

  const kernel = tgpu.computeFn({ workgroupSize: [1] })(() => {
    'use gpu';
    out.$[0] = live.slot(CUBE, d.u32(0));
    out.$[1] = cubes.slot(0, d.u32(0));
  });

  const pipeline = root
    .pipe(blocks.inject())
    .pipe(live.inject())
    .pipe(cubes.inject())
    .createComputePipeline({ compute: kernel });

  const wgsl = tgpu.resolve([pipeline]);

  expect(wgsl).toContain('var<storage, read> live_slots: array<u32, 2000>;');
  expect(wgsl).toContain('var<storage, read> live_slots_1: array<u32, 1000>;');
  expect(wgsl).toContain('return live_slots[((u32(group) * 1000u) + i)];');
  expect(wgsl).toContain('return live_slots_1[((u32(group) * 1000u) + i)];');
  expect(wgsl).toContain('out[0i] = slot(1i, 0u);\n  out[1i] = slot_1(0i, 0u);');
});

it('starts each group at its region with firstInstance', () => {
  const { root, bytes } = recordingRoot({ features: ['indirect-first-instance'] });
  const blocks = table.create(root, Block, { capacity: 1000 });
  const live = liveLists.create(root, {
    ...blocks.live(),
    group: Block.kind.pipe(blocks.inject()),
    draws: [{ vertexCount: 6 }, { indexCount: 36, firstIndex: 12, baseVertex: 4 }],
    firstInstance: true,
  });
  const args = new Uint32Array(bytes(root.unwrap(live.args)).buffer);

  // drawIndirect reads firstInstance from word 3, drawIndexedIndirect from word 4
  expect([...args]).toEqual([6, 0, 0, 0, 0, 36, 0, 12, 4, 1000]);

  live.setDraw(0, { vertexCount: 3, firstVertex: 9 });
  live.setDraw(1, { vertexCount: 3 });
  expect([...args]).toEqual([3, 0, 9, 0, 0, 3, 0, 0, 1000, 0]);
});

it('throws for firstInstance on a device without indirect-first-instance', () => {
  const { root } = recordingRoot();

  expect(() => liveLists.create(root, { capacity: 4, firstInstance: true })).toThrow(
    /indirect-first-instance/,
  );
});

it('throws for slots past the device limit', () => {
  const { root } = recordingRoot({ limits: { maxStorageBufferBindingSize: 8 } });

  expect(() => liveLists.create(root, { capacity: 4 })).toThrow(
    /need 16 bytes.*`maxStorageBufferBindingSize` is 8 bytes/,
  );
});

it('throws for a capacity past the workgroups compact dispatches, whatever the workgroup size', () => {
  const { root } = recordingRoot({ limits: { maxComputeWorkgroupsPerDimension: 4 } });

  // `dispatch` covers only the rows a group lists, so a small workgroup size doesn't lower it
  liveLists.create(root, { capacity: 1024, workgroupSize: 16 }).destroy();
  expect(() => liveLists.create(root, { capacity: 1025 })).toThrow(
    "liveLists.create needs 5 workgroups along x for a `capacity` of 1025. The device's " +
      '`maxComputeWorkgroupsPerDimension` is 4. The largest capacity it allows is 1024.',
  );
});

it('lets one pipeline draw every group with live.instanceSlot', () => {
  const { root } = recordingRoot({ features: ['indirect-first-instance'] });
  const blocks = table.create(root, Block, { capacity: 1000 });
  const live = liveLists.create(root, {
    ...blocks.live(),
    group: Block.kind.pipe(blocks.inject()),
    draws: [{ vertexCount: 6 }, { vertexCount: 36 }],
    firstInstance: true,
  });

  const vertex = tgpu.vertexFn({
    in: { iid: d.builtin.instanceIndex },
    out: { position: d.builtin.position },
  })(({ iid }) => {
    'use gpu';
    return { position: d.vec4f(Block.pos(live.instanceSlot(iid)), 1) };
  });

  const fragment = tgpu.fragmentFn({ out: d.vec4f })(() => {
    'use gpu';
    return d.vec4f(1);
  });

  const pipeline = root
    .pipe(blocks.inject())
    .pipe(live.inject())
    .createRenderPipeline({ vertex, fragment, targets: { format: 'rgba8unorm' } });

  const wgsl = tgpu.resolve([pipeline]);

  expect(vars(wgsl)).toEqual([
    'var<storage, read>: array<BlockHost, 1000>',
    'var<storage, read>: array<u32, 2000>',
  ]);
  expect(wgsl).not.toMatch(/1000u/); // no group offset baked in
});

it('fails to resolve live.instanceSlot without firstInstance, where it would read group 0', () => {
  const { root } = recordingRoot();
  const blocks = table.create(root, Block, { capacity: 1000 });
  const live = liveLists.create(root, {
    capacity: blocks.capacity,
    alive: Block.alive.pipe(blocks.inject()),
    group: Block.kind.pipe(blocks.inject()),
    draws: [{ vertexCount: 6 }, { vertexCount: 36 }],
  });

  const vertex = tgpu.vertexFn({
    in: { iid: d.builtin.instanceIndex },
    out: { position: d.builtin.position },
  })(({ iid }) => {
    'use gpu';
    return { position: d.vec4f(Block.pos(live.instanceSlot(iid)), 1) };
  });
  const fragment = tgpu.fragmentFn({ out: d.vec4f })(() => {
    'use gpu';
    return d.vec4f(1);
  });
  const pipeline = root
    .pipe(blocks.inject())
    .pipe(live.inject())
    .createRenderPipeline({ vertex, fragment, targets: { format: 'rgba8unorm' } });

  expect(() => tgpu.resolve([pipeline])).toThrow('`instanceSlot` needs live lists created with');
});

it('takes 1 to 256 groups', () => {
  const { root } = recordingRoot();

  const create = (draws: liveLists.IndirectDraw[]) =>
    liveLists.create(root, { capacity: 4, draws });

  expect(() => create([])).toThrow(RangeError);
  expect(() => create(Array.from({ length: 257 }, () => ({ vertexCount: 3 })))).toThrow(RangeError);
});

const firstFour = (i: number) => {
  'use gpu';
  return i < 4;
};

it('takes a live test that reads nothing to inject', async () => {
  const { root, log } = recordingRoot();

  const live = liveLists.create(root, {
    capacity: 8,
    alive: firstFour,
    draws: [{ vertexCount: 3 }],
  });
  await live.initAsync();

  expect(log.pipelines).toBe(2);
});

it('lists every slot without a live test, and keeps the capacity a table gave it', async () => {
  const { root, log } = recordingRoot();
  const every = liveLists.create(root, { capacity: 8, draws: [{ vertexCount: 3 }] });
  await every.initAsync();
  expect(every.capacity).toBe(8);
  expect(log.shaders.find((wgsl) => wgsl.includes('workgroupBarrier'))).toContain(
    'if ((i < 8u)) {\n    group = 0u;',
  );

  const blocks = table.create(root, Block, { capacity: 300 });
  const live = liveLists.create(root, { ...blocks.live(), draws: [{ vertexCount: 3 }] });
  expect(live.capacity).toBe(300);

  // @ts-expect-error `span` needs `alive`, which marks the slots above it dead
  void ({ capacity: 8, span: () => 4 } satisfies liveLists.Options);
});
