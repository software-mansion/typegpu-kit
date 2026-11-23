import { d, tgpu } from 'typegpu';
import { expect, it } from 'vitest';
import { liveLists, frames, table } from '../src/index.ts';
import { gpuRoot } from '@typegpu/kit/testing/dawn';

const Block = table.define('Block', { host: { kind: d.u32 } });
/** Multiples of 3 list in group 0, other odd rows in group 1, the rest nowhere. */
const groupOf = (i: number) => (i % 3 === 0 ? 0 : i % 2 === 1 ? 1 : 2);

it('lists every live row once, in its group, across workgroups', async () => {
  const root = await gpuRoot();
  const capacity = 5000;
  const blocks = table.create(root, Block, { capacity });

  const live = liveLists.create(root, {
    capacity,
    alive: Block.alive.pipe(blocks.inject()),
    group: Block.kind.pipe(blocks.inject()),
    draws: [{ vertexCount: 3 }, { vertexCount: 3 }, { indexCount: 6 }],
  });

  const ids = Array.from({ length: capacity }, (_, i) => blocks.add({ kind: i % 4 })); // 3 skips
  for (let i = 0; i < capacity; i += 3) blocks.remove(ids[i]);
  blocks.remove(ids[capacity - 1]); // shrinks the span

  const loop = frames.create(root, (frame) => {
    blocks.flush();
    const pass = frame.encoder.beginComputePass();
    live.compact(pass);
    pass.end();
  });

  loop.run(0);
  loop.run(1); // the counts reset between frames

  const [args, slots] = await Promise.all([live.args.read(), live.slots.read()]);

  const expected = [0, 1, 2].map((g) =>
    ids
      .map((id, slot) => ({ id, slot }))
      .filter(({ id, slot }) => blocks.has(id) && slot % 4 === g)
      .map(({ slot }) => slot),
  );

  for (const [g, rows] of expected.entries()) {
    expect(args[g * 5 + 1]).toBe(rows.length);
    const listed = slots.slice(g * capacity, g * capacity + rows.length).toSorted((a, b) => a - b);
    expect(listed).toEqual(rows);
  }
  expect([args[0], args[2], args[3], args[5], args[10]]).toEqual([3, 0, 0, 3, 6]);

  root.destroy();
});

it("runs a kernel over one group's live rows alone, from its indirect dispatch", async () => {
  const root = await gpuRoot();
  const capacity = 1000;
  const blocks = table.create(root, Block, { capacity });
  const live = liveLists.create(root, {
    ...blocks.live(),
    group: Block.kind.pipe(blocks.inject()),
    draws: [{ vertexCount: 3 }, { vertexCount: 3 }],
    workgroupSize: 32,
  });

  const ids = Array.from({ length: capacity }, (_, i) => blocks.add({ kind: i % 2 }));
  for (let i = 0; i < capacity; i += 5) blocks.remove(ids[i]);

  const marks = root.createBuffer(d.arrayOf(d.u32, capacity)).$usage('storage');
  const marked = marks.as('mutable');
  const kernel = tgpu.computeFn({
    workgroupSize: [32],
    in: { gid: d.builtin.globalInvocationId },
  })(({ gid }) => {
    'use gpu';
    if (gid.x >= live.count(1)) return;
    marked.$[live.slot(1, gid.x)] += 1;
  });
  const marking = root.pipe(live.inject()).createComputePipeline({ compute: kernel });

  const loop = frames.create(root, (frame) => {
    blocks.flush();
    const pass = frame.encoder.beginComputePass();
    live.compact(pass);
    pass.setPipeline(marking);
    live.dispatch(pass, 1);
    pass.end();
  });
  loop.run(0);

  const [counts, dispatches] = await Promise.all([marks.read(), live.dispatches.read()]);
  const expected = ids.map((id, slot) => (blocks.has(id) && slot % 2 === 1 ? 1 : 0));
  expect(counts).toEqual(expected);
  const odd = expected.reduce((sum: number, each) => sum + each, 0);
  expect(dispatches.slice(3, 6)).toEqual([Math.ceil(odd / 32), 1, 1]);

  root.destroy();
});

it('lists the rows a kernel of its own lists, and dispatches a kernel over each one once', async () => {
  const root = await gpuRoot();
  const capacity = 1000;
  const live = liveLists.create(root, { capacity, groups: 2 });

  const listing = tgpu.computeFn({
    workgroupSize: [64],
    in: { gid: d.builtin.globalInvocationId },
  })(({ gid }) => {
    'use gpu';
    const i = gid.x;
    let group = d.u32(2);
    if (i < capacity && i % 3 === 0) group = 0;
    else if (i < capacity && i % 2 === 1) group = 1;
    live.list(group, i);
  });
  const lister = root.createComputePipeline({ compute: listing });

  const marks = root.createBuffer(d.arrayOf(d.u32, capacity)).$usage('storage');
  const marked = marks.as('mutable');
  const mark = tgpu.computeFn({
    workgroupSize: [64],
    in: { gid: d.builtin.globalInvocationId },
  })(({ gid }) => {
    'use gpu';
    if (gid.x >= live.count(1)) return;
    marked.$[live.slot(1, gid.x)] += 1;
  });
  const marker = root.createComputePipeline({ compute: mark });

  const loop = frames.create(root, (frame) => {
    const pass = frame.encoder.beginComputePass();
    pass.setPipeline(lister);
    pass.dispatchWorkgroups(Math.ceil(capacity / 64));
    live.finish(pass);
    pass.setPipeline(marker);
    live.dispatch(pass, 1);
    pass.end();
  });
  loop.run(0);
  loop.run(1); // the counts reset between listings

  const [counts, args] = await Promise.all([marks.read(), live.args.read()]);
  const sizes = [0, 1].map((g) => counts.filter((_, i) => groupOf(i) === g).length);
  expect([args[1], args[6]]).toEqual(sizes);
  expect(counts).toEqual(counts.map((_, i) => (groupOf(i) === 1 ? 2 : 0)));

  root.destroy();
});

it('draws every group from one pipeline with firstInstance', async () => {
  const root = await gpuRoot({ features: ['indirect-first-instance'] });
  expect(root.device.features.has('indirect-first-instance')).toBe(true);

  const capacity = 64;
  const blocks = table.create(root, Block, { capacity });
  const live = liveLists.create(root, {
    capacity,
    alive: Block.alive.pipe(blocks.inject()),
    group: Block.kind.pipe(blocks.inject()),
    draws: [{ vertexCount: 1 }, { indexCount: 1 }],
    firstInstance: true,
  });

  const ids = Array.from({ length: 50 }, (_, i) => blocks.add({ kind: i % 3 })); // kind 2 skips
  blocks.remove(ids[4]);
  blocks.remove(ids[7]);

  // Each instance draws one point in a pixel of its own, and its fragment marks its row.
  const hits = root.createMutable(d.arrayOf(d.u32, capacity));
  const vertex = tgpu.vertexFn({
    in: { iid: d.builtin.instanceIndex },
    out: { position: d.builtin.position, slot: d.interpolate('flat', d.u32) },
  })(({ iid }) => {
    'use gpu';
    const slot = live.instanceSlot(iid);
    const x = ((d.f32(slot) + 0.5) / capacity) * 2 - 1;
    return { position: d.vec4f(x, 0, 0, 1), slot };
  });
  const fragment = tgpu.fragmentFn({
    in: { slot: d.interpolate('flat', d.u32) },
    out: d.vec4f,
  })(({ slot }) => {
    'use gpu';
    hits.$[slot] = Block.kind(slot) + 1;
    return d.vec4f(1);
  });

  const pipeline = root
    .pipe(blocks.inject())
    .pipe(live.inject())
    .createRenderPipeline({
      vertex,
      fragment,
      targets: { format: 'rgba8unorm' },
      primitive: { topology: 'point-list' },
    })
    .withIndexBuffer(root.createBuffer(d.arrayOf(d.u32, 1), [0]).$usage('index'));
  const target = root.createTexture({ size: [capacity, 1], format: 'rgba8unorm' }).$usage('render');

  const loop = frames.create(root, (frame) => {
    blocks.flush();
    const compute = frame.encoder.beginComputePass();
    live.compact(compute);
    compute.end();

    const pass = frame.encoder.beginRenderPass({ colorAttachments: { view: target } });
    pass.setPipeline(pipeline);
    live.draw(pass, 0);
    live.draw(pass, 1);
    pass.end();
  });
  loop.run(0);

  const marked = await hits.read();
  const expected = Array.from({ length: capacity }, (_, slot) => {
    const id = ids[slot];
    if (id === undefined || !blocks.has(id)) return 0;
    const kind = slot % 3;
    return kind < 2 ? kind + 1 : 0;
  });
  expect([...marked]).toEqual(expected);

  root.destroy();
});
