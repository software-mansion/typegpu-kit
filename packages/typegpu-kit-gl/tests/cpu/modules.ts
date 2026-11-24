import { cameras, grid, liveLists, machines, particles, table } from '@typegpu/kit';
import { d, tgpu, type TgpuRoot } from 'typegpu';

export async function lists(root: TgpuRoot) {
  const Block = table.define('CpuBlock', { host: { kind: d.u32 } });
  const blocks = table.create(root, Block, { capacity: 257 });
  const live = liveLists.create(root, {
    ...blocks.live(),
    group: Block.kind.pipe(blocks.inject()),
    draws: [{ vertexCount: 3 }, { vertexCount: 3 }],
    workgroupSize: 32,
  });
  const ids = Array.from({ length: 257 }, (_, i) => blocks.add({ kind: i % 2 }));
  for (let i = 0; i < ids.length; i += 5) blocks.remove(ids[i]!);
  const marks = root.createMutable(d.arrayOf(d.u32, 257));
  const kernel = tgpu.computeFn({ workgroupSize: [32], in: { id: d.builtin.globalInvocationId } })(
    ({ id }) => {
      'use gpu';
      if (id.x < live.count(1)) marks.$[live.slot(1, id.x)] += 1;
    },
  );
  const pipeline = root.pipe(live.inject()).createComputePipeline({ compute: kernel });
  blocks.flush();
  const encoder = root.createCommandEncoder();
  const pass = encoder.beginComputePass();
  live.compact(pass);
  pass.setPipeline(pipeline);
  live.dispatch(pass, 1);
  pass.end();
  encoder.submit();
  const result = await marks.read();
  live.destroy();
  blocks.destroy();
  return result;
}

export async function states(root: TgpuRoot) {
  const Mood = machines.define('CpuMood', {
    events: ['flip'],
    states: { off: { next: [{ to: 'on', on: 'flip' }] }, on: {} },
  });
  const moods = machines.create(root, Mood, { capacity: 64, step: 1 / 60 });
  for (let i = 0; i < 64; i++) moods.start(i);
  moods.flush();
  const step = () => {
    const encoder = root.createCommandEncoder();
    const pass = encoder.beginComputePass();
    moods.step(pass);
    pass.end();
    encoder.submit();
  };
  step();
  for (let i = 0; i < 64; i += 3) moods.send.flip(i);
  moods.flush();
  step();
  const result = (await moods.rows.read()).map((row) => (row.state === Mood.on ? 1 : 0));
  moods.destroy();
  return result;
}

export async function sparks(root: TgpuRoot) {
  const view = cameras.flat(root, { size: { width: 8, height: 8 }, height: 10 });
  const system = particles.create(root, view, {
    capacity: 128,
    step: 1 / 60,
    settings: { gravity: [0, -10, 0], drag: 0 },
  });
  system.burst({ count: 100, position: [1, 2, 0], velocity: [3, 0, 0], life: 2, variance: 0 });
  const encoder = root.createCommandEncoder();
  const pass = encoder.beginComputePass();
  system.spawn(pass);
  for (let i = 0; i < 30; i++) system.step(pass);
  pass.end();
  encoder.submit();
  const result = (await system.pool.read())
    .filter((p) => p.age < p.life)
    .flatMap((p) => [p.age, p.position.x, p.velocity.y]);
  system.destroy();
  return result;
}

export async function neighbours(root: TgpuRoot) {
  const count = 64;
  const positions = root.createReadonly(
    d.arrayOf(d.vec3f, count),
    Array.from({ length: count }, (_, i) =>
      d.vec3f((i % 4) + 0.5, (Math.floor(i / 4) % 4) + 0.5, Math.floor(i / 16) + 0.5),
    ),
  );
  const cells = grid.create(root, {
    capacity: count,
    low: d.vec3f(0),
    size: d.vec3f(4),
    sight: 1,
    item: d.u32,
    position: (i) => {
      'use gpu';
      return d.vec3f(positions.$[i]);
    },
    itemOf: (i) => {
      'use gpu';
      return d.u32(i);
    },
  });
  const output = root.createMutable(d.arrayOf(d.u32, count));
  const probe = tgpu.computeFn({ workgroupSize: [64], in: { id: d.builtin.globalInvocationId } })(({
    id,
  }) => {
    'use gpu';
    const from = cells.walk(positions.$[id.x]);
    let total = d.u32(0);
    for (let r = d.u32(0); r < grid.RANGES; r++) {
      const range = cells.range(from, r);
      for (let j = d.u32(0); j < range.count; j++) {
        const other = cells.neighbour(range, j);
        if (other === id.x) total++;
      }
    }
    output.$[id.x] = total;
  });
  const pipeline = root.pipe(cells.inject()).createComputePipeline({ compute: probe });
  const encoder = root.createCommandEncoder();
  const pass = encoder.beginComputePass();
  cells.sort(pass);
  pass.setPipeline(pipeline);
  pass.dispatchWorkgroups(1);
  pass.end();
  encoder.submit();
  const result = await output.read();
  cells.destroy();
  return result;
}

export const moduleCases = [
  [
    'table/liveLists indirect dispatch',
    lists,
    Array.from({ length: 257 }, (_, i) => (i % 5 !== 0 && i % 2 === 1 ? 1 : 0)),
  ],
  ['machines events', states, Array.from({ length: 64 }, (_, i) => (i % 3 === 0 ? 1 : 0))],
  ['particles integration', sparks, Array.from({ length: 100 }, () => [0.5, 2.5, -5]).flat()],
  ['grid neighbours', neighbours, Array(64).fill(1) as number[]],
] as const;
