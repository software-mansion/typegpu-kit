import { d, std, tgpu } from 'typegpu';
import { expect, it } from 'vitest';
import { frames, history, table } from '../src/index.ts';
import { gpuRoot } from '@typegpu/kit/testing/dawn';

const Enemy = table.define('Enemy', {
  host: { pos: d.vec2f, damage: d.f32 },
  device: { health: d.f32 },
});

const hurt = tgpu.computeFn({ workgroupSize: [64], in: { gid: d.builtin.globalInvocationId } })(({
  gid,
}) => {
  'use gpu';
  const i = gid.x;
  if (!Enemy.alive(i)) return;

  Enemy.device.$[i].health = Enemy.health(i) - Enemy.damage(i);
});

const vertex = tgpu.vertexFn({
  in: { iid: d.builtin.instanceIndex },
  out: { position: d.builtin.position, health: d.f32 },
})(({ iid }) => {
  'use gpu';
  return { position: d.vec4f(Enemy.pos(iid), 0, 1), health: Enemy.health(iid) };
});

const fragment = tgpu.fragmentFn({ in: { health: d.f32 }, out: d.vec4f })(({ health }) => {
  'use gpu';
  return d.vec4f(health, 0, 0, 1);
});

it('writes device rows in compute and reads them in render, with uploads in between', async () => {
  const root = await gpuRoot();
  const enemies = table.create(root, Enemy, { capacity: 100, defaults: { health: 10 } });

  const gpu = root.pipe(enemies.inject());
  const step = gpu.createComputePipeline({ compute: hurt });
  const draw = gpu.createRenderPipeline({ vertex, fragment, targets: { format: 'rgba8unorm' } });

  const target = root['~unstable']
    .createTexture({ size: [4, 4], format: 'rgba8unorm' })
    .$usage('render');
  const view = target.createView('render');

  const a = enemies.add({ damage: 1 });
  const b = enemies.add({ damage: 2 });
  const loop = frames.create(root, (frame) => {
    enemies.flush();

    const compute = frame.encoder.beginComputePass();
    compute.setPipeline(step);
    compute.dispatchWorkgroups(Math.ceil(enemies.span / 64));
    compute.end();

    const pass = frame.encoder.beginRenderPass({
      colorAttachments: [{ view, loadOp: 'clear', storeOp: 'store' }],
    });
    pass.setPipeline(draw);
    pass.draw(1, enemies.span);
    pass.end();
  });

  loop.run(0);
  loop.run(1);
  enemies.fields.health.set(b, 50); // a revive overwrites what the kernel wrote
  enemies.add({ damage: 4 });
  loop.run(2);

  const health = (await enemies.buffers.device!.read()).slice(0, 3).map((row) => row.health);
  expect(health).toEqual([7, 48, 6]);
  expect(enemies.has(a)).toBe(true);

  root.destroy();
});

it('moves rows between two tables of one definition with nothing piped', async () => {
  const root = await gpuRoot();
  const reds = table.create(root, Enemy, { capacity: 64 });
  const blues = table.create(root, Enemy, { capacity: 64 });
  const red = reds.gpu;
  const blue = blues.gpu;
  for (const [x, health] of [
    [1, 5],
    [-1, 6],
    [2, 7],
  ] as const)
    reds.add({ pos: [x, 0], health });
  blues.add();
  blues.add();

  const claims = root.createMutable(d.atomic(d.u32));
  const defect = tgpu.computeFn({ workgroupSize: [64], in: { gid: d.builtin.globalInvocationId } })(
    ({ gid }) => {
      'use gpu';
      const i = gid.x;
      if (!red.alive(i) || red.pos(i).x < 0) return;
      blue.device.$[std.atomicAdd(claims.$, 1)].health = red.health(i);
      red.device.$[i].health = 0;
    },
  );
  const both = tgpu.vertexFn({
    in: { iid: d.builtin.instanceIndex },
    out: { position: d.builtin.position, health: d.f32 },
  })(({ iid }) => {
    'use gpu';
    return { position: d.vec4f(red.pos(iid), 0, 1), health: red.health(iid) + blue.health(iid) };
  });
  const move = root.createComputePipeline({ compute: defect });
  const draw = root.createRenderPipeline({
    vertex: both,
    fragment,
    targets: { format: 'rgba8unorm' },
  });
  const view = root['~unstable']
    .createTexture({ size: [4, 4], format: 'rgba8unorm' })
    .$usage('render')
    .createView('render');

  frames
    .create(root, (frame) => {
      reds.flush();
      blues.flush();
      const compute = frame.encoder.beginComputePass();
      compute.setPipeline(move);
      compute.dispatchWorkgroups(1);
      compute.end();
      const pass = frame.encoder.beginRenderPass({
        colorAttachments: [{ view, loadOp: 'clear', storeOp: 'store' }],
      });
      pass.setPipeline(draw);
      pass.draw(1, 3);
      pass.end();
    })
    .run(0);

  const health = async (rows: typeof reds, n: number) =>
    (await rows.buffers.device!.read()).slice(0, n).map((row) => row.health);
  expect(await health(reds, 3)).toEqual([0, 6, 0]);
  expect((await health(blues, 2)).toSorted()).toEqual([5, 7]);
  expect(await root.takeErrors()).toEqual([]);

  root.destroy();
});

const Hit = table.define('Hit', { host: { victim: d.u32, amount: d.f32 } });

const applyHits = tgpu.computeFn({
  workgroupSize: [64],
  in: { gid: d.builtin.globalInvocationId },
})(({ gid }) => {
  'use gpu';
  const i = gid.x;
  if (!Enemy.alive(i)) return;

  const enemy = Enemy.device.$[i];
  for (let h = d.u32(0); h < Hit.span(); h++) {
    if (Hit.alive(h) && Enemy.slotOf(Hit.victim(h)) === i) enemy.health -= Hit.amount(h);
  }
});

it('finds the row of an id in a kernel, with the mask the table writes', async () => {
  const root = await gpuRoot();
  const enemies = table.create(root, Enemy, { capacity: 300, defaults: { health: 10 } }); // 9 bits
  const hits = table.create(root, Hit, { capacity: 8 });

  const ids = Array.from({ length: 300 }, () => enemies.add());
  enemies.remove(ids[257]);
  const reborn = enemies.add(); // slot 257, the next generation

  hits.add({ victim: reborn, amount: 3 });
  hits.add({ victim: ids[1], amount: 1 });
  hits.add({ victim: reborn, amount: 2 });

  const step = root
    .pipe(enemies.inject())
    .pipe(hits.inject())
    .createComputePipeline({ compute: applyHits });

  const loop = frames.create(root, (frame) => {
    enemies.flush();
    hits.flush();

    const pass = frame.encoder.beginComputePass();
    pass.setPipeline(step);
    pass.dispatchWorkgroups(Math.ceil(enemies.span / 64));
    pass.end();
  });
  loop.run(0);

  const health = (await enemies.buffers.device!.read()).map((row) => row.health);
  expect([reborn & 0x1ff, reborn >>> 9]).toEqual([257, (ids[257] >>> 9) + 1]);
  expect([health[0], health[1], health[257], health[299]]).toEqual([10, 9, 5, 10]);

  root.destroy();
});

it('keeps last frame’s rows for code compiled with previously', async () => {
  const root = await gpuRoot();
  const enemies = table.create(root, Enemy, {
    capacity: 4,
    history: true,
    defaults: { health: 10 },
  });
  const seen = root.createMutable(d.arrayOf(d.vec2f, 4));

  const before = history.previously((i: number) => {
    'use gpu';
    return d.vec2f(Enemy.pos(i).x, Enemy.health(i));
  });
  const look = tgpu.computeFn({
    workgroupSize: [64],
    in: { gid: d.builtin.globalInvocationId },
  })(({ gid }) => {
    'use gpu';
    const i = gid.x;
    if (!Enemy.alive(i)) return;
    seen.$[i] = before(i);
  });

  const gpu = root.pipe(enemies.inject());
  const looking = gpu.createComputePipeline({ compute: look });
  const hurting = gpu.createComputePipeline({ compute: hurt });

  const loop = frames.create(root, (frame) => {
    enemies.flush(frame.encoder);
    const pass = frame.encoder.beginComputePass();
    pass.setPipeline(looking);
    pass.dispatchWorkgroups(1);
    pass.setPipeline(hurting);
    pass.dispatchWorkgroups(1);
    pass.end();
  });

  const a = enemies.add({ pos: [1, 0], damage: 1 });
  loop.run(0);
  expect((await seen.read())[0]).toEqual(d.vec2f(1, 10));

  enemies.fields.pos.set(a, 2, 0);
  enemies.add({ pos: [7, 0], damage: 1 });
  loop.run(1);
  const [moved, born] = await seen.read();
  expect([moved, born]).toEqual([d.vec2f(1, 9), d.vec2f(7, 10)]);

  root.destroy();
});
