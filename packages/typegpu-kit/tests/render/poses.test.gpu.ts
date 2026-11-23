import { frames, history } from '@typegpu/kit';
import { gpuRoot } from '@typegpu/kit/testing/dawn';
import { d, std, tgpu, type TgpuComputePass, type TgpuRoot } from 'typegpu';
import { expect, it } from 'vitest';
import { meshPool, poses } from '../../src/index.ts';
import { jointHalf, Pose, poseJoints } from '../../src/render/skinning.ts';
import { bend, blade, leg, rest, shin, slow, sway } from './leg.ts';

/** The shin's three vertices, then the blade's. */
const VERTICES = 6;
const POSES = 4;
/** Two joints of three rows each. */
const ROWS = 6;

/** Where the shin's and the blade's vertices go with the knee turned `degrees` about +z. */
function bent(degrees: number) {
  const s = Math.sin((degrees * Math.PI) / 180);
  const c = Math.cos((degrees * Math.PI) / 180);
  return [
    [0, 0.5, 0],
    [-s, 1 + c, 0],
    // Half on the hip and half on the knee, which the pool keeps as 128 and 127 255ths.
    [(127 / 255) * -0.5 * s, (128 / 255) * 1.5 + (127 / 255) * (1 + 0.5 * c), 0],
    [-s, 1 + c, 0],
    [c - s, 1 + s + c, 0],
    [-s, 1 + c, 1],
  ];
}

type Kernel = (pass: TgpuComputePass) => void;

/** Row `row` of the joints that vertices read, this frame's or, through `previously`, last. */
const rowOf = (row: number) => {
  'use gpu';
  return d.vec4f(poseJoints.$[jointHalf.$ + row]);
};
const lastRowOf = history.previously(rowOf);

/** Poses of the leg playing `bend` and `rest`, with `local` from `hook`. */
const legPoses =
  (hook?: (root: TgpuRoot) => NonNullable<poses.Options['local']>) => (root: TgpuRoot) =>
    poses.create(
      root,
      leg,
      { bend, rest },
      { capacity: POSES, ...(hook && { local: hook(root) }) },
    );

/**
 * Poses that `make` makes, over a pool of the shin and of a blade the knee holds; and a reader of
 * where figures put their vertices and joints, each frame recording `before`'s kernel, the figures,
 * then reading every vertex and joint row in each pose asked for, in one compute pass.
 */
async function posing<P extends poses.Poses>(
  make: (root: TgpuRoot) => P,
  before?: (root: TgpuRoot, poses: P) => Kernel,
) {
  const root = await gpuRoot();
  const pool = meshPool.create(root, { vertices: 16, indices: 16, capacity: 2 });
  pool.add({ meshes: [shin], images: [] });
  pool.add(poses.attach({ meshes: [blade], images: [] }, leg, 'knee'));
  const figures = make(root);
  const kernel = before?.(root, figures);

  const ids = root.createBuffer(d.arrayOf(d.u32, POSES)).$usage('storage');
  const output = root.createBuffer(d.arrayOf(d.vec4f, POSES * VERTICES)).$usage('storage');
  const rows = root.createBuffer(d.arrayOf(d.vec4f, POSES * ROWS * 2)).$usage('storage');
  const asked = ids.as('readonly');
  const placed = output.as('mutable');
  const copied = rows.as('mutable');

  // A workgroup a pose, a thread a vertex.
  const vertices = tgpu.computeFn({
    workgroupSize: [VERTICES],
    in: { wid: d.builtin.workgroupId, lid: d.builtin.localInvocationIndex },
  })(({ wid, lid }) => {
    'use gpu';
    const position = poses.skinnedPosition(lid, meshPool.positionAt(lid), asked.$[wid.x]);
    placed.$[wid.x * VERTICES + lid] = d.vec4f(position, 1);
  });

  // A workgroup a pose, a thread a joint's row, this frame's and last.
  const joints = tgpu.computeFn({
    workgroupSize: [ROWS],
    in: { wid: d.builtin.workgroupId, lid: d.builtin.localInvocationIndex },
  })(({ wid, lid }) => {
    'use gpu';
    const row = Pose.slotOf(asked.$[wid.x]) * ROWS + lid;
    const at = (wid.x * ROWS + lid) * 2;
    copied.$[at] = rowOf(row);
    copied.$[at + 1] = lastRowOf(row);
  });
  const reading = root
    .pipe(pool.inject())
    .pipe(figures.inject())
    .createComputePipeline({ compute: vertices });
  const rowing = root.pipe(figures.inject()).createComputePipeline({ compute: joints });

  let now = 0;
  let count = 0;
  const loop = frames.create(root, (frame) => {
    const pass = frame.encoder.beginComputePass();
    kernel?.(pass);
    figures.animate(pass, now);
    if (count > 0) {
      pass.setPipeline(reading);
      pass.dispatchWorkgroups(count);
      pass.setPipeline(rowing);
      pass.dispatchWorkgroups(count);
    }
    pass.end();
  });

  const run = async (time: number, ...posed: poses.Id[]) => {
    now = time;
    count = posed.length;
    ids.write(Array.from({ length: POSES }, (_, k) => posed[k] ?? 0));
    loop.run(time);
    const [values, both] = await Promise.all([output.read(), rows.read()]);
    const flat = (from: number) =>
      posed.map((_, k) =>
        Array.from({ length: ROWS }, (__, r) => both[(k * ROWS + r) * 2 + from]).flatMap(
          ({ x, y, z, w }) => [x, y, z, w],
        ),
      );
    return {
      vertices: posed.map((_, k) =>
        values.slice(k * VERTICES, (k + 1) * VERTICES).map(({ x, y, z }) => [x, y, z]),
      ),
      /** Each pose's joint rows this frame, as flat floats. */
      now: flat(0),
      /** Each pose's joint rows last frame. */
      before: flat(1),
    };
  };

  /** Where each pose in `posed` puts the six vertices at `time`. */
  const at = async (time: number, ...posed: poses.Id[]) => (await run(time, ...posed)).vertices;

  return { root, figures, at, run };
}

/** Whether every coordinate is within a hundred-thousandth of what was expected. */
function expectNear(actual: number[][], expected: number[][]) {
  const off = actual.flatMap((vertex, v) =>
    vertex.flatMap((value, c) => (Math.abs(value - expected[v][c]) < 1e-5 ? [] : [[v, c]])),
  );
  expect(off, JSON.stringify({ actual, expected })).toEqual([]);
}

it('poses vertices as their joints move, a vertex on two between them', async () => {
  const { figures, at } = await posing(legPoses());
  const id = figures.add('bend', 0, { loop: false });

  for (const [time, degrees] of [
    [0, 0],
    [0.5, 45],
    [1, 90],
    // Held at its end.
    [3, 90],
  ] as const) {
    const [placed] = await at(time, id);
    expectNear(placed!, bent(degrees));
  }
});

it('plays a clip looping, at a speed, from a time and backwards', async () => {
  const { figures, at } = await posing(legPoses());
  const looping = figures.add('bend', 0);
  const slowed = figures.add('bend', 1, { speed: 0.5, from: 0.25 });
  const backwards = figures.add('bend', 1, { speed: -1, from: 1, loop: false });

  const [a, b, c] = await at(1.25, looping, slowed, backwards);
  expectNear(a!, bent(22.5));
  expectNear(b!, bent(33.75));
  expectNear(c!, bent(67.5));
});

it('blends from what a pose played before over the fade, then plays the new clip alone', async () => {
  const { figures, at } = await posing(legPoses());
  const id = figures.add('bend', 0, { loop: false });
  figures.play(id, 'rest', 1, { fade: 1 });

  expectNear((await at(1.5, id))[0], bent(45));
  expectNear((await at(2, id))[0], bent(0));
});

it('keeps each pose to its own, as poses come and go', async () => {
  const { figures, at } = await posing(legPoses());
  const bending = figures.add('bend', 0, { loop: false });
  const resting = figures.add('rest', 0);
  const [a, b] = await at(1, bending, resting);
  expectNear(a!, bent(90));
  expectNear(b!, bent(0));

  figures.remove(bending);
  const again = figures.add('bend', 1, { loop: false });
  const [halfway, still] = await at(1.5, again, resting);
  expectNear(halfway!, bent(45));
  expectNear(still!, bent(0));
});

it('leaves the joints of a removed pose below the span as they were', async () => {
  const { figures, run } = await posing(legPoses());
  const bending = figures.add('bend', 0, { loop: false });
  const resting = figures.add('rest', 0);
  const before = await run(0.5, bending, resting);
  figures.remove(bending);
  const after = await run(1, bending, resting);
  expect(after.now[0]).toEqual(before.now[0]);
});

it('places joints by a hook of your own, a pose at a time by its slot', async () => {
  const angles = { write: (_degrees: number[]) => {} };
  const { figures, at } = await posing(
    legPoses((root) => {
      const buffer = root.createBuffer(d.arrayOf(d.f32, POSES)).$usage('storage');
      const degrees = buffer.as('readonly');
      angles.write = (values) => buffer.write(values);
      return (played, joint, slot) => {
        'use gpu';
        if (joint !== 1) return poses.Placement(played);
        const half = (degrees.$[slot] * Math.PI) / 360;
        return poses.Placement({
          rotation: d.vec4f(0, 0, std.sin(half), std.cos(half)),
          position: played.position,
          scale: played.scale,
        });
      };
    }),
  );
  const resting = figures.add('rest', 0);
  const bending = figures.add('bend', 0, { loop: false });
  const turns = [0, 0, 0, 0];
  turns[figures.slotOf(resting)] = 30;
  turns[figures.slotOf(bending)] = 60;
  angles.write(turns);

  const [a, b] = await at(0.5, resting, bending);
  expectNear(a!, bent(30));
  expectNear(b!, bent(60));
});

/** Poses of the leg playing `bend`, `slow`, the line of both, `sway` and `rest`. */
const lined = (layers?: poses.Options<'over'>['layers']) => (root: TgpuRoot) =>
  poses.create(
    root,
    leg,
    { bend, slow, both: [bend, slow], sway, rest },
    { capacity: POSES, ...(layers && { layers }) },
  );

/** The hip's twelve floats of joint rows, then the knee's. */
const hipOf = (rows: number[]) => rows.slice(0, 12);
const kneeOf = (rows: number[]) => rows.slice(12, 24);

it('samples a line at one phase, each end its clip alone and halfway both', async () => {
  const { figures, at } = await posing(lined());
  const first = figures.add('both', 0);
  const second = figures.add('both', 0, { mix: 1 });
  const middle = figures.add('both', 0, { mix: 0.5 });

  // Half a cycle of bend's second, and a quarter of slow's two.
  const [a, b] = await at(0.5, first, second);
  expectNear(a!, bent(45));
  expectNear(b!, bent(45));

  // A cycle of a second and a half, both at their middle: 45 and 90 degrees, halfway 67.5.
  const [halfway] = await at(0.75, middle);
  expectNear(halfway!, bent(67.5));
});

it('keeps a line at its phase as its mix changes', async () => {
  const { figures, at } = await posing(lined());
  const id = figures.add('both', 0, { mix: 0.5 });
  expectNear((await at(0.75, id))[0], bent(67.5));

  // Half a cycle in, slow alone is at 90 degrees, and a second later three quarters in.
  figures.mix(id, 1, 0.75);
  expectNear((await at(0.75, id))[0], bent(90));
  expectNear((await at(1.25, id))[0], bent(135));

  // A small change moves the joints by less than one key's step.
  const [before] = await at(1.25, id);
  figures.mix(id, 0.98, 1.25);
  const [after] = await at(1.25, id);
  const moved = Math.max(
    ...before!.flatMap((vertex, v) => vertex.map((c, k) => Math.abs(c - after![v][k]))),
  );
  expect(moved).toBeLessThan(Math.sin((3 * Math.PI) / 180) * 2);
  expect(figures.seconds(id, 1.25)).toBeCloseTo(0.75 * (1 + 0.98), 4);
});

it('moves only the joints a layer masks, leaving the rest bit for bit', async () => {
  const { figures, run } = await posing(lined({ over: { joints: ['knee'] } }));
  const plain = figures.add('bend', 0);
  const layered = figures.add('bend', 0);
  figures.play(layered, 'sway', 0, { layer: 'over' });

  const { now, vertices } = await run(0.5, plain, layered);
  expect(hipOf(now[1])).toEqual(hipOf(now[0]));
  expect(kneeOf(now[1])).not.toEqual(kneeOf(now[0]));
  // The knee takes sway's turn back, a quarter at a time, over the hip's rest.
  expectNear(vertices[1], bent(-45));
});

it('adds nothing at weight 0 or at the first key, and the difference from it between', async () => {
  const { figures, run } = await posing(lined({ over: { add: true } }));
  const plain = figures.add('bend', 0);
  const light = figures.add('bend', 0);
  const first = figures.add('bend', 0);
  const added = figures.add('rest', 0);
  figures.play(light, 'sway', 0, { layer: 'over', weight: 0 });
  figures.play(first, 'sway', 0, { layer: 'over', speed: 0 });
  figures.play(added, 'slow', 0, { layer: 'over' });

  const { now, vertices } = await run(0.5, plain, light, first, added);
  expect(now[1]).toEqual(now[0]);
  now[2].forEach((value, i) => expect(value).toBeCloseTo(now[0][i], 6));
  // Slow's knee, a quarter turn in after a second of two, over the rest.
  expectNear(vertices[3], bent(45));
});

it('fades a stopped layer back to the base', async () => {
  const { figures, at } = await posing(lined({ over: {} }));
  const id = figures.add('bend', 0, { loop: false });
  figures.play(id, 'rest', 0, { layer: 'over' });
  expectNear((await at(1, id))[0], bent(0));

  figures.stop(id, 1, { layer: 'over', fade: 0.5 });
  figures.play(id, 'bend', 1, { loop: false, from: 1 });
  expectNear((await at(1.25, id))[0], bent(45));
  expectNear((await at(1.5, id))[0], bent(90));
});

/** Where the vertices go with the knee a share `w` of the way from rest to a quarter turn. */
const faded = (w: number) =>
  bent((2 * Math.atan2(w * Math.SQRT1_2, 1 - w + w * Math.SQRT1_2) * 180) / Math.PI);

/** A quarter turn held on the layer. */
const HELD = { layer: 'over', speed: 0, from: 1, loop: false } as const;

it('keeps the share a layer shows when a fade in or out is interrupted', async () => {
  const { figures, at } = await posing(lined({ over: {} }));
  // Stopped half way through fading in, from half the layer back to none.
  const stopping = figures.add('rest', 0);
  figures.play(stopping, 'bend', 0, { ...HELD, fade: 1 });
  figures.stop(stopping, 0.5, { layer: 'over', fade: 1 });
  // Played again half way through fading out, from half the layer to all of it.
  const replaying = figures.add('rest', 0);
  figures.play(replaying, 'bend', 0, HELD);
  figures.stop(replaying, 0, { layer: 'over', fade: 1 });
  figures.play(replaying, 'bend', 0.5, { ...HELD, fade: 1 });

  for (const [time, out, back] of [
    [0.5, 0.5, 0.5],
    [0.75, 0.25, 0.75],
    [1, 0, 1],
  ] as const) {
    const [a, b] = await at(time, stopping, replaying);
    expectNear(a!, faded(out));
    expectNear(b!, faded(back));
  }
});

it('stops and plays again from a kernel, mid fade, as the CPU does', async () => {
  const kernel = { next: (_id: poses.Id, _verb: 'stop' | 'play') => {} };
  const { figures, run } = await posing(lined({ over: {} }), (root, made) => {
    const target = root.createUniform(d.u32);
    const { over } = made.layers;
    const { bend: bending } = made.clips;
    const stopping = tgpu.computeFn({ workgroupSize: [1] })(() => {
      'use gpu';
      poses.stop(target.$, over, 1);
    });
    const playing = tgpu.computeFn({ workgroupSize: [1] })(() => {
      'use gpu';
      poses.play(target.$, bending, over, 1, 0.25);
    });
    const fills = root.pipe(made.inject());
    const verbs = {
      stop: fills.createComputePipeline({ compute: stopping }),
      play: fills.createComputePipeline({ compute: playing }),
    };
    let armed: 'stop' | 'play' | undefined;
    kernel.next = (id, verb) => {
      target.write(id);
      armed = verb;
    };
    return (pass) => {
      if (armed) {
        pass.setPipeline(verbs[armed]);
        pass.dispatchWorkgroups(1);
      }
      armed = undefined;
    };
  });
  const cpu = figures.add('rest', 0);
  const gpu = figures.add('rest', 0);
  figures.play(cpu, 'bend', 0, { ...HELD, fade: 1 });
  figures.play(gpu, 'bend', 0, { ...HELD, fade: 1 });
  await run(0.25, cpu, gpu);

  figures.stop(cpu, 0.5, { layer: 'over', fade: 1 });
  kernel.next(gpu, 'stop');
  const stopped = await run(0.5, cpu, gpu);
  expectNear(stopped.vertices[1], faded(0.5));

  figures.play(cpu, 'bend', 0.75, { layer: 'over', fade: 1, from: 0.25 });
  kernel.next(gpu, 'play');
  for (const time of [0.75, 1, 1.25]) {
    const { now } = await run(time, cpu, gpu);
    now[1].forEach((value, i) => expect(value).toBeCloseTo(now[0][i], 5));
  }
});

it('leaves a pose as it was for a kernel naming a layer past the third', async () => {
  const kernel = { arm: (_id: poses.Id) => {} };
  const { figures, run } = await posing(lined({ over: {} }), (root, made) => {
    const target = root.createUniform(d.u32);
    const { bend: bending } = made.clips;
    const start = tgpu.computeFn({ workgroupSize: [1] })(() => {
      'use gpu';
      poses.play(target.$, bending, 4, 0, 0.5);
      poses.once(target.$, bending, 5, 0);
      poses.mix(target.$, 1, 4);
      poses.weigh(target.$, 0, 4);
      poses.stop(target.$, 4, 0);
    });
    const starting = root.pipe(made.inject()).createComputePipeline({ compute: start });
    let armed = false;
    kernel.arm = (id) => {
      target.write(id);
      armed = true;
    };
    return (pass) => {
      if (armed) {
        pass.setPipeline(starting);
        pass.dispatchWorkgroups(1);
      }
      armed = false;
    };
  });
  const id = figures.add('rest', 0);
  const before = await run(0.5, id);
  kernel.arm(id);
  const after = await run(0.5, id);
  expect(after.now).toEqual(before.now);
});

it('plays from a kernel as the CPU plays, at the time animate writes', async () => {
  const kernel = { start: (_id: poses.Id) => {} };
  const { figures, run } = await posing(lined({ over: { joints: ['knee'] } }), (root, made) => {
    const target = root.createUniform(d.u32);
    const { over } = made.layers;
    const { sway: swaying } = made.clips;
    const start = tgpu.computeFn({ workgroupSize: [1] })(() => {
      'use gpu';
      poses.play(target.$, swaying, over, 0.4, 0.25);
    });
    const starting = root.pipe(made.inject()).createComputePipeline({ compute: start });
    let armed = false;
    kernel.start = (id) => {
      target.write(id);
      armed = true;
    };
    return (pass) => {
      if (armed) {
        pass.setPipeline(starting);
        pass.dispatchWorkgroups(1);
      }
      armed = false;
    };
  });
  const cpu = figures.add('bend', 0);
  const gpu = figures.add('bend', 0);
  await run(0.5, cpu, gpu);

  figures.play(cpu, 'sway', 1, { layer: 'over', fade: 0.4, from: 0.25 });
  kernel.start(gpu);
  for (const time of [1, 1.2, 1.6]) {
    const { now } = await run(time, cpu, gpu);
    now[1].forEach((value, i) => expect(value).toBeCloseTo(now[0][i], 5));
  }
});

it('keeps last frame in the other half, and both halves equal for a new pose', async () => {
  const { figures, run } = await posing(legPoses());
  const id = figures.add('bend', 0, { loop: false });
  const first = await run(0.25, id);
  expect(first.before[0]).toEqual(first.now[0]);

  const second = await run(0.5, id);
  expect(second.before[0]).toEqual(first.now[0]);
  expect(second.now[0]).not.toEqual(first.now[0]);

  const late = figures.add('bend', 0, { loop: false });
  const third = await run(0.75, id, late);
  expect(third.before[0]).toEqual(second.now[0]);
  expect(third.before[1]).toEqual(third.now[1]);
});

/** A joint's three rows of skinning, as `animate` writes them, from where `jointAt` places it. */
const rowsOf = (placed: d.m4x4f, inverseBind: ArrayLike<number>) =>
  Array.from({ length: 12 }, (_, i) => {
    const row = Math.floor(i / 4);
    const column = i % 4;
    let sum = 0;
    for (let k = 0; k < 4; k++) sum += placed[k * 4 + row] * inverseBind[column * 4 + k];
    return sum;
  });

it('places joints on the CPU where animate places them on the GPU', async () => {
  const { figures, run } = await posing((root) =>
    poses.create(
      root,
      leg,
      { bend, slow, both: [bend, slow], sway, rest },
      { capacity: POSES, layers: { upper: { joints: ['knee'] }, added: { add: true } } },
    ),
  );
  // A line between clips, slowed, with a masked layer fading in
  const along = figures.add('both', 0, { mix: 0.3, speed: 0.7 });
  figures.play(along, 'sway', 0.1, { layer: 'upper', fade: 0.5 });
  // A base blending to another clip under a weighed additive layer
  const blended = figures.add('bend', 0);
  figures.play(blended, 'slow', 0, { layer: 'added', weight: 0.6 });
  figures.play(blended, 'sway', 0.2, { fade: 0.4 });
  // A clip held at its end with a layer fading out
  const held = figures.add('sway', 0, { loop: false });
  figures.play(held, 'bend', 0, { layer: 'upper' });
  figures.stop(held, 0.3, { layer: 'upper', fade: 0.5 });

  const ids = [along, blended, held];
  for (const time of [0.35, 1.7]) {
    const { now } = await run(time, ...ids);
    ids.forEach((id, k) => {
      const cpu = leg.joints.flatMap(({ name, inverseBind }) =>
        rowsOf(figures.jointAt(id, name, time), inverseBind),
      );
      cpu.forEach((value, i) => expect(value).toBeCloseTo(now[k][i], 5));
    });
  }
});

it('gives a joint’s position in shaders where jointAt places it', async () => {
  const { root, figures, run } = await posing(lined());
  const id = figures.add('sway', 0);
  const asked = root.createUniform(d.u32, id);
  const out = root.createBuffer(d.arrayOf(d.vec4f, 2)).$usage('storage');
  const written = out.as('mutable');
  const reading = tgpu.computeFn({
    workgroupSize: [2],
    in: { lid: d.builtin.localInvocationIndex },
  })(({ lid }) => {
    'use gpu';
    written.$[lid] = d.vec4f(poses.jointPosition(asked.$, lid), 1);
  });
  const pipeline = root.pipe(figures.inject()).createComputePipeline({ compute: reading });

  await run(0.5, id);
  pipeline.dispatchWorkgroups(1);
  const gpu = (await out.read()).map(({ x, y, z }) => [x, y, z]);
  const cpu = leg.joints.map(({ name }) => {
    const placed = figures.jointAt(id, name, 0.5);
    return [placed[12], placed[13], placed[14]];
  });
  expectNear(gpu, cpu);
  // The hip has turned the knee an eighth about +z.
  expectNear(gpu.slice(1), [[-Math.SQRT1_2, Math.SQRT1_2, 0]]);
});
