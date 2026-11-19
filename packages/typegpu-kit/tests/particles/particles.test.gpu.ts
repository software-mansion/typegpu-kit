import { cameras, clock, passes, targets } from '@typegpu/kit';
import { gpuRoot } from '@typegpu/kit/testing/dawn';
import { d, std, tgpu, type TgpuRoot } from 'typegpu';
import { expect, it } from 'vitest';
import { Particle } from '../../src/particles/particles.ts';
import { particles } from '../../src/index.ts';

const STEP = 1 / 60;

it('reads every field of a burst particle through its injected shader accessor', async () => {
  const root = await gpuRoot();
  const view = cameras.flat(root, { size: { width: 8, height: 8 }, height: 10 });
  const sparks = particles.create(root, view, {
    capacity: 1,
    step: STEP,
    settings: { gravity: [0, 0, 0], drag: 0 },
  });
  const output = root.createBuffer(Particle).$usage('storage');
  const read = output.as('mutable');
  const kernel = tgpu.computeFn({ workgroupSize: [1], in: { gid: d.builtin.globalInvocationId } })(
    ({ gid }) => {
      'use gpu';
      if (gid.x >= sparks.span) return;
      read.$ = sparks.particle(gid.x);
    },
  );
  const probe = root.pipe(sparks.inject()).createComputePipeline({ compute: kernel });
  sparks.burst({
    count: 1,
    position: [1, 2, 3],
    velocity: [4, 5, 6],
    color: [0.5, 0.25, 1, 0.75],
    size: 0.5,
    life: 3,
    variance: 0,
  });
  const encoder = root.createCommandEncoder();
  const pass = encoder.beginComputePass();
  sparks.spawn(pass);
  pass.setPipeline(probe);
  pass.dispatchWorkgroups(1);
  pass.end();
  encoder.submit();

  const particle = await output.read();
  expect([...particle.position]).toEqual([1, 2, 3]);
  expect([...particle.velocity]).toEqual([4, 5, 6]);
  expect([...particle.color]).toEqual([0.5, 0.25, 1, 0.75]);
  expect([particle.age, particle.size, particle.life]).toEqual([0, 0.5, 3]);
  expect(particle.seed).toBeGreaterThanOrEqual(0);
  expect(particle.seed).toBeLessThan(1);
  root.destroy();
});

/** An 8-bit texture's pixels, four bytes each, in rows of a multiple of 256 bytes. */
async function readPixels(root: TgpuRoot, texture: GPUTexture, width: number, height: number) {
  const readback = root.device.createBuffer({
    size: width * 4 * height,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const copy = root.device.createCommandEncoder();
  copy.copyTextureToBuffer({ texture }, { buffer: readback, bytesPerRow: width * 4 }, [
    width,
    height,
  ]);
  root.device.queue.submit([copy.finish()]);
  await readback.mapAsync(GPUMapMode.READ);

  return new Uint8Array(readback.getMappedRange().slice(0));
}

/** How many of a system's particles are alive. */
async function aliveIn(system: particles.Particles) {
  return (await system.pool.read()).filter((p) => p.age < p.life).length;
}

async function setup(capacity: number, settings: NonNullable<particles.Options['settings']> = {}) {
  const root = await gpuRoot();
  const size = { width: 32, height: 32 };
  const screen = targets.create(root, size, { color: { format: 'rgba16float' } });
  const into = passes.create(root, screen, { color: passes.color });
  const view = cameras.flat(root, { size, height: 10 });
  view.update(0);

  const system = () =>
    particles.create(root, view, {
      capacity,
      step: STEP,
      dimensions: 2,
      settings,
    });
  const sparks = system();
  const drawn = into.add(sparks.drawing);
  await into.initAsync();

  /** One frame: queued bursts, `steps` steps, liveLists and a draw. */
  const frame = (
    steps: number,
    extra?: (encoder: ReturnType<(typeof root)['~unstable']['createCommandEncoder']>) => void,
  ) => {
    const encoder = root.createCommandEncoder();
    extra?.(encoder);

    const compute = encoder.beginComputePass();
    sparks.spawn(compute);
    for (let i = 0; i < steps; i++) sparks.step(compute);
    sparks.compact(compute, 0);
    compute.end();

    const pass = encoder.beginRenderPass({ colorAttachments: { view: screen.views.color } });
    drawn.draw(pass);
    pass.end();
    encoder.submit();
  };

  const living = async () => (await sparks.pool.read()).filter((p) => p.age < p.life);

  return { root, into, system, sparks, frame, living };
}

it('bursts, moves and draws particles on a real device', async () => {
  const { root, sparks, frame, living } = await setup(256, { gravity: [0, -10, 0], drag: 0 });

  sparks.burst({ count: 100, position: [1, 2, 0], velocity: [3, 0, 0], life: 2, variance: 0 });
  frame(30);

  const alive = await living();
  expect(alive).toHaveLength(100);

  // Half a second at 3 units per second, falling under gravity, with no scatter or area.
  for (const particle of alive) {
    expect(particle.age).toBeCloseTo(0.5, 4);
    expect(particle.position.x).toBeCloseTo(2.5, 3);
    expect(particle.position.y).toBeLessThan(2 - 0.5 * 10 * 0.25 + 0.2);
    expect(particle.velocity.y).toBeCloseTo(-5, 3);
  }

  frame(100);
  expect(await living()).toHaveLength(0);
  root.destroy();
});

it('runs born on a burst particle as it is born, spreads an aged one across its life, and counts it in args', async () => {
  const root = await gpuRoot();
  const view = cameras.flat(root, { size: { width: 8, height: 8 }, height: 10 });
  const Mote = particles.schema('Mote', { slot: d.u32 });
  const dust = particles.create(root, view, {
    capacity: 256,
    step: STEP,
    particle: Mote,
    born: (mote, slot) => {
      'use gpu';
      let next = Mote(mote);
      next.color = d.vec4f(0, 1, 0, 1);
      next.slot = slot;
      return next;
    },
  });

  dust.burst({ count: 200, position: [0, 0, 0], life: 4, variance: 0, aged: true });

  const encoder = root.createCommandEncoder();
  const compute = encoder.beginComputePass();
  dust.spawn(compute);
  dust.compact(compute, 0);
  compute.end();
  encoder.submit();

  // No step has run, and each mote is already green and knows its slot, at an age somewhere in its
  // four seconds.
  const born = (await dust.pool.read()).slice(0, 200);
  expect(born.every((mote) => mote.color.x === 0 && mote.color.y === 1)).toBe(true);
  expect(born.every((mote, i) => mote.slot === i)).toBe(true);

  const ages = born.map((mote) => mote.age);
  expect(Math.min(...ages)).toBeGreaterThanOrEqual(0);
  expect(Math.max(...ages)).toBeLessThan(4);
  expect(Math.max(...ages) - Math.min(...ages)).toBeGreaterThan(2);

  expect((await dust.args.read())[1]).toBe(200);

  root.destroy();
});

it('runs update once a step with its length, from the first, on fields a burst gives', async () => {
  const root = await gpuRoot();
  const view = cameras.flat(root, { size: { width: 8, height: 8 }, height: 10 });
  const Bubble = particles.schema('Bubble', { grown: d.f32, steps: d.u32 });
  const bubbles = particles.create(root, view, {
    capacity: 64,
    step: STEP,
    particle: Bubble,
    update: (bubble, dt) => {
      'use gpu';
      let next = Bubble(bubble);
      next.grown = bubble.grown + dt;
      next.steps = bubble.steps + 1;
      return next;
    },
  });

  bubbles.burst({ count: 4, position: [0, 0, 0], life: 4, fields: { grown: 1 } });
  bubbles.burst({ count: 4, position: [0, 0, 0], life: 4 });

  const frame = (steps: number) => {
    const encoder = root.createCommandEncoder();
    const compute = encoder.beginComputePass();
    bubbles.spawn(compute);
    for (let i = 0; i < steps; i++) bubbles.step(compute);
    compute.end();
    encoder.submit();
  };

  frame(1);
  let pool = (await bubbles.pool.read()).slice(0, 8);
  expect(pool.map((bubble) => bubble.steps)).toEqual([1, 1, 1, 1, 1, 1, 1, 1]);
  for (const [i, bubble] of pool.entries()) {
    expect(bubble.grown).toBeCloseTo((i < 4 ? 1 : 0) + STEP, 6);
  }

  frame(2);
  pool = (await bubbles.pool.read()).slice(0, 8);
  expect(pool.every((bubble) => bubble.steps === 3)).toBe(true);
  expect(pool[7].grown).toBeCloseTo(3 * STEP, 6);

  root.destroy();
});

it('takes particles emitted by a kernel into two systems, and replaces them a ring later', async () => {
  const { root, system, sparks, frame, living } = await setup(64, { gravity: [0, 0, 0] });
  const embers = system();

  const kernel = tgpu.computeFn({
    workgroupSize: [8],
    in: { gid: d.builtin.globalInvocationId },
  })(({ gid }) => {
    'use gpu';
    const particle = Particle({
      position: d.vec3f(d.f32(gid.x), 0, 0),
      age: 0,
      velocity: d.vec3f(),
      life: 5,
      color: d.vec4f(1),
      size: 0.2,
      seed: 0,
    });

    sparks.emit(particle);
    if (gid.x < 3) embers.emit(particle);
  });
  const emitter = root
    .pipe(sparks.inject())
    .pipe(embers.inject())
    .createComputePipeline({ compute: kernel });

  frame(1, (encoder) => {
    const pass = encoder.beginComputePass();
    pass.setPipeline(emitter);
    pass.dispatchWorkgroups(1);
    pass.end();
  });
  expect(await living()).toHaveLength(8);
  expect((await embers.pool.read()).filter((p) => p.age < p.life)).toHaveLength(3);

  sparks.burst({ count: 64, position: [0, 9, 0], life: 5 });
  frame(1);
  const alive = await living();
  expect(alive).toHaveLength(64);
  expect(alive.every((particle) => particle.position.y === 9)).toBe(true);

  sparks.burst({ count: 10, position: [0, 0, 0], life: 1 });
  frame(0, (encoder) => sparks.clear(encoder));
  expect(await living()).toHaveLength(10);

  root.destroy();
});

/** Where a particle moving at 4 units a second along x is drawn across, in pixels, and its step. */
async function drawnAcross(place?: particles.Options['place']) {
  const root = await gpuRoot();
  // 6.4 pixels a unit, with the origin in the middle.
  const size = { width: 64, height: 64 };
  const screen = targets.create(root, size, { color: { format: 'rgba8unorm' } });
  const view = cameras.flat(root, { size, height: 10 });
  const dust = particles.create(root, view, {
    capacity: 1,
    step: 1,
    dimensions: 2,
    ...(place && { place }),
  });
  const drawn = passes.create(root, screen, { color: passes.color }).add(dust.drawing);

  // No variance, so it is drawn 2 units across in every run.
  dust.burst({
    count: 1,
    position: [-3, 0, 0],
    velocity: [4, 0, 0],
    size: 2,
    life: 10,
    variance: 0,
  });

  /** Where the particle is drawn across `ahead` seconds on, in pixels: its brightness's middle. */
  const drawnAt = async (ahead = 0) => {
    const encoder = root.createCommandEncoder();
    const compute = encoder.beginComputePass();
    dust.spawn(compute);
    dust.compact(compute, ahead);
    compute.end();

    const pass = encoder.beginRenderPass({ colorAttachments: { view: screen.views.color } });
    drawn.draw(pass);
    pass.end();
    encoder.submit();

    const pixels = await readPixels(root, root.unwrap(screen.textures.color), 64, 64);
    let sum = 0;
    let weight = 0;
    for (let i = 0; i < pixels.length; i += 4) {
      sum += (((i / 4) % 64) + 0.5) * pixels[i];
      weight += pixels[i];
    }
    return sum / weight;
  };

  return { root, drawnAt };
}

it('draws a particle ahead of where the last step left it, by its velocity times the ahead', async () => {
  const { root, drawnAt } = await drawnAcross();
  expect(await drawnAt()).toBeCloseTo(32 - 3 * 6.4, 0);

  // Three quarters of a second not yet stepped, at 4 units a second, from -3 to the origin.
  expect(await drawnAt(0.75)).toBeCloseTo(32, 0);

  root.destroy();
});

it('places the corners of a particle where its place puts them, from where it is drawn', async () => {
  const { root, drawnAt } = await drawnAcross((particle, corner) => {
    'use gpu';
    return particles.facing(particle, corner) + d.vec3f(2, 0, 0);
  });
  expect(await drawnAt()).toBeCloseTo(32 - 1 * 6.4, 0);

  expect(await drawnAt(0.75)).toBeCloseTo(32 + 2 * 6.4, 0);

  root.destroy();
});

it('runs died on each particle as its life ends, once, here emitting an ember into another system', async () => {
  const { root, system, frame, living } = await setup(16, { gravity: [0, 0, 0] });
  const embers = system();
  const view = cameras.flat(root, { size: { width: 4, height: 4 }, height: 10 });
  const sparks = particles.create(root, view, {
    capacity: 16,
    step: STEP,
    died: tgpu
      .fn((spark: Particle) => {
        'use gpu';
        let ember = Particle(spark);
        ember.age = 0;
        ember.life = 5;
        embers.emit(ember);
      })
      .pipe(embers.inject()),
  });

  sparks.burst({ count: 5, position: [0, 0, 0], life: 1.5 * STEP, variance: 0 });
  const steps = (count: number) => {
    const encoder = root.createCommandEncoder();
    const pass = encoder.beginComputePass();
    sparks.spawn(pass);
    for (let i = 0; i < count; i++) sparks.step(pass);
    pass.end();
    encoder.submit();
  };

  steps(1);
  expect([await aliveIn(sparks), await aliveIn(embers)]).toEqual([5, 0]);
  steps(1);
  expect([await aliveIn(sparks), await aliveIn(embers)]).toEqual([0, 5]);
  steps(3);
  expect(await aliveIn(embers)).toBe(5);
  frame(0);
  expect(await living()).toHaveLength(0);

  root.destroy();
});

it('writes the motion of a particle since last frame into a description’s output, across a step', async () => {
  const root = await gpuRoot();
  // 3.2 pixels a unit, a fifth of clip space each way.
  const size = { width: 32, height: 32 };
  const screen = targets.create(root, size, {
    color: { format: 'rgba16float' },
    motion: { format: 'rg32float' },
  });
  const Clips = d.struct({ current: d.vec4f, previous: d.vec4f });
  const motion = passes.output(d.vec2f, {
    varyings: Clips,
    previous: true,
    vary: (current: { position: d.v4f }, previous: { position: d.v4f }) => {
      'use gpu';
      return Clips({ current: current.position, previous: previous.position });
    },
    write: (_seen: unknown, clips) => {
      'use gpu';
      return clips.current.xy / clips.current.w - clips.previous.xy / clips.previous.w;
    },
  });
  const into = passes.create(root, screen, { color: passes.color, motion });
  const view = cameras.flat(root, { size, height: 10 });
  view.update(0);
  const sim = clock.fixedStep({ step: 1 });
  // No gravity or drag, so the velocity a step leaves is the one it started with.
  const dust = particles.create(root, view, {
    capacity: 1,
    step: sim.step,
    dimensions: 2,
    settings: { gravity: [0, 0, 0], drag: 0 },
  });
  const drawn = into.add(dust.drawing);

  // No variance, so it is drawn 2 units across in every run.
  dust.burst({
    count: 1,
    position: [-3, 0, 0],
    velocity: [4, 0, 0],
    size: 2,
    life: 10,
    variance: 0,
  });

  /** A frame of `dt`, and the range of clip-space motion across where the particle is drawn. */
  const moved = async (dt: number) => {
    const steps = sim.update(dt);
    const encoder = root.createCommandEncoder();
    const compute = encoder.beginComputePass();
    dust.spawn(compute);
    for (let i = 0; i < steps; i++) dust.step(compute);
    dust.compact(compute, sim.ahead);
    compute.end();
    const pass = into.begin(encoder);
    drawn.draw(pass);
    pass.end();
    encoder.submit();

    const readback = root.device.createBuffer({
      size: 32 * 8 * 32,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const copy = root.device.createCommandEncoder();
    copy.copyTextureToBuffer(
      { texture: root.unwrap(screen.textures.motion) },
      { buffer: readback, bytesPerRow: 32 * 8 },
      [32, 32],
    );
    root.device.queue.submit([copy.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const texels = new Float32Array(readback.getMappedRange().slice(0));
    readback.destroy();
    const across = texels.filter((x, i) => i % 2 === 0 && x !== 0);
    return { texels: across.length, least: Math.min(...across), most: Math.max(...across) };
  };

  await moved(0);
  // Half a second on, 2 units along without a step: drawn at -1, last frame at -3.
  const unstepped = await moved(0.5);
  expect(unstepped.texels).toBeGreaterThan(20);
  expect(unstepped.least).toBeCloseTo(0.4, 5);
  expect(unstepped.most).toBeCloseTo(0.4, 5);
  // A second on, a step and half a step ahead: drawn at 3, last frame at -1.
  const stepped = await moved(1);
  expect(stepped.least).toBeCloseTo(0.8, 5);
  expect(stepped.most).toBeCloseTo(0.8, 5);

  root.destroy();
});

it('rebounds a drop off a plane and pushes a particle out of a sphere along its normal', async () => {
  const root = await gpuRoot();
  const view = cameras.flat(root, { size: { width: 8, height: 8 }, height: 10 });
  const center = d.vec3f(3, 1, 0);
  const ground = particles.collide(
    {
      distance: (p) => {
        'use gpu';
        return std.min(p.y, std.length(p - center) - 1);
      },
    },
    { bounce: 0.6, friction: 0.1 },
  );
  const drops = particles.create(root, view, {
    capacity: 2,
    step: STEP,
    update: ground,
    settings: { gravity: [0, 0, 0], drag: 0 },
  });
  // Half a size is 0.1: the drop sinks 0.05 into the plane, the other 0.6 into the sphere
  drops.burst({
    count: 1,
    position: [-3, 0.05, 0],
    velocity: [2, -4, 0],
    size: 0.2,
    life: 9,
    variance: 0,
  });
  drops.burst({ count: 1, position: [3.3, 1.4, 0], size: 0.2, life: 9, variance: 0 });
  const encoder = root.createCommandEncoder();
  const compute = encoder.beginComputePass();
  drops.spawn(compute);
  drops.step(compute);
  compute.end();
  encoder.submit();

  const pool = await drops.pool.read();
  const drop = pool.find((each) => each.position.x < 0)!;
  // Up at 0.6 of its fall, with 0.9 of its sideways speed, from the plane's surface
  expect([...drop.velocity]).toEqual([expect.closeTo(1.8, 5), expect.closeTo(2.4, 5), 0]);
  expect(drop.position.y).toBeCloseTo(0.1 + 2.4 * STEP, 5);
  const inside = pool.find((each) => each.position.x > 0)!;
  // From 0.5 along (0.6, 0.8) from the centre to 1.1 along it
  expect([...inside.position]).toEqual([
    expect.closeTo(3.66, 2),
    expect.closeTo(1.88, 2),
    expect.closeTo(0, 2),
  ]);
  expect([...inside.velocity]).toEqual([0, 0, 0]);
  root.destroy();
});

it('leaves a particle where the gradient is zero, at a sphere’s centre', async () => {
  const root = await gpuRoot();
  const view = cameras.flat(root, { size: { width: 8, height: 8 }, height: 10 });
  const ball = particles.collide({
    distance: (p) => {
      'use gpu';
      return std.length(p) - 1;
    },
    gradient: (p) => {
      'use gpu';
      return p * 2;
    },
  });
  const drops = particles.create(root, view, {
    capacity: 1,
    step: STEP,
    update: ball,
    settings: { gravity: [0, 0, 0], drag: 0 },
  });
  drops.burst({ count: 1, position: [0, 0, 0], size: 0.2, life: 9, variance: 0 });
  const encoder = root.createCommandEncoder();
  const compute = encoder.beginComputePass();
  drops.spawn(compute);
  drops.step(compute);
  compute.end();
  encoder.submit();

  const [drop] = await drops.pool.read();
  expect([...drop!.position, ...drop!.velocity]).toEqual([0, 0, 0, 0, 0, 0]);
  root.destroy();
});
