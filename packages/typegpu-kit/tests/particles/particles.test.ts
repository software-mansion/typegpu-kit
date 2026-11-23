import { cameras, clock, lighting, passes, pipelines, shading, table, targets } from '@typegpu/kit';
import { census } from '@typegpu/kit/debug';
import { offscreenSurface, recordingRoot } from '@typegpu/kit/testing';
import { d, std, tgpu, type TgpuRoot } from 'typegpu';
import { describe, expect, expectTypeOf, it, vi } from 'vitest';
import { Particle, type Particles } from '../../src/particles/particles.ts';
import { particles, trails } from '../../src/index.ts';
import { stagesOf } from '../stages.ts';

const step = 1 / 60;

/** A canvas's passes, which draw into it alone. */
const canvasOf = (root: TgpuRoot, format: GPUTextureFormat = 'rgba16float') =>
  passes.create(root, offscreenSurface(root, { width: 4, height: 3, format }));

function setup(capacity = 256) {
  const recording = recordingRoot();
  const { root } = recording;
  const view = cameras.orbit(root, {
    size: { width: 4, height: 3 },
    target: [0, 0, 0],
    distance: 5,
  });
  const sparks = particles.create(root, view, { capacity, step });

  return { recording, root, view, sparks };
}

const spark = () => {
  'use gpu';
  return Particle({
    position: d.vec3f(),
    age: 0,
    velocity: d.vec3f(0, 1, 0),
    life: 1,
    color: d.vec4f(1),
    size: 0.1,
    seed: 0.5,
  });
};

/** Lights a particle at its centre, facing the eye, with the lighting piped into it. */
const lit = (chip: Particle, local: d.v2f) => {
  'use gpu';
  const facing = cameras.towards(cameras.current.$, chip.position);
  const color = shading.shade(
    shading.Material({
      color: chip.color.xyz,
      roughness: 1,
      emission: d.vec3f(),
      metallic: 0,
      occlusion: 1,
    }),
    shading.pointAt(chip.position, facing),
  );
  const alpha = chip.color.w * std.saturate(1 - std.dot(local, local));
  return d.vec4f(color * alpha, alpha);
};

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
const id = passes.output(d.u32, {
  write: (pixel: { id: number }) => {
    'use gpu';
    return pixel.id;
  },
});

/** A description with colour, motion, ids and depth, as a scene's passes might write. */
function described(root: TgpuRoot) {
  const screen = targets.create(
    root,
    { width: 4, height: 3 },
    {
      hdr: { format: 'rgba16float' },
      motion: { format: 'rg16float' },
      id: { format: 'r32uint' },
      depth: { format: 'depth24plus' },
    },
  );
  return passes.create(root, screen, { hdr: passes.color, motion, id, depth: passes.depth });
}

/** The colour blend of a pipeline's one target. */
const blendOf = (descriptor: GPURenderPipelineDescriptor) =>
  [...descriptor.fragment!.targets][0]!.blend!.color;

describe('particles', () => {
  it('names its drawing from the label option, or particles by default', () => {
    const { root, view, sparks } = setup();
    const embers = particles.create(root, view, { capacity: 4, step, label: 'embers' });

    expect(sparks.drawing.label).toBe('particles');
    expect(embers.drawing.label).toBe('embers');
  });

  it('compiles a spawn, a step and live lists, and the draw each description adds', async () => {
    const { recording, sparks } = setup();
    await sparks.initAsync();
    expect(recording.log.pipelines).toBe(4);

    // A description's warm-up compiles its drawings' kernels too.
    const { log, root: second } = recordingRoot();
    const view = cameras.orbit(second, {
      size: { width: 4, height: 3 },
      target: [0, 0, 0],
      distance: 5,
    });
    const embers = particles.create(second, view, { capacity: 4, step });
    const canvas = canvasOf(second);
    canvas.add(embers.drawing);
    await canvas.initAsync();
    expect(log.pipelines).toBe(5);
    expect(log.shaders.filter((code) => code.includes('@vertex'))).toHaveLength(1);
  });

  it('starts its gravity and drag at the defaults, under the settings option', () => {
    const { recording, root, view, sparks } = setup();
    const bubbles = particles.create(root, view, {
      capacity: 4,
      step,
      settings: { gravity: [0, 1.2, 0] },
    });

    const floats = (buffer: GPUBuffer) => [
      ...new Float32Array(recording.bytes(buffer).slice().buffer),
    ];
    expect(floats(root.unwrap(sparks.settings.buffer))).toEqual([
      ...Float32Array.of(0, -9.8, 0, 1),
    ]);
    expect(floats(root.unwrap(bubbles.settings.buffer))).toEqual([
      ...Float32Array.of(0, 1.2, 0, 1),
    ]);
  });

  it('spawns queued bursts in one dispatch, then forgets them', () => {
    const { recording, root, sparks } = setup();
    expect(sparks.burst({ count: 50, position: [1, 2, 3] })).toBe(true);
    expect(sparks.burst({ count: 30, position: [0, 0, 0], color: [1, 0.5, 0, 1] })).toBe(true);

    const frame = () => {
      const encoder = root.createCommandEncoder();
      const pass = encoder.beginComputePass();
      sparks.spawn(pass);
      pass.end();
      encoder.submit();
    };

    frame();
    expect(recording.log.events.filter((event) => event === 'dispatch')).toHaveLength(1);

    // Two records of 80 bytes, each starting with its position and then its count.
    const written = recording.log.writes.find((write) => write.size === 2 * 80)!;
    const bytes = recording.bytes(written.buffer);
    const words = new Uint32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
    const floats = new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
    expect([...floats.subarray(0, 3)]).toEqual([1, 2, 3]);
    expect(words[3]).toBe(50);
    expect(words[20 + 3]).toBe(30);

    frame();
    expect(recording.log.events.filter((event) => event === 'dispatch')).toHaveLength(1);
  });

  it('streams the particles its steps owe at a rate, carrying the fraction to the next spawn', () => {
    const { recording, root, sparks } = setup();
    const at: [number, number, number] = [0, 0, 0];
    const smoke = sparks.stream(45, { position: at });
    const queued: number[][] = [];

    for (const steps of [4, 3, 1, 0]) {
      const before = recording.log.writes.length;
      const encoder = root.createCommandEncoder();
      const pass = encoder.beginComputePass();
      sparks.spawn(pass);
      for (let i = 0; i < steps; i++) sparks.step(pass);
      pass.end();
      encoder.submit();
      // One record of 80 bytes: its position, then its count
      const write = recording.log.writes.slice(before).find((written) => written.size === 80);
      if (write) {
        const bytes = recording.bytes(write.buffer);
        const words = new Uint32Array(bytes.buffer, bytes.byteOffset, 4);
        queued.push([new Float32Array(bytes.buffer, bytes.byteOffset, 1)[0], words[3]]);
      }
      at[0] += 1;
      smoke.rate = 90;
    }

    // 3 owed by four steps at 45 a second, then 4.5 by three at 90, then 0.5 + 1.5 by one, each
    // from where the burst stood at its spawn
    expect(queued).toEqual([
      [1, 3],
      [2, 4],
      [3, 2],
    ]);
    expect(() => (smoke.rate = -1)).toThrow('a stream takes 0 or more particles a second');
  });

  it('spawns in separately submitted simulation ticks without compaction', () => {
    const {
      root,
      recording: { log },
      sparks,
    } = setup();
    for (let tick = 0; tick < 2; tick++) {
      sparks.burst({ count: 10, position: [tick, 0, 0] });
      const encoder = root.createCommandEncoder();
      const pass = encoder.beginComputePass();
      sparks.spawn(pass);
      sparks.step(pass);
      pass.end();
      encoder.submit();
    }
    expect(log.events.filter((event) => event === 'dispatch')).toHaveLength(4);
  });

  it('refuses a second spawn of bursts in one pass, whose upload would replace the first', () => {
    const { root, sparks } = setup();
    const pass = root.createCommandEncoder().beginComputePass();
    sparks.burst({ count: 10, position: [0, 0, 0] });
    sparks.spawn(pass);
    sparks.spawn(pass);
    sparks.burst({ count: 10, position: [1, 0, 0] });
    expect(() => sparks.spawn(pass)).toThrow('a second `spawn` in one submission');
  });

  it('trims bursts to its capacity, and refuses them past its capacity or its queue', () => {
    const { recording, root, view } = setup();
    const small = particles.create(root, view, {
      capacity: 100,
      step,
      bursts: 3,
    });
    const at = [0, 0, 0] as const;

    expect(small.burst({ count: 80, position: at })).toBe(true);
    expect(small.burst({ count: 80, position: at })).toBe(true);
    expect(small.burst({ count: 1, position: at })).toBe(false);
    expect(small.burst({ count: 0, position: at })).toBe(false);
    expect(() => small.burst({ count: NaN, position: at })).toThrow(RangeError);

    const encoder = root.createCommandEncoder();
    const pass = encoder.beginComputePass();
    small.spawn(pass);
    pass.end();
    encoder.submit();

    // Two records of 80 bytes; the count is the fourth word of each.
    const written = recording.log.writes.find((write) => write.size === 2 * 80)!;
    const bytes = recording.bytes(written.buffer);
    const words = new Uint32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
    expect([words[3], words[20 + 3]]).toEqual([80, 20]);

    for (let i = 0; i < 3; i++) expect(small.burst({ count: 1, position: at })).toBe(true);
    expect(small.burst({ count: 1, position: at })).toBe(false);
  });

  it('lets another module’s kernel emit through its inject', async () => {
    const { recording, root, sparks } = setup();

    const kernel = tgpu.computeFn({ workgroupSize: [1] })(() => {
      'use gpu';
      sparks.emit(spark());
    });

    await root.pipe(sparks.inject()).createComputePipeline({ compute: kernel }).initAsync();
    expect(recording.log.shaders.at(-1)).toContain('atomicAdd');
  });

  it('gives its capacity and a filled live test to modules that run over its pool', async () => {
    const { recording, root, sparks } = setup(64);
    const live = sparks.live();
    expect(live.capacity).toBe(64);

    const kernel = tgpu.computeFn({ workgroupSize: [1] })(() => {
      'use gpu';
      if (live.alive(3)) return;
    });

    await root.createComputePipeline({ compute: kernel }).initAsync();
    expect(recording.log.shaders.at(-1)).toMatch(/array<Particle, 64>/);
  });

  it('lets one kernel emit into two systems, each through its own pool, unpiped', async () => {
    const { recording, root, view, sparks } = setup();
    const smoke = particles.create(root, view, { capacity: 64, step });

    const kernel = tgpu.computeFn({ workgroupSize: [1] })(() => {
      'use gpu';
      sparks.emit(spark());
      smoke.emit(spark());
    });

    await root.createComputePipeline({ compute: kernel }).initAsync();

    const wgsl = recording.log.shaders.at(-1)!;
    const pools = wgsl.matchAll(/var<storage, read_write> \w+: array<Particle, (\d+)>/g);
    expect([...pools].map(([, length]) => Number(length))).toEqual([256, 64]);
    expect(wgsl.match(/var<storage, read_write> \w+: atomic<u32>/g)).toHaveLength(2);
  });

  it('fills the camera in the step, and update carries the fill of what else it reads', async () => {
    const { recording, root, view } = setup();
    const wind = tgpu.accessor(d.vec3f);
    const breeze = root.createUniform(d.vec3f);

    const leaves = particles.create(root, view, {
      capacity: 64,
      step,
      update: tgpu
        .fn((leaf: Particle) => {
          'use gpu';
          let next = Particle(leaf);
          const away = std.normalize(leaf.position - cameras.current.$.position);
          next.velocity = leaf.velocity + wind.$ + away;
          return next;
        })
        .pipe((cfg) => cfg.with(wind, breeze)),
    });
    await leaves.initAsync();

    const [stepping] = recording.log.shaders.filter((code) => code.includes('fn update'));
    expect(stepping).toBeDefined();
    expect(stepping).toMatch(/update\(next, [\d.]+f, i\)/); // the slot too, for those that read it
    expect(stepping!.match(/var<uniform>/g)!.length).toBeGreaterThanOrEqual(2);
  });

  it('blends over and tests the depth of its passes without writing it, when its options say so', () => {
    const { recording, root, view, sparks } = setup();
    const screen = targets.create(
      root,
      { width: 4, height: 3 },
      { color: { format: 'rgba16float' }, depth: { format: 'depth24plus' } },
    );
    const scene = passes.create(root, screen, { color: passes.color, depth: passes.depth });
    const dust = particles.create(root, view, { capacity: 64, step, blend: 'over' });
    const created = vi.spyOn(root.device, 'createRenderPipeline');
    const canvas = passes.create(root, {
      format: 'rgba16float',
      width: 4,
      height: 4,
      view: () => screen.views.color,
    });
    const glowing = canvas.add(sparks.drawing);
    const dusty = scene.add(dust.drawing);

    const encoder = root.createCommandEncoder();
    const compute = encoder.beginComputePass();
    sparks.compact(compute, 0);
    dust.compact(compute, 0);
    compute.end();
    const light = encoder.beginRenderPass({ colorAttachments: { view: screen.views.color } });
    glowing.draw(light);
    light.end();
    const matter = scene.begin(encoder, { clear: false });
    dusty.draw(matter);
    matter.end();
    encoder.submit();

    expect(recording.log.events.filter((event) => event === 'draw-indirect')).toHaveLength(2);

    const [add, over] = created.mock.calls.map(([descriptor]) => descriptor);

    expect(add!.depthStencil).toBeUndefined();
    expect(blendOf(add!)).toEqual({ srcFactor: 'one', dstFactor: 'one', operation: 'add' });
    expect(over!.depthStencil).toEqual({
      format: 'depth24plus',
      depthWriteEnabled: false,
      depthCompare: 'less',
    });
    expect(blendOf(over!)).toEqual({
      srcFactor: 'one',
      dstFactor: 'one-minus-src-alpha',
      operation: 'add',
    });
  });

  it('draws as a translucent layer when its options say so, and so do trails', () => {
    const { root, view } = setup();
    const screen = targets.create(
      root,
      { width: 4, height: 3 },
      {
        hdr: { format: 'rgba16float', sampled: true },
        accum: { format: 'rgba16float', sampled: true },
        reveal: { format: 'r16float', sampled: true },
        depth: { format: 'depth24plus' },
      },
    );
    const layered = passes.create(root, screen, {
      hdr: passes.color,
      accum: passes.accumulation,
      reveal: passes.revealage,
      depth: passes.depth,
    });
    const bubbles = particles.create(root, view, { capacity: 4, step, blend: 'translucent' });
    const wakes = trails.create(root, { capacity: 1, length: 2, blend: 'translucent' });

    const layer = {
      hdr: { writeMask: 0 },
      accum: { blend: { color: { srcFactor: 'one', dstFactor: 'one' } } },
    };
    expect(stagesOf(layered, bubbles.drawing).targets).toMatchObject(layer);
    expect(stagesOf(layered, wakes.drawing).targets).toMatchObject(layer);
  });

  it('draws into a description’s passes, its colour added, motion since last frame and the depth tested, not written', () => {
    const { recording, root, view } = setup();
    const into = described(root);
    const created = vi.spyOn(root.device, 'createRenderPipeline');
    const embers = particles.create(root, view, { capacity: 64, step });
    const drawn = into.without('id').add(embers.drawing);

    const encoder = root.createCommandEncoder();
    const compute = encoder.beginComputePass();
    embers.compact(compute, 0);
    compute.end();
    const pass = into.begin(encoder);
    drawn.draw(pass);
    pass.end();
    encoder.submit();

    const [descriptor] = created.mock.calls.map(([call]) => call);
    expect(descriptor!.fragment!.targets).toEqual([
      { format: 'rgba16float', blend: pipelines.blend.add },
      { format: 'rg16float' },
      { format: 'r32uint', writeMask: 0 },
    ]);
    expect(descriptor!.depthStencil).toEqual({
      format: 'depth24plus',
      depthWriteEnabled: false,
      depthCompare: 'less',
    });
    // How far ahead it draws them now, and how far it drew them last frame, in one uniform.
    const drawing = recording.log.shaders.find((code) => code.includes('@vertex'))!;
    expect(drawing.match(/var<uniform> \w+: vec2f/g)).toHaveLength(1);
    expect(drawing).toMatch(/return \w+\.x;/);
    expect(drawing).toMatch(/return \w+\.y;/);
  });

  it('draws last frame’s corners a step back along the velocity for each step since that draw', () => {
    const { recording, root, view } = setup();
    const into = described(root);
    // A step and frames whose sums are exact in binary, so a steady frame repeats its numbers.
    const steady = clock.fixedStep({ step: 1 / 4 });
    const embers = particles.create(root, view, { capacity: 64, step: steady.step });
    // Drawn into two descriptions, which draw them alike.
    const drawn = [into.without('id').add(embers.drawing), canvasOf(root).add(embers.drawing)];
    const made = recording.log.writes.length;
    /** How far ahead it draws them, then how far it drew them last frame, as last written. */
    const aheadAndBefore = () => {
      const buffers = new Set(recording.log.writes.slice(made).map(({ buffer }) => buffer));
      return [...buffers].flatMap((buffer) => [
        ...new Float32Array(recording.bytes(buffer).slice().buffer),
      ]);
    };

    const frame = (dt: number) => {
      const steps = steady.update(dt);
      const encoder = root.createCommandEncoder();
      const compute = encoder.beginComputePass();
      for (let i = 0; i < steps; i++) embers.step(compute);
      embers.compact(compute, steady.ahead);
      compute.end();
      const pass = into.begin(encoder);
      for (const each of drawn) each.draw(pass);
      pass.end();
      encoder.submit();
    };

    frame(1 / 8);
    frame(1 / 4);
    // Drawn half a step ahead, then one step on, half a step ahead again.
    expect(aheadAndBefore()).toEqual([1 / 8, 1 / 8 - 1 / 4]);

    const writes = recording.log.writes.length;
    frame(1 / 4);
    expect(recording.log.writes.length).toBe(writes);
  });

  it('names an output it can’t feed, until it leaves it out', () => {
    const { root, view } = setup();
    const into = described(root);

    const embers = particles.create(root, view, { capacity: 4, step });
    // @ts-expect-error -- particles have no id.
    into.add(embers.drawing);
    into.without('id').add(embers.drawing);
  });

  it('runs died on each particle whose life ends in a step, only when given', async () => {
    const { recording, root, view, sparks } = setup();
    const embers = particles.create(root, view, {
      capacity: 64,
      step,
      died: tgpu
        .fn((ember: Particle) => {
          'use gpu';
          sparks.emit(ember);
        })
        .pipe(sparks.inject()),
    });
    await embers.initAsync();
    await sparks.initAsync();

    const steps = recording.log.shaders.filter((code) => code.includes('fn stepKernel'));
    const emitting = steps.filter((code) => code.includes('atomicAdd'));
    expect([steps.length, emitting.length]).toEqual([2, 1]);
  });

  it('defines a particle with fields of its own after a particle’s, and refuses one it has', () => {
    const Ember = particles.schema('Ember', { kind: d.u32, heat: d.f32 });
    expect(Object.keys(Ember.propTypes)).toEqual([
      ...Object.keys(Particle.propTypes),
      'kind',
      'heat',
    ]);
    expect(d.sizeOf(Ember)).toBe(64);

    // @ts-expect-error -- every particle has an age.
    expect(() => particles.schema('Old', { age: d.u32 })).toThrow(
      "'age' is a built-in particle field. Choose another name.",
    );
  });

  it('queues the own fields a burst gives after its record, and zero for the rest', () => {
    const { recording, root, view } = setup();
    const Ember = particles.schema('Ember', { kind: d.u32, heat: d.f32 });
    const embers = particles.create(root, view, { capacity: 64, step, particle: Ember });

    embers.burst({ count: 1, position: [0, 0, 0], fields: { kind: 3, heat: 0.5 } });
    embers.burst({ count: 1, position: [0, 0, 0], fields: { heat: 2 } });
    const encoder = root.createCommandEncoder();
    const pass = encoder.beginComputePass();
    embers.spawn(pass);
    pass.end();
    encoder.submit();

    // Two records of 80 bytes and an ember of 64, whose own fields start at its byte 56.
    const written = recording.log.writes.find((write) => write.size === 2 * 144)!;
    const bytes = recording.bytes(written.buffer);
    const words = new Uint32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
    const floats = new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
    expect([words[34], floats[35]]).toEqual([3, 0.5]);
    expect([words[36 + 34], floats[36 + 35]]).toEqual([0, 2]);
  });

  it('types a system by the schema it was given, and every system as a `particles.System`', () => {
    const { root, view, sparks } = setup();
    const Ember = particles.schema('Ember', { heat: d.f32 });
    const embers = particles.create(root, view, { capacity: 4, step, particle: Ember });

    expectTypeOf<ReturnType<typeof embers.particle>['heat']>().toEqualTypeOf<number>();
    expectTypeOf(sparks).toEqualTypeOf<Particles>();
    const systems: particles.System[] = [sparks, embers];
    expect(systems).toHaveLength(2);
    void (() => {
      // @ts-expect-error -- a system of embers needs `particle: Ember` to store embers
      particles.create<typeof Ember>(root, view, { capacity: 4, step });
    });
  });

  it('blends with a state of its own', () => {
    const { root, view } = setup();
    const created = vi.spyOn(root.device, 'createRenderPipeline');
    const lighten: GPUBlendState = {
      color: { srcFactor: 'one-minus-dst', dstFactor: 'one', operation: 'add' },
      alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
    };
    const glare = particles.create(root, view, { capacity: 4, step, blend: lighten });
    const screen = targets.create(
      root,
      { width: 4, height: 3 },
      { color: { format: 'rgba16float' } },
    );
    const canvas = passes.create(root, screen, { color: passes.color });
    const drawn = canvas.add(glare.drawing);
    const encoder = root.createCommandEncoder();
    const compute = encoder.beginComputePass();
    glare.compact(compute, 0);
    compute.end();
    const pass = canvas.begin(encoder);
    drawn.draw(pass);
    pass.end();
    encoder.submit();

    expect(blendOf(created.mock.calls[0][0])).toEqual(lighten.color);
  });

  it('feeds outputs that read where it is', () => {
    const { root, view, sparks } = setup();
    const depth = passes.output(d.f32, {
      write: (pixel: { world: d.v3f }) => {
        'use gpu';
        return std.length(pixel.world - cameras.current.$.position);
      },
    });
    const screen = targets.create(
      root,
      { width: 4, height: 3 },
      { hdr: { format: 'rgba16float' }, distance: { format: 'r32float' } },
    );
    const into = passes
      .create(root, screen, { hdr: passes.color, distance: depth })
      .pipe(view.inject());
    into.add(sparks.drawing);
  });

  it('takes a camera, not any module with an inject', () => {
    const { root, sparks } = setup();

    // @ts-expect-error -- a particle system fills no camera.
    particles.create(root, sparks, { capacity: 4, step });
  });

  it('shades through a lighting piped into shade, within the binding budget', async () => {
    const { log, root } = recordingRoot();
    using counted = census.create(root);
    const view = cameras.orbit(root, {
      size: { width: 4, height: 3 },
      target: [0, 0, 0],
      distance: 5,
    });
    const lamps = table.create(root, lighting.PointLight, { capacity: 8 });
    const light = lighting.create(root, { points: { table: lamps }, shadow: { size: 64 } });

    const debris = particles.create(root, view, {
      capacity: 64,
      step,
      blend: 'over',
      shade: tgpu.fn(lit).pipe(light.inject()),
    });
    const canvas = canvasOf(root);
    canvas.add(debris.drawing);
    await canvas.initAsync();

    const [drawing] = log.shaders.filter((code) => code.includes('fn lit'));
    expect(drawing).toMatch(/fn reaching\(/); // the point light loop runs per pixel
    expect(counted.overBudget).toEqual([]);
  });
});
