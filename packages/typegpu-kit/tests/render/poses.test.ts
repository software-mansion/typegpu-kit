import { cameras, frames, history, passes, targets, type CameraBase } from '@typegpu/kit';
import { recordingRoot } from '@typegpu/kit/testing';
import { meshes } from '@typegpu/kit/geometry';
import { d, tgpu, type TgpuRoot } from 'typegpu';
import { describe, expect, it } from 'vitest';
import { instances, lighting, meshPool, poses } from '../../src/index.ts';
import { bend, blade, leg, moving, rest, shin, slow, sway } from './leg.ts';

/** Reads -0 as 0, so a turned zero compares equal. */
const unsigned = (values: ArrayLike<number>) => Array.from(values, (value) => value + 0);

/** Passes into an HDR target and depth, of `samples`, seen through `view`. */
const hdr = (root: TgpuRoot, view: CameraBase, samples = 1) =>
  passes
    .create(
      root,
      targets.create(
        root,
        { width: 1, height: 1 },
        {
          hdr: { format: 'rgba16float', samples },
          depth: { format: 'depth24plus', samples },
        },
      ),
      { hdr: passes.color, depth: passes.depth },
    )
    .pipe(view.inject());

/** Adds `renderer` to passes seen through `view`, and to the lighting's shadow when it has one. */
function compiled(
  root: TgpuRoot,
  view: CameraBase,
  light: lighting.Lighting,
  renderer: instances.Renderer,
  samples = 1,
) {
  const lit = hdr(root, view, samples);
  lit.add(renderer);
  light.shadow?.add(renderer);
  return Promise.all([lit.initAsync(), light.shadow?.initAsync()]).then(() => undefined);
}

const FIRST_INSTANCE = ['indirect-first-instance'];

const Walker = instances.define('Walker', {
  host: { pose: d.u32 },
  pose: (walker) => {
    'use gpu';
    return walker.pose;
  },
});

/** A pool with a shin and a box, a camera and a lighting with a shadow. */
function scene(root: TgpuRoot) {
  const pool = meshPool.create(root, { vertices: 100, indices: 200, capacity: 4 });
  const model = pool.add({ meshes: [shin], images: [] });
  const box = pool.add(meshes.box());
  const view = cameras.orbit(root, {
    size: { width: 1, height: 1 },
    target: [0, 0, 0],
    distance: 5,
  });
  return { pool, model, box, view, light: lighting.create(root, { shadow: {} }) };
}

describe('poses', () => {
  it('find every pose in one dispatch a frame, a workgroup a pose and a thread a joint', async () => {
    const { root, log } = recordingRoot();
    const figures = poses.create(root, leg, { bend, rest }, { capacity: 8 });
    const loop = frames.create(root, (frame) => {
      const pass = frame.encoder.beginComputePass();
      figures.animate(pass, frame.time);
      pass.end();
    });

    // No pose, nothing to find.
    const start = log.events.length;
    loop.run(0);
    expect(log.events.slice(start)).toEqual(['begin-compute', 'end-compute', 'submit']);

    figures.add('bend', 0);
    figures.add('rest', 0);
    await figures.initAsync();
    expect(log.shaders.at(-1)).toContain('@workgroup_size(2)');
    expect(log.shaders.at(-1)).toMatch(/var<workgroup> \w+: array<mat4x4f, 2>/);

    loop.run(1);
    const counts = log.counts();
    const before = log.events.length;
    loop.run(2);
    expect(log.events.slice(before)).toEqual([
      'begin-compute',
      'dispatch',
      'end-compute',
      'submit',
    ]);
    expect(log.counts()).toEqual(counts);
    expect([figures.count, figures.joints, figures.capacity]).toEqual([2, 2, 8]);
  });

  it('free what they own but their uniforms when destroyed', () => {
    const { root, log } = recordingRoot();
    const before = log.alive({ uniforms: false });
    const figures = poses.create(root, leg, { bend }, { capacity: 256 });
    figures.add('bend', 0);
    figures.destroy();
    expect(log.alive({ uniforms: false })).toEqual(before);
  });

  it('play a clip from where they were, blending, and refuse clips they were not made with', () => {
    const { root } = recordingRoot();
    const figures = poses.create(root, leg, { bend, rest }, { capacity: 1 });
    const id = figures.add('bend', 0, { loop: false });
    figures.play(id, 'rest', 1, { fade: 0.5 });
    expect(figures.has(id)).toBe(true);

    expect(() => figures.add('rest', 0)).toThrow('The poses are full (1).');
    expect(() => figures.play(id, 'run' as never, 0)).toThrow(
      "The poses have no clip 'run'; they play bend, rest.",
    );
    figures.remove(id);
    expect(figures.has(id)).toBe(false);
    expect(() => figures.play(id, 'bend', 0)).toThrow(/stale/);
  });

  it('give a clip’s cycle in seconds, and a line’s between its clips at the mix', () => {
    const { root } = recordingRoot();
    const figures = poses.create(root, leg, { bend, gait: [bend, slow] }, { capacity: 1 });
    expect(figures.duration('bend')).toBe(1);
    expect([0, 0.25, 1, 3].map((mix) => figures.duration('gait', mix))).toEqual([1, 1.25, 2, 2]);
  });

  it('refuse skins, sizes and clips they cannot pose', () => {
    const { root } = recordingRoot();
    const joint = leg.joints[0];
    const many = { joints: Array.from({ length: 257 }, () => joint) };
    const backwards = { joints: [{ ...joint, parent: 1 }, joint] };
    const short = { joints: [{ ...joint, inverseBind: [1, 0, 0] }] };

    expect(() => poses.create(root, { joints: [] }, { bend }, { capacity: 256 })).toThrow(
      /1 to 256 joints, got 0/,
    );
    expect(() => poses.create(root, many, { bend }, { capacity: 256 })).toThrow(
      /1 to 256 joints, got 257/,
    );
    expect(() => poses.create(root, backwards, { bend }, { capacity: 256 })).toThrow(
      "Joint 0, 'hip', has parent 1; a parent comes before its children.",
    );
    expect(() => poses.create(root, short, { bend }, { capacity: 256 })).toThrow(
      /3 inverse-bind floats. Supply 16/,
    );
    expect(() => poses.create(root, leg, { bend }, { capacity: 65_536 })).toThrow(/1 to 65 535/);
    expect(() => poses.create(root, leg, { bend }, { capacity: 256, hz: 0 })).toThrow(
      /hz must be finite and greater than 0/,
    );
    expect(() => poses.create(root, leg, {}, { capacity: 256 })).toThrow(
      'Poses need a clip to play.',
    );

    const elsewhere = { ...bend, channels: bend.channels.map((c) => ({ ...c, joint: 'wing' })) };
    expect(() => poses.create(root, leg, { fly: elsewhere }, { capacity: 256 })).toThrow(
      "The clip 'fly' moves none of the skin's joints: it moves 'wing'.",
    );
    const twins = { joints: [joint, { ...leg.joints[1], name: 'hip' }] };
    const hips = { ...bend, channels: bend.channels.map((c) => ({ ...c, joint: 'hip' })) };
    expect(() => poses.create(root, twins, { hips }, { capacity: 256 })).toThrow(
      /a name two of the skin's joints/,
    );
    expect(() =>
      poses.create(root, leg, { odd: { ...rest, duration: -1 } }, { capacity: 256 }),
    ).toThrow("The clip 'odd' lasts -1 seconds.");
  });

  it('draw a posed kind through the poses given after the options, from its meshes with joints', async () => {
    const { root, log } = recordingRoot({ features: FIRST_INSTANCE });
    const { pool, model, box, view, light } = scene(root);
    const figures = poses.create(root, leg, { bend }, { capacity: 256 });
    const walkers = instances.create(root, pool, { kind: Walker, capacity: 1024 });
    walkers.add(model, { pose: figures.add('bend', 0) });
    expect(() => walkers.add(box)).toThrow(
      "The kind 'Walker' uses pose and requires a mesh with joints.",
    );

    // A lighting of its own: a description compiles every drawing added to it, again.
    const unposed = lighting.create(root, { shadow: {} });
    await expect(
      compiled(root, view, unposed, walkers.renderer().pipe(unposed.inject())),
    ).rejects.toThrow(
      'Nothing filled the poses: create the pipeline from `root.pipe(figures.inject())`, where `figures` is a `poses.create`.',
    );

    const shaders = log.shaders.length;
    await compiled(
      root,
      view,
      light,
      walkers.renderer().pipe(light.inject()).pipe(figures.inject()),
    );
    const made = log.shaders.slice(shaders);
    const color = made.find((code) => code.includes('@fragment'))!;
    const shadow = made.find((code) => !code.includes('@fragment'))!;
    expect(color).toContain('unpack4x8unorm');
    // The shadow's pass figures positions alone, without the normals.
    expect(shadow).toContain('unpack4x8unorm');
    expect(shadow).not.toContain('unpack2x16snorm');
  });

  it('let an unposed set draw meshes with joints as they were bound', async () => {
    const { root } = recordingRoot({ features: FIRST_INSTANCE });
    const { pool, model, view, light } = scene(root);
    const set = instances.create(root, pool, { capacity: 1024 });
    set.add(model);
    await expect(
      compiled(root, view, light, set.renderer().pipe(light.inject())),
    ).resolves.toBeUndefined();
  });

  it('attach a model to a joint, in the space the skin was bound in', () => {
    const [held] = poses.attach({ meshes: [blade], images: [] }, leg, 'knee').meshes;
    expect(Array.from(held!.positions)).toEqual([0, 2, 0, 1, 2, 0, 0, 2, 1]);
    expect(Array.from(held!.joints!)).toEqual([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]);
    expect(Array.from(held!.weights!)).toEqual([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]);
    expect(Array.from(held!.indices)).toEqual([0, 1, 2]);

    // A mirroring bind turns the triangle over, so it winds back the right way, and the normal
    // by the inverse transpose.
    const mirror = {
      joints: [
        { ...leg.joints[0], inverseBind: [-2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1] },
      ],
    };
    const [mirrored] = poses.attach({ meshes: [blade], images: [] }, mirror, 'hip').meshes;
    expect(Array.from(mirrored!.indices)).toEqual([0, 2, 1]);
    expect(Array.from(mirrored!.positions.subarray(3, 6))).toEqual([-0.5, 1, 0]);
    expect(Array.from(mirrored!.normals.subarray(0, 3))).toEqual([0, 1, 0]);

    // An offset turns the item half a turn about z in the knee's space before it binds, and one
    // that mirrors turns the triangle over.
    const turn = [-1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
    const [turned] = poses.attach({ meshes: [blade], images: [] }, leg, 'knee', {
      offset: turn,
    }).meshes;
    expect(unsigned(turned!.positions)).toEqual([0, 0, 0, -1, 0, 0, 0, 0, 1]);
    expect(unsigned(turned!.normals.subarray(0, 3))).toEqual([0, -1, 0]);
    expect(Array.from(turned!.indices)).toEqual([0, 1, 2]);
    const flip = [1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
    const [flipped] = poses.attach({ meshes: [blade], images: [] }, leg, 'knee', {
      offset: flip,
    }).meshes;
    expect(Array.from(flipped!.indices)).toEqual([0, 2, 1]);
    expect(unsigned(flipped!.normals.subarray(0, 3))).toEqual([0, -1, 0]);

    expect(() => poses.attach({ meshes: [blade], images: [] }, leg, 'hand')).toThrow(
      "The skin has no joint named 'hand'; it has 'hip', 'knee'.",
    );
    const flat = {
      joints: [
        {
          ...leg.joints[0],
          inverseBind: [...moving(0, 0, 0).slice(0, 10), 0, ...moving(0, 0, 0).slice(11)],
        },
      ],
    };
    expect(() => poses.attach({ meshes: [blade], images: [] }, flat, 'hip')).toThrow(
      "Joint 'hip''s inverse bind can't be inverted.",
    );
  });

  it('rebind a model by joint name to a skin of the same rig, listed in another order', () => {
    const [hip, knee] = leg.joints;
    const flipped = {
      joints: [
        { ...knee!, parent: -1 },
        { ...hip!, parent: 0 },
      ],
    };
    const [moved] = poses.rebind({ meshes: [shin, blade], images: [] }, leg, flipped).meshes;
    expect(Array.from(moved!.joints!)).toEqual([1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1]);
    expect(moved!.weights).toBe(shin.weights);
    expect(poses.rebind({ meshes: [blade], images: [] }, leg, flipped).meshes[0]).toBe(blade);

    expect(() => poses.rebind({ meshes: [shin], images: [] }, leg, { joints: [hip!] })).toThrow(
      "The skin has no joint named 'knee'.",
    );
    const lower = { joints: [hip!, { ...knee!, inverseBind: moving(0, -0.5, 0) }] };
    expect(() => poses.rebind({ meshes: [shin], images: [] }, leg, lower)).toThrow(
      "The skins bind their joint 'knee' in different places.",
    );
  });
  it('find every pose in one pipeline however many lines and layers they play', async () => {
    const { root, log } = recordingRoot();
    const figures = poses.create(
      root,
      leg,
      { bend, both: [bend, slow], sway },
      {
        capacity: 8,
        layers: { arms: { joints: ['knee'] }, chest: { add: true }, kick: { add: true } },
      },
    );
    figures.add('both', 0);
    figures.add('bend', 0);
    const pipelines = log.pipelines;
    await figures.initAsync();
    expect(log.pipelines).toBe(pipelines + 1);

    // The keys, spans, rig, rows and joints, and the clock.
    const kernel = log.shaders.at(-1)!;
    expect(kernel.match(/var<storage/g)).toHaveLength(5);
    expect(kernel.match(/var<uniform>/g)).toHaveLength(1);
    expect(figures.clips).toEqual({ bend: 0, both: 1, sway: 2 });
    expect(figures.layers).toEqual({ base: 0, arms: 1, chest: 2, kick: 3 });
  });

  it('pose a vertex through the joints and the clock, this frame and last', async () => {
    const { root, log } = recordingRoot();
    const pool = meshPool.create(root, { vertices: 100, indices: 200, capacity: 4 });
    pool.add({ meshes: [shin], images: [] });
    const figures = poses.create(root, leg, { bend }, { capacity: 8 });
    const which = root.createUniform(d.u32);
    const place = (i: number) => {
      'use gpu';
      return d.vec4f(poses.skinnedPosition(i, meshPool.positionAt(i), which.$), 1);
    };
    const placedBefore = history.previously(place);

    const shaded = (motion: boolean) => {
      const vertex = tgpu.vertexFn({
        in: { i: d.builtin.vertexIndex },
        out: { position: d.builtin.position, before: d.vec4f },
      })(({ i }) => {
        'use gpu';
        let before = d.vec4f();
        if (motion) before = placedBefore(i);
        return { position: place(i), before };
      });
      const fragment = tgpu.fragmentFn({ in: { before: d.vec4f }, out: d.vec4f })(({ before }) => {
        'use gpu';
        return before;
      });
      return root
        .pipe(pool.inject())
        .pipe(figures.inject())
        .createRenderPipeline({ vertex, fragment, targets: { format: 'rgba16float' } })
        .initAsync();
    };

    for (const motion of [false, true]) {
      await shaded(motion);
      // The pool's vertices and joints, the poses' joints; the pose's uniform, the slots and clock.
      const code = log.shaders.at(-1)!.split('@fragment')[0];
      expect(code.match(/var<storage/g)).toHaveLength(3);
      expect(code.match(/var<uniform>/g)).toHaveLength(3);
    }
  });

  it('let a kernel call every verb through the rows, the clips, the clock and the slots', async () => {
    const { root, log } = recordingRoot();
    const figures = poses.create(root, leg, { bend, sway }, { capacity: 8, layers: { arms: {} } });
    const { arms, base } = figures.layers;
    const { sway: swaying, bend: bending } = figures.clips;
    const out = root.createMutable(d.f32);
    const drive = tgpu.computeFn({ workgroupSize: [1] })(() => {
      'use gpu';
      const id = d.u32(1);
      poses.play(id, swaying, arms, 0.2, 0);
      poses.once(id, bending, arms, 0.2);
      poses.mix(id, 0.5, base);
      poses.weigh(id, 0.6, arms);
      poses.stop(id, arms, 0.3);
      out.$ = poses.seconds(id, base);
    });
    await root.pipe(figures.inject()).createComputePipeline({ compute: drive }).initAsync();
    const code = log.shaders.at(-1)!;
    // The rows, the spans and the output; the clock and the slots.
    expect(code.match(/var<storage/g)).toHaveLength(3);
    expect(code.match(/var<uniform>/g)).toHaveLength(2);
  });

  it('write nothing on a steady frame whose mix and weight stay as they were', () => {
    const { root, log } = recordingRoot();
    const figures = poses.create(
      root,
      leg,
      { both: [bend, slow], sway },
      { capacity: 8, layers: { arms: {} } },
    );
    const id = figures.add('both', 0, { mix: 0.3 });
    figures.play(id, 'sway', 0, { layer: 'arms', weight: 0.5 });
    let time = 0;
    const loop = frames.create(root, (frame) => {
      figures.mix(id, 0.3, time);
      figures.weigh(id, 0.5, { layer: 'arms' });
      const pass = frame.encoder.beginComputePass();
      figures.animate(pass, time);
      pass.end();
    });
    loop.run(0);
    const counts = log.counts();
    const from = log.writes.length;
    time = 1;
    loop.run(1);
    // The clock alone.
    expect(log.writes.slice(from).map(({ size }) => size)).toEqual([12]);
    expect(log.counts()).toEqual(counts);

    figures.mix(id, 0.4, 1);
    const changed = log.writes.length;
    loop.run(2);
    // One track of 52 bytes, then the clock.
    expect(log.writes.slice(changed).map(({ size }) => size)).toEqual([52, 12]);
  });

  it('refuse layers, lines and joints they cannot play', () => {
    const { root } = recordingRoot();
    expect(() =>
      poses.create(root, leg, { bend }, { capacity: 1, layers: { arms: { joints: ['hand'] } } }),
    ).toThrow(
      "The layer 'arms' moves 'hand', and the skin has no joint of that name; it has 'hip', 'knee'.",
    );
    expect(() =>
      poses.create(root, leg, { bend }, { capacity: 1, layers: { a: {}, b: {}, c: {}, d: {} } }),
    ).toThrow('Poses have at most 3 layers, got a, b, c, d.');
    expect(() =>
      poses.create(root, leg, { bend }, { capacity: 1, layers: { base: {} } as never }),
    ).toThrow("'base' names the base; name the layer otherwise.");
    expect(() => poses.create(root, leg, { one: [bend] }, { capacity: 1 })).toThrow(
      "The line 'one' has 1 clips; a line mixes two or more.",
    );

    const figures = poses.create(root, leg, { bend, sway }, { capacity: 1, layers: { arms: {} } });
    const id = figures.add('bend', 0);
    expect(() => figures.play(id, 'sway', 0, { layer: 'legs' as never })).toThrow(
      "The poses have no layer 'legs'; they have base, arms.",
    );
    expect(() => figures.stop(id, 0)).toThrow("The base doesn't stop; play another clip on it.");
    expect(() => figures.weigh(id, 0.5)).toThrow('The base always shows whole; weigh a layer.');
    expect(figures.seconds(id, 0, { layer: 'arms' })).toBe(0);
  });
});

const near = (actual: number[], expected: readonly number[]) =>
  expected.forEach((value, c) => expect(actual[c]).toBeCloseTo(value, 5));
const EIGHTH_Z = [0, 0, Math.sin(Math.PI / 8), Math.cos(Math.PI / 8)];

describe('clips on the CPU', () => {
  it('find a clip by name, or throw the names there are', () => {
    expect(poses.clip({ clips: [bend, sway] }, 'sway')).toBe(sway);
    expect(() => poses.clip({ clips: [bend, sway] }, 'run')).toThrow(
      "There is no clip named 'run'; the clips are 'bend', 'sway'.",
    );
  });

  it('sample rotations along the arc and cubic keys along their spline', () => {
    near(poses.sample(bend.channels[0], 0.5), EIGHTH_Z);
    // Flat tangents ease out of the first key: a quarter of the way in, 5/32 of the way along.
    const eased: poses.ChannelData = {
      joint: 'hip',
      path: 'position',
      interpolation: 'cubic',
      times: [0, 1],
      values: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0],
    };
    near(poses.sample(eased, 0.25), [0, 2 * (5 / 32), 0]);
  });

  it('shift a looping clip, wrapping past its end', () => {
    const shifted = poses.shift(sway, 0.75);
    near(poses.sample(shifted.channels[0], 0.5), poses.sample(sway.channels[0], 0.25));
  });

  it('mirror a clip onto the opposite joints, turning the other way', () => {
    const mirrored = poses.mirror(sway, (joint) => (joint === 'hip' ? 'knee' : 'hip'));
    expect(mirrored.channels.map(({ joint }) => joint)).toEqual(['knee', 'hip']);
    near(poses.sample(mirrored.channels[0], 1), [0, 0, -Math.SQRT1_2, Math.SQRT1_2]);
  });

  it('lean the top of a part by the turn above it and keep only the part', () => {
    // The hip turns a quarter and the knee a quarter back, so the knee stays upright in the model.
    const leaned = poses.lean(sway, leg, 'knee');
    expect(leaned.channels.map(({ joint }) => joint)).toEqual(['knee']);
    for (const time of [0, 0.5, 1]) near(poses.sample(leaned.channels[0], time), [0, 0, 0, 1]);
  });
});
