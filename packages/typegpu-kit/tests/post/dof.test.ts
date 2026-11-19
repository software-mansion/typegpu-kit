import { bindings, cameras, finish, targets } from '@typegpu/kit';
import { recordingRoot } from '@typegpu/kit/testing';
import { d, std, tgpu } from 'typegpu';
import { describe, expect, it } from 'vitest';
import { dof } from '../../src/index.ts';

function setup(options: { radius?: dof.Radius; rings?: 2 | 3 } = {}) {
  const recording = recordingRoot();
  const { root } = recording;
  const size = { width: 64, height: 32 };
  const screen = targets.create(root, size, {
    hdr: { format: 'rgba16float', sampled: true },
  });
  const early = targets.create(root, size, {
    depth: { format: 'depth32float', scale: 0.5, sampled: true },
  });
  const view = cameras.orbit(root, { size, target: [0, 0, 0], distance: 6 });
  const before = recording.log.alive();
  const blur = dof.create(root, view, {
    source: () => screen.textures.hdr,
    depth: () => early.textures.depth,
    ...options,
  });
  const frame = () => {
    const encoder = root.createCommandEncoder();
    blur.record(encoder);
    encoder.submit();
  };

  return { recording, root, size, screen, early, view, before, blur, frame };
}

const pass = ['begin-render', 'draw', 'end-render'];

/** The gather's WGSL with `rings`. */
const gatherWith = async (rings: 2 | 3) => {
  const { recording, blur } = setup({ rings });
  await blur.initAsync();
  return recording.log.shaders.find((wgsl) => wgsl.includes('fn gather'))!;
};

/** Tints the near field orange and the far field blue over the picture. */
const focusBand = (color: d.v3f, uv: d.v2f) => {
  'use gpu';
  const r = dof.circle(uv);
  const tint = std.select(d.vec3f(1, 0.45, 0.1), d.vec3f(0.1, 0.45, 1), r > 0);
  return std.mix(color, tint, std.saturate(std.abs(r) * 100));
};

/** Counts the gather's loads of the half-resolution picture. */
const loads = (wgsl: string) => wgsl.match(/textureLoad\(prepared,/g)?.length;

describe('dof', () => {
  it('prepares, tiles, gathers and composites in four passes, writing nothing in a steady frame', async () => {
    const { recording, blur, frame } = setup();
    await blur.initAsync();
    // The four passes and the kernel that moves the focus after an aim
    expect(recording.log.pipelines).toBe(5);
    const drawn = recording.log.shaders.filter((wgsl) => wgsl.includes('@fragment'));
    for (const wgsl of drawn) expect(wgsl).not.toContain('var<storage');

    frame();
    recording.log.events.length = 0;
    const counts = recording.log.counts();
    const writes = recording.log.writes.length;
    frame();
    expect(recording.log.events).toEqual([...pass, ...pass, ...pass, ...pass, 'submit']);
    expect(recording.log.counts()).toEqual(counts);
    expect(recording.log.writes.length).toBe(writes);

    // After an aim, each frame moves the focus and copies it into the settings, writing its dt
    blur.aim(0.5, 0.5);
    frame();
    recording.log.events.length = 0;
    const aimed = recording.log.writes.length;
    frame();
    expect(recording.log.events.slice(0, 5)).toEqual([
      'begin-compute',
      'dispatch',
      'end-compute',
      'copy-buffer',
      'begin-render',
    ]);
    expect(recording.log.writes.length).toBe(aimed + 1);
  });

  it('reads each ring four taps a quarter turn apart, without trigonometry', async () => {
    const three = await gatherWith(3);
    // The centre and four loads per ring, each in a loop of 2r steps
    expect(loads(three)).toBe(13);
    expect(loads(await gatherWith(2))).toBe(9);
    expect(three).not.toMatch(/\b(cos|sin)\(/);
  });

  it('calls the radius it is given, which reads the focus under a point', async () => {
    const recording = recordingRoot();
    const { root } = recording;
    const cursor = root.createUniform(d.vec2f, d.vec2f(0.5));
    const underCursor = (distance: number, _uv: d.v2f) => {
      'use gpu';
      return dof.thinLens(distance, dof.distanceAt(cursor.$));
    };
    const size = { width: 64, height: 32 };
    const screen = targets.create(root, size, { hdr: { format: 'rgba16float', sampled: true } });
    const early = targets.create(root, size, {
      depth: { format: 'depth32float', sampled: true },
    });
    const view = cameras.orbit(root, { size, target: [0, 0, 0], distance: 6 });
    const blur = dof.create(root, view, {
      source: () => screen.textures.hdr,
      depth: () => early.textures.depth,
      radius: underCursor,
    });
    await blur.initAsync();
    const reading = recording.log.shaders.filter((wgsl) => wgsl.includes('fn underCursor'));
    // The prepare and the composite
    expect(reading).toHaveLength(2);
  });

  it('refuses lens and distanceAt outside the radius', () => {
    const { root, view } = setup();
    const kernel = tgpu.computeFn({ workgroupSize: [1] })(() => {
      'use gpu';
      dof.thinLens(2, 3);
    });
    expect(() =>
      root.pipe(view.inject()).createComputePipeline({ compute: kernel }).initSync(),
    ).toThrow('Call `dof.thinLens`, `dof.distanceAt` and `dof.focus` only from the `radius`');
  });

  it('shows the circle through a shared part that follows a resize, keeping the pipelines', () => {
    const { recording, root, size, screen, early, blur, frame } = setup();
    const shared = bindings.share(root, { blur: blur.part });
    const focusView = finish.create(root, {
      format: 'bgra8unorm',
      source: blur,
      grade: tgpu.fn(focusBand).pipe(blur.inject()),
    });
    const output = root.createTexture({ size: [64, 32], format: 'bgra8unorm' }).$usage('render');
    const record = () => {
      const encoder = root.createCommandEncoder();
      blur.record(encoder);
      const shown = encoder.beginRenderPass({
        colorAttachments: [
          { view: output.createView('render'), loadOp: 'clear', storeOp: 'store' },
        ],
      });
      shown.setBindGroup(shared.layout, shared.group());
      focusView.draw(shown);
      shown.end();
      encoder.submit();
    };
    record();
    const wgsl = recording.log.shaders.find((code) => code.includes('fn focusBand'))!;
    expect(wgsl.match(/var blur_prepared: texture_2d<f32>/g)).toHaveLength(1);

    const pipelines = recording.log.pipelines;
    const textures = recording.log.textures;
    size.width = 128;
    screen.fit();
    early.fit();
    record();
    // The source and the depth, then the four of the depth of field
    expect(recording.log.textures - textures).toBe(2 + 4);
    expect(recording.log.pipelines).toBe(pipelines);
    frame();
  });

  it('asks for a shared part when a pipeline holds the picture and the source resizes', () => {
    const { root, size, screen, blur, frame } = setup();
    const kernel = tgpu.computeFn({ workgroupSize: [1] })(() => {
      'use gpu';
      dof.circle(d.vec2f());
    });
    root.pipe(blur.inject()).createComputePipeline({ compute: kernel }).initSync();
    size.width = 128;
    screen.fit();
    expect(frame).toThrow('bindings.share(root, { blur: blur.part })');
  });

  it('refuses rings other than 2 or 3', () => {
    expect(() => setup({ rings: 4 as 3 })).toThrow('Depth of field takes 2 or 3 rings, got 4.');
  });
});
