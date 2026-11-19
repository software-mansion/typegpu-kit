import { d, std, tgpu, type TgpuRenderCommands, type TgpuRoot } from 'typegpu';
import { describe, expect, it } from 'vitest';
import { census as censuses } from '../src/debug/index.ts';
import { bindings, frames, passes, type Inject } from '../src/index.ts';
import { offscreenSurface, recordingRoot } from '../src/testing.ts';

// Four owners that don't know about each other, one per kind of entry, and an output that the
// kernel's own code fills.
const palette = bindings.layout({
  colors: { storage: d.arrayOf(d.vec4f), access: 'readonly' },
  count: { uniform: d.u32 },
});
const glow = bindings.layout({
  image: { texture: d.texture2d(d.f32) },
  smooth: { sampler: 'filtering' },
});
const tint = bindings.layout({ tint: { uniform: d.vec4f } });
const canvas = bindings.layout({ pixels: { storageTexture: d.textureStorage2d('rgba8unorm') } });

const Out = d.struct({ color: d.vec4f, glow: d.vec4f });
const output = tgpu.mutableAccessor(Out);

const kernel = tgpu.computeFn({ workgroupSize: [1] })(() => {
  'use gpu';
  const last = d.vec4f(palette.$.colors[palette.$.count - 1]) * tint.$.tint;
  output.$.color = d.vec4f(last);
  output.$.glow = std.textureSampleLevel(glow.$.image, glow.$.smooth, d.vec2f(0.5), 0);
  std.textureStore(canvas.$.pixels, d.vec2u(0, 0), last);
});

const counted = tgpu.computeFn({ workgroupSize: [1] })(() => {
  'use gpu';
  output.$.color = d.vec4f(d.f32(palette.$.count));
});

function owners(root: TgpuRoot) {
  const colors = root.createBuffer(d.arrayOf(d.vec4f, 2)).$usage('storage');
  const count = root.createBuffer(d.u32, 2).$usage('uniform');
  const image = root.createTexture({ size: [2, 2], format: 'rgba8unorm' }).$usage('sampled');
  const smooth = root.createSampler({ magFilter: 'linear', minFilter: 'linear' });
  const color = root.createBuffer(d.vec4f, d.vec4f(1)).$usage('uniform');
  const pixels = root.createTexture({ size: [1, 1], format: 'rgba8unorm' }).$usage('storage');

  return {
    palette: palette.bind({ colors, count }),
    glow: glow.bind({ image, smooth }),
    tint: tint.bind({ tint: color }),
    canvas: canvas.bind({ pixels }),
  };
}

function setup() {
  const recording = recordingRoot();
  const { root } = recording;
  const census = censuses.create(root);
  const parts = owners(root);
  const out = root.createMutable(Out);

  const create = (...modules: Inject[]) =>
    modules
      .reduce((cfg, module) => cfg.pipe(module), root.with(output, out))
      .createComputePipeline({ compute: kernel });

  const own = () => [
    parts.palette.inject(),
    parts.glow.inject(),
    parts.tint.inject(),
    parts.canvas.inject(),
  ];

  return { ...recording, census, parts, create, own };
}

describe('bindings', () => {
  it('bind on their own as fixed bindings, in the automatic group, when nobody owns them', () => {
    const { census, create, own } = setup();

    create(...own()).initSync();

    expect(
      census.pipelines.map((p) => [p.groups, p.bindings.map((b) => b.kind).toSorted()]),
    ).toEqual([
      [1, ['sampler', 'storage', 'storage', 'storageTexture', 'texture', 'uniform', 'uniform']],
    ]);
  });

  it('take a group per owner that keeps its own layout, and go over the limit at four', () => {
    const { root, census, parts, create } = setup();

    const each = Object.values(parts).map((part) => bindings.share(root, { part }));
    create(...each.map((shared) => shared.inject())).initSync();

    expect(census.pipelines.map((p) => p.groups)).toEqual([5]);
    expect(census.overBudget).toHaveLength(1);
  });

  it('share one layout that the app owns, whichever order the fills come in', () => {
    const { root, census, parts, create, own } = setup();
    const shared = bindings.share(root, parts);

    create(...own(), shared.inject()).initSync();
    create(shared.inject(), ...own()).initSync();
    create(...own()).initSync();

    expect(census.pipelines.map((p) => p.groups)).toEqual([2, 2, 2]);
    expect(census.overBudget).toEqual([]);
  });

  it('share a part whose buffer a kernel writes', () => {
    const { root, census } = setup();
    const tally = bindings.layout({ counts: { storage: d.arrayOf(d.u32), access: 'mutable' } });
    const part = tally.bind({ counts: root.createBuffer(d.arrayOf(d.u32, 4)).$usage('storage') });
    const shared = bindings.share(root, { tally: part });
    const count = tgpu.computeFn({ workgroupSize: [1] })(() => {
      'use gpu';
      tally.$.counts[0] += 1;
    });

    root.pipe(shared.inject()).createComputePipeline({ compute: count }).initSync();
    shared.group();

    expect(census.pipelines.map((p) => p.bindings.map((b) => b.kind))).toEqual([['storage']]);
  });

  it('records a drawing that binds its own shared group', async () => {
    const { root, log, parts } = setup();
    const shared = bindings.share(root, { tint: parts.tint });
    const surface = offscreenSurface(root, { width: 8, height: 8, format: 'bgra8unorm' });
    const scene = passes.create(root, surface);
    const Vertex = d.struct({ position: d.vec4f });
    const drawing = scene.add({
      Vertex,
      at: () => {
        'use gpu';
        return Vertex({ position: d.vec4f(0, 0, 0, 1) });
      },
      pixelOf: () => {
        'use gpu';
        return Out({ color: tint.$.tint, glow: d.vec4f() });
      },
      color: passes.output(d.vec4f, {
        write: (pixel: { color: d.v4f }) => {
          'use gpu';
          return d.vec4f(pixel.color);
        },
      }),
      inject: () => parts.tint.inject(),
      draw(pass: TgpuRenderCommands) {
        pass.setBindGroup(shared.layout, shared.group());
        pass.draw(3);
      },
    });
    await scene.initAsync();
    frames.create(root, (frame) => scene.record(frame, drawing)).run(0);
    expect(log.pipelines).toBe(1);
    expect(log.events.filter((event) => event === 'draw')).toHaveLength(1);
  });

  it('make the group once, and again only after a resource is replaced', () => {
    const { root, log, parts, create, own } = setup();
    const shared = bindings.share(root, parts);
    const pipeline = create(...own());

    const made: unknown[] = [];
    const createBindGroup = root.device.createBindGroup.bind(root.device);
    root.device.createBindGroup = (descriptor) => {
      made.push(descriptor.layout);
      return createBindGroup(descriptor);
    };
    const groups = () => made.filter((layout) => layout === root.unwrap(shared.layout)).length;

    const loop = frames.create(root, ({ encoder }) => {
      const pass = encoder.beginComputePass();
      pass.setBindGroup(shared.layout, shared.group());
      pass.setPipeline(pipeline);
      pass.dispatchWorkgroups(1);
      pass.end();
    });

    loop.run(0);
    loop.run(1);
    const compiled = log.pipelines;
    expect(groups()).toBe(1);

    parts.palette.replace({ colors: root.createBuffer(d.arrayOf(d.vec4f, 8)).$usage('storage') });
    loop.run(2);
    loop.run(3);

    expect(groups()).toBe(2);
    expect(log.pipelines).toBe(compiled);
    expect(log.events.filter((event) => event === 'dispatch')).toHaveLength(4);
  });

  it('compare resources entry by entry, so a getter that makes a new object costs no group', () => {
    const { root } = recordingRoot();
    const size = root.createBuffer(d.u32, 4).$usage('uniform');
    const sizeOf = tgpu.accessor(d.u32);

    // The shape of @typegpu/noise's dynamic cache, whose `bindings` is a new object at every read.
    const cache: bindings.Part<{ size: { uniform: d.U32 } }> = {
      entries: { size: { uniform: d.u32 } },
      get resources() {
        return { size };
      },
      inject: (values) => (cfg) => cfg.with(sizeOf, () => values!.size),
    };

    const shared = bindings.share(root, { cache });
    expect(shared.group()).toBe(shared.group());
  });

  it('refuse what would leave a pipeline reading old resources', () => {
    const { root, parts, create, own } = setup();
    create(...own()).initSync();

    expect(() =>
      parts.tint.replace({ tint: root.createBuffer(d.vec4f).$usage('uniform') }),
    ).toThrow(/fixed bindings/);
    expect(() => bindings.share(root, { tint: parts.tint })).toThrow(/Share it before/);
  });

  it('throw as a pipeline builds without the share a part asks for, and build once it is shared', () => {
    const { root } = setup();
    const colors = root.createBuffer(d.arrayOf(d.vec4f, 2)).$usage('storage');
    const count = root.createBuffer(d.u32, 2).$usage('uniform');
    const asked = palette.bind({ colors, count }, { unshared: 'Share the palette first.' });
    const pipeline = root.with(output, root.createMutable(Out)).pipe(asked.inject());

    expect(() => pipeline.createComputePipeline({ compute: counted }).initSync()).toThrow(
      'Share the palette first.',
    );
    const shared = bindings.share(root, { palette: asked });
    pipeline.createComputePipeline({ compute: counted }).initSync();
    asked.replace({ count: root.createBuffer(d.u32, 3).$usage('uniform') });
    expect(shared.group()).toBeDefined();
  });

  it('refuse a second owner, and two entries with one name', () => {
    const { root, parts } = setup();
    bindings.share(root, { palette: parts.palette });

    expect(() => bindings.share(root, { palette: parts.palette })).toThrow(/already has an owner/);

    const entry = { uniform: d.u32 } as const;
    const loose = (name: string): bindings.Part => ({
      entries: { [name]: entry },
      resources: { [name]: root.createBuffer(d.u32).$usage('uniform') },
      inject: () => (cfg) => cfg,
    });

    expect(() => bindings.share(root, { x: loose('a_b'), x_a: loose('b') })).toThrow(/'x_a_b'/);
  });

  it('take no part when they refuse one, so the others can still be shared', () => {
    const { root, parts } = setup();
    bindings.share(root, { tint: parts.tint });

    expect(() => bindings.share(root, { palette: parts.palette, tint: parts.tint })).toThrow(
      /already has an owner/,
    );
    expect(() => bindings.share(root, { a: parts.glow, b: parts.glow })).toThrow(/twice/);
    expect(() => bindings.share(root, { palette: parts.palette, glow: parts.glow })).not.toThrow();
  });

  it('name a WebGPU object that only a bind group can take', () => {
    const { root } = recordingRoot();
    const count = root.unwrap(root.createBuffer(d.u32).$usage('uniform'));
    const colors = root.createBuffer(d.arrayOf(d.vec4f, 2)).$usage('storage');
    const part = palette.bind({ colors, count });

    const pipeline = root
      .with(output, root.createMutable(Out))
      .pipe(part.inject())
      .createComputePipeline({ compute: counted });

    expect(() => pipeline.initSync()).toThrow(/'count' is a WebGPU object/);
  });
});
