import { gpuRoot } from '@typegpu/kit/testing/dawn';
import { perlin3d } from '@typegpu/noise';
import { common, d, std, tgpu, type TgpuRoot } from 'typegpu';
import { afterEach, describe, expect, it } from 'vitest';
import { bindings, frames, type Inject } from '../src/index.ts';

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

const fragment = tgpu.fragmentFn({ in: { uv: d.vec2f }, out: d.vec4f })(() => {
  'use gpu';
  const seen = std.textureSampleLevel(glow.$.image, glow.$.smooth, d.vec2f(0.5), 0) * tint.$.tint;
  output.$.glow = d.vec4f(seen);
  return seen;
});

const Sample = d.struct({ noise: d.f32 });
const sample = tgpu.mutableAccessor(Sample);

const sampling = tgpu.computeFn({ workgroupSize: [1] })(() => {
  'use gpu';
  sample.$.noise = perlin3d.sample(d.vec3f(1.3, 2.7, 0.4));
});

type Rgba = readonly [number, number, number, number];
const RED: Rgba = [1, 0, 0, 1];
const BLUE: Rgba = [0, 0, 1, 1];

let cleanup: (() => void) | undefined;

afterEach(() => {
  cleanup?.();
  cleanup = undefined;
});

function image(root: TgpuRoot, value: number) {
  const texture = root.createTexture({ size: [2, 2], format: 'rgba8unorm' }).$usage('sampled');
  texture.write(new Uint8Array(2 * 2 * 4).fill(value));
  return texture;
}

const colorsOf = (root: TgpuRoot, list: readonly Rgba[]) =>
  root
    .createBuffer(
      d.arrayOf(d.vec4f, list.length),
      list.map((c) => d.vec4f(...c)),
    )
    .$usage('storage');

async function setup() {
  const root = await gpuRoot();
  cleanup = () => root.destroy();

  const count = root.createBuffer(d.u32, 2).$usage('uniform');
  const parts = {
    palette: palette.bind({ colors: colorsOf(root, [RED, BLUE]), count }),
    glow: glow.bind({
      image: image(root, 51),
      smooth: root.createSampler({ magFilter: 'linear', minFilter: 'linear' }),
    }),
    tint: tint.bind({ tint: root.createBuffer(d.vec4f, d.vec4f(0.5)).$usage('uniform') }),
    canvas: canvas.bind({
      pixels: root.createTexture({ size: [1, 1], format: 'rgba8unorm' }).$usage('storage'),
    }),
  };
  const own = () => Object.values(parts).map((part) => part.inject());

  const out = root.createMutable(Out);
  const create = (...modules: Inject[]) =>
    modules
      .reduce((cfg, module) => cfg.pipe(module), root.with(output, out))
      .createComputePipeline({ compute: kernel });

  // Counts what the device makes, to tell a new group from a new pipeline.
  const made = { pipelines: 0, groups: [] as GPUBindGroupLayout[] };
  const device = root.device;
  const createComputePipeline = device.createComputePipeline.bind(device);
  device.createComputePipeline = (descriptor) => {
    made.pipelines += 1;
    return createComputePipeline(descriptor);
  };
  const createRenderPipeline = device.createRenderPipeline.bind(device);
  device.createRenderPipeline = (descriptor) => {
    made.pipelines += 1;
    return createRenderPipeline(descriptor);
  };
  const createBindGroup = device.createBindGroup.bind(device);
  device.createBindGroup = (descriptor) => {
    made.groups.push(descriptor.layout);
    return createBindGroup(descriptor);
  };

  return { root, parts, own, out, create, made, count };
}

describe('bindings on a GPU', () => {
  it('read their resources as fixed bindings when nobody owns them', async () => {
    const { root, own, out, create } = await setup();
    const pipeline = create(...own());

    frames
      .create(root, ({ encoder }) => {
        const pass = encoder.beginComputePass();
        pass.setPipeline(pipeline);
        pass.dispatchWorkgroups(1);
        pass.end();
      })
      .run(0);
    const read = await out.read();

    expect([...read.color]).toEqual([0, 0, 0.5, 0.5]);
    expect(read.glow.x).toBeCloseTo(0.2, 3);
  });

  it('reads one texture through two fixed view schemas in the same part', async () => {
    const { root } = await setup();
    const both = bindings.layout({
      image: { texture: d.texture2d(d.f32) },
      layers: { texture: d.texture2dArray(d.f32) },
    });
    const texture = root.createTexture({ size: [1, 1, 1], format: 'rgba8unorm' }).$usage('sampled');
    texture.write(new Uint8Array([255, 0, 0, 255]));
    const part = both.bind({ image: texture, layers: texture });
    const result = root.createMutable(d.f32);
    const compute = tgpu.computeFn({ workgroupSize: [1] })(() => {
      'use gpu';
      const flat = std.textureLoad(both.$.image, d.vec2i(), 0);
      const layer = std.textureLoad(both.$.layers, d.vec2i(), 0, 0);
      result.$ = flat.x + layer.x;
    });
    const pipeline = root.pipe(part.inject()).createComputePipeline({ compute });
    frames
      .create(root, ({ encoder }) => {
        const pass = encoder.beginComputePass();
        pass.setPipeline(pipeline);
        pass.dispatchWorkgroups(1);
        pass.end();
      })
      .run(0);

    expect(await result.read()).toBe(2);
  });

  it('read them from a layout the app owns, and swap them with a new group and no new pipelines', async () => {
    const { root, parts, own, out, create, made, count } = await setup();
    const shared = bindings.share(root, parts);
    // The owner's fill first, as an instances renderer pipes its `modules` before the lighting.
    const pipeline = create(shared.inject(), ...own());

    const loop = frames.create(root, ({ encoder }) => {
      const pass = encoder.beginComputePass();
      pass.setBindGroup(shared.layout, shared.group());
      pass.setPipeline(pipeline);
      pass.dispatchWorkgroups(1);
      pass.end();
    });
    const groups = () => made.groups.filter((layout) => layout === root.unwrap(shared.layout));

    loop.run(0);
    const first = await out.read();
    loop.run(1);
    const pipelines = made.pipelines;

    expect(groups()).toHaveLength(1);
    expect([...first.color]).toEqual([0, 0, 0.5, 0.5]);
    expect(first.glow.x).toBeCloseTo(0.2, 3);

    parts.palette.replace({ colors: colorsOf(root, [BLUE, BLUE, RED]) });
    count.write(3);
    parts.glow.replace({ image: image(root, 255) });
    loop.run(2);
    const read = await out.read();

    expect(groups()).toHaveLength(2);
    expect(made.pipelines).toBe(pipelines);
    expect([...read.color]).toEqual([0.5, 0, 0, 0.5]);
    expect(read.glow.x).toBeCloseTo(1, 5);
  });

  it('bind a render pipeline from the pass, and swap what it samples', async () => {
    const { root, parts, out } = await setup();
    const shared = bindings.share(root, { glow: parts.glow, tint: parts.tint });

    const pipeline = root
      .with(output, out)
      .pipe(parts.glow.inject())
      .pipe(parts.tint.inject())
      .createRenderPipeline({
        vertex: common.fullScreenTriangle,
        fragment,
        targets: { format: 'rgba8unorm' },
      });

    const target = root.createTexture({ size: [1, 1], format: 'rgba8unorm' }).$usage('render');
    const view = target.createView('render');
    const loop = frames.create(root, ({ encoder }) => {
      const pass = encoder.beginRenderPass({
        colorAttachments: [{ view, loadOp: 'clear', storeOp: 'store' }],
      });
      pass.setBindGroup(shared.layout, shared.group());
      pass.setPipeline(pipeline);
      pass.draw(3);
      pass.end();
    });

    loop.run(0);
    const before = (await out.read()).glow.x;
    parts.glow.replace({ image: image(root, 255) });
    loop.run(1);
    const after = (await out.read()).glow.x;

    expect(before).toBeCloseTo(0.1, 3);
    expect(after).toBeCloseTo(0.5, 5);
  });

  it("share @typegpu/noise's dynamic cache as it ships, and take a resized cache with a new group", async () => {
    const { root, made } = await setup();
    const config = perlin3d.dynamicCacheConfig();
    const cache = config.instance(root, d.vec3u(4, 4, 4));
    // The cache never had a group of its own: it gives entries, and fills its shaders from
    // whatever layout holds them.
    const noise: bindings.Part<typeof config.layout> = {
      entries: config.layout,
      get resources() {
        return cache.bindings;
      },
      inject(values) {
        if (!values) throw new Error('The noise cache has no group of its own; share it.');
        return config.inject(values);
      },
    };
    const shared = bindings.share(root, { noise });

    const out = root.createMutable(Sample);
    const reference = root.createMutable(Sample);
    const cached = root
      .with(sample, out)
      .pipe(shared.inject())
      .createComputePipeline({ compute: sampling });
    const computed = root.with(sample, reference).createComputePipeline({ compute: sampling });

    const loop = frames.create(root, ({ encoder }) => {
      const pass = encoder.beginComputePass();
      pass.setBindGroup(shared.layout, shared.group());
      pass.setPipeline(cached);
      pass.dispatchWorkgroups(1);
      pass.setPipeline(computed);
      pass.dispatchWorkgroups(1);
      pass.end();
    });
    // The cache makes groups of its own to fill a bigger cache; count the shared ones.
    const groups = () => made.groups.filter((layout) => layout === root.unwrap(shared.layout));

    loop.run(0);
    const [first, expected] = await Promise.all([out.read(), reference.read()]);
    const pipelines = made.pipelines;

    expect(first.noise).toBeCloseTo(expected.noise, 5);
    expect(first.noise).not.toBe(0);

    cache.size = d.vec3u(8, 8, 8);
    loop.run(1);
    loop.run(2);

    expect(groups()).toHaveLength(2);
    expect(made.pipelines).toBe(pipelines);
    expect((await out.read()).noise).toBeCloseTo(expected.noise, 5);
  });
});
