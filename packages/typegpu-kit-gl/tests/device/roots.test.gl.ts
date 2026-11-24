import { d, std, tgpu, type TgpuRoot } from 'typegpu';
import { afterAll, beforeAll, describe, expect, test } from 'vitest';
import { compareRoots, roots } from './compare.ts';

let gpu: TgpuRoot;
let gl: TgpuRoot;

beforeAll(async () => {
  ({ gpu, gl } = await roots());
});

afterAll(() => {
  gpu?.destroy();
  gl?.destroy();
});

const SIZE = 32;

function colorTarget(root: TgpuRoot, format: 'rgba8unorm' | 'rgba16float' = 'rgba8unorm') {
  return root.createTexture({ size: [SIZE, SIZE], format }).$usage('render', 'sampled');
}

function depthTarget(root: TgpuRoot, format: 'depth24plus' | 'depth32float' = 'depth24plus') {
  return root.createTexture({ size: [SIZE, SIZE], format }).$usage('render', 'sampled');
}

const fullScreen = tgpu.vertexFn({
  in: { index: d.builtin.vertexIndex },
  out: { position: d.builtin.position, uv: d.vec2f },
})(({ index }) => {
  'use gpu';
  const corners = [d.vec2f(-1, -1), d.vec2f(3, -1), d.vec2f(-1, 3)];
  const corner = corners[index]!;
  return {
    position: d.vec4f(corner, 0, 1),
    uv: d.vec2f(corner.x * 0.5 + 0.5, 0.5 - corner.y * 0.5),
  };
});

const Vertex = d.struct({ position: d.vec3f, normal: d.vec3f });
const layout = tgpu.vertexLayout(d.arrayOf(Vertex));

function litMesh(root: TgpuRoot) {
  const vertices = root
    .createBuffer(d.arrayOf(Vertex, 6), [
      { position: d.vec3f(-0.9, -0.9, 0.6), normal: d.vec3f(0, 0, 1) },
      { position: d.vec3f(0.9, -0.9, 0.6), normal: d.vec3f(0, 0, 1) },
      { position: d.vec3f(0, 0.9, 0.6), normal: d.vec3f(0, 0, 1) },
      { position: d.vec3f(-0.5, -0.5, 0.2), normal: d.vec3f(0.6, 0, 0.8) },
      { position: d.vec3f(0.5, -0.5, 0.9), normal: d.vec3f(0.6, 0, 0.8) },
      { position: d.vec3f(0, 0.5, 0.2), normal: d.vec3f(0, 0.6, 0.8) },
    ])
    .$usage('vertex');
  const indices = root.createBuffer(d.arrayOf(d.u16, 6), [0, 1, 2, 3, 4, 5]).$usage('index');
  const camera = root.createUniform(d.struct({ transform: d.mat4x4f, light: d.vec3f }), {
    transform: d.mat4x4f(0.8, 0.1, 0, 0, -0.1, 0.8, 0, 0, 0, 0, 1, 0, 0.05, 0, 0, 1),
    light: d.vec3f(0.3, 0.4, 0.866),
  });
  const vertex = tgpu.vertexFn({
    in: { position: d.vec3f, normal: d.vec3f },
    out: { position: d.builtin.position, normal: d.vec3f },
  })((input) => {
    'use gpu';
    return {
      position: camera.$.transform * d.vec4f(input.position, 1),
      normal: input.normal,
    };
  });
  const fragment = tgpu.fragmentFn({ in: { normal: d.vec3f }, out: d.vec4f })(({ normal }) => {
    'use gpu';
    const lit = std.max(std.dot(std.normalize(normal), camera.$.light), 0);
    return d.vec4f(d.vec3f(lit, lit * 0.5, 1 - lit), 1);
  });
  const color = colorTarget(root);
  const depth = depthTarget(root);
  root
    .createRenderPipeline({
      attribs: layout.attrib,
      vertex,
      fragment,
      targets: { format: 'rgba8unorm' },
      depthStencil: { format: 'depth24plus', depthCompare: 'less', depthWriteEnabled: true },
      primitive: { cullMode: 'back' },
    })
    .with(layout, vertices)
    .withIndexBuffer(indices)
    .withColorAttachment({ view: color, clearValue: [0.1, 0.1, 0.1, 1] })
    .withDepthStencilAttachment({ view: depth })
    .drawIndexed(6);
  return color;
}

function instanced(root: TgpuRoot) {
  const offsets = root.createReadonly(d.arrayOf(d.vec4f, 4), [
    d.vec4f(-0.5, -0.5, 1, 0),
    d.vec4f(0.5, -0.5, 0, 1),
    d.vec4f(-0.5, 0.5, 0.5, 0.5),
    d.vec4f(0.5, 0.5, 1, 1),
  ]);
  const vertex = tgpu.vertexFn({
    in: { index: d.builtin.vertexIndex, instance: d.builtin.instanceIndex },
    out: { position: d.builtin.position, tint: d.vec2f },
  })(({ index, instance }) => {
    'use gpu';
    const corners = [d.vec2f(-0.3, -0.3), d.vec2f(0.3, -0.3), d.vec2f(0, 0.3)];
    const offset = offsets.$[instance]!;
    return { position: d.vec4f(corners[index]! + offset.xy, 0, 1), tint: offset.zw };
  });
  const fragment = tgpu.fragmentFn({ in: { tint: d.vec2f }, out: d.vec4f })(({ tint }) => {
    'use gpu';
    return d.vec4f(tint, 0.25, 1);
  });
  const color = colorTarget(root);
  root
    .createRenderPipeline({ vertex, fragment, targets: { format: 'rgba8unorm' } })
    .withColorAttachment({ view: color })
    .draw(3, 3, 0, 1);
  return color;
}

function texturedQuad(root: TgpuRoot) {
  const image = root
    .createTexture({ size: [64, 64], format: 'rgba8unorm', mipLevelCount: 7 })
    .$usage('sampled', 'render');
  const texels = new Uint8Array(64 * 64 * 4);
  for (let y = 0; y < 64; y++)
    for (let x = 0; x < 64; x++) {
      const on = ((x >> 2) + (y >> 2)) % 2 === 0;
      texels.set(on ? [255, 200, 0, 255] : [0, 40, 255, 255], (y * 64 + x) * 4);
    }
  image.write(texels);
  image.generateMipmaps();
  const view = image.createView(d.texture2d(d.f32));
  const sampler = root.createSampler({
    magFilter: 'linear',
    minFilter: 'linear',
    mipmapFilter: 'linear',
  });
  const fragment = tgpu.fragmentFn({ in: { uv: d.vec2f }, out: d.vec4f })(({ uv }) => {
    'use gpu';
    return std.textureSample(view.$, sampler.$, uv * d.vec2f(1, 0.5));
  });
  const color = colorTarget(root);
  root
    .createRenderPipeline({ vertex: fullScreen, fragment, targets: { format: 'rgba8unorm' } })
    .withColorAttachment({ view: color })
    .draw(3);
  return color;
}

function shadowMap(root: TgpuRoot) {
  const shadow = depthTarget(root, 'depth32float');
  const caster = tgpu.vertexFn({
    in: { index: d.builtin.vertexIndex },
    out: { position: d.builtin.position },
  })(({ index }) => {
    'use gpu';
    const corners = [d.vec2f(-0.6, -0.6), d.vec2f(0.6, -0.6), d.vec2f(0, 0.7)];
    return { position: d.vec4f(corners[index]!, 0.3, 1) };
  });
  root
    .createRenderPipeline({
      vertex: caster,
      depthStencil: { format: 'depth32float', depthCompare: 'less', depthWriteEnabled: true },
    })
    .withDepthStencilAttachment({ view: shadow })
    .draw(3);
  const view = shadow.createView(d.textureDepth2d());
  const sampler = root.createComparisonSampler({
    compare: 'less',
    magFilter: 'linear',
    minFilter: 'linear',
  });
  const fragment = tgpu.fragmentFn({ in: { uv: d.vec2f }, out: d.vec4f })(({ uv }) => {
    'use gpu';
    const lit = std.textureSampleCompare(view.$, sampler.$, uv, 0.5);
    return d.vec4f(lit, lit, lit, 1);
  });
  const color = colorTarget(root);
  root
    .createRenderPipeline({ vertex: fullScreen, fragment, targets: { format: 'rgba8unorm' } })
    .withColorAttachment({ view: color })
    .draw(3);
  return color;
}

function postProcess(root: TgpuRoot) {
  const hdr = colorTarget(root, 'rgba16float');
  const scene = tgpu.fragmentFn({ in: { uv: d.vec2f }, out: d.vec4f })(({ uv }) => {
    'use gpu';
    return d.vec4f(uv.x * 4, uv.y * 2, 0.5, 1);
  });
  root
    .createRenderPipeline({
      vertex: fullScreen,
      fragment: scene,
      targets: { format: 'rgba16float' },
    })
    .withColorAttachment({ view: hdr })
    .draw(3);
  const view = hdr.createView(d.texture2d(d.f32));
  const tonemap = tgpu.fragmentFn({ in: { position: d.builtin.position }, out: d.vec4f })(({
    position,
  }) => {
    'use gpu';
    const c = std.textureLoad(view.$, d.vec2i(position.xy), 0).rgb;
    return d.vec4f(c / (c + 1), 1);
  });
  const color = colorTarget(root);
  root
    .createRenderPipeline({
      vertex: fullScreen,
      fragment: tonemap,
      targets: { format: 'rgba8unorm' },
    })
    .withColorAttachment({ view: color })
    .draw(3);
  return color;
}

function computeThenDraw(root: TgpuRoot) {
  const values = root.createBuffer(d.arrayOf(d.f32, SIZE)).$usage('storage');
  const writable = values.as('mutable');
  const fill = tgpu.computeFn({ in: { id: d.builtin.globalInvocationId }, workgroupSize: [8] })(({
    id,
  }) => {
    'use gpu';
    if (id.x < SIZE) {
      writable.$[id.x] = std.fract(d.f32(id.x) * 0.37);
    }
  });
  root.createComputePipeline({ compute: fill }).dispatchWorkgroups(SIZE / 8);
  const readable = values.as('readonly');
  const fragment = tgpu.fragmentFn({ in: { position: d.builtin.position }, out: d.vec4f })(({
    position,
  }) => {
    'use gpu';
    const value = readable.$[d.u32(position.x)]!;
    return d.vec4f(value, 1 - value, 0, 1);
  });
  const color = colorTarget(root);
  root
    .createRenderPipeline({ vertex: fullScreen, fragment, targets: { format: 'rgba8unorm' } })
    .withColorAttachment({ view: color })
    .draw(3);
  return color;
}

describe('roots on the WebGL device match WebGPU', () => {
  test('a lit mesh with a uniform camera and a depth test', async () => {
    const result = await compareRoots(gpu, gl, litMesh);
    expect(result.first).toBeUndefined();
  });

  test('an instanced draw that reads a storage buffer in the vertex stage', async () => {
    const result = await compareRoots(gpu, gl, instanced);
    expect(result.mismatched).toBeLessThan(0.01);
  });

  test('a textured quad with generated mipmaps', async () => {
    const result = await compareRoots(gpu, gl, texturedQuad, { tolerance: 3 / 255 });
    expect(result.mismatched).toBeLessThan(0.02);
  });

  test('a shadow map read through a comparison sampler', async () => {
    const result = await compareRoots(gpu, gl, shadowMap);
    expect(result.mismatched).toBeLessThan(0.02);
  });

  test('an HDR target read by a post-process pass', async () => {
    const result = await compareRoots(gpu, gl, postProcess);
    expect(result.first).toBeUndefined();
  });

  test('a compute pass that fills a buffer a later draw reads', async () => {
    const result = await compareRoots(gpu, gl, computeThenDraw);
    expect(result.first).toBeUndefined();
  });
});
