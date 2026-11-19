import { gpuRoot } from '@typegpu/kit/testing/dawn';
import { d, std, tgpu, type Configurable, type TgpuRenderCommands, type TgpuRoot } from 'typegpu';
import { expect, it } from 'vitest';
import { mat4 } from 'wgpu-matrix';
import { cameras, frames, history, passes, targets } from '../src/index.ts';

const SIZE = 32;

const Clips = d.struct({ current: d.vec4f, previous: d.vec4f });

const uvOf = (clip: d.v4f) => {
  'use gpu';
  return (clip.xy / clip.w) * d.vec2f(0.5, -0.5);
};

/** Where the surface at each pixel was last frame, from where it is, in UV. */
const motion = passes.output(d.vec2f, {
  varyings: Clips,
  previous: true,
  vary: (current: { position: d.v4f }, previous: { position: d.v4f }) => {
    'use gpu';
    return Clips({ current: current.position, previous: previous.position });
  },
  write: (_pixel: unknown, clips) => {
    'use gpu';
    return uvOf(clips.previous) - uvOf(clips.current);
  },
});

const id = passes.output(d.u32, {
  clear: 0xffffffff,
  write: (pixel: { id: number }) => {
    'use gpu';
    return pixel.id;
  },
});

const shaded = passes.output(d.vec4f, {
  write: (pixel: { color: d.v4f }) => {
    'use gpu';
    return d.vec4f(pixel.color);
  },
});

// Triangles at offsets kept in a history, under a camera that keeps last frame's view.
const Vertex = d.struct({ position: d.vec4f, world: d.vec3f, id: d.u32 });
const Surface = d.struct({ world: d.vec3f, id: d.u32, color: d.vec4f });
const offsets = tgpu.accessor(d.arrayOf(d.vec3f, 2));

const at = (vertexIndex: number, instanceIndex: number) => {
  'use gpu';
  const corners = [d.vec3f(-0.25, -0.25, 0), d.vec3f(0.25, -0.25, 0), d.vec3f(0, 0.25, 0)];
  const world = corners[vertexIndex] + offsets.$[instanceIndex];
  return Vertex({
    position: cameras.current.$.viewProjection * d.vec4f(world, 1),
    world,
    id: instanceIndex + 1,
  });
};

const pixelOf = (vertex: d.Infer<typeof Vertex>) => {
  'use gpu';
  return Surface({ world: vertex.world, id: vertex.id, color: d.vec4f(0.5, 0.5, 0.5, 1) });
};

// A glow without an id or a motion of its own, over the first triangle.
const GlowVertex = d.struct({ position: d.vec4f });
const glowAt = (vertexIndex: number, _instanceIndex: number) => {
  'use gpu';
  const corners = [d.vec2f(-0.7, -0.3), d.vec2f(-0.3, -0.3), d.vec2f(-0.5, 0.3)];
  return GlowVertex({ position: d.vec4f(corners[vertexIndex], 0.5, 1) });
};
const glowSurface = (_vertex: d.Infer<typeof GlowVertex>) => {
  'use gpu';
  return Surface({ world: d.vec3f(), id: 0, color: d.vec4f(0.25, 0, 0, 0) });
};
const add: GPUBlendState = {
  color: { srcFactor: 'one', dstFactor: 'one' },
  alpha: { srcFactor: 'one', dstFactor: 'one' },
};

// A full-screen pass that marches a ray through each pixel to the plane z = 0.
const ScreenVertex = d.struct({ position: d.vec4f, uv: d.vec2f });
const screenAt = (vertexIndex: number, _instanceIndex: number) => {
  'use gpu';
  const corners = [d.vec2f(-1, -1), d.vec2f(3, -1), d.vec2f(-1, 3)];
  const corner = corners[vertexIndex];
  return ScreenVertex({ position: d.vec4f(corner, 0.9, 1), uv: corner * d.vec2f(0.5, -0.5) + 0.5 });
};
const planeSurface = (vertex: d.Infer<typeof ScreenVertex>) => {
  'use gpu';
  const ray = cameras.rayOf(cameras.current.$, d.vec2f(vertex.uv.x, 1 - vertex.uv.y) * 2 - 1);
  const world = ray.origin + ray.direction * (-ray.origin.z / ray.direction.z);
  return Surface({ world, id: 0, color: d.vec4f(0, 0, 0.5, 1) });
};

const clipOf = (world: d.v3f) => {
  'use gpu';
  return cameras.current.$.viewProjection * d.vec4f(world, 1);
};
const clipBefore = history.previously(clipOf);

/** For surfaces only the camera moves: where the world at the pixel was on screen last frame. */
const reprojection = passes.output(d.vec2f, {
  write: (pixel: { world: d.v3f }) => {
    'use gpu';
    return uvOf(clipBefore(pixel.world)) - uvOf(clipOf(pixel.world));
  },
});

/** A view from x along the x axis, without perspective. */
const from = (x: number) => mat4.translation([-x, 0, 0]);

function scene(root: TgpuRoot) {
  const screen = targets.create(
    root,
    { width: SIZE, height: SIZE },
    {
      hdr: { format: 'rgba8unorm', sampled: true },
      motion: { format: 'rg32float', sampled: true },
      id: { format: 'r32uint', sampled: true },
      depth: { format: 'depth24plus' },
    },
  );

  // Temporal antialiasing's part: an offset in clip space, new every frame.
  const jitter = root.createUniform(d.vec2f);
  const view = cameras.custom(root);
  view.write(from(0), mat4.identity());
  const pass = passes
    .create(
      root,
      screen,
      { hdr: passes.color, motion, id, depth: passes.depth },
      {
        jitter: (position) => {
          'use gpu';
          return position + d.vec4f(jitter.$ * position.w, 0, 0);
        },
      },
    )
    .pipe(view.inject());

  const placed = history.create(root, d.arrayOf(d.vec3f, 2), {
    initial: [d.vec3f(-0.3, 0, 0), d.vec3f(0.5, 0, 0)],
  });
  const fill = () => (cfg: Configurable) => cfg.with(offsets, placed.reading);

  // The description builds the pipeline, seeing through its camera.
  const meshes = pass.add({
    Vertex,
    at,
    pixelOf,
    color: shaded,
    inject: fill,
    draw: (begun) => begun.draw(3, 2),
  });

  return { root, screen, jitter, pass, view, placed, fill, meshes };
}

async function read(root: TgpuRoot, texture: GPUTexture, bytes: number) {
  const row = 256;
  const buffer = root.device.createBuffer({
    size: row * SIZE,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const encoder = root.device.createCommandEncoder();
  encoder.copyTextureToBuffer({ texture }, { buffer, bytesPerRow: row }, [SIZE, SIZE]);
  root.device.queue.submit([encoder.finish()]);
  await buffer.mapAsync(GPUMapMode.READ);
  const data = buffer.getMappedRange().slice(0);
  buffer.destroy();
  return (x: number, y: number) => new DataView(data, y * row + x * bytes);
}

/** The motion at each pixel, in UV, and the id. */
async function sampled(root: TgpuRoot, screen: ReturnType<typeof scene>['screen']) {
  const motions = await read(root, root.unwrap(screen.textures.motion), 8);
  const ids = await read(root, root.unwrap(screen.textures.id), 4);
  return {
    motion: (x: number, y: number) => [
      motions(x, y).getFloat32(0, true),
      motions(x, y).getFloat32(4, true),
    ],
    id: (x: number, y: number) => ids(x, y).getUint32(0, true),
  };
}

// The triangles' middles are at clip x -0.5 and 0.5, pixels 8 and 24; row 17 is clip y -0.09.
// Panned right by 0.1, they move to pixels 6 and 22.

it('writes motion from last frame through the camera and the instances, without the jitter', async () => {
  const { root, screen, jitter, pass, view, placed, meshes } = scene(await gpuRoot());
  // A pipeline of its own, built by the description.
  const glow = pass.without('motion', 'id').pipeline({
    Vertex: GlowVertex,
    at: glowAt,
    pixelOf: glowSurface,
    color: shaded,
    blend: add,
    depthStencil: { depthWriteEnabled: false, depthCompare: 'always' },
  });

  let withGlow = false;
  const loop = frames.create(root, (frame) => {
    placed.record(frame);
    const drawing = pass.begin(frame.encoder);
    meshes.draw(drawing);
    drawing.end();
    if (withGlow) {
      const late = pass.begin(frame.encoder, { clear: false });
      late.setPipeline(glow);
      late.draw(3, 1);
      late.end();
    }
  });
  const frame = (glowing: boolean) => {
    withGlow = glowing;
    loop.run(0);
  };

  // The first instance moves left by 0.2; the camera holds still.
  placed.write([d.vec3f(-0.5, 0, 0), d.vec3f(0.5, 0, 0)]);
  jitter.write(d.vec2f(0.03, -0.02));
  view.write(from(0), mat4.identity());
  frame(false);
  const first = await sampled(root, screen);

  // The camera pans right by 0.1; the instances hold still. Another jitter, and the glow.
  jitter.write(d.vec2f(-0.03, 0.02));
  view.write(from(0.1), mat4.identity());
  frame(true);
  const second = await sampled(root, screen);
  const colorAt = (await read(root, root.unwrap(screen.textures.hdr), 4))(6, 17);
  const color = [0, 1, 2, 3].map((i) => colorAt.getUint8(i));

  // Last frame minus this one: the first instance was 0.2 to the right, 0.1 in UV.
  expect(first.motion(8, 17)[0]).toBeCloseTo(0.1, 5);
  expect(first.motion(8, 17)[1]).toBeCloseTo(0, 5);
  expect(first.motion(24, 17)[0]).toBeCloseTo(0, 5);
  expect([first.id(8, 17), first.id(24, 17), first.id(16, 2)]).toEqual([1, 2, 0xffffffff]);
  // Both were 0.1 to the right, 0.05 in UV; the glow left the motion and the ids as they were.
  expect(second.motion(6, 17)[0]).toBeCloseTo(0.05, 5);
  expect(second.motion(22, 17)[0]).toBeCloseTo(0.05, 5);
  expect([second.id(6, 17), second.id(22, 17)]).toEqual([1, 2]);
  // The glow's quarter of red over the triangle's half grey.
  expect(color[0]).toBeGreaterThanOrEqual(190);
  expect(color[0]).toBeLessThanOrEqual(192);
  expect(color.slice(1)).toEqual([128, 128, 255]);
  root.destroy();
});

it('reprojects what only the camera moves, and shows a write only after a record', async () => {
  const { root, screen, jitter, pass, view, placed, fill, meshes } = scene(await gpuRoot());
  const plane = pass
    .without('id')
    .instead({ motion: reprojection })
    .add({
      Vertex: ScreenVertex,
      at: screenAt,
      pixelOf: planeSurface,
      color: shaded,
      inject: fill,
      draw: (begun) => begun.draw(3, 1),
    });

  let recording = true;
  const loop = frames.create(root, (frame) => {
    if (recording) placed.record(frame);
    const drawing = pass.begin(frame.encoder);
    plane.draw(drawing);
    meshes.draw(drawing);
    drawing.end();
  });
  const frame = (record = true) => {
    recording = record;
    loop.run(0);
  };

  // The first instance moves left by 0.2; the camera holds still.
  placed.write([d.vec3f(-0.5, 0, 0), d.vec3f(0.5, 0, 0)]);
  jitter.write(d.vec2f(0.03, -0.02));
  view.write(from(0), mat4.identity());
  frame();
  const first = await sampled(root, screen);
  // The camera pans right by 0.1.
  jitter.write(d.vec2f(-0.03, 0.02));
  view.write(from(0.1), mat4.identity());
  frame();
  const second = await sampled(root, screen);
  // Nothing moves.
  jitter.write(d.vec2f(0.01, 0.01));
  view.write(from(0.1), mat4.identity());
  frame();
  const third = await sampled(root, screen);
  // The first instance moves back, but no `record` lands it.
  placed.write([d.vec3f(-0.3, 0, 0), d.vec3f(0.5, 0, 0)]);
  view.write(from(0.1), mat4.identity());
  frame(false);
  const fourth = await sampled(root, screen);

  expect(first.motion(8, 17)[0]).toBeCloseTo(0.1, 5);
  expect(first.motion(16, 4)[0]).toBeCloseTo(0, 5);
  // The plane above the triangles moved with the camera, as they did.
  expect(second.motion(6, 17)[0]).toBeCloseTo(0.05, 5);
  expect(second.motion(16, 4)[0]).toBeCloseTo(0.05, 5);
  expect([second.id(6, 17), second.id(16, 4)]).toEqual([1, 0xffffffff]);
  expect(third.motion(6, 17)[0]).toBeCloseTo(0, 5);
  expect(third.motion(16, 4)[0]).toBeCloseTo(0, 5);
  expect([fourth.id(6, 17), fourth.motion(6, 17)[0]]).toEqual([1, expect.closeTo(0, 5)]);
  root.destroy();
});

// A plane at depth 0.75 over the screen, and a card whose pixels write 0.5 on its right half.
const Plane = d.struct({ position: d.vec4f });
const planeAt = (vertexIndex: number, _instanceIndex: number) => {
  'use gpu';
  const corners = [d.vec2f(-1, -1), d.vec2f(3, -1), d.vec2f(-1, 3)];
  return Plane({ position: d.vec4f(corners[vertexIndex], 0.75, 1) });
};
const Depth = d.struct({ depth: d.f32 });
const halfOf = (vertex: d.Infer<typeof Plane>) => {
  'use gpu';
  if (vertex.position.x < SIZE / 2) std.discard();
  return Depth({ depth: 0.5 });
};

it('writes depth alone, a drawing’s own depth with the bias added, and no colour', async () => {
  const root = await gpuRoot();
  const map = targets.create(
    root,
    { width: SIZE, height: SIZE },
    { depth: { format: 'depth32float', sampled: true } },
  );
  const shadow = passes.create(root, map, { depth: passes.depth }, { bias: { constant: 64 } });
  const plane = shadow.add({ Vertex: Plane, at: planeAt, draw: (pass) => pass.draw(3) });
  const card = shadow.add({
    Vertex: Plane,
    at: planeAt,
    pixelOf: halfOf,
    depthOf: (pixel: d.Infer<typeof Depth>) => {
      'use gpu';
      return pixel.depth;
    },
    depthStencil: { depthCompare: 'always' },
    draw: (pass) => pass.draw(3),
  });
  await shadow.initAsync();

  const encoder = root.createCommandEncoder();
  const pass = shadow.begin(encoder);
  plane.draw(pass);
  card.draw(pass);
  pass.end();
  encoder.submit();

  const depths = await read(root, root.unwrap(map.textures.depth), 4);
  // 64 steps of 2^-23 on the written depth; the rasterised one gets its own share of them.
  expect(depths(4, 4).getFloat32(0, true)).toBeCloseTo(0.75, 5);
  expect(depths(28, 4).getFloat32(0, true)).toBeCloseTo(0.5 + 64 * 2 ** -23, 7);
  root.destroy();
});

it('casts a cut-out shape into depth alone, and draws a drawing with its own depth into colour alone', async () => {
  const root = await gpuRoot();
  const size = { width: SIZE, height: SIZE };
  const map = targets.create(root, size, { depth: { format: 'depth32float', sampled: true } });
  const plain = targets.create(root, size, { color: { format: 'rgba8unorm' } });
  const shadow = passes.create(root, map, { depth: passes.depth });
  const canvas = passes.create(root, plain, { color: passes.color });
  const drawing = {
    Vertex: Plane,
    at: planeAt,
    cutout: (vertex: d.Infer<typeof Plane>) => {
      'use gpu';
      return vertex.position.x < SIZE / 2;
    },
    pixelOf: (_vertex: d.Infer<typeof Plane>) => {
      'use gpu';
      return Tinted({ color: d.vec4f(1) });
    },
    color: shaded,
    depthOf: () => {
      'use gpu';
      return d.f32(0.5);
    },
    draw: (pass: TgpuRenderCommands) => pass.draw(3),
  };
  const { depthOf: _, ...cast } = drawing;
  const casting = shadow.add(cast);
  const shown = canvas.add(drawing);
  await Promise.all([shadow.initAsync(), canvas.initAsync()]);

  const encoder = root.createCommandEncoder();
  shadow.record(encoder, casting);
  canvas.record(encoder, shown);
  encoder.submit();

  const depths = await read(root, root.unwrap(map.textures.depth), 4);
  expect(depths(4, 4).getFloat32(0, true)).toBe(1);
  expect(depths(28, 4).getFloat32(0, true)).toBeCloseTo(0.75, 5);
  const colors = await read(root, root.unwrap(plain.textures.color), 4);
  expect([colors(4, 4).getUint8(0), colors(28, 4).getUint8(0)]).toEqual([0, 255]);
  root.destroy();
});

// Flat layers across the screen, without perspective, so every translucent layer weighs the same.
const Layer = d.struct({ position: d.vec4f, tint: d.vec4f });
const tints = tgpu.accessor(d.arrayOf(d.vec4f, 3));
const depths = tgpu.accessor(d.arrayOf(d.f32, 3));
const layerAt = (vertexIndex: number, instanceIndex: number) => {
  'use gpu';
  const corners = [d.vec2f(-1, -1), d.vec2f(3, -1), d.vec2f(-1, 3)];
  return Layer({
    position: d.vec4f(corners[vertexIndex], depths.$[instanceIndex], 1),
    tint: d.vec4f(tints.$[instanceIndex]),
  });
};
const Tinted = d.struct({ color: d.vec4f });
const layerSurface = (layer: d.Infer<typeof Layer>) => {
  'use gpu';
  return Tinted({ color: d.vec4f(layer.tint) });
};

it('lays translucent drawings over the opaque ones in any order, and over in draw order without the targets', async () => {
  const root = await gpuRoot();
  const screen = targets.create(
    root,
    { width: SIZE, height: SIZE },
    {
      hdr: { format: 'rgba8unorm', sampled: true },
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
  const plain = passes.create(root, screen, { hdr: passes.color, depth: passes.depth });

  // Opaque blue at 0.9; half red at 0.5 and half green at 0.3, in either order.
  const fill = (order: readonly number[], under = false) => {
    const tint = root.createReadonly(d.arrayOf(d.vec4f, 3), [
      d.vec4f(0, 0, 1, 1),
      ...order.map((i) => [d.vec4f(0.5, 0, 0, 0.5), d.vec4f(0, 0.5, 0, 0.5)][i]),
    ]);
    const placed = under ? [0.9, 0.95, 0.95] : [0.9, ...order.map((i) => [0.5, 0.3][i])];
    const depth = root.createReadonly(d.arrayOf(d.f32, 3), placed);
    return () => (cfg: Configurable) => cfg.with(tints, tint).with(depths, depth);
  };
  const drawing = { Vertex: Layer, at: layerAt, pixelOf: layerSurface, color: shaded };

  const shown = async (into: passes.Description, order: readonly number[]) => {
    const opaque = into.add({
      ...drawing,
      inject: fill(order),
      draw: (pass) => pass.draw(3, 1, 0, 0),
    });
    const glass = into.add({
      ...drawing,
      translucent: true,
      inject: fill(order),
      draw: (pass) => pass.draw(3, 2, 0, 1),
    });
    // Red again, behind the blue
    const hidden = into.add({
      ...drawing,
      translucent: true,
      inject: fill(order, true),
      draw: (pass) => pass.draw(3, 1, 0, 1),
    });
    const encoder = root.createCommandEncoder();
    into.record(encoder, [opaque, glass, hidden]);
    if (into === layered) into.composite(encoder);
    encoder.submit();
    const pixel = (await read(root, root.unwrap(screen.textures.hdr), 4))(16, 16);
    return [0, 1, 2].map((i) => pixel.getUint8(i));
  };

  // An even average of red and green over a quarter of the blue, either way round.
  const forward = await shown(layered, [0, 1]);
  const backward = await shown(layered, [1, 0]);
  expect(forward).toEqual(backward);
  for (const [channel, value] of forward.entries()) {
    expect(Math.abs(value - [96, 96, 64][channel])).toBeLessThanOrEqual(1);
  }
  // In draw order: green over red over blue, or red over green over blue.
  expect(await shown(plain, [0, 1])).toEqual([64, 128, 64]);
  expect(await shown(plain, [1, 0])).toEqual([128, 64, 64]);
  expect(() => plain.composite(root.createCommandEncoder())).toThrow(/composite/);
  root.destroy();
});

// A triangle at z = 0, seen in layer 1 alone: the other layers look at a point far to its side.
const LayerVertex = d.struct({ position: d.vec4f });
const casterAt = (vertexIndex: number, _instanceIndex: number) => {
  'use gpu';
  const corners = [d.vec3f(-1, -1, 0), d.vec3f(1, -1, 0), d.vec3f(0, 1, 0)];
  return LayerVertex({
    position: cameras.current.$.viewProjection * d.vec4f(corners[vertexIndex], 1),
  });
};

it('draws into the layer each pass is for, through the view of that layer', async () => {
  const root = await gpuRoot();
  const map = targets.create(
    root,
    { width: SIZE, height: SIZE },
    { depth: { format: 'depth16unorm', layers: 3, sampled: true } },
  );
  const views = cameras.layered(root, 3);
  const projection = mat4.ortho(-2, 2, -2, 2, 0, 10);
  for (const k of [0, 1, 2]) {
    const x = k === 1 ? 0 : 20;
    views.write(k, mat4.lookAt([x, 0, 5], [x, 0, 0], [0, 1, 0]), projection);
  }
  const casters = passes.create(root, map, { depth: passes.depth }).pipe(views.inject());
  const drawn = casters.add({
    Vertex: LayerVertex,
    at: casterAt,
    draw: (pass) => pass.draw(3),
  });
  await casters.initAsync();

  const encoder = root.createCommandEncoder();
  casters.record(encoder, drawn);
  encoder.submit();

  const row = 256;
  const depthOf = async (layer: number, x: number, y: number) => {
    const buffer = root.device.createBuffer({
      size: row * SIZE,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const copy = root.device.createCommandEncoder();
    copy.copyTextureToBuffer(
      { texture: root.unwrap(map.textures.depth), origin: { x: 0, y: 0, z: layer } },
      { buffer, bytesPerRow: row },
      [SIZE, SIZE],
    );
    root.device.queue.submit([copy.finish()]);
    await buffer.mapAsync(GPUMapMode.READ);
    const value = new DataView(buffer.getMappedRange()).getUint16(y * row + x * 2, true);
    buffer.destroy();
    return value / 0xffff;
  };

  // The eye is 5 from the triangle, half way through the depth range.
  expect(await depthOf(1, SIZE / 2, SIZE / 2)).toBeCloseTo(0.5, 3);
  expect(await depthOf(1, 1, 1)).toBe(1);
  expect(await depthOf(0, SIZE / 2, SIZE / 2)).toBe(1);
  expect(await depthOf(2, SIZE / 2, SIZE / 2)).toBe(1);
  expect(await root.takeErrors()).toEqual([]);

  casters.destroy();
  map.destroy();
  root.destroy();
});
