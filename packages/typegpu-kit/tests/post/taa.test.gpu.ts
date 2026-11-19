import { cameras, frames, history, passes, targets, type SizeSource } from '@typegpu/kit';
import { gpuRoot } from '@typegpu/kit/testing/dawn';
import { d, std, tgpu, type TgpuRoot } from 'typegpu';
import { expect, it } from 'vitest';
import { taa } from '../../src/index.ts';

const SIZE = 32;

// A white rectangle from `left` to `right` in clip space, full height, at an offset kept in a
// history so its motion is known.
const Corner = d.struct({ position: d.vec4f });
const span = tgpu.accessor(d.vec2f);

const at = (vertexIndex: number, _instanceIndex: number) => {
  'use gpu';
  const xs = [0, 1, 0, 0, 1, 1];
  const ys = [-1, -1, 1, 1, -1, 1];
  const x = std.mix(span.$.x, span.$.y, d.f32(xs[vertexIndex]));
  return Corner({ position: d.vec4f(x, d.f32(ys[vertexIndex]), 0.5, 1) });
};
const pixelOf = (_corner: d.Infer<typeof Corner>) => {
  'use gpu';
  return Corner();
};
const white = passes.output(d.vec4f, {
  write: () => {
    'use gpu';
    return d.vec4f(1);
  },
});

const clipX = (pixel: number) => (pixel / SIZE) * 2 - 1;

async function setup() {
  const root = await gpuRoot();
  const screen = targets.create(
    root,
    { width: SIZE, height: SIZE },
    {
      hdr: { format: 'rgba16float', sampled: true },
      motion: { format: 'rg16float', sampled: true },
      depth: { format: 'depth24plus', sampled: true },
    },
  );
  const aa = taa.create(root, {
    source: () => screen.textures.hdr,
    motion: () => screen.textures.motion,
    depth: () => screen.textures.depth,
  });
  const scene = passes.create(
    root,
    screen,
    { hdr: passes.color, motion: taa.motion, depth: passes.depth },
    { jitter: aa.jitter },
  );
  const placed = history.create(root, d.vec2f, { usage: 'uniform' });
  const rectangle = scene.add({
    Vertex: Corner,
    at,
    pixelOf,
    color: white,
    inject: () => (cfg) => cfg.with(span, placed.reading),
    draw: (pass) => pass.draw(6),
  });
  await Promise.all([aa.initAsync(), scene.initAsync()]);

  const loop = frames.create(root, (frame) => {
    placed.record(frame);
    const pass = scene.begin(frame.encoder);
    rectangle.draw(pass);
    pass.end();
    aa.record(frame.encoder);
  });
  const frame = () => loop.run(0);

  return { root, aa, placed, frame };
}

/** The red channel of a row of an `rgba16float` SIZE × SIZE texture. */
async function row(root: TgpuRoot, texture: GPUTexture, y: number) {
  const bytesPerRow = SIZE * 8;
  const buffer = root.device.createBuffer({
    size: bytesPerRow * SIZE,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const encoder = root.device.createCommandEncoder();
  encoder.copyTextureToBuffer({ texture }, { buffer, bytesPerRow }, [SIZE, SIZE]);
  root.device.queue.submit([encoder.finish()]);
  await buffer.mapAsync(GPUMapMode.READ);
  const halves = new Float16Array(buffer.getMappedRange().slice(0));
  buffer.destroy();
  return Array.from({ length: SIZE }, (_, x) => halves[(y * SIZE + x) * 4]);
}

it('never resolves to a negative channel, even from a source that holds one', async () => {
  const root = await gpuRoot();
  const screen = targets.create(
    root,
    { width: SIZE, height: SIZE },
    { hdr: { format: 'rgba16float', sampled: true } },
  );
  const aa = taa.create(root, { source: () => screen.textures.hdr, depth: null });
  await aa.initAsync();
  const loop = frames.create(root, (frame) => {
    frame.encoder
      .beginRenderPass({ colorAttachments: { view: screen.views.hdr, clearValue: [-1, 2, -1, 1] } })
      .end();
    aa.record(frame);
  });

  // The first frame passes the source through; later ones clamp history to the box around it
  for (const count of [1, 4]) {
    for (let i = 0; i < count; i++) loop.run(0);
    expect(await row(root, root.unwrap(aa.output), 16)).toEqual(Array<number>(SIZE).fill(0));
  }
  root.destroy();
});

it('settles an edge between two pixels on a share of each, where one frame shows all or none', async () => {
  const { root, aa, placed, frame } = await setup();
  placed.write(d.vec2f(clipX(10.3), 1));

  aa.enable(false);
  frame();
  const hard = await row(root, root.unwrap(aa.output), 16);
  aa.enable(true);
  for (let i = 0; i < 48; i++) frame();
  const soft = await row(root, root.unwrap(aa.output), 16);

  // The edge crosses pixel 10 three tenths in, so seven tenths of it are white.
  expect([hard[9], hard[10], hard[11]]).toEqual([0, 1, 1]);
  expect(soft[10]).toBeGreaterThan(0.4);
  expect(soft[10]).toBeLessThan(0.95);
  expect(soft[5]).toBeCloseTo(0, 2);
  expect(soft[20]).toBeCloseTo(1, 2);
  root.destroy();
});

it('follows a moving rectangle, leaving no trail behind it', async () => {
  const { root, aa, placed, frame } = await setup();

  // Eight pixels wide, moving right a pixel a frame.
  for (let i = 0; i < 16; i++) {
    placed.write(d.vec2f(clipX(4 + i), clipX(12 + i)));
    frame();
  }
  const seen = await row(root, root.unwrap(aa.output), 16);

  // It now covers pixels 19 to 26; three pixels behind it is what it left.
  expect(seen[23]).toBeGreaterThan(0.95);
  expect(seen[16]).toBeLessThan(0.05);
  expect(seen[30]).toBeLessThan(0.05);
  root.destroy();
});

// A rectangle standing still in the world, x from 0 to 2.5, that writes no motion, under a camera
// at z = 5 with a 90° view, so a pixel at z = 0 spans 10 / SIZE.
const Point = d.struct({ position: d.vec4f });
const still = (vertexIndex: number, _instanceIndex: number) => {
  'use gpu';
  const xs = [0, 1, 0, 0, 1, 1];
  const ys = [-1, -1, 1, 1, -1, 1];
  const world = d.vec3f(d.f32(xs[vertexIndex]) * 2.5, d.f32(ys[vertexIndex]) * 10, 0);
  return Point({ position: cameras.current.$.viewProjection * d.vec4f(world, 1) });
};
const placeless = (_point: d.Infer<typeof Point>) => {
  'use gpu';
  return Point();
};

async function panning(withCamera: boolean, withMotion = true, reversed = false) {
  const root = await gpuRoot();
  const screen = targets.create(
    root,
    { width: SIZE, height: SIZE },
    {
      hdr: { format: 'rgba16float', sampled: true },
      motion: { format: 'rg16float', sampled: true },
      depth: { format: 'depth24plus', sampled: true },
    },
  );
  const view = cameras.custom(root, { reversed });
  const aa = taa.create(root, {
    source: () => screen.textures.hdr,
    ...(withMotion && { motion: () => screen.textures.motion }),
    depth: () => screen.textures.depth,
    ...(withCamera && { camera: view }),
  });
  const scene = passes
    .create(
      root,
      screen,
      { hdr: passes.color, motion: taa.motion, depth: passes.depth },
      { jitter: aa.jitter, ...(reversed && { depthClear: 0, depthCompare: 'greater' as const }) },
    )
    .pipe(view.inject());
  const rectangle = scene.without('motion').add({
    Vertex: Point,
    at: still,
    pixelOf: placeless,
    color: white,
    draw: (pass) => pass.draw(6),
  });
  await Promise.all([aa.initAsync(), scene.initAsync()]);

  // Column-major: a 90° perspective into WebGPU's depth from 0 to 1, near 0.1 and far 100, or
  // from 1 to 0 when reversed, and a view from (x, 0, 5) down -z.
  const [near, far] = reversed ? [100, 0.1] : [0.1, 100];
  const projection = new Float32Array(16);
  projection.set([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, far / (near - far), -1]);
  projection[14] = (near * far) / (near - far);
  const frame = (x: number) => {
    const eye = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, -x, 0, -5, 1]);
    view.write(eye, projection);
    const encoder = root.createCommandEncoder();
    const pass = scene.begin(encoder);
    rectangle.draw(pass);
    pass.end();
    aa.record(encoder);
    encoder.submit();
  };

  // The camera pans left a pixel a frame, so the rectangle moves right across the screen.
  for (let i = 0; i < 16; i++) frame(5 - (i * 10) / SIZE);
  const seen = await row(root, root.unwrap(aa.output), 16);
  root.destroy();
  return seen;
}

it('moves what writes no motion as the camera moves it, from its depth', async () => {
  const followed = await panning(true);
  const stuck = await panning(false);
  const motionless = await panning(true, false);

  // The rectangle, 8 pixels wide, covers pixels 15 to 22 at the last frame. Held still, the
  // history smears it a pixel behind and lags its leading edge.
  expect(followed.slice(15, 23).every((shade) => shade > 0.95)).toBe(true);
  expect(followed[14]).toBeLessThan(0.05);
  expect(followed[23]).toBeLessThan(0.05);
  expect(stuck[14]).toBeGreaterThan(0.3);
  expect(stuck[22]).toBeLessThan(0.7);
  // A TAA given no motion moves every pixel with the camera.
  expect(motionless).toEqual(followed);
});

it('follows the nearer surface around each pixel under reversed depth', async () => {
  const followed = await panning(true, true, true);

  expect(followed.slice(15, 23).every((shade) => shade > 0.95)).toBe(true);
  expect(followed[14]).toBeLessThan(0.05);
  expect(followed[23]).toBeLessThan(0.05);
});

/** The red channel of the top-left `width` × `height` texels of an `rgba16float` texture. */
async function reds(root: TgpuRoot, texture: GPUTexture, width: number, height: number) {
  const bytesPerRow = Math.ceil((texture.width * 8) / 256) * 256;
  const buffer = root.device.createBuffer({
    size: bytesPerRow * texture.height,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const encoder = root.device.createCommandEncoder();
  encoder.copyTextureToBuffer({ texture }, { buffer, bytesPerRow }, [
    texture.width,
    texture.height,
  ]);
  root.device.queue.submit([encoder.finish()]);
  await buffer.mapAsync(GPUMapMode.READ);
  const halves = new Float16Array(buffer.getMappedRange().slice(0));
  buffer.destroy();
  return Array.from({ length: height }, (_row, y) =>
    Array.from({ length: width }, (_column, x) => halves[(y * bytesPerRow) / 2 + x * 4]!),
  );
}

/** Samples `image` at `width` × `height` pixel centres, between its texels by bilinear weights. */
function stretch(image: number[][], width: number, height: number) {
  const rows = image.length;
  const columns = image[0]!.length;
  const texel = (x: number, y: number) =>
    image[Math.min(rows - 1, Math.max(0, y))]![Math.min(columns - 1, Math.max(0, x))]!;
  return Array.from({ length: height }, (_row, y) =>
    Array.from({ length: width }, (_column, x) => {
      const u = ((x + 0.5) * columns) / width - 0.5;
      const v = ((y + 0.5) * rows) / height - 0.5;
      const [i, j] = [Math.floor(u), Math.floor(v)];
      const [s, t] = [u - i, v - j];
      const top = texel(i, j) * (1 - s) + texel(i + 1, j) * s;
      const bottom = texel(i, j + 1) * (1 - s) + texel(i + 1, j + 1) * s;
      return top * (1 - t) + bottom * t;
    }),
  );
}

/** Averages each `n` × `n` block of `image`. */
function shrink(image: number[][], n: number) {
  return Array.from({ length: image.length / n }, (_row, y) =>
    Array.from({ length: image[0]!.length / n }, (_column, x) => {
      let sum = 0;
      for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) sum += image[y * n + j]![x * n + i]!;
      return sum / (n * n);
    }),
  );
}

const closeTo = (seen: number[], printed: number[]) =>
  seen.forEach((shade, x) => expect(shade).toBeCloseTo(printed[x]!, 3));

const rms = (a: number[][], b: number[][]) => {
  const squares = a.flatMap((line, y) => line.map((value, x) => (value - b[y]![x]!) ** 2));
  return Math.sqrt(squares.reduce((sum, value) => sum + value, 0) / squares.length);
};

/** Output pixels across the upscaling tests' picture. */
const OUTPUT = 48;
/** Rings whose spacing narrows from many source pixels at the middle to under one at the edge. */
const RINGS = 70;

// A full-screen triangle carrying its picture's UV, so jitter moves what each pixel sees
const Texel = d.struct({ position: d.vec4f, uv: d.vec2f });
const Tinted = d.struct({ color: d.vec4f });
const plate = (vertexIndex: number) => {
  'use gpu';
  const xs = [-1, 3, -1];
  const ys = [-1, -1, 3];
  const x = d.f32(xs[vertexIndex]);
  const y = d.f32(ys[vertexIndex]);
  return Texel({ position: d.vec4f(x, y, 0.5, 1), uv: d.vec2f(x + 1, 1 - y) * 0.5 });
};
const rings = (texel: d.Infer<typeof Texel>) => {
  'use gpu';
  const from = texel.uv - 0.5;
  const shade = 0.5 + 0.5 * std.cos(RINGS * std.dot(from, from) * Math.PI);
  return Tinted({ color: d.vec4f(shade, shade, shade, 1) });
};

/**
 * Draws the rings into an `hdr` target of `side` pixels, or the `resolution` of it, and resolves
 * them `frames` times, at `size` if given, or shows one frame as drawn for none.
 */
async function ringsThrough(
  root: TgpuRoot,
  side: number,
  options: { frames: number; resolution?: number; size?: SizeSource },
) {
  const screen = targets.create(
    root,
    { width: side, height: side },
    { hdr: { format: 'rgba16float', sampled: true } },
    { resolution: options.resolution ?? 1 },
  );
  const aa = taa.create(root, {
    source: () => screen.textures.hdr,
    depth: null,
    region: screen.regions.hdr,
    ...(options.size && { size: options.size }),
  });
  const scene = passes.create(root, screen, { hdr: passes.color }, { jitter: aa.jitter });
  const drawn = scene.add({
    Vertex: Texel,
    at: plate,
    pixelOf: rings,
    color: passes.shaded,
    draw: (pass) => pass.draw(3),
  });
  if (options.frames === 0) aa.enable(false);
  for (let i = 0; i < Math.max(1, options.frames); i++) {
    const encoder = root.createCommandEncoder();
    scene.record(encoder, drawn);
    aa.record(encoder);
    encoder.submit();
  }
  const { width, height } = aa.region;
  return reds(root, root.unwrap(aa.output), width, height);
}

it('resolves rings finer than a source pixel closer to the full picture than a stretch does', async () => {
  const root = await gpuRoot();
  const truth = shrink(await ringsThrough(root, OUTPUT * 4, { frames: 0 }), 4);
  const size = { width: OUTPUT, height: OUTPUT };

  // At 1.5, a texture two thirds the size; at 1.3, the region a resolution draws of a full one.
  // Five cycles of jitter each.
  for (const [side, resolution] of [
    [OUTPUT / 1.5, 1],
    [OUTPUT, 1 / 1.3],
  ] as const) {
    const up = await ringsThrough(root, side, { frames: 90, resolution, size });
    const resolved = await ringsThrough(root, side, { frames: 90, resolution });
    const drawn = await ringsThrough(root, side, { frames: 0, resolution });
    expect(rms(up, truth)).toBeLessThan(rms(stretch(resolved, OUTPUT, OUTPUT), truth));
    expect(rms(up, truth)).toBeLessThan(rms(stretch(drawn, OUTPUT, OUTPUT), truth));
  }
});

it('lets a changed pixel follow the current frame where `reactive` is high, and accumulates as before at 0', async () => {
  const root = await gpuRoot();
  const side = 16;
  const picture = root
    .createTexture({ size: [side, side], format: 'rgba16float' })
    .$usage('sampled');
  // Reactive on the right half only
  const quick = root.createTexture({ size: [side, side], format: 'r8unorm' }).$usage('sampled');
  root.device.queue.writeTexture(
    { texture: root.unwrap(quick) },
    Uint8Array.from({ length: side * side }, (_, i) => (i % side >= side / 2 ? 255 : 0)),
    { bytesPerRow: side },
    [side, side],
  );
  const responsive = taa.create(root, {
    source: () => picture,
    depth: null,
    reactive: () => quick,
  });
  const plain = taa.create(root, { source: () => picture, depth: null });

  // A checkerboard, as dithering draws, whose neighbourhood spans black to white, so the colour
  // clamp keeps any history; then the same board inverted
  const board = (phase: number) =>
    Float16Array.from({ length: side * side * 4 }, (_, i) => {
      const texel = i >> 2;
      return (i & 3) === 3 ? 1 : ((texel % side) + Math.floor(texel / side) + phase) % 2;
    });
  const show = (phase: number) => {
    root.device.queue.writeTexture(
      { texture: root.unwrap(picture) },
      board(phase).buffer,
      {
        bytesPerRow: side * 8,
      },
      [side, side],
    );
    const encoder = root.createCommandEncoder();
    responsive.record(encoder);
    plain.record(encoder);
    encoder.submit();
  };

  for (let i = 0; i < 40; i++) show(0);
  for (let i = 0; i < 3; i++) show(1);
  const quickly = await reds(root, root.unwrap(responsive.output), side, side);
  const usually = await reds(root, root.unwrap(plain.output), side, side);

  // Three frames after the change, the reactive half is close to the new board, and the other
  // half has moved less than halfway, as it does without `reactive`
  const off = (image: number[][], half: 0 | 1) =>
    Math.max(
      ...image.flatMap((line, y) =>
        line.map((value, x) =>
          Math.floor((x * 2) / side) === half ? Math.abs(value - ((x + y + 1) % 2)) : 0,
        ),
      ),
    );
  expect(quickly.map((line) => line.slice(0, side / 2))).toEqual(
    usually.map((line) => line.slice(0, side / 2)),
  );
  expect(off(quickly, 1)).toBeLessThan(0.25);
  expect(off(quickly, 0)).toBeGreaterThan(0.5);
});

it('resolves without `size` or `reactive` exactly as it did before they existed', async () => {
  const { root, aa, placed, frame } = await setup();

  // An edge settling, then a rectangle moving 0.7 pixels a frame, printed by the resolve before
  // `size` and `reactive`
  placed.write(d.vec2f(clipX(10.3), 1));
  for (let i = 0; i < 12; i++) frame();
  const settled = await row(root, root.unwrap(aa.output), 16);
  for (let i = 0; i < 16; i++) {
    placed.write(d.vec2f(clipX(4 + i * 0.7), clipX(12 + i * 0.7)));
    frame();
  }
  const moving = await row(root, root.unwrap(aa.output), 16);

  closeTo(settled, [...Array<number>(10).fill(0), 0.630859375, ...Array<number>(21).fill(1)]);
  closeTo(
    moving,
    [
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0498046875, 0.49462890625, 0.99951171875, 1, 1, 1, 1,
      1, 0.93408203125, 0.41357421875, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ],
  );
  root.destroy();
});
