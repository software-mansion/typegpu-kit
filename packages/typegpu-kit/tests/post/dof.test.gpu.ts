import { gpuRoot } from '@typegpu/kit/testing/dawn';
import { common, d, std, tgpu, type TgpuRoot } from 'typegpu';
import { expect, it } from 'vitest';
import { mat4 } from 'wgpu-matrix';
import { cameras, dof, passes, targets } from '../../src/index.ts';

const SIZE = 64;
const FOV = 0.8;
const DISTANCE = 4;
const COLOR = [0.5, 0.25, 1, 1] as const;

const Vertex = d.struct({ position: d.vec4f });

/** A wall across the whole view, `DISTANCE` in front of the eye. */
const wall: passes.Drawing<typeof Vertex, unknown> = {
  Vertex,
  at: (vertexIndex: number) => {
    'use gpu';
    const xs = [-1, 1, -1, -1, 1, 1];
    const ys = [-1, -1, 1, 1, -1, 1];
    const world = d.vec3f(xs[vertexIndex] * 10, ys[vertexIndex] * 10, -DISTANCE);
    return Vertex({ position: cameras.current.$.viewProjection * d.vec4f(world, 1) });
  },
  pixelOf: () => {
    'use gpu';
    return 0;
  },
  color: passes.output(d.vec4f, {
    write: () => {
      'use gpu';
      return d.vec4f(1);
    },
  }),
  primitive: { cullMode: 'none' },
  draw: (pass) => pass.draw(6),
};

/** The radius `dof.thinLens` gives on the CPU, as a fraction of the picture height. */
const thinLens = (aperture: number, focus: number, distance: number) =>
  0.5 * aperture * (1 / Math.tan(FOV / 2) / 2) * (1 / focus - 1 / distance);

/** White on the left half of the picture and black on the right, or `COLOR` all over. */
const halves = tgpu.fragmentFn({ in: { position: d.builtin.position }, out: d.vec4f })(({
  position,
}) => {
  'use gpu';
  return std.select(d.vec4f(0, 0, 0, 1), d.vec4f(1), position.x < SIZE / 2);
});

interface Run {
  readonly settings?: Partial<d.Infer<typeof dof.Settings>>;
  readonly radius?: dof.Radius;
  readonly split?: boolean;
  /** Aims at the middle and records with this `dt`. */
  readonly dt?: number;
  readonly pull?: number;
}

/** The radius stored at the middle, and the output's middle row, after one `record`. */
async function blurred(root: TgpuRoot, { settings, radius, split = false, dt, pull }: Run) {
  const view = cameras.custom(root);
  view.write(mat4.lookAt([0, 0, 0], [0, 0, -1], [0, 1, 0]), mat4.perspective(FOV, 1, 0.1, 100));

  const screen = targets.create(
    root,
    { width: SIZE, height: SIZE },
    { hdr: { format: 'rgba16float', sampled: true } },
  );
  const early = targets.create(
    root,
    { width: SIZE, height: SIZE },
    { depth: { format: 'depth32float', scale: 0.5, sampled: true } },
  );
  const prepass = passes.create(root, early, { depth: passes.depth }).pipe(view.inject());
  const drawn = [prepass.add(wall as never)];
  const blur = dof.create(root, view, {
    source: () => screen.textures.hdr,
    depth: () => early.textures.depth,
    ...(radius && { radius }),
    ...(settings && { settings }),
    ...(pull !== undefined && { pull }),
  });
  const paint = root.createRenderPipeline({
    vertex: common.fullScreenTriangle,
    fragment: halves,
    targets: { format: 'rgba16float' },
  });
  await Promise.all([blur.initAsync(), prepass.initAsync()]);

  const encoder = root.createCommandEncoder();
  const picture = encoder.beginRenderPass({
    colorAttachments: [
      { view: screen.views.hdr, loadOp: 'clear', storeOp: 'store', clearValue: COLOR },
    ],
  });
  if (split) {
    picture.setPipeline(paint);
    picture.draw(3);
  }
  picture.end();
  prepass.record(encoder, drawn);
  if (dt !== undefined) blur.aim(0.5, 0.5);
  blur.record(encoder, dt === undefined ? {} : { dt });
  encoder.submit();

  // The radius the prepare pass stored, through `dof.circle`
  const out = root.createBuffer(d.f32).$usage('storage');
  const written = out.as('mutable');
  const kernel = tgpu.computeFn({ workgroupSize: [1] })(() => {
    'use gpu';
    written.$ = dof.circle(d.vec2f(0.75, 0.5));
  });
  root.pipe(blur.inject()).createComputePipeline({ compute: kernel }).dispatchWorkgroups(1);
  const stored = await out.read();

  const bytesPerRow = SIZE * 8;
  const buffer = root.device.createBuffer({
    size: bytesPerRow * SIZE,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const copy = root.device.createCommandEncoder();
  copy.copyTextureToBuffer({ texture: root.unwrap(blur.output) }, { buffer, bytesPerRow }, [
    SIZE,
    SIZE,
  ]);
  root.device.queue.submit([copy.finish()]);
  await buffer.mapAsync(GPUMapMode.READ);
  const pixels = new Float16Array(buffer.getMappedRange().slice(0));
  buffer.destroy();

  const row = pixels.subarray((SIZE / 2) * SIZE * 4, (SIZE / 2 + 1) * SIZE * 4);
  const at = (x: number) => [...row.subarray(x * 4, x * 4 + 4)];
  return { radius: stored, at };
}

it('has no radius at the focus, and keeps an even picture as it is', async () => {
  const root = await gpuRoot();
  const { radius, at } = await blurred(root, { settings: { focus: DISTANCE, aperture: 0.5 } });
  expect(radius).toBeCloseTo(0, 4);
  for (const [i, channel] of at(SIZE / 2).entries()) expect(channel).toBeCloseTo(COLOR[i], 2);
  root.destroy();
});

it('gives the thin-lens radius in front of and behind the focus, within the limit', async () => {
  const root = await gpuRoot();
  const behind = await blurred(root, { settings: { focus: 2, aperture: 0.1, limit: 1 } });
  expect(behind.radius).toBeCloseTo(thinLens(0.1, 2, DISTANCE), 4);
  expect(behind.radius).toBeGreaterThan(0);

  const front = await blurred(root, { settings: { focus: 8, aperture: 0.1, limit: 1 } });
  expect(front.radius).toBeCloseTo(thinLens(0.1, 8, DISTANCE), 4);
  expect(front.radius).toBeLessThan(0);

  // A wide blur of an even picture averages to the same colour
  const limited = await blurred(root, { settings: { focus: 1, aperture: 1, limit: 0.02 } });
  expect(limited.radius).toBeCloseTo(0.02, 4);
  for (const [i, channel] of limited.at(SIZE / 2).entries()) {
    expect(channel).toBeCloseTo(COLOR[i], 2);
  }
  root.destroy();
});

it('pulls the focus towards what is under the aim, evenly in blur', async () => {
  const root = await gpuRoot();
  const settings = { focus: 10, aperture: 0.1, limit: 1 };
  const halfway = await blurred(root, { settings, pull: 1, dt: Math.LN2 });
  expect(halfway.radius).toBeCloseTo(thinLens(0.1, 2 / (1 / 10 + 1 / DISTANCE), DISTANCE), 4);

  const there = await blurred(root, { settings, dt: 0 });
  expect(there.radius).toBeCloseTo(0, 4);
  root.destroy();
});

// The left half blurs by a tenth of the picture height, 3 pixels from the edge of the halves
const nearLeft = (_distance: number, uv: d.v2f) => {
  'use gpu';
  return std.select(d.f32(0), d.f32(-0.1), uv.x < 0.5);
};
const farLeft = (_distance: number, uv: d.v2f) => {
  'use gpu';
  return std.select(d.f32(0), d.f32(0.1), uv.x < 0.5);
};

it('spreads a blurred near field over what is sharp behind it', async () => {
  const root = await gpuRoot();
  const { at } = await blurred(root, {
    radius: nearLeft,
    settings: { limit: 0.1 },
    split: true,
  });
  // Sharp black, 3 pixels into the right half, under the white near field's edge
  expect(at(SIZE / 2 + 3)[0]).toBeGreaterThan(0.1);
  // Far from the edge, the right half stays black
  expect(at(SIZE - 4)[0]).toBeLessThan(0.01);
  root.destroy();
});

it('keeps a blurred far field off what is sharp in front of it', async () => {
  const root = await gpuRoot();
  const { at } = await blurred(root, { radius: farLeft, settings: { limit: 0.1 }, split: true });
  expect(at(SIZE / 2 + 3)[0]).toBeLessThan(0.01);
  // The far field blurs only itself, so its edge stays white
  expect(at(SIZE / 2 - 2)[0]).toBeGreaterThan(0.99);
  root.destroy();
});
