import { gpuRoot } from '@typegpu/kit/testing/dawn';
import { d, std, tgpu, type TgpuRoot } from 'typegpu';
import { expect, it } from 'vitest';
import { mat4 } from 'wgpu-matrix';
import { bindings, cameras, ao, passes, targets } from '../../src/index.ts';

const SIZE = 64;

const Vertex = d.struct({ position: d.vec4f, normal: d.vec3f });
type Vec3 = readonly [number, number, number];

/** A quad from `origin` along `u` and `v`, facing along `u × v`. */
const quad = (origin: Vec3, u: Vec3, v: Vec3): passes.Drawing<typeof Vertex, unknown> => {
  const o = d.vec3f(...origin);
  const a = d.vec3f(...u);
  const b = d.vec3f(...v);
  return {
    Vertex,
    at: (vertexIndex: number) => {
      'use gpu';
      const xs = [0, 1, 0, 0, 1, 1];
      const ys = [0, 0, 1, 1, 0, 1];
      const world = o + a * d.f32(xs[vertexIndex]) + b * d.f32(ys[vertexIndex]);
      return Vertex({
        position: cameras.current.$.viewProjection * d.vec4f(world, 1),
        normal: std.normalize(std.cross(a, b)),
      });
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
};

const floor = quad([-10, 0, 10], [20, 0, 0], [0, 0, -20]);
const wall = quad([0, 0, 10], [0, 0, -20], [0, 10, 0]);

/** Visibility at the middle of the occlusion of `shapes` seen from `eye`, looking at the origin. */
async function visibilityAt(
  root: TgpuRoot,
  eye: Vec3,
  shapes: readonly passes.Drawing<typeof Vertex, unknown>[],
  normals: boolean,
  projection = mat4.perspective(0.8, 1, 0.1, 100),
) {
  const view = cameras.custom(root);
  const up: Vec3 = eye[0] === 0 && eye[2] === 0 ? [0, 0, -1] : [0, 1, 0];
  view.write(mat4.lookAt(eye, [0, 0, 0], up), projection);

  const early = targets.create(
    root,
    { width: SIZE, height: SIZE },
    {
      normal: { format: 'rg16float', sampled: true },
      depth: { format: 'depth32float', sampled: true },
    },
  );
  const prepass = (
    normals
      ? passes.create(root, early, { normal: ao.normal, depth: passes.depth })
      : passes.create(root, early, { depth: passes.depth })
  ).pipe(view.inject());
  const drawn = shapes.map((shape) => prepass.add(shape as never));
  const occlusion = ao.create(root, view, {
    depth: () => early.textures.depth,
    ...(normals && { normal: () => early.textures.normal }),
  });
  await Promise.all([occlusion.initAsync(), prepass.initAsync()]);

  const encoder = root.createCommandEncoder();
  prepass.record(encoder, drawn);
  occlusion.record(encoder);
  encoder.submit();

  const bytesPerRow = SIZE * 4;
  const buffer = root.device.createBuffer({
    size: bytesPerRow * SIZE,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const copy = root.device.createCommandEncoder();
  copy.copyTextureToBuffer({ texture: root.unwrap(occlusion.output) }, { buffer, bytesPerRow }, [
    SIZE,
    SIZE,
  ]);
  root.device.queue.submit([copy.finish()]);
  await buffer.mapAsync(GPUMapMode.READ);
  const halves = new Float16Array(buffer.getMappedRange().slice(0));
  buffer.destroy();

  const middle = (SIZE / 2) * SIZE + SIZE / 2;
  return { visibility: halves[middle * 2], distance: halves[middle * 2 + 1], view, occlusion };
}

for (const normals of [false, true]) {
  const from = normals ? 'from the normal output' : 'from depth';

  it(`sees an open floor as unoccluded, and its distance, with normals ${from}`, async () => {
    const root = await gpuRoot();
    const above = await visibilityAt(root, [0, 4, 0], [floor], normals);
    expect(above.visibility).toBeGreaterThan(0.98);
    expect(above.distance).toBeCloseTo(4, 1);

    const slanted = await visibilityAt(root, [3, 2, 3], [floor], normals);
    expect(slanted.visibility).toBeGreaterThan(0.98);
    root.destroy();
  });

  it(`darkens a right-angled corner, with normals ${from}`, async () => {
    const root = await gpuRoot();
    const { visibility } = await visibilityAt(root, [2, 1.5, 2], [floor, wall], normals);
    expect(visibility).toBeLessThan(0.7);
    expect(visibility).toBeGreaterThan(0.03);
    root.destroy();
  });
}

it('sees the same floor and corner through an orthographic view', async () => {
  const root = await gpuRoot();
  const flat = mat4.ortho(-3, 3, -3, 3, 0.1, 100);
  const above = await visibilityAt(root, [0, 4, 0], [floor], false, flat);
  expect(above.visibility).toBeGreaterThan(0.98);
  expect(above.distance).toBeCloseTo(4, 1);

  const slanted = await visibilityAt(root, [3, 2, 3], [floor], false, flat);
  expect(slanted.visibility).toBeGreaterThan(0.98);

  const { visibility } = await visibilityAt(root, [2, 1.5, 2], [floor, wall], false, flat);
  expect(visibility).toBeLessThan(0.7);
  expect(visibility).toBeGreaterThan(0.03);
  root.destroy();
});

it('reads the occlusion at a world position through ao.at', async () => {
  const root = await gpuRoot();
  const { visibility, view, occlusion } = await visibilityAt(
    root,
    [2, 1.5, 2],
    [floor, wall],
    false,
  );

  const points = [d.vec3f(0.02, 0, 0), d.vec3f(1, 0, 0.5), d.vec3f(0, 3, 0), d.vec3f(0.6, 0.6, 0)];
  const out = root.createBuffer(d.arrayOf(d.f32, 4)).$usage('storage');
  const written = out.as('mutable');
  const kernel = tgpu.computeFn({ workgroupSize: [1] })(() => {
    'use gpu';
    const at = [points[0], points[1], points[2], points[3]];
    for (let i = 0; i < 4; i++) written.$[i] = ao.at(at[i]);
  });
  const shared = bindings.share(root, { ao: occlusion.part });
  const pipeline = root
    .pipe(view.inject())
    .pipe(occlusion.inject())
    .createComputePipeline({ compute: kernel });
  pipeline.with(shared.group()).dispatchWorkgroups(1);
  const [crease, open, onWall, inAir] = await out.read();

  // By the crease as the texture holds it, open floor, and two points no texel holds
  expect(crease).toBeCloseTo(visibility, 1);
  expect(open).toBeGreaterThan(0.98);
  expect(onWall).toBeGreaterThan(0.9);
  expect(inAir).toBe(1);
  root.destroy();
});
