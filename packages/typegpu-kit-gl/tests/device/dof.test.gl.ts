import { cameras, dof, passes, targets } from '@typegpu/kit';
import { common, d, std, tgpu, type TgpuRoot } from 'typegpu';
import { afterAll, afterEach, beforeAll, expect, it } from 'vitest';
import type { GLDevice } from '../../src/device/index.ts';
import { compare, readTexture, roots, type Readback } from './compare.ts';

const SIZE = 64;
const DISTANCE = 4;

/** The eye at the origin looking down -z, and a perspective of 0.8 radians from 0.1 to 100. */
const LOOK = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
const f = 1 / Math.tan(0.4);
const PERSPECTIVE = new Float32Array([
  f,
  0,
  0,
  0,
  0,
  f,
  0,
  0,
  0,
  0,
  100 / (0.1 - 100),
  -1,
  0,
  0,
  (0.1 * 100) / (0.1 - 100),
  0,
]);

let gpu: TgpuRoot;
let gl: TgpuRoot;
beforeAll(async () => {
  ({ gpu, gl } = await roots());
});
afterAll(() => {
  gpu.destroy();
  gl.destroy();
});
afterEach(() => {
  expect((gl.device as unknown as GLDevice).gl.getError()).toBe(0);
});

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

/** White on the left half of the picture and black on the right. */
const halves = tgpu.fragmentFn({ in: { position: d.builtin.position }, out: d.vec4f })(({
  position,
}) => {
  'use gpu';
  return std.select(d.vec4f(0, 0, 0, 1), d.vec4f(1), position.x < SIZE / 2);
});

/**
 * Pulls the focus from 10 to the wall over frames, and reads the output after the last few.
 * `steady` runs after the frame that reads the aimed distance, and `measured` three frames later.
 */
async function pulled(root: TgpuRoot, steady?: () => void, measured?: () => void) {
  const view = cameras.custom(root);
  view.write(LOOK, PERSPECTIVE);
  const size = { width: SIZE, height: SIZE };
  const screen = targets.create(root, size, { hdr: { format: 'rgba16float', sampled: true } });
  const early = targets.create(root, size, {
    depth: { format: 'depth32float', scale: 0.5, sampled: true },
  });
  const prepass = passes.create(root, early, { depth: passes.depth }).pipe(view.inject());
  const drawn = [prepass.add(wall as never)];
  const blur = dof.create(root, view, {
    source: () => screen.textures.hdr,
    depth: () => early.textures.depth,
    pull: 2,
    settings: { focus: 10, aperture: 1, limit: 1 },
  });
  const paint = root.createRenderPipeline({
    vertex: common.fullScreenTriangle,
    fragment: halves,
    targets: { format: 'rgba16float' },
  });
  const frame = () => {
    const encoder = root.createCommandEncoder();
    const picture = encoder.beginRenderPass({
      colorAttachments: [{ view: screen.views.hdr, loadOp: 'clear', storeOp: 'store' }],
    });
    picture.setPipeline(paint);
    picture.draw(3);
    picture.end();
    prepass.record(encoder, drawn);
    blur.record(encoder, { dt: 0.1 });
    encoder.submit();
  };

  frame();
  blur.aim(0.5, 0.5);
  frame();
  steady?.();
  for (let i = 0; i < 3; i++) frame();
  measured?.();
  const seen: Readback[] = [];
  for (let i = 0; i < 4; i++) {
    frame();
    seen.push(await readTexture(root.device, root.unwrap(blur.output)));
  }
  blur.destroy();
  screen.destroy();
  early.destroy();
  return seen;
}

/** The red of the pixel right of the middle of a `rgba16float` readback, as half-float bits. */
const edge = (picture: Readback) =>
  new Uint16Array(picture.bytes.buffer, picture.bytes.byteOffset)[
    ((SIZE / 2) * SIZE + SIZE / 2 + 1) * 4
  ]!;

it('pulls the focus on the fallback without reading it back, as WebGPU draws it', async () => {
  const stats = (gl.device as unknown as GLDevice).stats;
  let reads = -1;
  let textures = -1;
  const actual = await pulled(
    gl,
    () => stats.reset(),
    () => {
      reads = [...stats.buffers.values()].reduce((sum, buffer) => sum + buffer.reads, 0);
      textures = stats.readback.textures;
    },
  );
  const expected = await pulled(gpu);

  // Three steady frames read nothing back
  expect(reads).toBe(0);
  expect(textures).toBe(0);
  for (const [k, picture] of expected.entries()) {
    const { mismatched } = compare(picture, actual[k]!);
    expect(mismatched, `frame ${k}`).toBeLessThan(0.01);
  }
  // The pull blurs the edge between the halves, and less as the focus nears the wall
  expect(edge(actual[0]!)).toBeGreaterThan(edge(actual[3]!));
});
