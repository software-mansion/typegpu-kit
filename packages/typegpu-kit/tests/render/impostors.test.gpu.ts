import {
  Camera,
  cameras,
  frames,
  passes,
  targets,
  type CameraBase,
  type Drawn,
} from '@typegpu/kit';
import { gpuRoot } from '@typegpu/kit/testing/dawn';
import { meshes } from '@typegpu/kit/geometry';
import { d, tgpu, type TgpuRoot } from 'typegpu';
import { describe, expect, it } from 'vitest';
import {
  impostors,
  instances,
  lighting,
  materials,
  meshPool,
  shading,
  poses,
} from '../../src/index.ts';
import { leg } from './leg.ts';

const SIZE = 64;

/** A colour target and depth, and passes that draw into them, seen through `view`. */
function described(root: TgpuRoot, view: CameraBase) {
  const screen = targets.create(
    root,
    { width: SIZE, height: SIZE },
    { color: { format: 'rgba8unorm' }, depth: { format: 'depth24plus' } },
  );
  return {
    screen,
    into: passes
      .create(root, screen, { color: passes.color, depth: passes.depth })
      .pipe(view.inject()),
  };
}

/** Draws the casters' shadow and then the renderers into the cleared targets, and reads them. */
function drawer(
  root: TgpuRoot,
  { screen, into }: ReturnType<typeof described>,
  sets: { flush(): void }[],
  light: lighting.Lighting & { shadow: lighting.Shadow },
  renderers: Drawn[],
  casters: Drawn[],
) {
  const { device } = root;
  const color = root.unwrap(screen.textures.color);

  const loop = frames.create(root, (frame) => {
    for (const set of sets) set.flush();
    light.shadow.record(frame.encoder, casters);
    const pass = into.begin(frame);
    for (const renderer of renderers) renderer.draw(pass);
    pass.end();
  });

  return async () => {
    loop.run(0);
    const buffer = device.createBuffer({
      size: SIZE * SIZE * 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const encoder = device.createCommandEncoder();
    encoder.copyTextureToBuffer({ texture: color }, { buffer, bytesPerRow: SIZE * 4 }, [
      SIZE,
      SIZE,
    ]);
    device.queue.submit([encoder.finish()]);
    await buffer.mapAsync(GPUMapMode.READ);
    const pixels = new Uint8Array(buffer.getMappedRange().slice(0));
    buffer.destroy();
    return pixels;
  };
}

/** How many pixels of two images are far apart, and how far apart they are on average. */
function apart(seen: Uint8Array, shown: Uint8Array) {
  let far = 0;
  let total = 0;
  for (let i = 0; i < SIZE * SIZE; i++) {
    let gap = 0;
    for (let c = 0; c < 3; c++) gap = Math.max(gap, Math.abs(seen[i * 4 + c] - shown[i * 4 + c]));
    total += gap;
    if (gap > 40) far++;
  }
  return { far: far / (SIZE * SIZE), mean: total / (SIZE * SIZE) };
}

/** A box with a ball on it, on the ground under a sun, as meshes and as one impostor. */
async function scene(features: GPUFeatureName[], sampling: impostors.Sampling) {
  const root = await gpuRoot({ features });
  const pool = meshPool.create(root, { capacity: 64, vertices: 4096, indices: 16384 });
  const box = pool.add(meshes.box({ width: 1, height: 2, depth: 1 }));
  const ball = pool.add(meshes.sphere({ radius: 0.6 }));
  const ground = pool.add(meshes.plane({ width: 20, depth: 20 }));
  const card = pool.add(impostors.CARD).meshes[0];
  const light = lighting.create(root, {
    shading: { reflect: shading.lambert },
    settings: {
      sunDirection: [0.5, 1, 0.3],
      sunIntensity: 1,
      skyTop: [0.2, 0.2, 0.2],
      skyBottom: [0.1, 0.1, 0.1],
    },
    shadow: { size: 512, extent: 10, filter: 1 },
  });
  // Between the atlas's views: they lie at whole steps of the hemisphere's grid.
  const view = cameras.orbit(root, {
    size: { width: SIZE, height: SIZE },
    target: [0, 1.2, 0],
    distance: 7,
    yaw: 0.37,
    pitch: 0.31,
  });

  const parts = [
    { mesh: box, position: [0, 1, 0], color: [0.8, 0.3, 0.2] },
    { mesh: ball, position: [0, 2.4, 0], color: [0.2, 0.6, 0.9] },
  ] as const;
  const atlas = impostors.create(root, pool, parts, { card, views: 12, size: 128 });
  const capturing = root.createCommandEncoder();
  atlas.record(capturing);
  capturing.submit();

  const floor = instances.create(root, pool, { capacity: 1 });
  floor.add(ground, materials.matte(d.vec3f(0.7)));
  const solid = instances.create(root, pool, { capacity: 2 });
  for (const { mesh, position, color } of parts) {
    solid.add(mesh, { position: [...position], color: [...color] });
  }
  const cards = instances.create(root, pool, { kind: impostors.Impostor, capacity: 1 });
  cards.add(card, materials.matte(d.vec3f(1)));

  const drawn = described(root, view);
  const sets = [floor, solid, cards];
  const renderers = [
    floor.renderer().pipe(light.inject()),
    solid.renderer().pipe(light.inject()),
    cards.renderer().pipe(light.inject()).pipe(atlas.inject(sampling)),
  ];
  const [grounds, asMeshes, asCards] = renderers.map((renderer) => drawn.into.add(renderer));
  const [groundsCast, meshesCast, cardsCast] = renderers.map((renderer) =>
    light.shadow.add(renderer),
  );

  return {
    root,
    meshesSeen: drawer(
      root,
      drawn,
      sets,
      light,
      [grounds!, asMeshes!],
      [groundsCast!, meshesCast!],
    ),
    cardsSeen: drawer(root, drawn, sets, light, [grounds!, asCards!], [groundsCast!, cardsCast!]),
    meshesUnderCards: drawer(
      root,
      drawn,
      sets,
      light,
      [grounds!, asMeshes!],
      [groundsCast!, cardsCast!],
    ),
  };
}

const MODES = [
  ['with indirect-first-instance', ['indirect-first-instance'] as GPUFeatureName[]],
  ['without it', [] as GPUFeatureName[]],
] as const;
const SAMPLINGS = ['nearest', 'blended', 'parallax'] as const;
/** The mean gap from the meshes each sampling stays under: blending views without parallax blurs them. */
const MEAN = { nearest: 2.2, blended: 2.8, parallax: 2 };

describe.each(
  MODES.flatMap(([mode, features]) =>
    SAMPLINGS.map((sampling) => [mode, sampling, features] as const),
  ),
)('impostors (%s, %s)', (_mode, sampling, features) => {
  it('show a subject as its meshes do from a view between the atlas’s, standing in its shadow', async () => {
    const { root, meshesSeen, cardsSeen } = await scene([...features], sampling);
    // Only the outline's pixels may differ much.
    const { far, mean } = apart(await meshesSeen(), await cardsSeen());
    expect(far).toBeLessThan(0.015);
    expect(mean).toBeLessThan(MEAN[sampling]);
    root.destroy();
  });

  it('cast the shadow its meshes do, on themselves and the ground, and none on lit faces', async () => {
    const { root, meshesSeen, meshesUnderCards } = await scene([...features], sampling);
    const { far, mean } = apart(await meshesSeen(), await meshesUnderCards());
    expect(far).toBeLessThan(0.005);
    expect(mean).toBeLessThan(0.5);
    root.destroy();
  });
});

/** A square of side 2 facing +z, of `material`, with its uvs' (0, 0) at the top left. */
const square = (material: Partial<meshPool.MaterialData>): meshPool.MeshData => ({
  positions: new Float32Array([-1, 1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0]),
  normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1]),
  uvs: new Float32Array([0, 0, 0, 1, 1, 1, 1, 0]),
  indices: new Uint32Array([0, 1, 2, 0, 2, 3]),
  material: {
    color: [1, 1, 1, 1],
    roughness: 1,
    metallic: 0,
    emission: [0, 0, 0],
    textures: {},
    ...material,
  },
});

it('captures every pose phase beyond the bound mesh’s box, preserving each and borrowing the poses', async () => {
  const root = await gpuRoot({ features: ['indirect-first-instance'] });
  const pool = meshPool.create(root, { vertices: 16, indices: 16, capacity: 2 });
  const card = pool.add(impostors.CARD).meshes[0];
  const source = square({ color: [0.2, 0.7, 0.3, 1] });
  const model = pool.add({
    meshes: [
      {
        ...source,
        joints: new Uint16Array(16),
        weights: new Float32Array([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]),
      },
    ],
    images: [],
  });
  const move: poses.ClipData = {
    name: 'move',
    duration: 1,
    channels: [
      {
        joint: 'hip',
        path: 'position',
        interpolation: 'linear',
        times: [0, 1],
        values: [0, 0, 0, 3, 0, 0],
      },
    ],
  };
  const figures = poses.create(root, leg, { move }, { capacity: 256 });
  const posed = [0.5, 1].map((from) => figures.add('move', 0, { from, speed: 0, loop: false }));
  const pose = posed[0];
  const atlas = impostors.create(
    root,
    pool,
    model.meshes.map((mesh) => ({ mesh, pose: posed })),
    { card, poses: figures, views: 8 },
  );
  // The figures move in the encoder the capture records into, before it.
  const encoder = root.createCommandEncoder();
  const compute = encoder.beginComputePass();
  figures.animate(compute, 0);
  compute.end();
  atlas.record(encoder);
  encoder.submit();
  const Kind = instances.define('CapturedPose', {
    host: { pose: d.u32 },
    pose: (row) => {
      'use gpu';
      return row.pose;
    },
  });
  const solid = instances.create(root, pool, { kind: Kind, capacity: 1024 });
  const body = solid.add(model, { pose });
  const cards = instances.create(root, pool, { kind: impostors.Impostor, capacity: 1024 });
  const shown = cards.add(card);
  const light = lighting.create(root, {
    shading: { reflect: shading.lambert },
    settings: { sunIntensity: 0, skyTop: [0.4, 0.4, 0.4], skyBottom: [0.4, 0.4, 0.4] },
    shadow: {},
  });
  const view = orthographic(root);
  view.uniform.patch({
    viewProjection: d.mat4x4f(0.5, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, -0.1, 0, -1.5, 0, 0.5, 1),
    position: [3, 0, 10],
  });
  const into = described(root, view);
  const asMesh = into.into.add(solid.renderer().pipe(light.inject()).pipe(figures.inject()));
  const asCard = into.into.add(cards.renderer().pipe(light.inject()).pipe(atlas.inject('nearest')));
  expect(atlas.phases).toBe(posed.length);
  for (const [phase, current] of posed.entries()) {
    solid.write(body, { pose: current });
    cards.write(shown, { phase });
    const meshesSeen = await drawer(root, into, [solid], light, [asMesh], [])();
    const cardsSeen = await drawer(root, into, [cards], light, [asCard], [])();
    expect(meshesSeen.some((value, i) => i % 4 === 1 && value > 40)).toBe(true);
    expect(apart(meshesSeen, cardsSeen).mean).toBeLessThan(2);
  }
  atlas.destroy();
  expect(figures.has(pose)).toBe(true);
  expect(figures.count).toBe(posed.length);
  root.destroy();
});

/** Looks down -z with no perspective: a world unit is 16 pixels, the origin centred. */
function orthographic(root: TgpuRoot): CameraBase {
  const uniform = root.createUniform(Camera, {
    viewProjection: d.mat4x4f(0.5, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, -0.1, 0, 0, 0, 0.5, 1),
    position: d.vec3f(0, 0, 10),
    right: d.vec3f(1, 0, 0),
    up: d.vec3f(0, 1, 0),
  });
  return { uniform, inject: () => (cfg) => cfg.with(cameras.current, uniform) };
}

it("captures a two-sided mesh's back, and leaves out what a mask cuts away", async () => {
  const root = await gpuRoot({ features: ['indirect-first-instance'] });
  const pool = meshPool.create(root, { vertices: 64, indices: 64, capacity: 4 });
  const white = [255, 255, 255, 255];
  const clear = [255, 255, 255, 0];
  const nearest = { magFilter: 'nearest', minFilter: 'nearest' } as const;
  const model = pool.add({
    meshes: [
      square({ doubleSided: true }),
      square({ alpha: 'mask', textures: { color: { image: 0, sampler: nearest } } }),
    ],
    images: [{ width: 2, height: 1, pixels: new Uint8Array([...white, ...clear]) }],
  });
  await model.ready;
  const [twoSided, halved] = model.meshes;
  const card = pool.add(impostors.CARD).meshes[0];
  const views = { card, views: 3, size: 32 };
  const back = impostors.create(root, pool, twoSided!, views);
  const half = impostors.create(root, pool, halved!, views);
  const capturing = root.createCommandEncoder();
  back.record(capturing);
  half.record(capturing);
  capturing.submit();

  // Lit by nothing, a card shows its emission wherever its atlas covers.
  const dark = { sunIntensity: 0, skyTop: [0, 0, 0], skyBottom: [0, 0, 0] } as const;
  const light = lighting.create(root, { settings: dark });
  const glowing = { emission: [0, 1, 0], scale: 0.5 } as const;
  const behind = instances.create(root, pool, { kind: impostors.Impostor, capacity: 1 });
  // Turned away from the eye, so it shows the view from behind the square.
  behind.add(card, { ...glowing, position: [-1, 0, 0], rotation: [0, 1, 0, 0] });
  const front = instances.create(root, pool, { kind: impostors.Impostor, capacity: 1 });
  front.add(card, { ...glowing, position: [1, 0, 0] });

  const screen = targets.create(
    root,
    { width: SIZE, height: SIZE },
    { color: { format: 'rgba8unorm' }, depth: { format: 'depth24plus' } },
  );
  const into = passes
    .create(root, screen, { color: passes.color, depth: passes.depth })
    .pipe(orthographic(root).inject());
  const drawn = [
    into.add(behind.renderer().pipe(light.inject()).pipe(back.inject('nearest'))),
    into.add(front.renderer().pipe(light.inject()).pipe(half.inject('nearest'))),
  ];
  const loop = frames.create(root, (frame) => {
    behind.flush();
    front.flush();
    const pass = into.begin(frame);
    for (const each of drawn) each.draw(pass);
    pass.end();
  });
  loop.run(0);

  const buffer = root.device.createBuffer({
    size: SIZE * SIZE * 4,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const encoder = root.device.createCommandEncoder();
  encoder.copyTextureToBuffer(
    { texture: root.unwrap(screen.textures.color) },
    { buffer, bytesPerRow: SIZE * 4 },
    [SIZE, SIZE],
  );
  root.device.queue.submit([encoder.finish()]);
  await buffer.mapAsync(GPUMapMode.READ);
  const pixels = new Uint8Array(buffer.getMappedRange().slice(0));
  const at = (x: number, y: number) => {
    const i = (Math.floor(32 - 16 * y) * SIZE + Math.floor(16 * x + 32)) * 4;
    return [...pixels.subarray(i, i + 4)];
  };

  expect(at(-1, 0)).toEqual([0, 255, 0, 255]);
  expect(at(0.75, 0)).toEqual([0, 255, 0, 255]);
  expect(at(1.25, 0)).toEqual([0, 0, 0, 0]);

  buffer.destroy();
  root.destroy();
});

it('draws a device-transformed card where a kernel places it, with its own host fields', async () => {
  const root = await gpuRoot({ features: ['indirect-first-instance'] });
  const pool = meshPool.create(root, { vertices: 32, indices: 64, capacity: 2 });
  const box = pool.add(meshes.box());
  const card = pool.add(impostors.CARD).meshes[0];
  const atlas = impostors.create(root, pool, box, { card, views: 3, size: 64 });
  const capturing = root.createCommandEncoder();
  atlas.record(capturing);
  capturing.submit();
  const Card = impostors.define('MovingCard', { transform: 'device', host: { goal: d.vec3f } });
  const cards = instances.create(root, pool, { kind: Card, capacity: 1 });
  cards.add(card, { goal: [1, 0, 0], emission: [0, 1, 0] });
  cards.flush();
  const move = tgpu.computeFn({ workgroupSize: [1], in: { gid: d.builtin.globalInvocationId } })(({
    gid,
  }) => {
    'use gpu';
    if (gid.x >= Card.span() || !Card.alive(gid.x)) return;
    Card.device.$[gid.x].position = Card.goal(gid.x);
  });
  root.pipe(cards.inject()).createComputePipeline({ compute: move }).dispatchWorkgroups(1);
  const light = lighting.create(root, {
    settings: { sunIntensity: 0, skyTop: [0, 0, 0], skyBottom: [0, 0, 0] },
    shadow: {},
  });
  const view = orthographic(root);
  const into = described(root, view);
  const shown = into.into.add(cards.renderer().pipe(light.inject()).pipe(atlas.inject()));
  const pixels = await drawer(root, into, [cards], light, [shown], [])();
  const green = (x: number) => pixels[(32 * SIZE + x) * 4 + 1];
  expect(green(48)).toBe(255);
  expect(green(32)).toBe(0);
  root.destroy();
});

it("places and hides cards through their kind's hooks", async () => {
  const root = await gpuRoot({ features: ['indirect-first-instance'] });
  const pool = meshPool.create(root, { vertices: 32, indices: 64, capacity: 2 });
  const box = pool.add(meshes.box());
  const card = pool.add(impostors.CARD).meshes[0];
  const atlas = impostors.create(root, pool, box, { card, views: 3, size: 64 });
  const capturing = root.createCommandEncoder();
  atlas.record(capturing);
  capturing.submit();
  const Card = impostors.define('HookedCard', {
    host: { goal: d.vec3f, shown: d.u32 },
    place: (row) => {
      'use gpu';
      return instances.Placement({
        rotation: d.vec4f(0, 0, 0, 1),
        position: row.goal,
        scale: d.vec3f(1),
      });
    },
    visible: (row) => {
      'use gpu';
      return row.shown === 1;
    },
  });
  const cards = instances.create(root, pool, { kind: Card, capacity: 2 });
  const glowing = { emission: [0, 1, 0], position: [0, 0, 0] } as const;
  cards.add(card, { ...glowing, goal: [1, 0, 0], shown: 1 });
  cards.add(card, { ...glowing, goal: [-1, 0, 0], shown: 0 });
  const light = lighting.create(root, {
    settings: { sunIntensity: 0, skyTop: [0, 0, 0], skyBottom: [0, 0, 0] },
    shadow: {},
  });
  const into = described(root, orthographic(root));
  const shown = into.into.add(cards.renderer().pipe(light.inject()).pipe(atlas.inject()));
  const pixels = await drawer(root, into, [cards], light, [shown], [])();
  const green = (x: number) => pixels[(32 * SIZE + x) * 4 + 1];
  expect(green(48)).toBe(255);
  expect(green(32)).toBe(0);
  expect(green(16)).toBe(0);
  root.destroy();
});
