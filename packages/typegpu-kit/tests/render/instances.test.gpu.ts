import {
  bindings,
  Camera,
  cameras,
  frames,
  passes,
  targets,
  type CameraBase,
  type Frame,
  type SharedBindings,
} from '@typegpu/kit';
import { gpuRoot } from '@typegpu/kit/testing/dawn';
import { meshes } from '@typegpu/kit/geometry';
import { d, std, tgpu, type TgpuRenderPass, type TgpuRoot } from 'typegpu';
import { describe, expect, it } from 'vitest';
import { ao, instances, lighting, materials, meshPool, shading } from '../../src/index.ts';

// 64 RGBA8 pixels make a 256-byte row, the alignment texture copies need.
const SIZE = 64;
const FIRST_INSTANCE: GPUFeatureName[] = ['indirect-first-instance'];

const dark = { sunIntensity: 0, skyTop: [0, 0, 0], skyBottom: [0, 0, 0] } as const;

/** Looks down -z from z = 10 with no perspective; a world unit is 16 pixels, the origin centred. */
function orthographic(root: TgpuRoot): CameraBase {
  const uniform = root.createUniform(Camera, {
    viewProjection: d.mat4x4f(0.5, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, -0.1, 0, 0, 0, 0.5, 1),
    position: d.vec3f(0, 0, 10),
    right: d.vec3f(1, 0, 0),
    up: d.vec3f(0, 1, 0),
  });
  return { uniform, inject: () => (cfg) => cfg.with(cameras.current, uniform) };
}

/** A colour target and depth per device, and passes that draw into them. */
const described = new WeakMap<TgpuRoot, ReturnType<typeof screenFor>>();
const screenFor = (root: TgpuRoot) => {
  const screen = targets.create(
    root,
    { width: SIZE, height: SIZE },
    { color: { format: 'rgba8unorm' }, depth: { format: 'depth24plus' } },
  );
  return {
    screen,
    into: passes.create(root, screen, { color: passes.color, depth: passes.depth }),
  };
};
const screenOf = (root: TgpuRoot) => {
  const made = described.get(root) ?? screenFor(root);
  described.set(root, made);
  return made;
};
/** Each renderer added to passes into the root's targets, seen through `view`. */
const addedTo = (root: TgpuRoot, view: CameraBase, ...renderers: instances.Renderer[]) => {
  const into = passes
    .create(root, screenOf(root).screen, { color: passes.color, depth: passes.depth })
    .pipe(view.inject());
  return renderers.map((renderer) => into.add(renderer));
};

/**
 * Updates the sets, runs `before` and draws the renderers into the cleared targets, then reads
 * their pixels.
 */
function drawer(
  root: TgpuRoot,
  sets: { flush(): void }[],
  renderers: { draw(pass: TgpuRenderPass): void }[],
  before?: (frame: Frame) => void,
  shared?: SharedBindings,
) {
  const { device } = root;
  const { screen, into } = screenOf(root);
  const color = root.unwrap(screen.textures.color);

  const loop = frames.create(root, (frame) => {
    for (const set of sets) set.flush();
    before?.(frame);

    const pass = into.begin(frame);
    if (shared) pass.setBindGroup(shared.layout, shared.group());
    for (const renderer of renderers) renderer.draw(pass);
    pass.end();
  });

  let time = 0;
  return async () => {
    loop.run(time++);

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

    /** The pixel at world `x, y`. */
    return (x: number, y: number) => {
      const at = (Math.floor(32 - 16 * y) * SIZE + Math.floor(16 * x + 32)) * 4;
      return [...pixels.subarray(at, at + 4)];
    };
  };
}

/** A quarter turn about z, and an eighth about y. */
const QUARTER_Z = [0, 0, Math.SQRT1_2, Math.SQRT1_2] as const;
const EIGHTH_Y = [0, Math.sin(Math.PI / 8), 0, Math.cos(Math.PI / 8)] as const;

it.each([false, true])(
  'culls with a moved uniform range and keeps an individual row beyond it (firstInstance %s)',
  async (firstInstance) => {
    const root = await gpuRoot({ features: firstInstance ? FIRST_INSTANCE : [] });
    const pool = meshPool.create(root, { vertices: 64, indices: 128, capacity: 2 });
    const box = pool.add(meshes.box());
    const ball = pool.add(meshes.box({ width: 0.6, height: 0.6, depth: 0.6 }));
    const things = instances.create(root, pool, { capacity: 2 });
    things.add(box, { position: [-1, 0, 0], emission: [1, 0, 0] });
    things.add(ball, { position: [1, 0, 0], emission: [0, 1, 0] });
    const keep = root.createUniform(d.u32, 0xffffffff);
    const view = orthographic(root);
    const distance = root.createUniform(d.f32, 9);
    const renderer = things
      .renderer({
        cull: true,
        to: (row) => {
          'use gpu';
          return row === keep.$ ? d.f32(100) : distance.$;
        },
        fade: 2,
        eye: view,
      })
      .pipe(lighting.create(root, { settings: dark }).inject());
    const shown = drawer(root, [things], addedTo(root, view, renderer), (frame) => {
      const pass = frame.encoder.beginComputePass();
      renderer.cull(pass);
      pass.end();
    });
    let pixels = await shown();
    expect(pixels(-1, 0)).toEqual([0, 0, 0, 0]);
    expect(pixels(1, 0)).toEqual([0, 0, 0, 0]);
    distance.write(12);
    pixels = await shown();
    expect(pixels(-1, 0)).toEqual([255, 0, 0, 255]);
    expect(pixels(1, 0)).toEqual([0, 255, 0, 255]);
    distance.write(9);
    keep.write(0);
    pixels = await shown();
    expect(pixels(-1, 0)).toEqual([255, 0, 0, 255]);
    expect(pixels(1, 0)).toEqual([0, 0, 0, 0]);
    root.destroy();
  },
);

it('lists a caster inside the fade band exactly once, handing over at the callback distance', async () => {
  const root = await gpuRoot({ features: FIRST_INSTANCE });
  const pool = meshPool.create(root, { vertices: 32, indices: 36, capacity: 1 });
  const box = pool.add(meshes.box());
  const counted = root.createMutable(d.arrayOf(d.atomic(d.u32), 2));
  const Kind = instances.define('CountedCaster', {
    host: { counter: d.u32 },
    bounds: (row, base) => {
      'use gpu';
      std.atomicAdd(counted.$[row.counter], d.u32(1));
      return instances.Sphere(base);
    },
  });
  const near = instances.create(root, pool, { kind: Kind, capacity: 1 });
  const far = instances.create(root, pool, { kind: Kind, capacity: 1 });
  near.add(box, { counter: 0 });
  far.add(box, { counter: 1 });
  const view = orthographic(root);
  const distance = root.createUniform(d.f32, 10.5);
  const handoff = () => {
    'use gpu';
    return distance.$;
  };
  const light = lighting.create(root, { shadow: { size: 64, extent: 8 } });
  const renderers = [
    near.renderer({ cull: true, to: handoff, fade: 2, eye: view }).pipe(light.inject()),
    far.renderer({ cull: true, from: handoff, fade: 2, eye: view }).pipe(light.inject()),
  ];
  renderers.forEach((renderer) => light.shadow.add(renderer));
  const cull = () => {
    near.flush();
    far.flush();
    const encoder = root.createCommandEncoder();
    const pass = encoder.beginComputePass();
    for (const renderer of renderers) renderer.cull(pass);
    pass.end();
    encoder.submit();
  };
  cull();
  expect(await counted.buffer.read()).toEqual([1, 0]);
  counted.buffer.clear();
  distance.write(9.5);
  cull();
  expect(await counted.buffer.read()).toEqual([0, 1]);
  root.destroy();
});

it('decodes the vertices and indices it baked as the shapes give them', async () => {
  const root = await gpuRoot();
  const pool = meshPool.create(root, { capacity: 64, vertices: 1024, indices: 4096 });
  const shapes = [meshes.sphere({ segments: 8, rings: 6 }), meshes.unindexed(meshes.box())];
  const added = shapes.map((shape) => pool.add(shape));

  const count = added.reduce((sum, mesh) => sum + mesh.vertexCount, 0);
  const output = root.createBuffer(d.arrayOf(meshes.Vertex, count)).$usage('storage');
  const decoded = output.as('mutable');

  const kernel = tgpu.computeFn({
    workgroupSize: [64],
    in: { gid: d.builtin.globalInvocationId },
  })(({ gid }) => {
    'use gpu';
    if (gid.x < count) decoded.$[gid.x] = meshPool.vertexAt(gid.x);
  });
  root
    .pipe(pool.inject())
    .createComputePipeline({ compute: kernel })
    .dispatchWorkgroups(Math.ceil(count / 64));

  const [vertices, indices] = await Promise.all([output.read(), pool.indices.read()]);

  shapes.forEach((shape, s) => {
    const mesh = added[s];

    for (let i = 0; i < mesh.vertexCount; i++) {
      const want = shape.vertexAt(i);
      const got = vertices[mesh.baseVertex + i];

      for (let c = 0; c < 3; c++) {
        expect(got.position[c]).toBeCloseTo(want.position[c], 5);
        expect(got.normal[c]).toBeCloseTo(want.normal[c], 3);
      }
      expect(got.uv.x).toBeCloseTo(want.uv.x, 2);
      expect(got.uv.y).toBeCloseTo(want.uv.y, 2);
    }

    for (let i = 0; i < mesh.indexCount; i++) {
      const want = meshes.isIndexed(shape) ? shape.indexAt(i) : i;
      expect(indices[mesh.firstIndex + i]).toBe(want);
    }
  });

  root.destroy();
});

/** A square from -1 to 1 facing +z, uv (0, 0) at its top left, made of `material`. */
function square(material: Partial<meshPool.MaterialData>): meshPool.MeshData {
  return {
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
  };
}

/** One RGBA pixel per row, top to bottom, left to right. */
const image = (width: number, ...pixels: number[][]) => ({
  width,
  height: pixels.length / width,
  pixels: new Uint8Array(pixels.flat()),
});

const NEAREST = { magFilter: 'nearest', minFilter: 'nearest' } as const;

it("decodes a model's vertices as it was given them", async () => {
  const root = await gpuRoot();
  const pool = meshPool.create(root, { capacity: 64, vertices: 64, indices: 64 });
  pool.add(meshes.box());

  const given = square({});
  given.normals.set([0.6, 0, -0.8, 0, -1, 0], 0);
  given.uvs.set([0.25, 0.75], 4);
  const [mesh] = pool.add({ meshes: [given], images: [] }).meshes;

  const output = root.createBuffer(d.arrayOf(meshes.Vertex, 4)).$usage('storage');
  const decoded = output.as('mutable');
  const base = mesh!.baseVertex;
  const kernel = tgpu.computeFn({
    workgroupSize: [4],
    in: { gid: d.builtin.globalInvocationId },
  })(({ gid }) => {
    'use gpu';
    decoded.$[gid.x] = meshPool.vertexAt(base + gid.x);
  });
  root.pipe(pool.inject()).createComputePipeline({ compute: kernel }).dispatchWorkgroups(1);

  const [vertices, indices] = await Promise.all([output.read(), pool.indices.read()]);
  vertices.forEach((vertex, i) => {
    for (let c = 0; c < 3; c++) {
      expect(vertex.position[c]).toBeCloseTo(given.positions[i * 3 + c], 6);
      expect(vertex.normal[c]).toBeCloseTo(given.normals[i * 3 + c], 3);
    }
    expect(vertex.uv.x).toBeCloseTo(given.uvs[i * 2], 3);
    expect(vertex.uv.y).toBeCloseTo(given.uvs[i * 2 + 1], 3);
  });
  expect(indices.slice(mesh!.firstIndex, mesh!.firstIndex + 6)).toEqual([0, 1, 2, 0, 2, 3]);

  root.destroy();
});

it("finds each mesh's box: a shape's as it bakes, over many workgroups, and a model's", async () => {
  const root = await gpuRoot();
  const pool = meshPool.create(root, { capacity: 64, vertices: 1024, indices: 4096 });
  pool.add(meshes.box({ width: 2, height: 1, depth: 3 }));
  // 441 vertices, from (1, -3, 0.5) to (5, -1, 0.5).
  pool.add(
    meshes.parametric(
      {
        at: (u, v) => {
          'use gpu';
          return d.vec3f(1 + u * 4, -3 + v * 2, 0.5);
        },
      },
      { cols: 20, rows: 20 },
    ),
  );
  pool.add({ meshes: [square({})], images: [] });

  const output = root.createBuffer(d.arrayOf(meshPool.Box, 3)).$usage('storage');
  const boxes = output.as('mutable');
  const kernel = tgpu.computeFn({
    workgroupSize: [3],
    in: { gid: d.builtin.globalInvocationId },
  })(({ gid }) => {
    'use gpu';
    boxes.$[gid.x] = meshPool.boundsOf(gid.x);
  });
  root.pipe(pool.inject()).createComputePipeline({ compute: kernel }).dispatchWorkgroups(1);

  const found = (await output.read()).map(({ center, half }) => [[...center], [...half]]);
  expect(found).toEqual([
    [
      [0, 0, 0],
      [1, 0.5, 1.5],
    ],
    [
      [3, -2, 0.5],
      [2, 1, 0],
    ],
    [
      [0, 0, 0],
      [1, 1, 0],
    ],
  ]);

  root.destroy();
});

it("multiplies an instance's factors by its mesh's textures, where the uv says", async () => {
  const root = await gpuRoot({ features: FIRST_INSTANCE });
  const pool = meshPool.create(root, { capacity: 64, vertices: 64, indices: 64 });
  const red = [255, 0, 0, 255];
  const black = [0, 0, 0, 255];
  const model = pool.add({
    meshes: [
      square({ emission: [1, 1, 1], textures: { emission: { image: 0, sampler: NEAREST } } }),
    ],
    images: [image(2, red, black, black, red)],
  });
  await model.ready;

  const things = instances.create(root, pool, { capacity: 1024 });
  const draw = drawer(
    root,
    [things],
    addedTo(
      root,
      orthographic(root),
      things.renderer().pipe(lighting.create(root, { settings: dark }).inject()),
    ),
  );
  const id = things.add(model, { position: [-1.5, 0, 0], scale: 0.5 });
  things.add(model, { position: [1.5, 0, 0], scale: 0.5, emission: [0, 0.5, 0] });

  let at = await draw();
  expect(at(-1.75, 0.25)).toEqual([255, 0, 0, 255]);
  expect(at(-1.25, 0.25)).toEqual([0, 0, 0, 255]);
  expect(at(-1.25, -0.25)).toEqual([255, 0, 0, 255]);
  // The second instance's emission replaces the factor, and the texture still multiplies it.
  expect(at(1.25, 0.25)).toEqual([0, 0, 0, 255]);
  expect(at(1.75, 0.25)).toEqual([0, 0, 0, 255]);

  things.write(id, { emission: [1, 1, 0] });
  at = await draw();
  expect(at(-1.75, 0.25)).toEqual([255, 0, 0, 255]);

  root.destroy();
});

it('moves where a texture reads by its transform: scale, then rotation, then offset', async () => {
  const root = await gpuRoot({ features: FIRST_INSTANCE });
  const pool = meshPool.create(root, { capacity: 64, vertices: 64, indices: 64 });
  const red = [255, 0, 0, 255];
  const black = [0, 0, 0, 255];
  const turned = (transform: meshPool.TextureTransformData) =>
    pool.add({
      meshes: [
        square({
          emission: [1, 1, 1],
          textures: { emission: { image: 0, sampler: NEAREST, transform } },
        }),
      ],
      images: [image(2, red, black, black, black)],
    });
  const quarter = turned({ rotation: Math.PI / 2 });
  const shifted = turned({ rotation: Math.PI / 2, offset: [0.5, 0] });
  await Promise.all([quarter.ready, shifted.ready]);

  const things = instances.create(root, pool, { capacity: 1024 });
  const draw = drawer(
    root,
    [things],
    addedTo(
      root,
      orthographic(root),
      things.renderer().pipe(lighting.create(root, { settings: dark }).inject()),
    ),
  );
  things.add(quarter, { position: [-1.5, 0, 0], scale: 0.5 });
  things.add(shifted, { position: [1.5, 0, 0], scale: 0.5 });

  const at = await draw();
  // A quarter turn reads (u, v) at (v, -u), so the image's red corner shows top right.
  expect(at(-1.25, 0.25)).toEqual(red);
  expect(at(-1.75, 0.25)).toEqual(black);
  // The offset moves the turned uvs, not the ones before turning.
  expect(at(1.75, -0.25)).toEqual(red);
  expect(at(1.25, 0.25)).toEqual(black);

  root.destroy();
});

it('reads a texture through the second uv set where it asks for one', async () => {
  const root = await gpuRoot({ features: FIRST_INSTANCE });
  const pool = meshPool.create(root, { capacity: 64, vertices: 64, indices: 64, uvs1: true });
  const red = [255, 0, 0, 255];
  const black = [0, 0, 0, 255];
  const texture = { image: 0, sampler: NEAREST, uvSet: 1 } as const;
  const lightmapped = square({ emission: [1, 1, 1], textures: { emission: texture } });
  // The first set reads one texel everywhere, so only the second can draw the pattern
  const model = pool.add({
    meshes: [{ ...lightmapped, uvs: new Float32Array(8), uvs1: lightmapped.uvs }],
    images: [image(2, red, black, black, red)],
  });
  await model.ready;

  const things = instances.create(root, pool, { capacity: 1024 });
  const draw = drawer(
    root,
    [things],
    addedTo(
      root,
      orthographic(root),
      things.renderer().pipe(lighting.create(root, { settings: dark }).inject()),
    ),
  );
  things.add(model, { position: [0, 0, 0], scale: 0.5 });

  const at = await draw();
  expect(at(-0.25, 0.25)).toEqual(red);
  expect(at(0.25, 0.25)).toEqual(black);
  expect(at(0.25, -0.25)).toEqual(red);

  root.destroy();
});

it("gives a kind's material the default, textures included, to change", async () => {
  const root = await gpuRoot({ features: FIRST_INSTANCE });
  const pool = meshPool.create(root, { capacity: 64, vertices: 64, indices: 64 });
  const red = [255, 0, 0, 255];
  const black = [0, 0, 0, 255];
  const model = pool.add({
    meshes: [
      square({ emission: [1, 1, 1], textures: { emission: { image: 0, sampler: NEAREST } } }),
    ],
    images: [image(2, red, black, black, red)],
  });
  await model.ready;

  // The textured emission, turned from red to blue.
  const Turned = instances.define('Turned', {
    material: (_pixel, base) => {
      'use gpu';
      const turned = shading.Material(base);
      turned.emission = base.emission.zyx;
      return turned;
    },
  });
  const things = instances.create(root, pool, { kind: Turned, capacity: 1024 });
  things.add(model, { scale: 0.5 });

  const at = await drawer(
    root,
    [things],
    addedTo(
      root,
      orthographic(root),
      things.renderer().pipe(lighting.create(root, { settings: dark }).inject()),
    ),
  )();
  expect(at(-0.25, 0.25)).toEqual([0, 0, 255, 255]);
  expect(at(0.25, 0.25)).toEqual([0, 0, 0, 255]);
  expect(at(0.25, -0.25)).toEqual([0, 0, 255, 255]);

  root.destroy();
});

it("gives a kind's material an extra of the pool, carried from each vertex", async () => {
  const root = await gpuRoot({ features: FIRST_INSTANCE });
  const pool = meshPool.create(root, {
    capacity: 4,
    vertices: 64,
    indices: 64,
    extra: { tint: d.vec4f },
  });
  pool.add(meshes.box());
  // Red along the top edge, blue along the bottom.
  const tint = new Float32Array([1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1]);
  const model = pool.add({ meshes: [{ ...square({}), extra: { tint } }], images: [] });

  const Tinted = instances.define('Tinted', {
    carry: (vertexIndex) => {
      'use gpu';
      return pool.extra.tint(vertexIndex);
    },
    material: (pixel, base) => {
      'use gpu';
      const tinted = shading.Material(base);
      tinted.emission = pixel.carried.rgb;
      return tinted;
    },
  });
  const things = instances.create(root, pool, { kind: Tinted, capacity: 4 });
  things.add(model, {});
  const [box] = pool.meshes;
  things.add(box!, { position: [0, 0, -2], scale: 4 });

  const at = await drawer(
    root,
    [things],
    addedTo(
      root,
      orthographic(root),
      things.renderer().pipe(lighting.create(root, { settings: dark }).inject()),
    ),
  )();
  const [r, g, b] = at(0, 0.9);
  expect([r! > 230, g, b! < 25]).toEqual([true, 0, true]);
  const [low, , high] = at(0, -0.9);
  expect([low! < 25, high! > 230]).toEqual([true, true]);
  const [middleRed, , middleBlue] = at(0, 0);
  expect(Math.abs(middleRed! - middleBlue!)).toBeLessThan(20);
  // The box has no tint, so its extras are zero: black around the square.
  expect(at(1.5, 1.5)).toEqual([0, 0, 0, 255]);

  root.destroy();
});

it('bends normals by a normal texture: +x along u, and +y up the image', async () => {
  const root = await gpuRoot({ features: FIRST_INSTANCE });
  const pool = meshPool.create(root, { capacity: 64, vertices: 64, indices: 64 });
  const flat = { reflect: shading.lambert };
  const lit = Math.round(Math.SQRT1_2 * 255);

  // Normals leaning 45° towards +x, and towards +y, in tangent space.
  for (const [texel, towards] of [
    [
      [218, 128, 218, 255],
      [1, 0, 0],
    ],
    [
      [128, 218, 218, 255],
      [0, 1, 0],
    ],
  ] as const) {
    const model = pool.add({
      meshes: [square({ textures: { normal: { image: 0 } } })],
      images: [image(1, [...texel])],
    });
    await model.ready;

    for (const [sign, expected] of [
      [1, lit],
      [-1, 0],
    ] as const) {
      const sunDirection = towards.map((axis) => axis * sign) as [number, number, number];
      const light = lighting.create(root, {
        settings: { skyTop: [0, 0, 0], skyBottom: [0, 0, 0], sunDirection },
        shading: flat,
      });
      const things = instances.create(root, pool, { capacity: 1024 });
      things.add(model);

      const drawn = addedTo(root, orthographic(root), things.renderer().pipe(light.inject()));
      const [r] = (await drawer(root, [things], drawn)())(0, 0);
      expect(Math.abs(r! - expected), `towards ${towards} times ${sign}`).toBeLessThanOrEqual(2);
    }
  }

  root.destroy();
});

it("scales a normal texture's lean by its material's normalScale", async () => {
  const root = await gpuRoot({ features: FIRST_INSTANCE });
  const pool = meshPool.create(root, { capacity: 64, vertices: 64, indices: 64 });
  const light = lighting.create(root, {
    settings: { skyTop: [0, 0, 0], skyBottom: [0, 0, 0], sunDirection: [1, 0, 0] },
    shading: { reflect: shading.lambert },
  });
  const lit = Math.round(Math.SQRT1_2 * 255);

  // A normal leaning 45° towards +x, where the sun is: unscaled it faces the sun halfway, and
  // scaled to nothing it is the flat normal, which the sun only grazes.
  for (const [normalScale, expected] of [
    [1, lit],
    [0, 0],
  ] as const) {
    const model = pool.add({
      meshes: [square({ normalScale, textures: { normal: { image: 0 } } })],
      images: [image(1, [218, 128, 218, 255])],
    });
    await model.ready;
    const things = instances.create(root, pool, { capacity: 1024 });
    things.add(model);

    const drawn = addedTo(root, orthographic(root), things.renderer().pipe(light.inject()));
    const [r] = (await drawer(root, [things], drawn)())(0, 0);
    expect(Math.abs(r! - expected), `normalScale ${normalScale}`).toBeLessThanOrEqual(2);
  }

  root.destroy();
});

it("eases an occlusion texture's darkening by its material's occlusionStrength", async () => {
  const root = await gpuRoot({ features: FIRST_INSTANCE });
  const pool = meshPool.create(root, { capacity: 64, vertices: 64, indices: 64 });
  // Ambient light of 1 times how open the surface is, with no sun.
  const light = lighting.create(root, {
    settings: dark,
    shading: {
      ambient: (material) => {
        'use gpu';
        return d.vec3f(material.occlusion);
      },
    },
  });

  // A texture that closes the surface fully, eased as glTF's strength eases it.
  for (const [occlusionStrength, expected] of [
    [1, 0],
    [0.25, 191],
    [0, 255],
  ] as const) {
    const model = pool.add({
      meshes: [square({ occlusionStrength, textures: { occlusion: { image: 0 } } })],
      images: [image(1, [0, 0, 0, 255])],
    });
    await model.ready;
    const things = instances.create(root, pool, { capacity: 1024 });
    things.add(model);

    const drawn = addedTo(root, orthographic(root), things.renderer().pipe(light.inject()));
    const [r] = (await drawer(root, [things], drawn)())(0, 0);
    expect(Math.abs(r! - expected), `occlusionStrength ${occlusionStrength}`).toBeLessThanOrEqual(
      2,
    );
  }

  root.destroy();
});

it.each([
  ['with indirect-first-instance', FIRST_INSTANCE],
  ['without it', []],
])(
  "cuts away what a material's mask leaves, and shows a two-sided mesh's back (%s)",
  async (_mode, features) => {
    const root = await gpuRoot({ features });
    const pool = meshPool.create(root, { capacity: 64, vertices: 64, indices: 64 });
    const red = [255, 0, 0, 255];
    const clear = [255, 0, 0, 0];
    const texel = { image: 0, sampler: NEAREST };
    const glowing = { emission: [1, 1, 1] as const, textures: { color: texel, emission: texel } };
    const model = pool.add({
      meshes: [
        square({ ...glowing, alpha: 'mask' }),
        square(glowing),
        square({ emission: [0, 1, 0], doubleSided: true }),
        square({ emission: [0, 1, 0] }),
      ],
      images: [image(2, red, clear, clear, red)],
    });
    await model.ready;
    const [cut, whole, twoSided, oneSided] = model.meshes;

    const things = instances.create(root, pool, { capacity: 1024 });
    const draw = drawer(
      root,
      [things],
      addedTo(
        root,
        orthographic(root),
        things.renderer().pipe(lighting.create(root, { settings: dark }).inject()),
      ),
    );
    things.add(cut!, { position: [-1.5, 0.75, 0], scale: 0.5 });
    things.add(whole!, { position: [1.5, 0.75, 0], scale: 0.5 });
    // Turned to face away from the eye.
    const away = [0, 1, 0, 0] as const;
    things.add(twoSided!, { position: [-1.5, -0.75, 0], scale: 0.5, rotation: away });
    things.add(oneSided!, { position: [1.5, -0.75, 0], scale: 0.5, rotation: away });

    const at = await draw();
    expect(at(-1.75, 1)).toEqual([255, 0, 0, 255]);
    expect(at(-1.25, 1)).toEqual([0, 0, 0, 0]);
    expect(at(-1.25, 0.5)).toEqual([255, 0, 0, 255]);
    expect(at(1.25, 1)).toEqual([255, 0, 0, 255]);
    expect(at(-1.5, -0.75)).toEqual([0, 255, 0, 255]);
    expect(at(1.5, -0.75)).toEqual([0, 0, 0, 0]);

    root.destroy();
  },
);

it('lights the back of a two-sided mesh from its own side, and casts only what a mask leaves', async () => {
  const root = await gpuRoot({ features: FIRST_INSTANCE });
  const pool = meshPool.create(root, { capacity: 64, vertices: 64, indices: 64 });
  const box = pool.add(meshes.box());
  const white = [255, 255, 255, 255];
  const clear = [255, 255, 255, 0];
  const model = pool.add({
    meshes: [
      square({ alpha: 'mask', textures: { color: { image: 0, sampler: NEAREST } } }),
      square({ doubleSided: true }),
    ],
    images: [image(2, white, clear)],
  });
  await model.ready;
  const [halved, twoSided] = model.meshes;
  const light = lighting.create(root, {
    shading: { reflect: shading.lambert },
    settings: { skyTop: [0, 0, 0], skyBottom: [0, 0, 0], sunDirection: [1, 0, 1] },
    shadow: { size: 256, extent: 8, filter: 1 },
  });

  // A wall facing the camera, and before it a square whose right half its mask cuts away. The sun,
  // from the right, casts the half that is left from x -1.4 to -0.9 on the wall, and none of the
  // rest, from -0.9 to -0.4.
  const things = instances.create(root, pool, { capacity: 1024 });
  things.add(box, { position: [0, 0, -1], scale: [4, 4, 0.2], ...materials.matte(d.vec3f(1)) });
  things.add(halved!, { scale: 0.5 });
  // Facing away from the eye and the sun, lit as if it faced them.
  things.add(twoSided!, { position: [1.5, 1.5, 1], scale: 0.25, rotation: [0, 1, 0, 0] });

  const renderer = things.renderer().pipe(light.inject());
  const cast = light.shadow.add(renderer);
  const draw = drawer(root, [things], addedTo(root, orthographic(root), renderer), (frame) =>
    light.shadow.record(frame.encoder, [cast]),
  );

  const lit = Math.round(Math.SQRT1_2 * 255);
  const at = await draw();
  expect(at(-1.15, 0)[0]).toBe(0);
  expect(Math.abs(at(-0.7, 0)[0] - lit)).toBeLessThanOrEqual(1);
  expect(Math.abs(at(0.25, 0)[0] - lit)).toBeLessThanOrEqual(1);
  expect(Math.abs(at(1.5, 1.5)[0] - lit)).toBeLessThanOrEqual(1);

  root.destroy();
});

describe.each([
  ['with indirect-first-instance', FIRST_INSTANCE],
  ['without it', []],
])('drawing %s', (_name, features) => {
  it('puts each instance where its transform says, until it is removed', async () => {
    const root = await gpuRoot({ features });
    expect(root.device.features.has('indirect-first-instance')).toBe(features.length > 0);

    const pool = meshPool.create(root, { capacity: 64, vertices: 1024, indices: 4096 });
    const box = pool.add(meshes.box());
    const ball = pool.add(meshes.sphere({ segments: 16, rings: 8 }));
    const light = lighting.create(root, { settings: dark });

    const things = instances.create(root, pool, { capacity: 8 });
    const draw = drawer(
      root,
      [things],
      addedTo(root, orthographic(root), things.renderer().pipe(light.inject())),
    );

    things.add(box, { position: [-1.2, 0, 0], scale: 0.8, ...materials.glow(d.vec3f(1, 0, 0), 1) });
    const green = things.add(ball, {
      position: [1.2, 0, 0],
      scale: 0.8,
      ...materials.glow(d.vec3f(0, 1, 0), 1),
    });
    // A long thin box, stood upright.
    things.add(box, {
      rotation: QUARTER_Z,
      scale: [1.2, 0.2, 0.2],
      ...materials.glow(d.vec3f(0, 0, 1), 1),
    });

    let at = await draw();
    expect(at(-1.2, 0)).toEqual([255, 0, 0, 255]);
    expect(at(1.2, 0)).toEqual([0, 255, 0, 255]);
    expect(at(0, 0.45)).toEqual([0, 0, 255, 255]);
    expect(at(0.3, 0)).toEqual([0, 0, 0, 0]);

    things.remove(green);
    at = await draw();
    expect(at(1.2, 0)).toEqual([0, 0, 0, 0]);
    expect(at(-1.2, 0)).toEqual([255, 0, 0, 255]);

    root.destroy();
  });
});

it.each([
  ['with indirect-first-instance', FIRST_INSTANCE],
  ['without it', []],
])(
  'replays a render bundle recorded once, with the instances live in each frame (%s)',
  async (_mode, features) => {
    const root = await gpuRoot({ features });
    const pool = meshPool.create(root, { capacity: 64, vertices: 1024, indices: 4096 });
    const ball = pool.add(meshes.sphere({ segments: 16, rings: 8 }));
    const box = pool.add(meshes.box());
    const light = lighting.create(root, { settings: dark });
    const things = instances.create(root, pool, { capacity: 8 });
    const [drawn] = addedTo(root, orthographic(root), things.renderer().pipe(light.inject()));

    // A bundle records the meshes in use, so the ball and the box are added before it; the box's
    // rows start after the ball's.
    things.add(ball, { position: [0, 1.2, 0], scale: 0.6, ...materials.glow(d.vec3f(0, 0, 1), 1) });
    const red = things.add(box, {
      position: [-1.2, 0, 0],
      scale: 0.8,
      ...materials.glow(d.vec3f(1, 0, 0), 1),
    });
    const bundler = root['~unstable'].createRenderBundleEncoder({
      colorFormats: ['rgba8unorm'],
      depthStencilFormat: 'depth24plus',
    });
    drawn!.draw(bundler);
    const bundle = bundler.finish();
    const draw = drawer(root, [things], [{ draw: (pass) => pass.executeBundles([bundle]) }]);

    let at = await draw();
    expect(at(-1.2, 0)).toEqual([255, 0, 0, 255]);
    expect(at(0, 1.2)).toEqual([0, 0, 255, 255]);

    things.add(box, { position: [1.2, 0, 0], scale: 0.8, ...materials.glow(d.vec3f(0, 1, 0), 1) });
    things.remove(red);
    at = await draw();
    expect(at(1.2, 0)).toEqual([0, 255, 0, 255]);
    expect(at(-1.2, 0)).toEqual([0, 0, 0, 0]);
    expect(at(0, 1.2)).toEqual([0, 0, 255, 255]);

    root.destroy();
  },
);

it('draws into a render bundle for a description of one layer, and throws for several', async () => {
  const root = await gpuRoot();
  const pool = meshPool.create(root, { capacity: 64, vertices: 1024, indices: 4096 });
  const box = pool.add(meshes.box());
  const plane = pool.add(meshes.plane());
  const light = lighting.create(root, { settings: dark });
  const things = instances.create(root, pool, { capacity: 8 });
  const renderer = things.renderer().pipe(light.inject());
  things.add(box, {});
  things.add(plane, {});
  things.flush();

  const castersOf = (layers: number) =>
    passes
      .create(
        root,
        targets.create(
          root,
          { width: 4, height: 4 },
          { depth: { format: 'depth16unorm', layers } },
        ),
        { depth: passes.depth },
      )
      .pipe(cameras.layered(root, layers).inject());
  const bundleOf = () =>
    root['~unstable'].createRenderBundleEncoder({
      colorFormats: [],
      depthStencilFormat: 'depth16unorm',
    });

  const one = castersOf(1).add(renderer);
  const bundler = bundleOf();
  one.draw(bundler);
  bundler.finish();
  const two = castersOf(2).add(renderer);
  expect(() => two.draw(bundleOf())).toThrow("render bundle can't draw");

  root.destroy();
});

it('lights each face by its normal, turned with the instance', async () => {
  const root = await gpuRoot({ features: FIRST_INSTANCE });
  const pool = meshPool.create(root, { capacity: 64, vertices: 64, indices: 64 });
  const box = pool.add(meshes.box());
  const light = lighting.create(root, {
    shading: { reflect: shading.lambert },
    settings: { skyTop: [0, 0, 0], skyBottom: [0, 0, 0], sunDirection: [0, 0, 1] },
  });

  const things = instances.create(root, pool, { capacity: 1024 });
  things.add(box, { rotation: EIGHTH_Y, ...materials.matte(d.vec3f(1)) });
  const at = await drawer(
    root,
    [things],
    addedTo(root, orthographic(root), things.renderer().pipe(light.inject())),
  )();

  // Both faces in view lean 45° away from the sun.
  const lit = Math.round(Math.SQRT1_2 * 255);
  for (const x of [-0.3, 0.3]) {
    const [r, g, b] = at(x, 0);
    for (const channel of [r, g, b]) expect(Math.abs(channel! - lit)).toBeLessThanOrEqual(1);
  }

  root.destroy();
});

it.each([1, 4])('writes normals with no colour and no lighting, at %i samples', async (samples) => {
  const root = await gpuRoot({ features: FIRST_INSTANCE });
  const pool = meshPool.create(root, { capacity: 64, vertices: 64, indices: 64 });
  const things = instances.create(root, pool, { capacity: 4 });
  things.add(pool.add(meshes.box()), { rotation: EIGHTH_Y });

  const early = targets.create(
    root,
    { width: SIZE, height: SIZE },
    {
      drawn: { format: 'rg16float', samples },
      normal: { format: 'rg16float' },
      depth: { format: 'depth24plus', samples },
    },
  );
  const prepass = passes
    .create(
      root,
      early,
      { drawn: ao.normal, depth: passes.depth },
      samples > 1 ? { resolve: { drawn: 'normal' } } : {},
    )
    .pipe(orthographic(root).inject());
  const drawn = prepass.add(things.renderer());
  await prepass.initAsync();
  frames
    .create(root, (frame) => {
      things.flush();
      prepass.record(frame, drawn);
    })
    .run(0);

  const buffer = root.device.createBuffer({
    size: SIZE * SIZE * 4,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const encoder = root.device.createCommandEncoder();
  encoder.copyTextureToBuffer(
    { texture: root.unwrap(early.textures[samples > 1 ? 'normal' : 'drawn']) },
    { buffer, bytesPerRow: SIZE * 4 },
    [SIZE, SIZE],
  );
  root.device.queue.submit([encoder.finish()]);
  await buffer.mapAsync(GPUMapMode.READ);
  const halves = new Float16Array(buffer.getMappedRange().slice(0));
  buffer.destroy();
  const at = (x: number, y: number) => {
    const pixel = Math.floor(32 - 16 * y) * SIZE + Math.floor(16 * x + 32);
    return [halves[pixel * 2], halves[pixel * 2 + 1]];
  };

  // Each face in view leans 45° to its side, folded onto the octahedron at ±0.5
  for (const [x, folded] of [
    [-0.3, -0.5],
    [0.3, 0.5],
  ]) {
    const [u, v] = at(x!, 0);
    expect(u).toBeCloseTo(folded!, 2);
    expect(v).toBeCloseTo(0, 2);
  }
  expect(at(1.5, 0)).toEqual([4, 4]);
  root.destroy();
});

it('blends a blend renderer by opacity, premultiplied, over the opaque ones', async () => {
  const root = await gpuRoot({ features: FIRST_INSTANCE });
  const pool = meshPool.create(root, { capacity: 64, vertices: 64, indices: 64 });
  const box = pool.add(meshes.box());
  const light = lighting.create(root, { settings: dark });
  const view = orthographic(root);

  const solid = instances.create(root, pool, { capacity: 1024 });
  const ghosts = instances.create(root, pool, { capacity: 1024 });
  solid.add(box, {
    position: [0, 0, -1],
    scale: [1, 2, 1],
    ...materials.glow(d.vec3f(1, 0, 0), 1),
  });
  ghosts.add(box, { scale: [2, 1, 1], ...materials.glow(d.vec3f(0, 0, 1), 1), opacity: 0.5 });

  const at = await drawer(
    root,
    [solid, ghosts],
    addedTo(
      root,
      view,
      solid.renderer().pipe(light.inject()),
      ghosts.renderer({ blend: 'translucent' }).pipe(light.inject()),
    ),
  )();

  expect(at(0, 0.8)).toEqual([255, 0, 0, 255]);
  expect(at(0, 0)).toEqual([128, 0, 128, 255]);
  expect(at(0.8, 0)).toEqual([0, 0, 128, 128]);

  root.destroy();
});

it('tints what shows through a clear renderer with transmit, apart from its own light', async () => {
  const root = await gpuRoot({ features: FIRST_INSTANCE });
  const pool = meshPool.create(root, { capacity: 64, vertices: 64, indices: 64 });
  const box = pool.add(meshes.box());
  const light = lighting.create(root, { settings: dark });
  const view = orthographic(root);

  const solid = instances.create(root, pool, { capacity: 1 });
  const film = instances.create(root, pool, { capacity: 1 });
  solid.add(box, { position: [0, 0, -1], scale: [1, 2, 1], ...materials.glow(d.vec3f(1), 1) });
  film.add(box, { scale: [2, 1, 1], ...materials.glow(d.vec3f(0, 0, 1), 1), opacity: 0.5 });
  const tinted = film.renderer({ blend: 'clear', transmit: [0.25, 1, 0] }).pipe(light.inject());

  const at = await drawer(
    root,
    [solid, film],
    addedTo(root, view, solid.renderer().pipe(light.inject()), tinted),
  )();

  // White behind passes as the tint, not by half; the film's own light is still halved.
  expect(at(0, 0.8)).toEqual([255, 255, 255, 255]);
  expect(at(0, 0)).toEqual([64, 255, 128, 255]);
  expect(at(0.8, 0)).toEqual([0, 0, 128, 128]);

  root.destroy();
});

it('shadows what is behind a caster from the sun, inside the square it follows', async () => {
  const root = await gpuRoot({ features: FIRST_INSTANCE });
  const pool = meshPool.create(root, { capacity: 64, vertices: 64, indices: 64 });
  const box = pool.add(meshes.box());
  let on = true;
  const light = lighting.create(root, {
    shading: { reflect: shading.lambert },
    settings: { skyTop: [0, 0, 0], skyBottom: [0, 0, 0], sunDirection: [1, 0, 1] },
    shadow: { size: 256, extent: 8, filter: 1, enabled: () => on },
  });

  // A wall facing the camera, and a box in front of it that the sun, from the right, casts left.
  const things = instances.create(root, pool, { capacity: 1024 });
  things.add(box, { position: [0, 0, -1], scale: [4, 4, 0.2], ...materials.matte(d.vec3f(1)) });
  things.add(box, { position: [0, 0, 1], ...materials.matte(d.vec3f(1)) });

  const renderer = things.renderer().pipe(light.inject());
  const cast = light.shadow.add(renderer);
  const draw = drawer(root, [things], addedTo(root, orthographic(root), renderer), (frame) =>
    light.shadow.record(frame.encoder, [cast]),
  );

  // The box's shadow on the wall spans x from -2.9 to -0.9, and y from -0.5 to 0.5.
  const lit = Math.round(Math.SQRT1_2 * 255);
  let at = await draw();
  expect(at(-1.5, 0)[0]).toBe(0);
  expect(Math.abs(at(1.5, 0)[0] - lit)).toBeLessThanOrEqual(1);
  expect(Math.abs(at(-1.5, 1.2)[0] - lit)).toBeLessThanOrEqual(1);

  light.shadow.follow([20, 0, 0]);
  at = await draw();
  expect(Math.abs(at(-1.5, 0)[0] - lit)).toBeLessThanOrEqual(1);

  // Off, the box casts nothing even with the square moved back over it; on again, it does.
  on = false;
  light.shadow.follow([0, 0, 0]);
  at = await draw();
  expect(Math.abs(at(-1.5, 0)[0] - lit)).toBeLessThanOrEqual(1);

  on = true;
  at = await draw();
  expect(at(-1.5, 0)[0]).toBe(0);

  root.destroy();
});

it.each([
  ['with indirect-first-instance', FIRST_INSTANCE],
  ['without it', []],
])(
  'culls nothing it would draw, and casts the shadows of what only the sun sees (%s)',
  async (_mode, features) => {
    const root = await gpuRoot({ features });
    const pool = meshPool.create(root, { capacity: 64, vertices: 64, indices: 64 });
    const box = pool.add(meshes.box());
    const light = lighting.create(root, {
      shading: { reflect: shading.lambert },
      settings: { skyTop: [0, 0, 0], skyBottom: [0, 0, 0], sunDirection: [1, 0, 1] },
      shadow: { size: 256, extent: 8, filter: 1 },
    });

    // A wall facing the camera; a box before it, one across the view's right edge, and one past
    // that edge whose shadow falls on the wall from x 0.4 to 2.4. A row of boxes far off.
    const things = instances.create(root, pool, { capacity: 64 });
    const white = materials.matte(d.vec3f(1));
    things.add(box, { position: [0, 0, -1], scale: [4, 4, 0.2], ...white });
    for (const x of [0, 2.3, 3.4]) things.add(box, { position: [x, 0, 1], ...white });
    for (let i = 0; i < 40; i++) things.add(box, { position: [20 + i, 0, 1], ...white });

    const view = orthographic(root);
    const rendered = await Promise.all(
      [false, true].map(async (cull) => {
        const renderer = things.renderer({ cull }).pipe(light.inject());
        const cast = light.shadow.add(renderer);
        const draw = drawer(root, [things], addedTo(root, view, renderer), (frame) => {
          if (cull) {
            const pass = frame.encoder.beginComputePass();
            renderer.cull(pass);
            pass.end();
          }
          light.shadow.record(frame.encoder, [cast]);
        });
        const at = await draw();
        return Array.from({ length: SIZE * SIZE }, (_, i) =>
          at(((i % SIZE) + 0.5 - 32) / 16, (32 - Math.floor(i / SIZE) - 0.5) / 16),
        );
      }),
    );

    const [plain, culled] = rendered;
    expect(culled).toEqual(plain);
    // The wall lit above the boxes, in the shadow of the box past the edge alone between the edge
    // box's shadow and the edge box, and the edge box itself lit.
    const pixel = (x: number, y: number) =>
      culled![Math.floor(32 - 16 * y) * SIZE + Math.floor(16 * x + 32)];
    expect(pixel(1.55, 1)[0]).toBeGreaterThan(0);
    expect(pixel(1.55, 0)[0]).toBe(0);
    expect(pixel(1.95, 0)[0]).toBeGreaterThan(0);

    root.destroy();
  },
);

it.each([
  ['with indirect-first-instance', FIRST_INSTANCE],
  ['without it', []],
])(
  'hands its instances to another renderer by distance, pixel by pixel across the fade (%s)',
  async (_mode, features) => {
    const root = await gpuRoot({ features });
    const pool = meshPool.create(root, { capacity: 64, vertices: 64, indices: 64 });
    const box = pool.add(meshes.box());
    const light = lighting.create(root, {
      shading: { reflect: shading.lambert },
      settings: { skyTop: [0, 0, 0], skyBottom: [0, 0, 0], sunDirection: [1, 0, 1] },
      shadow: { size: 256, extent: 8, filter: 1 },
    });

    // A wall far off, and boxes that stand from the eye nearer than the fade, in each half of it,
    // and past it.
    const things = instances.create(root, pool, { capacity: 8 });
    const white = materials.matte(d.vec3f(1));
    things.add(box, { position: [0, 0, -4], scale: [4, 4, 0.2], ...white });
    const xs = [-1.5, -0.5, 0.5, 1.5];
    for (const [i, distance] of [8.5, 9.6, 10.4, 11.5].entries()) {
      const x = xs[i];
      const z = 10 - Math.sqrt(distance ** 2 - x ** 2);
      things.add(box, { position: [x, 0, z], scale: 0.8, ...white });
    }

    const view = orthographic(root);
    const whole = things.renderer({ cull: true }).pipe(light.inject());
    const near = things.renderer({ cull: true, to: 10, fade: 2, eye: view }).pipe(light.inject());
    const far = things.renderer({ cull: true, from: 10, fade: 2, eye: view }).pipe(light.inject());
    const all = [whole, near, far];
    const drawn = addedTo(root, view, ...all);
    const cast = all.map((renderer) => light.shadow.add(renderer));
    const each = (renderers: instances.Renderer[], built: typeof drawn) =>
      renderers.map((renderer) => built[all.indexOf(renderer)]);

    const seen = async (renderers: instances.Renderer[]) => {
      const draw = drawer(root, [things], each(renderers, drawn), (frame) => {
        const pass = frame.encoder.beginComputePass();
        for (const renderer of renderers) renderer.cull(pass);
        pass.end();
        light.shadow.record(frame.encoder, each(renderers, cast));
      });
      const at = await draw();
      return Array.from({ length: SIZE * SIZE }, (_, i) =>
        at(((i % SIZE) + 0.5 - 32) / 16, (32 - Math.floor(i / SIZE) - 0.5) / 16),
      );
    };

    expect(await seen([near, far])).toEqual(await seen([whole]));

    // Alone, the near renderer draws the first box whole, a share of the next two by how far
    // into the fade each stands, and nothing of the last or the wall.
    const alone = await seen([near]);
    const shares = xs.map((x) => {
      let covered = 0;
      let inside = 0;
      for (let i = 0; i < SIZE * SIZE; i++) {
        const px = ((i % SIZE) + 0.5 - 32) / 16;
        const py = (32 - Math.floor(i / SIZE) - 0.5) / 16;
        if (Math.abs(px - x) > 0.3 || Math.abs(py) > 0.3) continue;
        inside++;
        if (alone[i][3] > 0) covered++;
      }
      return covered / inside;
    });
    expect(shares[0]).toBe(1);
    expect(shares[1]).toBeCloseTo(0.7, 1);
    expect(shares[2]).toBeCloseTo(0.3, 1);
    expect(shares[3]).toBe(0);
    expect(alone[0][3]).toBe(0);

    root.destroy();
  },
);

it('keeps the shadow when an app that owns its map resizes it, with no new pipelines', async () => {
  const root = await gpuRoot({ features: FIRST_INSTANCE });
  const pool = meshPool.create(root, { capacity: 64, vertices: 64, indices: 64 });
  const box = pool.add(meshes.box());
  const light = lighting.create(root, {
    shading: { reflect: shading.lambert },
    settings: { skyTop: [0, 0, 0], skyBottom: [0, 0, 0], sunDirection: [1, 0, 1] },
    shadow: { size: 64, extent: 8, filter: 1 },
  });
  const shared = bindings.share(root, { shadow: light.shadow.part });

  const things = instances.create(root, pool, { capacity: 1024 });
  things.add(box, { position: [0, 0, -1], scale: [4, 4, 0.2], ...materials.matte(d.vec3f(1)) });
  things.add(box, { position: [0, 0, 1], ...materials.matte(d.vec3f(1)) });

  const renderer = things.renderer().pipe(light.inject());
  const cast = light.shadow.add(renderer);
  const scene = passes
    .create(root, screenOf(root).screen, { color: passes.color, depth: passes.depth }, { shared })
    .pipe(orthographic(root).inject());
  const draw = drawer(
    root,
    [things],
    [scene.add(renderer)],
    (frame) => light.shadow.record(frame.encoder, [cast]),
    shared,
  );

  const lit = Math.round(Math.SQRT1_2 * 255);
  const check = (at: Awaited<ReturnType<typeof draw>>) => {
    expect(at(-1.5, 0)[0]).toBe(0);
    expect(Math.abs(at(1.5, 0)[0] - lit)).toBeLessThanOrEqual(1);
  };

  check(await draw());
  const pipelines = root.device.createRenderPipeline;
  let compiled = 0;
  root.device.createRenderPipeline = (descriptor) => {
    compiled += 1;
    return pipelines.call(root.device, descriptor);
  };

  light.shadow.resize(512);
  check(await draw());

  expect(compiled).toBe(0);
  root.destroy();
});

/** Whether a channel is within a step of what was expected. */
const near = (value: number, expected: number) =>
  expect(Math.abs(value - expected)).toBeLessThanOrEqual(1);

it('covers a wider square once its extent is set, in its own format and bias, with no new pipelines', async () => {
  const root = await gpuRoot({ features: FIRST_INSTANCE });
  const pool = meshPool.create(root, { capacity: 64, vertices: 64, indices: 64 });
  const box = pool.add(meshes.box());
  const light = lighting.create(root, {
    shading: { reflect: shading.lambert },
    settings: { skyTop: [0, 0, 0], skyBottom: [0, 0, 0], sunDirection: [1, 0, 1] },
    shadow: {
      size: 256,
      extent: 0.5,
      filter: 1,
      format: 'depth32float',
      bias: { constant: 64 },
      lift: 2,
    },
  });

  const things = instances.create(root, pool, { capacity: 1024 });
  things.add(box, { position: [0, 0, -1], scale: [4, 4, 0.2], ...materials.matte(d.vec3f(1)) });
  things.add(box, { position: [0, 0, 1], ...materials.matte(d.vec3f(1)) });

  const renderer = things.renderer().pipe(light.inject());
  const cast = light.shadow.add(renderer);
  const draw = drawer(root, [things], addedTo(root, orthographic(root), renderer), (frame) =>
    light.shadow.record(frame.encoder, [cast]),
  );

  const lit = Math.round(Math.SQRT1_2 * 255);
  // Outside a square half a unit across, the sun reaches everything.
  let at = await draw();
  near(at(-1.5, 0)[0], lit);
  near(at(1.5, 0)[0], lit);

  const pipelines = root.device.createRenderPipeline;
  let compiled = 0;
  root.device.createRenderPipeline = (descriptor) => {
    compiled += 1;
    return pipelines.call(root.device, descriptor);
  };

  light.shadow.setExtent(8);
  expect([light.shadow.extent, light.shadow.reach]).toEqual([8, 16]);
  at = await draw();
  expect(at(-1.5, 0)[0]).toBe(0);
  near(at(1.5, 0)[0], lit);

  expect(compiled).toBe(0);
  expect(() => light.shadow.setExtent(0)).toThrow(RangeError);
  root.destroy();
});

it('draws a kind that its kernels move, through its own deform and material', async () => {
  const root = await gpuRoot({ features: FIRST_INSTANCE });
  const pool = meshPool.create(root, { capacity: 64, vertices: 64, indices: 64 });
  const box = pool.add(meshes.box());
  const light = lighting.create(root, { settings: dark });

  // Under the dark lighting, only the kind's material can make it red.
  const Mover = instances.define('Mover', {
    host: { tint: d.vec3f },
    device: { velocity: d.vec3f },
    transform: 'device',
    deform: (vertex) => {
      'use gpu';
      return vertex.position * d.vec3f(3, 1, 1);
    },
    material: (pixel) => {
      'use gpu';
      return materials.glow(pixel.instance.tint, 1);
    },
  });

  const movers = instances.create(root, pool, { kind: Mover, capacity: 4 });
  movers.add(box, { position: [-1.5, 0, 0], scale: 0.5, tint: [1, 0, 0], velocity: [1, 0, 0] });

  const kernel = tgpu.computeFn({
    workgroupSize: [64],
    in: { gid: d.builtin.globalInvocationId },
  })(({ gid }) => {
    'use gpu';
    const i = gid.x;
    if (!Mover.alive(i)) return;

    const mover = Mover.device.$[i];
    mover.position = Mover.position(i) + Mover.velocity(i);
  });
  const moving = root.pipe(movers.inject()).createComputePipeline({ compute: kernel });

  const draw = drawer(
    root,
    [movers],
    addedTo(root, orthographic(root), movers.renderer().pipe(light.inject())),
    (frame) => {
      const pass = frame.encoder.beginComputePass();
      pass.setPipeline(moving);
      pass.dispatchWorkgroups(1);
      pass.end();
    },
  );

  // Stretched three times along x, the box spans 1.5 by 0.5, around -0.5 and then 0.5.
  let at = await draw();
  expect(at(-1.1, 0)).toEqual([255, 0, 0, 255]);
  expect(at(0.1, 0)).toEqual([255, 0, 0, 255]);
  expect(at(0.5, 0)).toEqual([0, 0, 0, 0]);
  expect(at(-0.5, 0.4)).toEqual([0, 0, 0, 0]);

  at = await draw();
  expect(at(1.1, 0)).toEqual([255, 0, 0, 255]);
  expect(at(-0.5, 0)).toEqual([0, 0, 0, 0]);

  root.destroy();
});

const spots = tgpu.accessor(d.arrayOf(d.vec3f, 2));

/** A kind placed from a buffer of its own, hidden by a field, shaded unlit by its renderer. */
/** A material's colour, as it is. */
const unlit = (material: shading.Material) => {
  'use gpu';
  return d.vec3f(material.color);
};

const Marker = instances.define('Marker', {
  host: { spot: d.u32, hidden: d.u32 },
  place: (marker) => {
    'use gpu';
    return instances.Placement({
      rotation: d.vec4f(0, 0, 0, 1),
      position: spots.$[marker.spot],
      scale: d.vec3f(0.8),
    });
  },
  visible: (marker) => {
    'use gpu';
    return marker.hidden === 0;
  },
});

it.each([
  ['culled', true],
  ['drawn directly', false],
])(
  'places a kind from data of its own and hides what it says, shaded as the renderer says (%s)',
  async (_mode, cull) => {
    const root = await gpuRoot({ features: FIRST_INSTANCE });
    const pool = meshPool.create(root, { capacity: 64, vertices: 64, indices: 64 });
    const box = pool.add(meshes.box());
    const light = lighting.create(root, { settings: dark });
    const where = root
      .createBuffer(d.arrayOf(d.vec3f, 2), [d.vec3f(-1.2, 0, 0), d.vec3f(1.2, 0, 0)])
      .$usage('storage');

    const markers = instances.create(root, pool, { kind: Marker, capacity: 4 });
    const renderer = markers
      .renderer({ shade: unlit, cull })
      .pipe(light.inject())
      .pipe((cfg) => cfg.with(spots, where.as('readonly')));
    const drawn = addedTo(root, orthographic(root), renderer);
    const draw = drawer(root, [markers], drawn, (frame) => {
      if (!cull) return;
      const pass = frame.encoder.beginComputePass();
      renderer.cull(pass);
      pass.end();
    });

    // Their own position fields say the origin; the buffer says where they are.
    markers.add(box, { spot: 0, color: [1, 0, 0] });
    const right = markers.add(box, { spot: 1, color: [0, 1, 0] });
    let at = await draw();
    expect(at(-1.2, 0)).toEqual([255, 0, 0, 255]);
    expect(at(1.2, 0)).toEqual([0, 255, 0, 255]);
    expect(at(0, 0)).toEqual([0, 0, 0, 0]);

    markers.write(right, { hidden: 1 });
    at = await draw();
    expect(at(1.2, 0)).toEqual([0, 0, 0, 0]);
    expect(at(-1.2, 0)).toEqual([255, 0, 0, 255]);

    root.destroy();
  },
);
