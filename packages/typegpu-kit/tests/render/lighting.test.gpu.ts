import { cameras, frames, profiler, table } from '@typegpu/kit';
import * as grid from '@typegpu/kit/grid';
import { gpuRoot } from '@typegpu/kit/testing/dawn';
import { d, std, tgpu, type TgpuRoot } from 'typegpu';
import { expect, it } from 'vitest';
import { mat4 } from 'wgpu-matrix';
import { lighting, materials, pointShadow, shading } from '../../src/index.ts';
import { shadowShape, shadowView } from '../../src/render/shared.ts';

const Probe = d.struct({ position: d.vec3f, normal: d.vec3f, material: shading.Material });

const up = d.vec3f(0, 1, 0);
const white = materials.matte(d.vec3f(1));
const dark = { sunIntensity: 0, skyTop: [0, 0, 0], skyBottom: [0, 0, 0] } as const;

/** A table of white point lights, or spot lights aimed down, for `points.table`. */
const pointsIn = (root: TgpuRoot, capacity: number) =>
  table.create(root, lighting.PointLight, { capacity, defaults: { ...lighting.WHITE } });
const spotsIn = (root: TgpuRoot, capacity: number) =>
  table.create(root, lighting.SpotLight, { capacity, defaults: { ...lighting.WHITE_SPOT } });

/** Something to flush before each frame, such as a table of lights. */
type Flushed = { flush(): void } | undefined;

/** Shades one point on the GPU, seen from above, through a pipeline made once. */
function probe(root: TgpuRoot, light: lighting.Lighting, lamps?: Flushed) {
  const input = root.createBuffer(Probe).$usage('storage');
  const output = root.createBuffer(d.vec3f).$usage('storage');
  const view = cameras.custom(root);
  view.write(mat4.lookAt([0, 5, 0], [0, 0, 0], [0, 0, -1]), mat4.perspective(0.8, 1, 0.1, 50));

  const point = input.as('readonly');
  const color = output.as('mutable');
  const kernel = tgpu.computeFn({ workgroupSize: [1] })(() => {
    'use gpu';
    color.$ = shading.shade(point.$.material, shading.pointAt(point.$.position, point.$.normal));
  });

  const pipeline = root
    .pipe(view.inject())
    .pipe(light.inject())
    .createComputePipeline({ compute: kernel });

  const loop = frames.create(root, (frame) => {
    lamps?.flush();
    const pass = frame.encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.dispatchWorkgroups(1);
    pass.end();
  });

  let time = 0;
  return async (material = white) => {
    input.write({ position: d.vec3f(), normal: up, material });
    loop.run(time++);
    return output.read();
  };
}

it('lights a surface by the sun and the sky, as the CPU does', async () => {
  const root = await gpuRoot();

  const flat = lighting.create(root, {
    shading: { reflect: shading.lambert },
    settings: { sunDirection: [0, 1, 0], skyTop: [0, 0, 0], skyBottom: [0, 0, 0] },
  });
  expect([...(await probe(root, flat)())]).toEqual([1, 1, 1]);

  const lit = lighting.create(root, {
    settings: { sunDirection: [0, 1, 0], skyTop: [0.2, 0.2, 0.2], skyBottom: [0, 0, 0] },
  });
  const zero = d.vec2f();
  const facing = shading.Point({ position: d.vec3f(), normal: up, view: up, uv: zero, uv1: zero });
  const sun = shading.pbr(white, facing, up, d.vec3f(1)).x;

  // The sky's diffuse light, plus Karis's reflectance for roughness 1 times the same light.
  expect((await probe(root, lit)()).x).toBeCloseTo(sun + 0.2 + 0.0031, 3);

  // skyReflection multiplies the reflection alone.
  const bright = lighting.create(root, {
    settings: {
      sunDirection: [0, 1, 0],
      skyTop: [0.2, 0.2, 0.2],
      skyBottom: [0, 0, 0],
      skyReflection: 10,
    },
  });
  expect((await probe(root, bright)()).x).toBeCloseTo(sun + 0.2 + 0.031, 3);

  root.destroy();
});

it('shades by the light around a point, darkening its reflection alone by the occlusion', async () => {
  const root = await gpuRoot();
  const facing = 0.6;
  const out = root.createMutable(d.arrayOf(d.vec3f, 2));
  const read = tgpu.computeFn({ workgroupSize: [1] })(() => {
    'use gpu';
    const zero = d.vec2f();
    const view = d.vec3f(std.sqrt(1 - facing * facing), facing, 0);
    const point = shading.Point({ position: d.vec3f(), normal: up, view, uv: zero, uv1: zero });
    const shaded = materials.matte(d.vec3f(0.5));
    shaded.occlusion = 0.4;
    out.$[0] = shading.surrounded(shaded, point, d.vec3f(2), d.vec3f());
    out.$[1] = shading.surrounded(shaded, point, d.vec3f(), d.vec3f(2));
  });
  root.createComputePipeline({ compute: read }).dispatchWorkgroups(1);
  const [diffuse, reflected] = await out.read();

  expect(diffuse!.x).toBeCloseTo(1, 5);
  const { roughness } = materials.matte(d.vec3f(0.5));
  const seen = shading.specularOcclusion(facing, 0.4, roughness);
  const weight = shading.reflectance(d.vec3f(0.04), roughness, facing).x;
  expect(seen).toBeLessThan(1);
  expect(reflected!.x).toBeCloseTo(2 * weight * seen, 5);
  root.destroy();
});

it('makes the sun unit length and multiplies the intensities, and lights nothing from no direction', async () => {
  const root = await gpuRoot();
  const flat = { reflect: shading.lambert };

  const strong = lighting.create(root, {
    shading: flat,
    settings: { skyTop: [0, 0, 0], skyBottom: [0, 0, 0], sunDirection: [0, 3, 0], sunIntensity: 2 },
  });
  expect([...(await probe(root, strong)())]).toEqual([2, 2, 2]);

  const nowhere = lighting.create(root, {
    shading: flat,
    settings: { skyTop: [0, 0, 0], skyBottom: [0, 0, 0], sunDirection: [0, 0, 0] },
  });
  expect([...(await probe(root, nowhere)())]).toEqual([0, 0, 0]);

  // Halving is exact in floats, so the two skies light the same.
  const sky = { top: [0.4, 0.4, 0.4], bottom: [0.1, 0.1, 0.1] } as const;
  const halved = lighting.create(root, {
    settings: { sunIntensity: 0, skyTop: sky.top, skyBottom: sky.bottom, skyIntensity: 0.5 },
  });
  const half = lighting.create(root, {
    settings: { sunIntensity: 0, skyTop: [0.2, 0.2, 0.2], skyBottom: [0.05, 0.05, 0.05] },
  });
  expect([...(await probe(root, halved)())]).toEqual([...(await probe(root, half)())]);

  root.destroy();
});

it('caps what shade returns at the largest half float, under a mirror sun and a light 1 mm away', async () => {
  const root = await gpuRoot();
  const mirror = materials.metal(d.vec3f(1), 0);

  const sunny = lighting.create(root, { settings: { sunDirection: [0, 1, 0], sunIntensity: 3 } });
  const glare = [...(await probe(root, sunny)(mirror))];

  const lamps = pointsIn(root, 1);
  const near = lighting.create(root, { settings: dark, points: { table: lamps } });
  lamps.add({ position: [0, 0.001, 0] });
  const close = [...(await probe(root, near, lamps)())];

  for (const value of [...glare, ...close]) {
    expect(value).toBeGreaterThan(1000);
    expect(value).toBeLessThanOrEqual(65504);
  }
  root.destroy();
});

it('turns point lights on and off without a new pipeline', async () => {
  const root = await gpuRoot();
  const lamps = pointsIn(root, 4);
  const light = lighting.create(root, {
    settings: dark,
    shading: { reflect: shading.lambert },
    points: { table: lamps },
  });
  const shade = probe(root, light, lamps);

  const lamp = lamps.add({ position: [0, 2, 0], intensity: 4, range: 10 });
  lamps.add({ position: [0, 12, 0], intensity: 100, range: 10 });
  expect((await shade()).x).toBeCloseTo(4 * shading.falloff(2, 10), 5);

  lamps.remove(lamp);
  expect((await shade()).x).toBe(0);

  root.destroy();
});

it('gives the light arriving from the sun, the sky and each point light in reach, from every side', async () => {
  const root = await gpuRoot();
  const lamps = spotsIn(root, 4);
  const light = lighting.create(root, {
    settings: { sunDirection: [0, -1, 0], skyTop: [0.2, 0.2, 0.2], skyBottom: [0, 0, 0] },
    points: { table: lamps },
  });
  lamps.add({ position: [0, 2, 0], color: [1, 0.5, 0], intensity: 4, aim: [0, -1, 0] });
  lamps.add({ position: [0, -2, 0], intensity: 3, aim: [0, -1, 0], cone: lighting.cone(0, 0.5) });
  lamps.add({ position: [0, 12, 0], intensity: 100 });

  const output = root.createBuffer(d.vec3f).$usage('storage');
  const out = output.as('mutable');
  const view = cameras.custom(root);
  view.write(mat4.lookAt([0, 5, 0], [0, 0, 0], [0, 0, -1]), mat4.perspective(0.8, 1, 0.1, 50));
  const pipeline = root
    .pipe(view.inject())
    .pipe(light.inject())
    .createComputePipeline({
      compute: tgpu.computeFn({ workgroupSize: [1] })(() => {
        'use gpu';
        out.$ = shading.incoming(d.vec3f());
      }),
    });
  frames
    .create(root, (frame) => {
      lamps.flush();
      const pass = frame.encoder.beginComputePass();
      pass.setPipeline(pipeline);
      pass.dispatchWorkgroups(1);
      pass.end();
    })
    .run(0);

  // The sun from below lights the point as fully as from above; the spot below aims away.
  const lamp = 4 * shading.falloff(2, 10);
  const arriving = [...(await output.read())];
  [1 + 0.1 + lamp, 1 + 0.1 + lamp * 0.5, 1 + 0.1].forEach((value, c) =>
    expect(arriving[c]).toBeCloseTo(value, 5),
  );
  root.destroy();
});

/** The light in a slot of a `lighting.PointLight` table, for `points.light`. */
const lampAt = (slot: number) => {
  'use gpu';
  return lighting.Light({
    position: lighting.PointLight.position(slot),
    range: lighting.PointLight.range(slot),
    color: lighting.PointLight.color(slot),
    intensity: lighting.PointLight.intensity(slot),
  });
};

it('lights the same from a table, from a light function, and from the candidates a grid holds', async () => {
  const root = await gpuRoot();
  const LAMPS = 32;
  const SIGHT = 3;
  const SIDE = 12;

  // Lamps scattered over a floor, each reaching at most SIGHT, and points to shade on the floor
  // and on walls facing four ways.
  let seed = 3;
  const next = () => {
    seed = (seed * 16807) % 2147483647;
    return seed / 2147483647;
  };
  const inputs = Array.from({ length: LAMPS - 1 }, () => ({
    position: [next() * 16 - 8, 0.3 + next() * 2, next() * 16 - 8] as const,
    range: 1 + next() * (SIGHT - 1),
    color: [next(), next(), next()] as const,
    intensity: 1 + next() * 3,
  }));
  const normals = [d.vec3f(0, 1, 0), d.vec3f(1, 0, 0), d.vec3f(0, 0, -1), d.vec3f(-1, 0.5, 0)];
  const probes = Array.from({ length: SIDE * SIDE }, (_, i) => ({
    position: d.vec3f(
      ((i % SIDE) / (SIDE - 1)) * 17 - 8.5,
      0,
      (Math.floor(i / SIDE) / (SIDE - 1)) * 17 - 8.5,
    ),
    normal: normals[i % normals.length],
  }));

  const Surface = d.struct({ position: d.vec3f, normal: d.vec3f });
  const surfaces = root.createBuffer(d.arrayOf(Surface, probes.length), probes).$usage('storage');
  const colors = root.createBuffer(d.arrayOf(d.vec3f, probes.length)).$usage('storage');
  const view = cameras.custom(root);
  view.write(mat4.lookAt([0, 9, 4], [0, 0, 0], [0, 1, 0]), mat4.perspective(0.8, 1, 0.1, 50));

  const surface = surfaces.as('readonly');
  const color = colors.as('mutable');
  const kernel = tgpu.computeFn({
    workgroupSize: [64],
    in: { gid: d.builtin.globalInvocationId },
  })(({ gid }) => {
    'use gpu';
    if (gid.x >= probes.length) return;
    const at = surface.$[gid.x];
    const matte = materials.matte(d.vec3f(1));
    color.$[gid.x] = shading.shade(matte, shading.pointAt(at.position, at.normal));
  });

  const lamps = table.create(root, lighting.PointLight, { capacity: LAMPS });
  const rows = lamps.inject();
  const cells = grid.create(root, {
    capacity: LAMPS,
    low: d.vec3f(-9, 0, -9),
    size: d.vec3f(18, 3, 18),
    sight: SIGHT,
    alive: lighting.PointLight.alive.pipe(rows),
    position: lighting.PointLight.position.pipe(rows),
  });

  const candidates: grid.Candidates = cells.candidates();
  const everySlot = (_position: d.v3f, _r: number) => {
    'use gpu';
    return lighting.Span({ first: 0, count: LAMPS });
  };
  const listed: grid.Candidates = {
    ranges: 1,
    range: everySlot,
    slot: (index) => {
      'use gpu';
      return index;
    },
  };

  const lampLight = tgpu.fn(lampAt).pipe(rows);
  const lightings = {
    table: lighting.create(root, { settings: dark, points: { table: lamps } }),
    listed: lighting.create(root, { settings: dark, points: { table: lamps, candidates: listed } }),
    light: lighting.create(root, { settings: dark, points: { count: LAMPS, light: lampLight } }),
    cells: lighting.create(root, { settings: dark, points: { light: lampLight, candidates } }),
  };
  for (const input of inputs) lamps.add(input);
  // A bright light of range 0 among them, which lights nothing.
  const unlit = lamps.add({ position: [0, 1, 0], color: [1, 1, 1], intensity: 100, range: 0 });

  const shade = async (light: lighting.Lighting) => {
    const pipeline = root
      .pipe(view.inject())
      .pipe(light.inject())
      .createComputePipeline({ compute: kernel });
    frames
      .create(root, (frame) => {
        lamps.flush();
        const pass = frame.encoder.beginComputePass();
        cells.sort(pass);
        pass.setPipeline(pipeline);
        pass.dispatchWorkgroups(Math.ceil(probes.length / 64));
        pass.end();
      })
      .run(0);
    return (await colors.read()).flatMap((each) => [...each]);
  };

  const expected = await shade(lightings.table);
  expect(expected.filter((value) => value > 0.01).length).toBeGreaterThan(probes.length / 2);
  for (const each of [lightings.listed, lightings.light, lightings.cells]) {
    const shaded = await shade(each);
    for (const [i, value] of shaded.entries()) {
      expect(Math.abs(value - expected[i])).toBeLessThanOrEqual(1e-6 * Math.max(1, expected[i]));
    }
  }
  lamps.remove(unlit);
  expect(await shade(lightings.table)).toEqual(expected);

  root.destroy();
});

it('skips a point light exactly at the surface, which has no direction to light it from', async () => {
  const root = await gpuRoot();
  const lamps = pointsIn(root, 2);
  const light = lighting.create(root, { settings: dark, points: { table: lamps } });
  const shade = probe(root, light, lamps);

  lamps.add({ position: [0, 0, 0], intensity: 4, range: 10 });
  expect([...(await shade())]).toEqual([0, 0, 0]);

  root.destroy();
});

it('lights from where a kernel moved a light, which a change of its intensity leaves in place', async () => {
  const root = await gpuRoot();
  const movers = table.create(root, lighting.MovingPointLight, {
    capacity: 2,
    defaults: { ...lighting.WHITE },
  });
  const light = lighting.create(root, {
    settings: dark,
    shading: { reflect: shading.lambert },
    points: { table: movers },
  });
  const shade = probe(root, light, movers);
  const lamp = movers.add({ position: [0, 9, 0], intensity: 4, range: 10 });
  expect((await shade()).x).toBeCloseTo(4 * shading.falloff(9, 10), 5);

  const mover = root.pipe(movers.inject()).createComputePipeline({
    compute: tgpu.computeFn({ workgroupSize: [1] })(() => {
      'use gpu';
      lighting.MovingPointLight.device.$[0].position = d.vec3f(0, 2, 0);
    }),
  });
  frames
    .create(root, (frame) => {
      const pass = frame.encoder.beginComputePass();
      pass.setPipeline(mover);
      pass.dispatchWorkgroups(1);
      pass.end();
    })
    .run(0);

  movers.write(lamp, { intensity: 2 });
  expect((await shade()).x).toBeCloseTo(2 * shading.falloff(2, 10), 5);

  movers.write(lamp, { position: [0, 5, 0] });
  expect((await shade()).x).toBeCloseTo(2 * shading.falloff(5, 10), 5);

  root.destroy();
});

it('adds what a clear surface reflects whole, and scales its diffuse light and fog by its opacity', async () => {
  const root = await gpuRoot();
  const Out = d.struct({ full: d.vec3f, reflected: d.vec3f, clear: d.vec4f, opaque: d.vec4f });
  const output = root.createBuffer(Out).$usage('storage');
  const out = output.as('mutable');
  const view = cameras.custom(root);
  view.write(mat4.lookAt([0, 5, 0], [0, 0, 0], [0, 0, -1]), mat4.perspective(0.8, 1, 0.1, 50));

  // Seen from straight above under a sun straight above: the highlight's middle.
  const kernel = tgpu.computeFn({ workgroupSize: [1] })(() => {
    'use gpu';
    const point = shading.pointAt(d.vec3f(), d.vec3f(0, 1, 0));
    const glass = materials.glossy(d.vec3f(0.8, 0.9, 1), 0.2);
    out.$ = Out({
      full: shading.shade(glass, point),
      // A dielectric's reflectance doesn't depend on its colour: black, it only reflects.
      reflected: shading.shade(materials.glossy(d.vec3f(), 0.2), point),
      clear: shading.shadeClear(glass, point, 0.1),
      opaque: shading.shadeClear(glass, point, 1),
    });
  });

  const run = async (light: lighting.Lighting) => {
    const pipeline = root
      .pipe(view.inject())
      .pipe(light.inject())
      .createComputePipeline({ compute: kernel });
    const encoder = root.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.dispatchWorkgroups(1);
    pass.end();
    encoder.submit();
    return output.read();
  };

  const clearAir = lighting.create(root, {
    settings: { ...dark, sunIntensity: 1, sunDirection: [0, 1, 0] },
  });
  const { full, reflected, clear, opaque } = await run(clearAir);
  for (let c = 0; c < 3; c++) {
    expect(opaque[c]).toBeCloseTo(full[c], 4);
    expect(clear[c]).toBeCloseTo(full[c] * 0.1 + reflected[c] * 0.9, 4);
  }
  expect([clear.w, opaque.w]).toEqual([...Float32Array.of(0.1, 1)]);
  // The highlight outshines the diffuse light, so a surface that only dims it would lose it.
  expect(clear.x).toBeGreaterThan(full.x * 0.1 * 5);

  // Fog over the point: the opacity's share of it, the rest left to the fog behind, seen through.
  const foggy = lighting.create(root, {
    settings: {
      ...dark,
      sunIntensity: 1,
      sunDirection: [0, 1, 0],
      fogColor: [1, 1, 1],
      fogDensity: 0.2,
    },
  });
  const fogged = await run(foggy);
  const haze = 1 - Math.exp(-0.2 * 5);
  expect(fogged.clear.x).toBeCloseTo(
    fogged.full.x * 0.1 + fogged.reflected.x * 0.9 - haze * 0.9,
    4,
  );

  root.destroy();
});

it('fogs by distance from where it starts, thinning with height', async () => {
  const root = await gpuRoot();
  const glowing = materials.glow(d.vec3f(1), 1);

  const even = lighting.create(root, {
    settings: { ...dark, fogColor: [0, 0, 0], fogDensity: 0.1 },
  });
  expect((await probe(root, even)(glowing)).x).toBeCloseTo(Math.exp(-0.5), 4);

  // From y = 5 down to y = 0, the density 0.1 e^(-y/2) integrates to 0.2 (1 - e^-2.5).
  const low = lighting.create(root, {
    settings: { ...dark, fogColor: [0, 0, 0], fogDensity: 0.1, fogFalloff: 0.5 },
  });
  const depth = 0.2 * (1 - Math.exp(-2.5));
  expect((await probe(root, low)(glowing)).x).toBeCloseTo(Math.exp(-depth), 4);

  // Starting 2 from the eye, the fog spans y = 3 down to y = 0; starting past the point, nothing.
  const late = lighting.create(root, {
    settings: { ...dark, fogColor: [0, 0, 0], fogDensity: 0.1, fogFalloff: 0.5, fogStart: 2 },
  });
  expect((await probe(root, late)(glowing)).x).toBeCloseTo(
    Math.exp(-0.2 * (1 - Math.exp(-1.5))),
    4,
  );
  late.settings.patch({ fogStart: 6 });
  expect((await probe(root, late)(glowing)).x).toBeCloseTo(1, 4);

  // haze is the fog's share, by which heightFog adds the fog's colour.
  const tinted = lighting.create(root, {
    settings: { ...dark, fogColor: [0.5, 0.25, 1], fogDensity: 0.1, fogFalloff: 0.5 },
  });
  const output = root.createBuffer(d.vec4f).$usage('storage');
  const out = output.as('mutable');
  const view = cameras.custom(root);
  view.write(mat4.lookAt([0, 5, 0], [0, 0, 0], [0, 0, -1]), mat4.perspective(0.8, 1, 0.1, 50));
  const pipeline = root
    .pipe(view.inject())
    .pipe(tinted.inject())
    .createComputePipeline({
      compute: tgpu.computeFn({ workgroupSize: [1] })(() => {
        'use gpu';
        const point = shading.pointAt(d.vec3f(), d.vec3f(0, 1, 0));
        out.$ = d.vec4f(shading.heightFog(point).added, shading.haze(point));
      }),
    });
  const encoder = root.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.dispatchWorkgroups(1);
  pass.end();
  encoder.submit();
  const [r, g, b, haze] = [...(await output.read())];
  expect(haze).toBeCloseTo(1 - Math.exp(-depth), 4);
  [r, g, b].forEach((value, c) => expect(value).toBeCloseTo([0.5, 0.25, 1][c] * haze!, 5));

  root.destroy();
});

/** The sky's top colour on surfaces facing up, read from the settings. */
const skylit: shading.AmbientHook = (material, point) => {
  'use gpu';
  return material.color * shading.settings().skyTop * std.max(point.normal.y, 0);
};

it('lets your own ambient read the settings, and follows a patch without a new pipeline', async () => {
  const root = await gpuRoot();
  const light = lighting.create(root, {
    settings: { sunIntensity: 0, skyTop: [0.25, 0.5, 1] },
    shading: { ambient: skylit },
  });
  const shade = probe(root, light);
  expect([...(await shade())]).toEqual([0.25, 0.5, 1]);

  light.settings.patch({ skyTop: [1, 0, 0.5] });
  expect([...(await shade())]).toEqual([1, 0, 0.5]);

  root.destroy();
});

it("times the shadow's pass with a profiler's writes", async () => {
  const root = await gpuRoot({ features: ['timestamp-query'] });
  const light = lighting.create(root, { shadow: { size: 1024 } });
  const timer = profiler.create(root, { skip: 1, samples: 3 });
  const shadow = timer.timer('shadow');

  // A triangle over the whole map, drawn many times, so the pass takes measurable time.
  const Corner = d.struct({ position: d.vec4f });
  const caster = light.shadow.add({
    Vertex: Corner,
    at: (i: number, _instance: number) => {
      'use gpu';
      const x = d.f32(i % 2) * 4 - 1;
      const y = d.f32(i / 2) * 4 - 1;
      return Corner({ position: d.vec4f(x, y, 0.5, 1) });
    },
    draw: (pass) => pass.draw(3, 64),
  });

  const loop = frames.create(root, (frame) => {
    light.shadow.record(frame, [caster], { timestampWrites: shadow.writes() });
    timer.capture(frame);
  });

  // The timestamps come back a few frames later, later still on a busy machine.
  for (let i = 0; i < 100 && shadow.ms === undefined; i++) {
    loop.run(i);
    await new Promise((resolve) => setTimeout(resolve, 20));
  }
  expect(shadow.ms).toBeGreaterThan(0);

  root.destroy();
});

it('takes how much sun reaches a point, and how much of a point light, from hooks', async () => {
  const root = await gpuRoot();
  const flat = { reflect: shading.lambert };
  const sunOnly = { skyTop: [0, 0, 0], skyBottom: [0, 0, 0], sunDirection: [0, 1, 0] } as const;

  const focused = lighting.create(root, {
    settings: sunOnly,
    shading: {
      ...flat,
      sunShadow: (point) => {
        'use gpu';
        return shading.sunShadowed(point) * 2.5;
      },
    },
  });
  expect((await probe(root, focused)()).x).toBeCloseTo(2.5, 5);

  // A cone of half-angle 60° about straight down lets the lamp overhead through, at full strength.
  const lamps = pointsIn(root, 2);
  const spot = lighting.create(root, {
    settings: dark,
    shading: {
      ...flat,
      attenuation: (_point, toLight, distance, range, _slot) => {
        'use gpu';
        const inCone = std.step(0.5, toLight.y);
        return shading.falloff(distance, range) * inCone;
      },
    },
    points: { table: lamps },
  });
  const shade = probe(root, spot, lamps);
  lamps.add({ position: [0, 2, 0], intensity: 4, range: 10 });
  lamps.add({ position: [3, 0.5, 0], intensity: 4, range: 10 });
  expect((await shade()).x).toBeCloseTo(4 * shading.falloff(2, 10), 5);

  root.destroy();
});

it('reads a table of point lights made before it, which it leaves alive', async () => {
  const root = await gpuRoot();
  const lamps = table.create(root, lighting.PointLight, {
    capacity: 4,
    defaults: { color: [1, 1, 1], intensity: 1, range: 10 },
  });
  const light = lighting.create(root, {
    settings: dark,
    shading: { reflect: shading.lambert },
    points: { table: lamps },
  });
  expect('points' in light).toBe(false);
  const shade = probe(root, light);

  lamps.add({ position: [0, 2, 0], intensity: 4, range: 10 });
  lamps.flush();
  expect((await shade()).x).toBeCloseTo(4 * shading.falloff(2, 10), 5);

  light.destroy();
  expect(lamps.count).toBe(1);
  lamps.add({ position: [0, 3, 0] });
  lamps.destroy();
  root.destroy();
});

it('shades from the cascade that holds each point, sharp near the eye and out to far', async () => {
  const root = await gpuRoot();
  const eye = cameras.custom(root);
  eye.write(mat4.lookAt([0, 2, 0], [0, 2, -1], [0, 1, 0]), mat4.perspective(0.8, 1, 0.1, 500));
  const light = lighting.create(root, {
    settings: { sunDirection: [0, 1, 0] },
    shadow: { eye, cascades: 3, far: 120, size: 512, filter: 1 },
  });

  // A roof 2 units across, 1 above the ground, over each of three points down the view.
  const depths = [3, 30, 100];
  const Corner = d.struct({ position: d.vec4f });
  const corners = tgpu.const(
    d.arrayOf(d.vec2f, 6),
    [
      [-1, -1],
      [1, -1],
      [1, 1],
      [-1, -1],
      [1, 1],
      [-1, 1],
    ].map(([x, z]) => d.vec2f(x!, z!)),
  );
  const along = tgpu.const(
    d.arrayOf(d.f32, 3),
    depths.map((depth) => -depth),
  );
  const roofs = light.shadow.add({
    Vertex: Corner,
    at: (vertex: number, instance: number) => {
      'use gpu';
      const corner = corners.$[vertex];
      const world = d.vec3f(corner.x, 1, corner.y + along.$[instance]);
      return Corner({ position: cameras.current.$.viewProjection * d.vec4f(world, 1) });
    },
    draw: (pass) => pass.draw(6, 3),
  });

  const probes = depths.flatMap((depth) => [d.vec3f(0, 0, -depth), d.vec3f(3, 0, -depth)]);
  const output = root.createBuffer(d.arrayOf(d.vec2f, probes.length)).$usage('storage');
  const out = output.as('mutable');
  const at = root.createBuffer(d.arrayOf(d.vec3f, probes.length), probes).$usage('storage');
  const points = at.as('readonly');
  const kernel = tgpu.computeFn({ workgroupSize: [1], in: { gid: d.builtin.globalInvocationId } })(
    ({ gid }) => {
      'use gpu';
      const point = shading.pointAt(points.$[gid.x], d.vec3f(0, 1, 0));
      out.$[gid.x] = d.vec2f(d.f32(shading.cascadeOf(point).index), shading.sunShadowed(point));
    },
  );
  const probing = root
    .pipe(eye.inject())
    .pipe(light.inject())
    .createComputePipeline({ compute: kernel });

  frames
    .create(root, (frame) => {
      light.shadow.record(frame, roofs);
      const pass = frame.encoder.beginComputePass();
      pass.setPipeline(probing);
      pass.dispatchWorkgroups(probes.length);
      pass.end();
    })
    .run(0);

  const read = await output.read();
  expect(read.map((each) => each.x)).toEqual([0, 0, 1, 1, 2, 2]);
  expect(read.map((each) => each.y)).toEqual([0, 1, 0, 1, 0, 1]);
  root.destroy();
});

it('picks a cascade only where the filter stays inside it from the lifted point, and gives 1 outside a cascade asked for', async () => {
  const root = await gpuRoot();
  const eye = cameras.custom(root);
  eye.write(mat4.lookAt([0, 2, 0], [0, 2, -1], [0, 1, 0]), mat4.perspective(0.8, 1, 0.1, 500));
  const light = lighting.create(root, {
    settings: { sunDirection: [0, 1, 0] },
    shadow: { eye, cascades: 3, far: 120, size: 512, filter: 9, lift: 24 },
  });

  // A roof over everything, so every cascade's map is dark at its edges too.
  const Corner = d.struct({ position: d.vec4f });
  const corners = tgpu.const(
    d.arrayOf(d.vec2f, 6),
    [
      [-1, -1],
      [1, -1],
      [1, 1],
      [-1, -1],
      [1, 1],
      [-1, 1],
    ].map(([x, z]) => d.vec2f(x!, z!)),
  );
  const roof = light.shadow.add({
    Vertex: Corner,
    at: (vertex: number) => {
      'use gpu';
      const corner = corners.$[vertex] * 1000;
      return Corner({
        position: cameras.current.$.viewProjection * d.vec4f(corner.x, 1, corner.y, 1),
      });
    },
    draw: (pass) => pass.draw(6),
  });

  // Points down the view, lifted away from the eye towards each cascade's far edge.
  const PROBES = 600;
  const Probed = d.struct({
    index: d.f32,
    reach: d.f32,
    inside: d.f32,
    outside: d.f32,
    across: d.f32,
  });
  const output = root.createBuffer(d.arrayOf(Probed, PROBES)).$usage('storage');
  const out = output.as('mutable');
  const kernel = tgpu.computeFn({ workgroupSize: [1], in: { gid: d.builtin.globalInvocationId } })(
    ({ gid }) => {
      'use gpu';
      const depth = 0.5 + d.f32(gid.x) * 0.1;
      const point = shading.pointAt(d.vec3f(0, 0, -depth), d.vec3f(0, 0, -1));
      const k = shading.cascadeOf(point).index;
      let reach = d.f32(0);
      let across = d.f32(0);
      if (k >= 0 && k < 2) {
        // How far out the point lies in the next cascade's clip space
        const next = shadowView.$.cascades[k + 1];
        const clip =
          (shadowView.$.light * d.vec4f(point.position, 1)).xyz * next.scale + next.offset;
        across = std.max(std.abs(clip.x), std.abs(clip.y));
      }
      if (k >= 0) {
        // How far out the filter's last tap reaches, in the cascade's clip space
        const cascade = shadowView.$.cascades[k];
        const lifted = point.position + point.normal * (cascade.texel * shadowShape.$!.lift);
        const clip = (shadowView.$.light * d.vec4f(lifted, 1)).xyz * cascade.scale + cascade.offset;
        reach =
          std.max(std.abs(clip.x), std.abs(clip.y)) +
          shadowShape.$!.side * 0.5 * cascade.texel * cascade.scale.x;
      }
      // Lifted down, away from the roof, in the cascade that holds the point and in the next
      const below = shading.pointAt(point.position, d.vec3f(0, -1, 0));
      out.$[gid.x] = Probed({
        index: d.f32(k),
        reach,
        inside: shading.cascadeShadowed(below, k),
        outside: shading.cascadeShadowed(below, k + 1),
        across,
      });
    },
  );
  const probing = root
    .pipe(eye.inject())
    .pipe(light.inject())
    .createComputePipeline({ compute: kernel });

  frames
    .create(root, (frame) => {
      light.shadow.record(frame, roof);
      const pass = frame.encoder.beginComputePass();
      pass.setPipeline(probing);
      pass.dispatchWorkgroups(PROBES);
      pass.end();
    })
    .run(0);

  const read = await output.read();
  const held = read.filter(({ index }) => index >= 0);
  expect(new Set(held.map(({ index }) => index))).toEqual(new Set([0, 1]));
  for (const { reach } of held) expect(reach).toBeLessThanOrEqual(1);
  // Under the roof, a point is dark in its cascade, and in the next only where that reaches it.
  for (const { inside } of held) expect(inside).toBe(0);
  const next = held.filter(({ index }) => index < 2);
  expect(next.filter(({ across }) => across > 1).map(({ outside }) => outside)).toContain(1);
  for (const { across, outside } of next) expect(outside).toBe(across > 1 ? 1 : 0);
  root.destroy();
});

/** Under a roof over everything lit from straight above, probes the ground down the view. */
async function underRoof(
  root: TgpuRoot,
  eye: cameras.CustomCamera,
  light: lighting.Shadowed,
  probes: number,
) {
  const Corner = d.struct({ position: d.vec4f });
  const corners = tgpu.const(
    d.arrayOf(d.vec2f, 6),
    [
      [-1, -1],
      [1, -1],
      [1, 1],
      [-1, -1],
      [1, 1],
      [-1, 1],
    ].map(([x, z]) => d.vec2f(x!, z!)),
  );
  const roof = light.shadow.add({
    Vertex: Corner,
    at: (vertex: number) => {
      'use gpu';
      const corner = corners.$[vertex] * 1000;
      return Corner({
        position: cameras.current.$.viewProjection * d.vec4f(corner.x, 1, corner.y, 1),
      });
    },
    draw: (pass) => pass.draw(6),
  });

  const Probed = d.struct({ index: d.f32, edge: d.f32, kit: d.f32, hook: d.f32 });
  const output = root.createBuffer(d.arrayOf(Probed, probes)).$usage('storage');
  const out = output.as('mutable');
  const kernel = tgpu.computeFn({ workgroupSize: [1], in: { gid: d.builtin.globalInvocationId } })(
    ({ gid }) => {
      'use gpu';
      const point = shading.pointAt(d.vec3f(0, 0, -1 - d.f32(gid.x)), d.vec3f(0, 1, 0));
      const at = shading.cascadeOf(point);
      out.$[gid.x] = Probed({
        index: d.f32(at.index),
        edge: at.edge,
        kit: shading.sunShadowed(point),
        hook: shading.sunShare(point),
      });
    },
  );
  const probing = root
    .pipe(eye.inject())
    .pipe(light.inject())
    .createComputePipeline({ compute: kernel });

  frames
    .create(root, (frame) => {
      light.shadow.record(frame, roof);
      const pass = frame.encoder.beginComputePass();
      pass.setPipeline(probing);
      pass.dispatchWorkgroups(probes);
      pass.end();
    })
    .run(0);
  return output.read();
}

/** An eye 2 above the ground looking down -z. */
const lookingOut = (root: TgpuRoot) => {
  const eye = cameras.custom(root);
  eye.write(mat4.lookAt([0, 2, 0], [0, 2, -1], [0, 1, 0]), mat4.perspective(0.8, 1, 0.1, 500));
  return eye;
};

it('fades into sunlight over the last cascade through a sunShadow hook, where the default ends hard', async () => {
  const root = await gpuRoot();
  const eye = lookingOut(root);
  const light = lighting.create(root, {
    settings: { sunDirection: [0, 1, 0] },
    shadow: { eye, cascades: 3, far: 120, size: 512 },
    shading: {
      sunShadow: (point) => {
        'use gpu';
        const at = shading.cascadeOf(point);
        const lit = shading.cascadeShadowed(point, at.index);
        if (at.edge <= 0) return lit;
        return std.mix(lit, shading.cascadeShadowed(point, at.index + 1), at.edge);
      },
    },
  });

  const read = await underRoof(root, eye, light, 250);
  // The default is dark in every cascade and lit past the last.
  for (const { index, kit } of read) expect(kit).toBe(index < 0 ? 1 : 0);
  // The hook lights the last cascade's edge by how near it lies, and agrees elsewhere.
  const fading = read.filter(({ index, edge }) => index === 2 && edge > 0);
  expect(fading.length).toBeGreaterThan(3);
  for (const { edge, hook } of fading) expect(hook).toBeCloseTo(edge, 5);
  expect(Math.max(...fading.map(({ hook }) => hook))).toBeGreaterThan(0.5);
  for (const { edge, kit, hook } of read) if (edge === 0) expect(hook).toBe(kit);
  root.destroy();
});

it('places its cascades where a fit says, from the eye, light space and settings it hands the fit', async () => {
  const root = await gpuRoot();
  const eye = lookingOut(root);
  const handed: lighting.CascadeFitInput[] = [];
  const light = lighting.create(root, {
    settings: { sunDirection: [0, 1, 0] },
    shadow: {
      eye,
      far: 50,
      size: 512,
      // One cube 320 wide along the view, which reaches past `far`.
      fit: (input, places) => {
        handed.push(input);
        const { right, up: across, sun } = input;
        const centre = [0, 0, -150] as const;
        const along = (axis: readonly number[]) =>
          axis[0]! * centre[0] + axis[1]! * centre[1] + axis[2]! * centre[2];
        places[0] = { x: along(right), y: along(across), z: along(sun), radius: 160 };
        return true;
      },
    },
  });

  const read = await underRoof(root, eye, light, 300);
  expect(handed.at(-1)).toMatchObject({ eye, far: 50, splits: [0, 1], size: 512, sun: [0, 1, 0] });
  // Shadowed out to the cube's far face, six times `far`.
  for (const { index, kit } of read.slice(0, 290)) {
    expect(index).toBe(0);
    expect(kit).toBe(0);
  }
  root.destroy();
});

/** A spot straight down from 4 above the origin, from a function. */
const spotAt = (_slot: number) => {
  'use gpu';
  return lighting.Spot({
    position: d.vec3f(0, 4, 0),
    range: 20,
    color: d.vec3f(1),
    intensity: 4,
    aim: d.vec3f(0, -2, 0),
    cone: lighting.cone(0.3, 0.5),
  });
};

it("lights inside a spot's cone, dims on its edge and lights nothing outside, from a table or a function", async () => {
  const root = await gpuRoot();
  const flat = { reflect: shading.lambert };
  const ramp = lighting.cone(0.3, 0.5);
  const spot = {
    position: [0, 4, 0],
    aim: [0, -2, 0],
    cone: ramp,
    intensity: 4,
    range: 20,
  } as const;
  const lamps = spotsIn(root, 2);
  const tabled = lighting.create(root, { settings: dark, shading: flat, points: { table: lamps } });
  lamps.add(spot);
  // A light beside it, with the full cone, lights every direction.
  const lamp = lamps.add({ position: [9, 4, 0], intensity: 0, range: 20, cone: [0, 1] });

  const given = lighting.create(root, {
    settings: dark,
    shading: flat,
    points: { count: 1, spot: spotAt },
  });

  const Probe2 = d.struct({ position: d.vec3f });
  const angles = [0.2, 0.4, 0.6];
  const probes = angles.map((angle) => ({ position: d.vec3f(Math.tan(angle) * 4, 0, 0) }));
  const input = root.createBuffer(d.arrayOf(Probe2, 3), probes).$usage('storage');
  const output = root.createBuffer(d.arrayOf(d.vec3f, 3)).$usage('storage');
  const view = cameras.custom(root);
  view.write(mat4.lookAt([0, 5, 0], [0, 0, 0], [0, 0, -1]), mat4.perspective(0.8, 1, 0.1, 50));
  const at = input.as('readonly');
  const out = output.as('mutable');
  const kernel = tgpu.computeFn({ workgroupSize: [1], in: { gid: d.builtin.globalInvocationId } })(
    ({ gid }) => {
      'use gpu';
      const point = shading.pointAt(at.$[gid.x].position, d.vec3f(0, 1, 0));
      out.$[gid.x] = shading.shade(materials.matte(d.vec3f(1)), point);
    },
  );
  const shade = async (light: lighting.Lighting) => {
    const pipeline = root
      .pipe(view.inject())
      .pipe(light.inject())
      .createComputePipeline({ compute: kernel });
    frames
      .create(root, (frame) => {
        lamps.flush();
        const pass = frame.encoder.beginComputePass();
        pass.setPipeline(pipeline);
        pass.dispatchWorkgroups(3);
        pass.end();
      })
      .run(0);
    return (await output.read()).map((each) => each.x);
  };

  const expected = angles.map((angle) => {
    const distance = 4 / Math.cos(angle);
    const edge = Math.min(Math.max(Math.cos(angle) * ramp.x + ramp.y, 0), 1) ** 2;
    return 4 * shading.falloff(distance, 20) * edge * Math.cos(angle);
  });
  for (const light of [tabled, given]) {
    const [inside, edge, outside] = await shade(light);
    expect(inside).toBeCloseTo(expected[0], 5);
    expect(edge).toBeCloseTo(expected[1], 5);
    expect(edge).toBeGreaterThan(0);
    expect(edge).toBeLessThan(inside!);
    expect(outside).toBe(0);
  }

  // A light without a cone lights the point outside the spot's.
  lamps.write(lamp, { intensity: 4 });
  expect((await shade(tabled))[2]).toBeGreaterThan(0);
  root.destroy();
});

it('shadows an awning from a point light on both sides of where two cube faces meet, and from a narrow spot', async () => {
  const root = await gpuRoot();
  const lamps = spotsIn(root, 2);
  const light = lighting.create(root, {
    settings: dark,
    points: { table: lamps },
    pointShadow: pointShadow.define({ count: 2, size: 512, filter: 1 }),
  });
  const lamp = lamps.add({ position: [0, 0, 0], range: 10 });
  const spot = lamps.add({
    position: [20, 0, 0],
    range: 10,
    aim: [0, 1, 0],
    cone: lighting.cone(0.3, 0.5),
  });
  light.pointShadow.cast([lamp, spot]);

  // A tile 1 above each light, whose shadow on the awning 2 above the lamp crosses x = 2, where
  // the lamp's +y face meets its +x face.
  const Corner = d.struct({ position: d.vec4f });
  const corners = tgpu.const(
    d.arrayOf(d.vec2f, 6),
    [
      [-1, -1],
      [1, -1],
      [1, 1],
      [-1, -1],
      [1, 1],
      [-1, 1],
    ].map(([x, z]) => d.vec2f(x!, z!)),
  );
  const tiles = tgpu.const(d.arrayOf(d.vec4f, 2), [
    d.vec4f(1, 0, 0.4, 0.5),
    d.vec4f(20, 0, 0.2, 0.2),
  ]);
  const cast = light.pointShadow.add({
    Vertex: Corner,
    at: (vertex: number, instance: number) => {
      'use gpu';
      const corner = corners.$[vertex];
      const tile = tiles.$[instance];
      const world = d.vec3f(tile.x + corner.x * tile.z, 1, tile.y + corner.y * tile.w);
      return Corner({ position: cameras.current.$.viewProjection * d.vec4f(world, 1) });
    },
    draw: (pass) => pass.draw(6, 2),
  });

  const Probe2 = d.struct({ position: d.vec3f, light: d.vec4f });
  const fromLamp = d.vec4f(0, 0, 0, 0);
  const fromSpot = d.vec4f(20, 0, 0, 1);
  const probes = [
    [0.5, 0],
    [1.9, 0],
    [1.98, 0],
    [2.02, 0],
    [2.1, 0],
    [3.5, 0],
    [1.98, 1.5],
    [2.02, 1.5],
  ]
    .map(([x, z]) => ({ position: d.vec3f(x!, 2, z!), light: fromLamp }))
    .concat([
      { position: d.vec3f(20, 2, 0), light: fromSpot },
      { position: d.vec3f(20.7, 2, 0), light: fromSpot },
    ]);
  const input = root.createBuffer(d.arrayOf(Probe2, probes.length), probes).$usage('storage');
  const output = root.createBuffer(d.arrayOf(d.f32, probes.length)).$usage('storage');
  const at = input.as('readonly');
  const out = output.as('mutable');
  const kernel = tgpu.computeFn({ workgroupSize: [1], in: { gid: d.builtin.globalInvocationId } })(
    ({ gid }) => {
      'use gpu';
      const sample = at.$[gid.x];
      const point = shading.pointAt(sample.position, d.vec3f(0, -1, 0));
      const toLight = sample.light.xyz - sample.position;
      const distance = std.length(toLight);
      out.$[gid.x] = shading.lampShadowed(
        point,
        toLight / distance,
        distance,
        d.u32(sample.light.w),
      );
    },
  );
  const view = cameras.custom(root);
  view.write(mat4.lookAt([0, 5, 0], [0, 0, 0], [0, 0, -1]), mat4.perspective(0.8, 1, 0.1, 50));
  const probing = root
    .pipe(view.inject())
    .pipe(light.inject())
    .createComputePipeline({ compute: kernel });

  frames
    .create(root, (frame) => {
      lamps.flush();
      light.pointShadow.record(frame, cast);
      const pass = frame.encoder.beginComputePass();
      pass.setPipeline(probing);
      pass.dispatchWorkgroups(probes.length);
      pass.end();
    })
    .run(0);

  // The tile's shadow runs from x = 1.2 to 2.8 and z = -1 to 1 on the awning.
  expect([...(await output.read())]).toEqual([1, 0, 0, 0, 0, 1, 1, 1, 0, 1]);
  root.destroy();
});

it('gives the same falloff, cone and beam on the CPU as in a shader', async () => {
  const root = await gpuRoot();
  const Case = d.struct({ toLight: d.vec3f, aim: d.vec3f, angles: d.vec2f, distance: d.f32 });
  const Result = d.struct({ falloff: d.f32, cone: d.vec2f, beam: d.f32 });
  const cases = [
    { toLight: d.vec3f(0, 1, 0), aim: d.vec3f(0, -1, 0), angles: d.vec2f(0.2, 0.4), distance: 0.5 },
    {
      toLight: d.vec3f(0.3, 0.9, 0),
      aim: d.vec3f(0, -2, 0),
      angles: d.vec2f(0.2, 0.4),
      distance: 6,
    },
    {
      toLight: d.vec3f(0, 0.8, 0.6),
      aim: d.vec3f(0.1, -1, 0.4),
      angles: d.vec2f(0, 0.9),
      distance: 9.9,
    },
    { toLight: d.vec3f(1, 0, 0), aim: d.vec3f(0, -1, 0), angles: d.vec2f(0.5, 0.6), distance: 12 },
  ].map(({ toLight, ...rest }) => ({ ...rest, toLight: std.normalize(toLight) }));
  const RANGE = 10;

  const input = root.createBuffer(d.arrayOf(Case, cases.length), cases).$usage('storage');
  const output = root.createBuffer(d.arrayOf(Result, cases.length)).$usage('storage');
  const read = input.as('readonly');
  const written = output.as('mutable');
  const kernel = tgpu.computeFn({ workgroupSize: [4], in: { gid: d.builtin.globalInvocationId } })(
    ({ gid }) => {
      'use gpu';
      if (gid.x >= cases.length) return;
      const each = read.$[gid.x];
      const cone = lighting.cone(each.angles.x, each.angles.y);
      written.$[gid.x] = Result({
        falloff: shading.falloff(each.distance, RANGE),
        cone,
        beam: shading.beam(each.toLight, each.aim, cone),
      });
    },
  );
  const pipeline = root.createComputePipeline({ compute: kernel });
  const encoder = root.createCommandEncoder();
  const pass = encoder.beginComputePass({});
  pass.setPipeline(pipeline);
  pass.dispatchWorkgroups(1);
  pass.end();
  encoder.submit();

  const gpu = await output.read();
  cases.forEach(({ toLight, aim, angles, distance }, i) => {
    const cone = lighting.cone(angles.x, angles.y);
    const cpu = [
      shading.falloff(distance, RANGE),
      cone.x,
      cone.y,
      shading.beam(toLight, aim, cone),
    ];
    const { falloff, cone: shaded, beam } = gpu[i];
    // Within float rounding, relative to the value: a narrow cone's scale is large
    [falloff, shaded.x, shaded.y, beam].forEach((value, k) =>
      expect(Math.abs(value - cpu[k])).toBeLessThanOrEqual(1e-6 * Math.max(1, Math.abs(cpu[k]))),
    );
  });
  // The cases cover full light, an edge, none outside the cone and none past the range
  expect(gpu.map(({ beam }) => beam)).toContain(1);
  expect(gpu.map(({ beam }) => beam > 0 && beam < 1)).toContain(true);
  expect(gpu.map(({ beam }) => beam)).toContain(0);
  expect(gpu.at(-1)!.falloff).toBe(0);
  root.destroy();
});
