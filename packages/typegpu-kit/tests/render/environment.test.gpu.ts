import { gpuRoot } from '@typegpu/kit/testing/dawn';
import { d, std, tgpu, type TgpuRoot } from 'typegpu';
import { expect, it } from 'vitest';
import { environment, lighting, materials, shading, images } from '../../src/index.ts';

const AXES = [
  d.vec3f(1, 0, 0),
  d.vec3f(-1, 0, 0),
  d.vec3f(0, 1, 0),
  d.vec3f(0, -1, 0),
  d.vec3f(0, 0, 1),
  d.vec3f(0, 0, -1),
];
const ROUGHNESS = [0, 0.2, 0.4, 0.6, 0.8, 1];

/** Bakes `source`, then reads radiance along each axis at each roughness, and irradiance. */
async function bakeAndRead(
  root: TgpuRoot,
  sky: environment.Environment,
  source: environment.Source,
) {
  const added = sky.add(source);
  await sky.initAsync();
  const encoder = root.createCommandEncoder();
  sky.bake(encoder, added);
  encoder.submit();

  const reflected = root.createMutable(d.arrayOf(d.vec3f, 36));
  const diffuse = root.createMutable(d.arrayOf(d.vec3f, 6));
  const read = tgpu.computeFn({ workgroupSize: [1] })(() => {
    'use gpu';
    for (const i of tgpu.unroll([0, 1, 2, 3, 4, 5])) {
      diffuse.$[i] = environment.irradiance(AXES[i]);
      for (const r of tgpu.unroll([0, 1, 2, 3, 4, 5])) {
        reflected.$[i * 6 + r] = environment.reflection(AXES[i], d.f32(ROUGHNESS[r]));
      }
    }
  });
  root.pipe(sky.inject()).createComputePipeline({ compute: read }).dispatchWorkgroups(1);
  return { reflected: await reflected.read(), diffuse: await diffuse.read() };
}

const uniform = () => {
  'use gpu';
  return d.vec3f(0.5, 1, 2);
};

const axis = (direction: d.v3f) => {
  'use gpu';
  return std.max(direction, d.vec3f());
};

/** Half-float bits of a small whole number. */
const half = (n: number) =>
  n === 0
    ? 0
    : ((Math.floor(Math.log2(n)) + 15) << 10) | ((n / 2 ** Math.floor(Math.log2(n)) - 1) * 1024);

const near = (actual: d.v3f, expected: readonly number[], digits = 2) => {
  for (const [i, value] of expected.entries()) expect(actual[i]).toBeCloseTo(value, digits);
};

it('bakes a uniform source into every reflection level and the diffuse light unchanged', async () => {
  const root = await gpuRoot();
  const sky = environment.create(root, { size: 32 });

  const { reflected, diffuse } = await bakeAndRead(root, sky, uniform);
  for (const value of reflected) near(value, [0.5, 1, 2]);
  for (const value of diffuse) near(value, [0.5, 1, 2]);
});

const squared = (direction: d.v3f) => {
  'use gpu';
  return d.vec3f(direction.y * direction.y);
};

const COS_10 = Math.cos((10 * Math.PI) / 180);
const cap = (direction: d.v3f) => {
  'use gpu';
  return std.select(d.vec3f(), d.vec3f(1), direction.y > COS_10);
};

it('convolves the second band of the harmonics by a quarter', async () => {
  const root = await gpuRoot();
  const { diffuse } = await bakeAndRead(root, environment.create(root, { size: 32 }), squared);
  // E / π is 1/3 + (3 n.y² − 1) / 12 for a radiance of y²
  near(diffuse[2], [0.5, 0.5, 0.5]);
  near(diffuse[0], [0.25, 0.25, 0.25]);
});

/** The share of the GGX lobe facing +y, weighted by n·l, inside the cap, at each of `ROUGHNESS`. */
const CAPPED = [1, 0.838, 0.264, 0.086, 0.0435, 0.0301];

it('blurs a small bright cap as the GGX lobe at each roughness does', async () => {
  const root = await gpuRoot();
  const { reflected } = await bakeAndRead(root, environment.create(root), cap);
  // Filtered importance sampling blurs the cap, which sits at the lobe's peak, a little further
  for (const [r, share] of CAPPED.entries()) {
    const seen = reflected[12 + r].x / share;
    expect(seen).toBeGreaterThan(0.7);
    expect(seen).toBeLessThan(1.02);
  }
});

it("dims reflections at grazing angles by Lagarde's specular occlusion in α", async () => {
  const root = await gpuRoot();
  const sky = environment.create(root, { size: 32 });
  await bakeAndRead(root, sky, uniform);
  const light = lighting.create(root);
  const facing = 0.1;
  const seen = root.createMutable(d.arrayOf(d.vec3f, 2));
  const read = tgpu.computeFn({ workgroupSize: [1] })(() => {
    'use gpu';
    const point = shading.Point({
      position: d.vec3f(),
      normal: d.vec3f(0, 1, 0),
      view: d.vec3f(std.sqrt(1 - facing * facing), facing, 0),
      uv: d.vec2f(),
      uv1: d.vec2f(),
    });
    const metal = materials.metal(d.vec3f(1), 0.3);
    seen.$[0] = environment.ambient(metal, point);
    seen.$[1] = environment.occluded(metal, point, 0.5);
  });
  root
    .pipe(sky.inject())
    .pipe(light.inject())
    .createComputePipeline({ compute: read })
    .dispatchWorkgroups(1);
  const [open, hidden] = await seen.read();
  const alpha = 0.3 * 0.3;
  const expected = Math.min(Math.max((facing + 0.5) ** (2 ** (-16 * alpha - 1)) - 0.5, 0), 1);
  expect(hidden!.x / open!.x).toBeCloseTo(expected, 3);
});

it('places directions on WebGPU cube faces and reads a panorama with its centre facing +x', async () => {
  const root = await gpuRoot();
  const sky = environment.create(root, { size: 16 });
  const { reflected } = await bakeAndRead(root, sky, axis);
  // The mirror level along +x, +y and +z
  near(reflected[0], [1, 0, 0], 1);
  near(reflected[12], [0, 1, 0], 1);
  near(reflected[24], [0, 0, 1], 1);

  // A panorama whose red counts columns and green counts rows from the top
  const pixels = new Uint16Array(16 * 8 * 4);
  for (let y = 0; y < 8; y++) {
    for (let x = 0; x < 16; x++) pixels.set([half(x), half(y), 0, 0x3c00], (y * 16 + x) * 4);
  }
  const panorama = images.create(root, { width: 16, height: 8, pixels }, { mips: false });
  const read = await bakeAndRead(root, environment.create(root, { size: 16 }), panorama);
  // +x reads the middle column, -z a quarter of the way across and +z three quarters
  expect(read.reflected[0].x).toBeCloseTo(7.5, 0);
  expect(read.reflected[30].x).toBeCloseTo(3.5, 0);
  expect(read.reflected[24].x).toBeCloseTo(11.5, 0);
  // Up reads the top row and down the bottom one
  expect(read.reflected[12].y).toBeLessThan(0.6);
  expect(read.reflected[18].y).toBeGreaterThan(6.4);
});

it("bakes the lighting's gradient into the diffuse light that shading.hemisphere gives", async () => {
  const root = await gpuRoot();
  const light = lighting.create(root, {
    settings: { skyTop: [0.3, 0.4, 0.5], skyBottom: [0.1, 0.1, 0.12], skyIntensity: 2 },
  });
  const sky = environment.create(root, { size: 32 });
  const { diffuse } = await bakeAndRead(
    root,
    sky,
    tgpu.fn(environment.gradient).pipe(light.inject()),
  );

  near(diffuse[2], [0.6, 0.8, 1]);
  near(diffuse[3], [0.2, 0.2, 0.24]);
  near(diffuse[0], [0.4, 0.5, 0.62]);
});

it('lights the ground with sunlight that the air reddens as the sun sets', () => {
  const noon = environment.sunColor(d.vec3f(0, 1, 0));
  const dusk = environment.sunColor(d.vec3f(1, 0.03, 0));
  const night = environment.sunColor(d.vec3f(1, -0.1, 0));

  expect(noon.x).toBeGreaterThan(0.9);
  expect(noon.z).toBeGreaterThan(0.75);
  expect(dusk.x / dusk.z).toBeGreaterThan(4);
  expect([...night]).toEqual([0, 0, 0]);
});

it('bakes a blue daylight sky, brighter at the horizon, that only the gradient lights at night', async () => {
  const root = await gpuRoot();
  const dark = { skyTop: [0, 0, 0], skyBottom: [0, 0, 0], sunIntensity: 1 } as const;
  const noon = lighting.create(root, { settings: { ...dark, sunDirection: [0.3, 1, 0] } });
  const day = await bakeAndRead(
    root,
    environment.create(root, { size: 32 }),
    tgpu.fn(environment.daylight).pipe(noon.inject()),
  );
  const zenith = day.reflected[12];
  const horizon = day.reflected[24];
  expect(zenith.z).toBeGreaterThan(zenith.x * 2);
  expect(horizon.x + horizon.y + horizon.z).toBeGreaterThan(zenith.x + zenith.y + zenith.z);
  // The sky lights a white matte surface facing up to a fraction of the sun
  expect(day.diffuse[2].z).toBeGreaterThan(0.05);
  expect(day.diffuse[2].z).toBeLessThan(0.5);

  const night = lighting.create(root, {
    settings: {
      sunDirection: [0, -1, 0],
      skyTop: [0.01, 0.02, 0.04],
      skyBottom: [0.01, 0.01, 0.01],
    },
  });
  const dark2 = await bakeAndRead(
    root,
    environment.create(root, { size: 32 }),
    tgpu.fn(environment.daylight).pipe(night.inject()),
  );
  near(dark2.diffuse[2], [0.01, 0.02, 0.04], 3);
});
