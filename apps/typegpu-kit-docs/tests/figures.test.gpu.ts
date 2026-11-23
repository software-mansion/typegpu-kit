import { frames } from '@typegpu/kit';
import { offscreenSurface, recordingRoot } from '@typegpu/kit/testing';
import type { CreateScene, Scene } from '@typegpu/kit/hosting';
import { gpuRoot } from '@typegpu/kit/testing/dawn';
import { linearToSrgb } from '@typegpu/color';
import { common, d, std, tgpu, type TgpuRoot } from 'typegpu';
import { describe, expect, it } from 'vitest';
import type { Loading } from 'typegpu-kit-examples/contract';
import { vec3 } from 'wgpu-matrix';
import { createGlow, rulesOf } from '../src/components/site/glow.scene.ts';
import { COLORS } from '../src/components/site/dither.ts';
import { createPrint } from '../src/components/viz/gpu.ts';
import * as bloomFigure from '../src/components/viz/Bloom.scene.ts';
import * as dominoStageFigure from '../src/components/viz/DominoStage.scene.ts';
import * as impostorsFigure from '../src/components/viz/Impostors.scene.ts';
import * as instancesFigure from '../src/components/viz/Instances.scene.ts';
import * as lightsFigure from '../src/components/viz/PointLights.scene.ts';
import * as sphereFigure from '../src/components/viz/LitSphere.scene.ts';
import * as radianceFigure from '../src/components/viz/Radiance.scene.ts';
import * as shadowFigure from '../src/components/viz/Shadow.scene.ts';
import * as posesFigure from '../src/components/viz/Poses.scene.ts';
import * as sparksFigure from '../src/components/viz/Sparks.scene.ts';
import * as textFigure from '../src/components/viz/Text.scene.ts';
import * as toneFigure from '../src/components/viz/ToneCurves.scene.ts';
import * as loupe from '../src/components/viz/loupe.ts';
import * as gridWalk from '../src/components/viz/GridWalk.program.ts';
import { fromDisk } from 'typegpu-kit-examples/testing';

// 64 RGBA8 pixels make a 256-byte row, the alignment texture copies need.
const SIZE = 64;

/** Every figure's scene, by its file; each must run with no options, or with what its `load` reads. */
const scenes = import.meta.glob<{
  createScene: CreateScene<object | undefined, Scene>;
  load?: (loading: Loading) => Promise<object>;
}>('../src/components/viz/*.scene.ts', { eager: true });

/** The pixels of an RGBA8 texture whose width is a multiple of 64. */
async function read(root: TgpuRoot, texture: GPUTexture) {
  const { device } = root;
  const { width, height } = texture;
  const buffer = device.createBuffer({
    size: width * height * 4,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const encoder = device.createCommandEncoder();
  encoder.copyTextureToBuffer({ texture }, { buffer, bytesPerRow: width * 4 }, [width, height]);
  device.queue.submit([encoder.finish()]);

  await buffer.mapAsync(GPUMapMode.READ);
  return new Uint8Array(buffer.getMappedRange().slice(0));
}

/** A scene on an offscreen surface of `width` by SIZE, with frames 1/60 s apart and a read. */
async function staged<O, S extends Scene>(
  create: CreateScene<O, S>,
  options: O,
  width = SIZE,
  features: GPUFeatureName[] = [],
) {
  const root = await gpuRoot({ features });
  const surface = offscreenSurface(root, { width, height: SIZE, format: 'rgba8unorm' });
  const scene = await create(root, surface, options);
  const loop = frames.create(root, scene.frame);
  let frame = 0;

  /** Runs a frame at `time` and reads what it drew. */
  const at = (time: number) => {
    loop.run(time);
    return read(root, surface.texture);
  };

  return {
    scene,
    at,
    /** Runs the next frame and reads what it drew. */
    draw: () => at(frame++ / 60),
    destroy() {
      loop.destroy();
      scene.destroy?.();
      root.destroy();
    },
  };
}

const sum = (pixels: Uint8Array) => pixels.reduce((total, value) => total + value, 0);

/** What an opaque picture's colours add up to: its sum without the alpha. */
const lightOf = (pixels: Uint8Array) => sum(pixels) - (pixels.length / 4) * 255;

/** The brightest green of any pixel. */
const greenest = (pixels: Uint8Array) => Math.max(...pixels.filter((_, i) => i % 4 === 1));

const pixel = (pixels: Uint8Array, x: number, y: number, width = SIZE) => [
  ...pixels.subarray((y * width + x) * 4, (y * width + x) * 4 + 4),
];

it('records the print in the scene frame, creates nothing in a steady one and frees its resources', () => {
  const { root, log } = recordingRoot();
  const surface = offscreenSurface(root, { width: SIZE, height: SIZE, format: 'rgba8unorm' });
  const before = log.alive({ uniforms: false });
  const print = createPrint(root, surface);
  const scene = sphereFigure.createScene(root, print.surface);
  const loop = frames.create(root, (frame) => {
    scene.frame(frame);
    print.draw(frame);
  });
  log.events.length = 0;
  loop.run(0);
  loop.run(1 / 60);
  const steady = log.counts();
  loop.run(2 / 60);
  expect(log.counts()).toEqual(steady);
  expect(log.events.filter((event) => event === 'submit')).toHaveLength(3);
  loop.destroy();
  scene.destroy();
  print.destroy();
  expect(log.alive({ uniforms: false })).toEqual(before);
  root.destroy();
});

it('prints display colour at seven levels a channel over the ground, in CSS-pixel cells, keeping clear pixels clear', async () => {
  const root = await gpuRoot();
  const surface = offscreenSurface(root, {
    width: SIZE,
    height: SIZE,
    pixelRatio: 2,
    format: 'rgba8unorm',
  });
  const print = createPrint(root, surface, 2);
  const fragment = tgpu.fragmentFn({ in: { uv: d.vec2f }, out: d.vec4f })(({ uv }) => {
    'use gpu';
    if (uv.y < 0.25) return d.vec4f();
    if (uv.y < 0.5) return d.vec4f(0, 0, 0, 1);
    const color = std.select(d.vec3f(uv.x), d.vec3f(0.1, 0.85, 0.95) * uv.x, uv.y > 0.75);
    return d.vec4f(color, 1);
  });
  const drawing = root.createRenderPipeline({
    vertex: common.fullScreenTriangle,
    fragment,
    targets: { format: print.surface.format },
  });
  const loop = frames.create(root, (frame) => {
    const pass = frame.encoder.beginRenderPass({
      colorAttachments: { view: print.surface.view(frame) },
    });
    pass.setPipeline(drawing);
    pass.draw(3);
    pass.end();
    print.draw(frame);
  });
  loop.run(0);
  const pixels = await read(root, surface.texture);
  const ground = COLORS.grey[0].map((v) => Math.round(v * 255));
  const levels = Array.from({ length: 7 }, (_, k) => Math.round((k / 6) * 255));
  expect(pixel(pixels, 0, 0)[3]).toBe(0);
  expect(pixel(pixels, 0, 24)).toEqual([...ground, 255]);
  for (let y = 32; y < SIZE; y++) {
    for (let x = 0; x < SIZE; x++) {
      const color = pixel(pixels, x, y);
      for (const [i, v] of color.slice(0, 3).entries()) {
        expect([ground[i], ...levels].some((level) => Math.abs(level - v) <= 1)).toBe(true);
      }
      expect(color).toEqual(pixel(pixels, x - (x % 4), y - (y % 4)));
    }
  }
  expect(
    Array.from({ length: SIZE }, (_, x) => pixel(pixels, x, 56)).some(
      (rgba) => rgba[1] > rgba[0] + 64,
    ),
  ).toBe(true);
  loop.destroy();
  print.destroy();
  root.destroy();
});

/** The pixels of a SIZE by SIZE square whose left edge is at `left`. */
const square = (pixels: Uint8Array, width: number, left: number) => {
  const out = new Uint8Array(SIZE * SIZE * 4);
  for (let y = 0; y < SIZE; y++) {
    out.set(pixels.subarray((y * width + left) * 4, (y * width + left + SIZE) * 4), y * SIZE * 4);
  }
  return out;
};

// Culled instances draw through one pipeline with indirect-first-instance, which nearly every
// device has for a program that asks, and through one that reads each draw's mesh from a bind group
// without it.
const devices: [string, GPUFeatureName[]][] = [
  ['with indirect-first-instance', ['indirect-first-instance']],
  ['without it', []],
];

describe.each(devices)('%s', (_name, features) => {
  it.each(Object.entries(scenes))('runs the frames of %s', async (_path, { createScene, load }) => {
    const options = await load?.(fromDisk);
    const { draw, destroy } = await staged(createScene, options, SIZE, features);
    for (let i = 0; i < 3; i++) await draw();
    destroy();
  });
});

it('draws a sphere of each preset in its square, and drops the highlights under lambert', async () => {
  const { PRESETS } = sphereFigure;
  const width = SIZE * PRESETS.length;
  const { scene, draw, destroy } = await staged(sphereFigure.createScene, undefined, width);
  scene.set(0.2, 0);

  const pbr = await draw();
  scene.reflect('lambert');
  const lambert = await draw();
  const spheres = (pixels: Uint8Array) => PRESETS.map((_, i) => square(pixels, width, i * SIZE));
  const [lit, flat] = [spheres(pbr), spheres(lambert)];

  for (const pixels of [...lit, ...flat]) {
    expect(pixel(pixels, 0, 0)).toEqual([0, 0, 0, 0]);
    expect(pixel(pixels, SIZE / 2, SIZE / 2)[3]).toBe(255);
  }

  // The glossy sphere loses its highlight; the glow reflects almost nothing either way.
  const glossy = PRESETS.indexOf('glossy');
  const glow = PRESETS.indexOf('glow');
  expect(greenest(lit[glossy])).toBeGreaterThan(greenest(flat[glossy]) + 40);
  expect(Math.abs(sum(lit[glow]) - sum(flat[glow]))).toBeLessThan(sum(lit[glow]) * 0.01);
  expect(pixel(lit[glow], SIZE / 2, SIZE / 2)[1]).toBeGreaterThan(200);

  // Laid out again, the glow and the matte sphere swap squares.
  const swapped = PRESETS.map((_, i) => (i === glow ? 0 : i === 0 ? glow : i));
  scene.layout(
    swapped.map((at): sphereFigure.Rect => [at / PRESETS.length, 0, 1 / PRESETS.length, 1]),
  );
  scene.reflect('pbr');
  const moved = spheres(await draw());
  for (const [i, at] of swapped.entries()) {
    expect(Math.abs(sum(moved[at]) - sum(lit[i]))).toBeLessThan(sum(lit[i]) * 0.001);
  }

  destroy();
});

it('draws a strip per tone curve, as the CPU computes it, with gaps between them', async () => {
  const { STRIPS, CURVES, FILL, LOW, HIGH, HUES } = toneFigure;
  const { scene, draw, destroy } = await staged(toneFigure.createScene, undefined);
  scene.set(HUES.blue, 1);

  const pixels = await draw();
  const rows = STRIPS.length - 1 + FILL;

  STRIPS.forEach((tone, row) => {
    const y = Math.floor(((row + FILL / 2) / rows) * SIZE);
    const x = SIZE / 2;

    // An 8 by 8 window holds every threshold of the Bayer screen once, so it averages to the colour
    const mean = (of: (x: number, y: number) => number) => {
      let total = 0;
      for (let dy = -4; dy < 4; dy++) for (let dx = -4; dx < 4; dx++) total += of(x + dx, y + dy);
      return total / 64;
    };
    for (const channel of [0, 1, 2] as const) {
      const printed = mean((px, py) => pixel(pixels, px, py)[channel] / 255);
      const curve = mean((px) => {
        const light = 2 ** (LOW + (HIGH - LOW) * ((px + 0.5) / SIZE));
        const [red, green, blue] = HUES.blue;
        const shown = linearToSrgb(CURVES[tone](d.vec3f(red * light, green * light, blue * light)));
        return Math.min([shown.x, shown.y, shown.z][channel], 1);
      });
      expect(printed).toBeCloseTo(curve, 1);
    }
  });

  const gap = Math.floor(((FILL + 1) / 2 / rows) * SIZE);
  expect(pixel(pixels, SIZE / 2, gap)[3]).toBe(0);

  destroy();
});

it('draws the shapes and their shadows, which move with the sun', async () => {
  const { scene, draw, destroy } = await staged(shadowFigure.createScene, undefined);

  scene.aim(0.6);
  const before = await draw();
  scene.remake(256, 1);
  const remade = await draw();
  scene.aim(2.4);
  const after = await draw();

  expect(pixel(before, SIZE / 2, SIZE / 2)[3]).toBe(255);
  expect(Math.abs(sum(remade) - sum(before))).toBeLessThan(sum(before) * 0.02);
  expect(after).not.toEqual(remade);

  destroy();
});

it('lights the floor from the glows it counts, at any size of field, and none without them', async () => {
  const { MAX } = radianceFigure;
  const { scene, draw, destroy } = await staged(radianceFigure.createScene, { motion: false });

  const dark = await draw();
  scene.count(MAX);
  const lit = await draw();
  scene.resize(64);
  const coarse = await draw();
  scene.count(0);
  const cleared = await draw();

  expect(lightOf(lit)).toBeGreaterThan(lightOf(dark) * 4);
  expect(Math.abs(lightOf(coarse) - lightOf(lit))).toBeLessThan(lightOf(lit) * 0.25);
  expect(cleared).toEqual(dark);

  destroy();
});

it('draws the sparks it bursts onto its ground, and clears them', async () => {
  const { CLEAR } = sparksFigure;
  const { scene, draw, destroy } = await staged(sparksFigure.createScene, { capacity: 256 });

  const empty = await draw();
  expect(scene.burst(0, 0, 400, 1)).toBe(true);
  const lit = await draw();
  scene.clear();
  const cleared = await draw();

  // The first burst is blue, and its sparks have moved a fraction of a pixel from the centre.
  const ground = CLEAR.map((channel) => Math.round(channel * 255));
  expect(pixel(empty, 0, 0)).toEqual(ground);
  expect(pixel(empty, SIZE / 2, SIZE / 2)).toEqual(ground);
  expect(scene.stats.born).toBe(256);
  expect(pixel(lit, SIZE / 2, SIZE / 2)[2]).toBeGreaterThan(ground[2] + 64);
  expect(pixel(lit, 0, 0)).toEqual(ground);
  expect(cleared).toEqual(empty);

  destroy();
});

it('adds the glow of the discs past the threshold, and none past every disc', async () => {
  const { DISCS } = bloomFigure;
  const { scene, draw, destroy } = await staged(bloomFigure.createScene, {});

  scene.set({ threshold: 1, glowIntensity: 0 });
  const plain = await draw();
  scene.set({ glowIntensity: 0.5 });
  const glowing = await draw();
  scene.set({ threshold: 2 ** DISCS });
  const above = await draw();
  scene.set({ threshold: 1 });
  scene.remake(1);
  const close = await draw();

  // Beside the brightest disc, at the right edge, and halfway down.
  const blue = pixel(plain, SIZE - 1, SIZE / 2)[2];
  expect(pixel(glowing, SIZE - 1, SIZE / 2)[2]).toBeGreaterThan(blue! + 8);
  expect(above).toEqual(plain);
  expect(close).not.toEqual(glowing);

  destroy();
});

it('draws the instances it counts, and moves every one', async () => {
  const { MAX } = instancesFigure;
  const { scene, draw, destroy } = await staged(instancesFigure.createScene, {});

  const empty = await draw();
  scene.count(MAX);
  const full = await draw();
  const moved = await draw();
  scene.count(0);
  const cleared = await draw();

  expect(pixel(full, SIZE / 2, SIZE / 2)).not.toEqual(pixel(empty, SIZE / 2, SIZE / 2));
  expect(moved).not.toEqual(full);
  expect(cleared).toEqual(empty);

  destroy();
});

it('lights the scene from the point lights it counts, moves them, and reaches less far with less range', async () => {
  const { MAX } = lightsFigure;
  const { scene, draw, destroy } = await staged(lightsFigure.createScene, undefined);

  scene.count(0);
  const dark = await draw();
  scene.count(MAX);
  const lit = await draw();
  const moved = await draw();
  scene.range(1);
  const near = await draw();
  scene.count(0);
  const cleared = await draw();

  expect(lightOf(lit)).toBeGreaterThan(lightOf(dark) * 1.5);
  expect(moved).not.toEqual(lit);
  expect(lightOf(near)).toBeLessThan(lightOf(moved) * 0.8);
  expect(lightOf(near)).toBeGreaterThan(lightOf(dark));
  expect(cleared).toEqual(dark);

  destroy();
});

/** A glow marker on a SIZE square, drawn 4 s in over a clear ground beside a lit entry. */
async function marked() {
  const root = await gpuRoot();
  const surface = offscreenSurface(root, { width: SIZE, height: SIZE, format: 'rgba8unorm' });
  const glow = createGlow(root, surface.format);
  const marker = glow.marker();
  const loop = frames.create(root, (frame) => {
    glow.time.write(4);
    const pass = frame.encoder.beginRenderPass({
      colorAttachments: { view: surface.view(frame), clearValue: [0, 0, 0, 0] },
    });
    marker.draw(pass);
    pass.end();
  });

  return {
    marker,
    /** Lights the entry at `strength` in cyan, then draws a frame and reads it. */
    async draw(entry: d.v4f, strength: number) {
      marker.box.write({
        color: [0, 1, 1, 1],
        before: entry,
        after: entry,
        strength: [strength, strength],
        start: -10,
        ratio: 1,
      });
      loop.run(0);
      return read(root, surface.texture);
    },
    destroy() {
      glow.destroy();
      root.destroy();
    },
  };
}

it('lights grains of smoke where the light comes in, fewer across, and none off its entry', async () => {
  const { draw, destroy } = await marked();
  const pixels = await draw(d.vec4f(0, 8, SIZE, 24), 1);

  // Grains are cyan at 0.22 alpha or more, premultiplied; the wash under them stays below 0.08.
  const grains = (from: number, to: number) => {
    let count = 0;
    for (let y = 8; y < 24; y++) {
      for (let x = from; x < to; x++) if (pixel(pixels, x, y)[3] >= 50) count++;
    }
    return count;
  };

  expect(grains(0, SIZE / 2)).toBeGreaterThan(grains(SIZE / 2, SIZE));
  expect(grains(0, SIZE / 4)).toBeGreaterThan(0);
  expect(pixel(pixels, 8, 16)[0]).toBe(0);
  expect(sum(pixels.subarray(0, 8 * SIZE * 4))).toBe(0);
  expect(sum(pixels.subarray(24 * SIZE * 4))).toBe(0);

  destroy();
});

it("dots a section's rule, filled in and in the light's colour beside its lit entry", async () => {
  const { marker, draw, destroy } = await marked();
  marker.rules.write(rulesOf([1, 0, 0, 0.5], [d.vec4f(4.5, 0, 56, 0)]));
  const drawLit = async (strength: number) => {
    const pixels = await draw(d.vec4f(8, 4, SIZE, 12), strength);
    return (from: number, to: number, x = 4) =>
      Array.from({ length: to - from }, (_, i) => pixel(pixels, x, from + i));
  };

  // Beside the lit entry every dot is on, in the light's cyan.
  const lit = await drawLit(1);
  expect(lit(4, 12).every((rgba) => rgba[3] === 255 && rgba[1] > rgba[0])).toBe(true);

  // Unlit, every other dot is on, in the rules' red at half alpha, down to the rule's end.
  const unlit = await drawLit(0);
  const dots = unlit(0, 56).filter((rgba) => rgba[3] > 0);
  expect(dots.length).toBe(28);
  expect(dots.every((rgba) => rgba[1] === 0 && Math.abs(rgba[3] - 128) <= 1)).toBe(true);
  expect(unlit(57, SIZE).every((rgba) => rgba[3] === 0)).toBe(true);
  expect([...unlit(0, SIZE, 3), ...unlit(0, SIZE, 5)].every((rgba) => rgba[3] === 0)).toBe(true);

  destroy();
});

it('outlines a glyph within reach, and fills its whole quad past it', async () => {
  const options = await textFigure.load(fromDisk);
  const { scene, draw, destroy } = await staged(textFigure.createScene, options);
  const covered = async (shade: textFigure.Shade, width: number) => {
    scene.shade(shade);
    scene.widen(width);
    const pixels = await draw();
    return pixels.filter((value, i) => i % 4 === 3 && value > 0).length;
  };

  // 20 pixels to the em, with the shipped bake's 40 texels an em and a range of 8.
  const em = 20;
  scene.write('o');
  scene.resize(em);
  expect(scene.reach).toBe(2);
  const [left, bottom, right, top] = options.font.data.glyphs.o!.plane!;
  const quad = (right - left) * (top - bottom) * em * em;

  const inked = await covered('fill', 1);
  const outlined = await covered('outline', 1);
  expect(inked).toBeLessThan(outlined);
  expect(outlined).toBeLessThan(quad * 0.85);
  expect(await covered('outline', 3)).toBeGreaterThan(quad * 0.95);
  expect(await covered('round', 1)).toBeGreaterThan(quad * 0.95);

  destroy();
});

it('draws the tree as its meshes and as its impostor alike, and lights the views it reads', async () => {
  const { IMPOSTOR_AT, viewsRead } = impostorsFigure;
  const width = 2 * SIZE;
  const { scene, draw, destroy } = await staged(impostorsFigure.createScene, undefined, width);
  // Straight between the two, so each stands as far from the eye.
  scene.view.goal.yaw = 0;

  /** How many pixels in columns `from` to `to` are the leaves' green. */
  const leaves = (pixels: Uint8Array, from: number, to: number) => {
    let count = 0;
    for (let y = 0; y < SIZE; y++) {
      for (let x = from; x < to; x++) {
        const [r, g, b] = pixel(pixels, x, y, width);
        if (g! > r! + 8 && g! > b! + 8) count++;
      }
    }
    return count;
  };

  const blended = await draw();
  const meshes = leaves(blended, 0, SIZE);
  expect(meshes).toBeGreaterThan(80);
  expect(Math.abs(leaves(blended, SIZE, width) - meshes)).toBeLessThan(meshes * 0.15);

  await scene.read('nearest');
  expect(await draw()).not.toEqual(blended);

  // Four views a side: each a cell of 16 pixels in the square at the right.
  await scene.capture(4);
  scene.showAtlas(true);
  const shown = await draw();
  const cell = SIZE / 4;
  const brightness = (x: number, y: number) =>
    Math.max(...pixel(shown, SIZE + x, y, width).slice(0, 3));
  const background = brightness(0, 0);
  let brightest = { value: 0, x: 0, y: 0 };
  const covered = Array.from({ length: 16 }, () => 0);
  for (let y = 0; y < SIZE; y++) {
    for (let x = 0; x < SIZE; x++) {
      const value = brightness(x, y);
      if (value > brightest.value) brightest = { value, x, y };
      if (value > background + 6) covered[Math.floor(y / cell) * 4 + Math.floor(x / cell)]++;
    }
  }
  // Every view shows the tree.
  for (const count of covered) expect(count).toBeGreaterThan(cell * cell * 0.05);

  // The one view the impostor reads is lit, and none other is as bright.
  const { yaw, pitch, distance, target } = scene.view.pose;
  const eye = [
    target[0] + distance * Math.cos(pitch) * Math.sin(yaw),
    target[1] + distance * Math.sin(pitch),
    target[2] + distance * Math.cos(pitch) * Math.cos(yaw),
  ];
  const toward = vec3.normalize(vec3.subtract(eye, vec3.add(IMPOSTOR_AT, scene.subject.center)));
  const { layer } = viewsRead(toward, 4, 'nearest')[0];
  for (let y = 0; y < SIZE; y++) {
    for (let x = 0; x < SIZE; x++) {
      const inside = Math.floor(y / cell) * 4 + Math.floor(x / cell) === layer;
      if (!inside) expect(brightness(x, y)).toBeLessThan(brightest.value * 0.75);
    }
  }
  expect(Math.floor(brightest.y / cell) * 4 + Math.floor(brightest.x / cell)).toBe(layer);

  destroy();
});

it('plays one pose on both: a walk moves them, speed 0 holds them, and death holds its ends', async () => {
  const options = await posesFigure.load(fromDisk);
  const death = options.models['skeleton-clips'].clips.find(
    (clip) => clip.name === 'Skeletons_Death',
  )!.duration;
  const { scene, at, destroy } = await staged(posesFigure.createScene, options, SIZE, [
    'indirect-first-instance',
  ]);

  const idle = await at(0);
  scene.play('walk', 0);
  const walking = await at(0.5);
  expect(walking).not.toEqual(idle);
  expect(await at(0.75)).not.toEqual(walking);

  scene.speed(0);
  const held = await at(1);
  expect(await at(1.5)).toEqual(held);

  // Death's first frame, then its last, held, then played back to its first.
  scene.speed(1);
  scene.play('death', 0);
  const standing = await at(1.5);
  const fallen = await at(1.5 + death + 0.2);
  expect(fallen).not.toEqual(standing);
  expect(await at(1.5 + death + 1)).toEqual(fallen);

  scene.speed(-1);
  expect(await at(2.5 + 2 * death + 0.5)).toEqual(standing);

  destroy();
});

it.each(dominoStageFigure.STAGES)('draws the %s stage of the domino run', async (stage) => {
  const { draw, destroy } = await staged(dominoStageFigure.createScene, { stage });
  expect(lightOf(await draw())).toBeGreaterThan(0);
  destroy();
});

it("darkens the floor with the run's shadow", async () => {
  const lit = await staged(dominoStageFigure.createScene, { stage: 'run' });
  const run = lightOf(await lit.draw());
  lit.destroy();
  const shaded = await staged(dominoStageFigure.createScene, { stage: 'shadow' });
  expect(lightOf(await shaded.draw())).toBeLessThan(run);
  shaded.destroy();
});

it('topples the run after its push', async () => {
  const { draw, destroy } = await staged(dominoStageFigure.createScene, { stage: 'falling' });
  // The push comes 0.6 s in; by 1.5 s, the wave has run some 20 dominoes.
  for (let frame = 0; frame < 30; frame++) await draw();
  const standing = await draw();
  for (let frame = 0; frame < 60; frame++) await draw();
  expect(await draw()).not.toEqual(standing);
  destroy();
});

it('magnifies the square around a point as blocks of its pixels, kept inside the source', async () => {
  const root = await gpuRoot();
  const source = offscreenSurface(root, { width: SIZE, height: SIZE, format: 'rgba8unorm' });
  const target = offscreenSurface(root, { width: SIZE, height: SIZE, format: 'rgba8unorm' });
  // Each pixel's x and y, times 4, in red and green
  const coordinates = tgpu.fragmentFn({ in: { position: d.builtin.position }, out: d.vec4f })(({
    position,
  }) => {
    'use gpu';
    return d.vec4f((std.floor(position.xy) * 4) / 255, 0, 1);
  });
  const pattern = root.createRenderPipeline({
    vertex: common.fullScreenTriangle,
    fragment: coordinates,
    targets: { format: source.format },
  });
  const lens = loupe.create(root, source.format, 16);

  /** What the loupe draws around `x`, `y`, as a pixel lookup. */
  const magnify = async (x: number, y: number) => {
    const loop = frames.create(root, (frame) => {
      const pass = frame.encoder.beginRenderPass({
        colorAttachments: { view: source.view(frame) },
      });
      pass.setPipeline(pattern);
      pass.draw(3);
      pass.end();
      lens.draw(frame, source.texture, target, x, y);
    });
    loop.run(0);
    loop.destroy();
    const pixels = await read(root, target.texture);
    return (px: number, py: number) => pixel(pixels, px, py);
  };

  // 16 pixels across 64, so each is a block of 4 by 4, from the corner at 32, 12
  const middle = await magnify(40, 20);
  for (const [x, y] of [
    [0, 0],
    [3, 3],
    [4, 0],
    [37, 61],
    [63, 63],
  ] as const) {
    expect(middle(x, y)).toEqual([
      (32 + Math.floor(x / 4)) * 4,
      (12 + Math.floor(y / 4)) * 4,
      0,
      255,
    ]);
  }

  // Near the edge, the square stays inside the source
  const corner = await magnify(63, 0);
  expect(corner(0, 0)).toEqual([48 * 4, 0, 0, 255]);
  expect(corner(63, 63)).toEqual([63 * 4, 15 * 4, 0, 255]);

  lens.destroy();
  root.destroy();
});

it('walks each item past every neighbour once, and leans no way where a corner-first walk leans', async () => {
  const root = await gpuRoot();
  const sparse = await gridWalk.createProgram({ root });
  const walk = await sparse.walk({ count: 90, stop: 5, seed: 3 });
  expect(walk.sorted.toSorted((a, b) => a - b)).toEqual([...Array(90).keys()]);
  for (let i = 0; i < walk.count; i++) expect(walk.visits[i]).toContain(i);
  sparse.destroy();

  const cols = 10;
  const rows = 6;
  const dense = await gridWalk.createProgram({
    root,
    cols,
    rows,
    capacity: 24 * cols * rows,
    visits: 1,
  });
  const stopped = await dense.walk({ stop: 8 });
  for (const axis of stopped.mean) expect(Math.abs(axis)).toBeLessThan(0.02);
  for (const axis of stopped.corner) expect(axis).toBeLessThan(-0.1);
  dense.destroy();
  root.destroy();
});
