import { cameras, passes, targets } from '@typegpu/kit';
import { gpuRoot } from '@typegpu/kit/testing/dawn';
import { d, std, tgpu, type TgpuRoot } from 'typegpu';
import { expect, it } from 'vitest';
import { font, typeset, text, type Font, type Place, type TextSet } from '../../src/index.ts';
import { inter } from './inter.ts';

const WIDTH = 128;
const HEIGHT = 64;
const SIZE = 40;
const WORD = 'Fly';

/** Where the word's first baseline starts, in pixels from the top-left. */
const ANCHOR = [10, 44] as const;

/** The anchor at the left of the word's baseline, where the checks measure from. */
const LEFT = { align: 'left', baseline: 'alphabetic' } as const;

/** An 8-bit texture's pixels, four bytes each. */
async function readPixels(root: TgpuRoot, texture: GPUTexture) {
  const readback = root.device.createBuffer({
    size: WIDTH * 4 * HEIGHT,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const copy = root.device.createCommandEncoder();
  copy.copyTextureToBuffer({ texture }, { buffer: readback, bytesPerRow: WIDTH * 4 }, [
    WIDTH,
    HEIGHT,
  ]);
  root.device.queue.submit([copy.finish()]);
  await readback.mapAsync(GPUMapMode.READ);

  return new Uint8Array(readback.getMappedRange().slice(0));
}

/** A target per device and pixel ratio, and passes that draw into it. */
const targetsOf = new WeakMap<TgpuRoot, Map<number, ReturnType<typeof described>>>();
const described = (root: TgpuRoot, pixelRatio: number) => {
  const screen = targets.create(
    root,
    { width: WIDTH, height: HEIGHT, pixelRatio },
    { color: { format: 'rgba8unorm' } },
  );
  return { screen, into: passes.create(root, screen, { color: passes.color }) };
};
const intoOf = (root: TgpuRoot, pixelRatio: number) => {
  const made = targetsOf.get(root) ?? new Map<number, ReturnType<typeof described>>();
  targetsOf.set(root, made);
  const into = made.get(pixelRatio) ?? described(root, pixelRatio);
  made.set(pixelRatio, into);
  return into;
};

/** Draws `labels` alone into a cleared target of `pixelRatio` and reads it back. */
async function drawn(root: TgpuRoot, labels: TextSet, pixelRatio = 1) {
  const { screen, into } = intoOf(root, pixelRatio);
  const encoder = root.createCommandEncoder();
  const words = into.add(labels.drawing);
  const pass = into.begin(encoder);
  words.draw(pass);
  pass.end();
  encoder.submit();

  return readPixels(root, root.unwrap(screen.textures.color));
}

/** The atlas's median at a uv, filtered as a linear sampler filters it, from 0 to 1. */
function median(u: number, v: number) {
  const { width, height } = inter.data.atlas;
  const x = u * width - 0.5;
  const y = v * height - 0.5;
  const x0 = Math.floor(x);
  const y0 = Math.floor(y);

  const channel = (c: number) => {
    const at = (tx: number, ty: number) =>
      inter.atlas[
        (Math.min(Math.max(ty, 0), height - 1) * width + Math.min(Math.max(tx, 0), width - 1)) * 4 +
          c
      ] / 255;
    const top = at(x0, y0) * (x0 + 1 - x) + at(x0 + 1, y0) * (x - x0);
    const bottom = at(x0, y0 + 1) * (x0 + 1 - x) + at(x0 + 1, y0 + 1) * (x - x0);
    return top * (y0 + 1 - y) + bottom * (y - y0);
  };

  const [r, g, b] = [channel(0), channel(1), channel(2)];
  return Math.max(Math.min(r, g), Math.min(Math.max(r, g), b));
}

/** A uv held half a texel inside a glyph's cell, from `from` to `to` texels, as the shader holds it. */
const clamp = (value: number, from: number, to: number, size: number) =>
  Math.min(Math.max(value, (from + 0.5) / size), (to - 0.5) / size);

/**
 * The word's glyphs, sampled on the CPU at each pixel's centre as the shader samples them: the
 * atlas's median, the greatest where quads overlap, or -1 where none lies.
 */
function reference(sans: Font, size = SIZE) {
  const laid = typeset.create(8).set(sans, WORD, { em: size, ...LEFT });
  const { width, height } = inter.data.atlas;
  const glyphs = Object.values(inter.data.glyphs);
  const medians = new Float32Array(WIDTH * HEIGHT).fill(-1);

  for (let i = 0; i < laid.count; i++) {
    const { plane, atlas } = glyphs[laid.glyphs[i]];
    const [left, bottom, right, top] = plane!;
    const [l, t, r, b] = atlas!;

    for (let py = 0; py < HEIGHT; py++) {
      for (let px = 0; px < WIDTH; px++) {
        // The pixel's offset from the anchor in em, with y up, and where it falls in the quad.
        const x = (px + 0.5 - ANCHOR[0] - laid.pens[i * 2]) / size;
        const y = (ANCHOR[1] - py - 0.5 - laid.pens[i * 2 + 1]) / size;
        const across = (x - left) / (right - left);
        const up = (y - bottom) / (top - bottom);
        // Pixel centres on a quad's edge may or may not rasterise, so neither side counts them
        if (across <= 0 || across >= 1 || up <= 0 || up >= 1) continue;

        const u = (l + (r - l) * across) / width;
        const v = (b + (t - b) * up) / height;
        const at = median(clamp(u, l, r, width), clamp(v, t, b, height));
        medians[py * WIDTH + px] = Math.max(medians[py * WIDTH + px], at);
      }
    }
  }

  return medians;
}

/**
 * A set of one label in `place`, its anchor at `position`, and the word written in white, `em`
 * to the em.
 */
function word(
  root: TgpuRoot,
  camera: cameras.CustomCamera,
  sans: Font,
  place: Place,
  position: readonly number[],
  { em = SIZE, shade }: Partial<Pick<text.Options, 'shade'>> & { em?: number } = {},
) {
  const labels = text.create(root, sans, {
    camera: camera,
    capacity: 2,
    length: 8,
    place,
    ...(shade && { shade }),
  });
  labels.labels.patch({ 0: { position: [position[0], position[1], position[2]] } });
  labels.write(0, WORD, { em, ...LEFT });
  return labels;
}

/** The docs' outline: two pixels of the rounded distance around the fill, or what it reaches. */
const outlined = (fragment: text.Fragment) => {
  'use gpu';
  const width = std.min(2, fragment.reach - 0.5);
  const fill = std.saturate(fragment.distance + 0.5);
  const outline = std.saturate(fragment.round + width + 0.5) * fragment.color.w;
  return d.vec4f(fragment.color.xyz * fill * outline, outline);
};

/** The outline the docs showed before, two pixels whatever the size. */
const fixedOutline = (fragment: text.Fragment) => {
  'use gpu';
  const fill = std.saturate(fragment.distance + 0.5);
  const outline = std.saturate(fragment.round + 2.5) * fragment.color.w;
  return d.vec4f(fragment.color.xyz * fill * outline, outline);
};

/** Column-major matrices: an orthographic projection with one world unit to the pixel. */
const ortho = (near: number, far: number) =>
  new Float32Array([
    2 / WIDTH,
    0,
    0,
    0,
    0,
    2 / HEIGHT,
    0,
    0,
    0,
    0,
    -1 / (far - near),
    0,
    -1,
    1,
    -near / (far - near),
    1,
  ]);

/** Looking down -z from z = 10, x right and y up: pixel (x, y) is world (x, -y, 0). */
const front = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, -10, 1]);

/** Looking down -y from y = 10, x right and -z up: pixel (x, y) is world (x, 0, y). */
const above = new Float32Array([1, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, -10, 1]);

it('covers the pixels the atlas says the glyphs cover, and no others', async () => {
  const root = await gpuRoot();
  const sans = font.create(root, inter);
  const camera = cameras.custom(root);
  const labels = word(root, camera, sans, text.screen, [ANCHOR[0], ANCHOR[1], 0.5]);

  const pixels = await drawn(root, labels);
  const medians = reference(sans);
  const alpha = (i: number) => pixels[i * 4 + 3];

  const inside = [...medians.keys()].filter((i) => medians[i] > 0.62);
  const outside = [...medians.keys()].filter((i) => medians[i] < 0.38);
  expect(inside.length).toBeGreaterThan(300);
  expect(inside.filter((i) => alpha(i) < 200)).toEqual([]);
  expect(outside.filter((i) => alpha(i) > 55)).toEqual([]);

  // White, premultiplied.
  expect(inside.filter((i) => pixels[i * 4] !== alpha(i))).toEqual([]);

  // The F's bar is at its top and its stem at its left: the glyph is neither flipped nor mirrored.
  const lit = (x: number, y: number) => alpha(y * WIDTH + x) > 128;
  const rows = [...Array(HEIGHT).keys()].filter((y) =>
    [...Array(30).keys()].some((x) => lit(x, y)),
  );
  const [top, bottom] = [rows[0], rows.at(-1)!];
  expect(bottom).toBeGreaterThanOrEqual(ANCHOR[1] - 2);
  expect(bottom).toBeLessThan(ANCHOR[1] + 1);
  const stem = [...Array(30).keys()].find((x) => lit(x, bottom))!;
  expect(stem).toBeLessThan(ANCHOR[0] + 6);
  expect(lit(stem + 16, top + 1)).toBe(true);
  expect(lit(stem + 16, bottom - 1)).toBe(false);
  root.destroy();
});

it('lands the same pixels from the camera, facing it, pinned and on the ground', async () => {
  const root = await gpuRoot();
  const sans = font.create(root, inter);
  const camera = cameras.custom(root);
  const [x, y] = ANCHOR;

  camera.write(front, ortho(1, 20));
  const onScreen = await drawn(root, word(root, camera, sans, text.screen, [x, y, 0.5]));
  const facing = await drawn(root, word(root, camera, sans, text.facing, [x, -y, 0]));
  const pinned = await drawn(root, word(root, camera, sans, text.pinned, [x, -y, 0]));

  camera.write(above, ortho(1, 20));
  const ground = await drawn(root, word(root, camera, sans, text.ground, [x, 0, y]));

  const differences = (pixels: Uint8Array) =>
    pixels.filter((value, i) => i % 4 === 3 && Math.abs(value - onScreen[i]) > 24).length;

  expect(onScreen.filter((value, i) => i % 4 === 3 && value > 128).length).toBeGreaterThan(400);
  expect(differences(facing)).toBe(0);
  expect(differences(pinned)).toBe(0);
  expect(differences(ground)).toBe(0);
  root.destroy();
});

it('sizes screen and pinned text in points, times the pixel ratio of the pass', async () => {
  const root = await gpuRoot();
  const sans = font.create(root, inter);
  const camera = cameras.custom(root);
  const [x, y] = ANCHOR;
  camera.write(front, ortho(1, 20));

  const pixels = await drawn(root, word(root, camera, sans, text.screen, [x, y, 0.5]));
  const halves = { em: SIZE / 2 };
  const onScreen = await drawn(root, word(root, camera, sans, text.screen, [x, y, 0.5], halves), 2);
  const pinned = await drawn(root, word(root, camera, sans, text.pinned, [x, -y, 0], halves), 2);

  const differences = (drawnAt: Uint8Array) =>
    drawnAt.filter((value, i) => i % 4 === 3 && Math.abs(value - pixels[i]) > 24).length;
  expect(pixels.filter((value, i) => i % 4 === 3 && value > 128).length).toBeGreaterThan(400);
  expect(differences(onScreen)).toBe(0);
  expect(differences(pinned)).toBe(0);
  root.destroy();
});

it('outlines small text only as far as the atlas reaches, where a fixed width boxes each glyph', async () => {
  const root = await gpuRoot();
  const sans = font.create(root, inter);
  const camera = cameras.custom(root);
  const small = 10; // a reach of one pixel with the shipped bake
  const at = [ANCHOR[0], ANCHOR[1], 0.5] as const;

  const options = { em: small, shade: outlined };
  const scaled = await drawn(root, word(root, camera, sans, text.screen, at, options));
  const fixed = await drawn(
    root,
    word(root, camera, sans, text.screen, at, { ...options, shade: fixedOutline }),
  );

  // Inside a glyph's quad, where every texel around is farther from the edge than the atlas spans.
  const medians = reference(sans, small);
  const far = [...medians.keys()].filter((i) => medians[i] === 0);
  expect(far.length).toBeGreaterThan(20);
  expect(far.filter((i) => scaled[i * 4 + 3] > 8)).toEqual([]);
  expect(far.filter((i) => fixed[i * 4 + 3] < 128)).toEqual([]);

  // The outline still draws around the glyphs.
  const ink = [...medians.keys()].filter((i) => medians[i] > 0.62);
  expect(ink.filter((i) => scaled[i * 4 + 3] < 200)).toEqual([]);
  root.destroy();
});

it('moves and recolours a label with patch, and clears what shorter text leaves', async () => {
  const root = await gpuRoot();
  const sans = font.create(root, inter);
  const camera = cameras.custom(root);
  const labels = word(root, camera, sans, text.screen, [ANCHOR[0], ANCHOR[1], 0.5]);
  const before = await drawn(root, labels);

  labels.labels.patch({ 0: { position: [ANCHOR[0] + 40, ANCHOR[1], 0.5], color: [1, 0, 0, 1] } });
  const moved = await drawn(root, labels);

  const shifted = [...Array(WIDTH * HEIGHT).keys()].filter((i) => {
    const x = i % WIDTH;
    const was = x >= 40 ? before[(i - 40) * 4 + 3] : 0;
    return Math.abs(moved[i * 4 + 3] - was) > 24;
  });
  expect(shifted).toEqual([]);
  expect(moved.filter((value, i) => i % 4 === 1 && value > 0).length).toBe(0);
  expect(moved.filter((value, i) => i % 4 === 0 && value > 128).length).toBeGreaterThan(400);

  labels.write(0, '', { em: SIZE });
  expect((await drawn(root, labels)).some((value) => value > 0)).toBe(false);
  root.destroy();
});

it('draws quads a kernel copies from another set like CPU layout, and hides a label it erases', async () => {
  const root = await gpuRoot();
  const sans = font.create(root, inter);
  const camera = cameras.custom(root);
  const expected = word(root, camera, sans, text.screen, [ANCHOR[0], ANCHOR[1], 0.5]);
  const baseline = await drawn(root, expected);
  const labels = text.create(root, sans, {
    camera: camera,
    capacity: 1,
    length: 8,
    place: text.screen,
  });
  const visible = root.createUniform(d.u32, 1);
  const write = tgpu.computeFn({ workgroupSize: [8], in: { gid: d.builtin.globalInvocationId } })(({
    gid,
  }) => {
    'use gpu';
    if (gid.x >= 8) return;
    if (visible.$ === 1) {
      labels.setQuad(0, gid.x, expected.quadAt(0, gid.x));
      if (gid.x === 0) labels.move(0, d.vec3f(ANCHOR[0], ANCHOR[1], 0.5));
    } else if (gid.x === 0) labels.erase(0);
  });
  const writing = root
    .pipe(labels.inject())
    .pipe(expected.inject())
    .createComputePipeline({ compute: write });
  writing.dispatchWorkgroups(1);
  expect(await drawn(root, labels)).toEqual(baseline);
  visible.write(0);
  writing.dispatchWorkgroups(1);
  expect((await drawn(root, labels)).some((value) => value > 0)).toBe(false);
  root.destroy();
});

/** Alpha that differs from `other`'s, in rows `from` to `to`. */
const differences = (pixels: Uint8Array, other: Uint8Array, from = 0, to = HEIGHT) =>
  pixels.filter((value, i) => {
    const y = Math.floor(i / 4 / WIDTH);
    return i % 4 === 3 && y >= from && y < to && Math.abs(value - other[i]) > 24;
  }).length;
const lit = (pixels: Uint8Array, from = 0) =>
  pixels.filter((value, i) => i % 4 === 3 && Math.floor(i / 4 / WIDTH) >= from && value > 128)
    .length;

/** Shades the first glyph of a label alone. */
const leading = (fragment: text.Fragment) => {
  'use gpu';
  const alpha =
    std.saturate(fragment.distance + 0.5) * std.select(d.f32(0), 1, fragment.glyph === 0);
  return d.vec4f(fragment.color.xyz * alpha, alpha);
};
/** Shades a label above its baseline alone. */
const upper = (fragment: text.Fragment) => {
  'use gpu';
  const alpha =
    std.saturate(fragment.distance + 0.5) * std.select(d.f32(0), 1, fragment.local.y > 0);
  return d.vec4f(fragment.color.xyz * alpha, alpha);
};

it('places by a scale per label, and shades per glyph and by where in the label', async () => {
  const root = await gpuRoot();
  const sans = font.create(root, inter);
  const camera = cameras.custom(root);
  const at = [ANCHOR[0], ANCHOR[1], 0.5] as const;
  const baseline = await drawn(root, word(root, camera, sans, text.screen, at));

  // Half the size at twice the scale: the same pixels.
  const scale = root.createReadonly(d.arrayOf(d.f32, 2), [2, 2]);
  const swelled: Place = (position, offset, glyph) => {
    'use gpu';
    return text.screen(position, offset * scale.$[glyph.label]);
  };
  const halved = await drawn(root, word(root, camera, sans, swelled, at, { em: SIZE / 2 }));
  expect(differences(halved, baseline)).toBe(0);

  // The first glyph alone, as `F` draws it.
  const typed = await drawn(root, word(root, camera, sans, text.screen, at, { shade: leading }));
  const alone = word(root, camera, sans, text.screen, at);
  alone.write(0, 'F', { em: SIZE, ...LEFT });
  const f = await drawn(root, alone);
  expect(lit(f)).toBeGreaterThan(100);
  expect(differences(typed, f)).toBe(0);

  // Above the baseline alone: the y's tail goes, and the rest stays.
  const cut = await drawn(root, word(root, camera, sans, text.screen, at, { shade: upper }));
  expect(lit(baseline, ANCHOR[1] + 1)).toBeGreaterThan(10);
  expect(lit(cut, ANCHOR[1] + 1)).toBe(0);
  expect(differences(cut, baseline, 0, ANCHOR[1] - 1)).toBe(0);
  root.destroy();
});
