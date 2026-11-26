import { hable } from '@typegpu/color';
import { bindings, targets } from '@typegpu/kit';
import { gpuRoot } from '@typegpu/kit/testing/dawn';
import { common, d, std, tgpu, type TgpuRoot } from 'typegpu';
import { expect, it } from 'vitest';
import { bloom, fxaa, finish, tone } from '../../src/index.ts';

const WIDTH = 64;
const HEIGHT = 32;

/** The final pass's settings and its bloom's, whose fields don't overlap. */
type Look = d.Infer<typeof finish.Settings> & d.Infer<typeof bloom.Settings>;

/** A tone curve and a grade that give different colours in either order. */
const halve = (color: d.v3f) => {
  'use gpu';
  return color * 0.5;
};

const lift = (color: d.v3f, _uv: d.v2f) => {
  'use gpu';
  return color + 0.25;
};

/** An 8-bit WIDTH × HEIGHT texture's pixels, four bytes each. */
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

/**
 * Runs one frame: clears the source to `colour`, opaque unless it gives an alpha, blooms it and
 * draws every finish into `output`, in the first finish's format, which has 8 bits a channel.
 */
async function frame(
  colour: [number, number, number] | [number, number, number, number],
  build: Omit<finish.Options, 'source' | 'glow'>[],
  look: Partial<Look> = {},
  bright?: bloom.Options['bright'],
) {
  const root = await gpuRoot();
  const size = { width: WIDTH, height: HEIGHT };
  const screen = targets.create(root, size, {
    hdr: { format: 'rgba16float', sampled: true },
    output: { format: build[0].format },
  });

  const source = () => screen.textures.hdr;
  const glow = bloom.create(root, { source, settings: look, ...(bright && { bright }) });
  const posts = build.map((options) =>
    finish.create(root, { ...options, source, settings: look, glow }),
  );
  await Promise.all([glow, ...posts].map((part) => part.initAsync()));

  const encoder = root.createCommandEncoder();
  const clear = encoder.beginRenderPass({
    colorAttachments: { view: screen.views.hdr, clearValue: [...colour, 1].slice(0, 4) },
  });
  clear.end();

  glow.record(encoder);
  for (const final of posts) final.record(encoder, screen.views.output);
  encoder.submit();

  // The last finish's pixels.
  const pixels = await readPixels(root, root.unwrap(screen.textures.output));
  const at = ((HEIGHT / 2) * WIDTH + WIDTH / 2) * 4;
  const middle = [...pixels.subarray(at, at + 3)];

  root.destroy();
  return { pixels, middle };
}

it('runs the bloom chain and every tone curve and filter on a real device', async () => {
  const tones = [tone.aces, tone.agx, tone.neutral, tone.reinhard, hable, null];
  const filters: finish.Filter[] = ['linear', 'sharp'];
  const builds = tones.flatMap((curve) =>
    filters.map((filter) => ({ format: 'rgba8unorm' as const, tone: curve, filter })),
  );

  await frame([4, 2, 1], builds);
});

it('passes mid grey through as its sRGB encoding when nothing else touches it', async () => {
  const { middle } = await frame([0.18, 0.18, 0.18], [{ format: 'rgba8unorm', tone: null }]);

  // Below the bloom's threshold, so only the vignette darkens it, and not at the middle.
  for (const channel of middle) expect(channel).toBeGreaterThanOrEqual(116);
  for (const channel of middle) expect(channel).toBeLessThanOrEqual(119);
});

it('stores the source as it is with encode: false and no tone curve', async () => {
  const { pixels } = await frame(
    [0.4, 0.4, 0.4],
    [{ format: 'rgba8unorm', tone: null, encode: false }],
    { vignette: 0, dither: 0 },
  );

  // Encoded, 0.4 would store as 170.
  expect(new Set(pixels.filter((_, i) => i % 4 !== 3))).toEqual(new Set([102]));
});

it('grades after the tone curve and before the vignette', async () => {
  const vignette = 0.5;
  const { pixels } = await frame(
    [0.4, 0.4, 0.4],
    [{ format: 'rgba8unorm', tone: halve, grade: lift, encode: false }],
    { vignette, dither: 0 },
  );

  // Halved, lifted, then darkened towards the corners. Lifted first, the middle would be 83;
  // lifted after the vignette, the corners would be 90 rather than 60.
  const errors = Array.from({ length: WIDTH * HEIGHT }, (_, i) => {
    const u = ((i % WIDTH) + 0.5) / WIDTH - 0.5;
    const v = (Math.floor(i / WIDTH) + 0.5) / HEIGHT - 0.5;
    const expected = (0.4 * 0.5 + 0.25) * Math.max(0, 1 - (u * u + v * v) * 2 * vignette);
    return Math.abs(pixels[i * 4] - expected * 255);
  });
  expect(Math.max(...errors)).toBeLessThanOrEqual(1);
});

/** A grade that reads elsewhere what it was given here, so a flat picture leaves only the lift. */
const elsewhere = (color: d.v3f, uv: d.v2f) => {
  'use gpu';
  return finish.picture(uv + d.vec2f(0.1, -0.2)) - color + finish.glow(uv) + 0.25;
};

/** A bright pass that keeps half the red alone. */
const redOnly = (color: d.v3f) => {
  'use gpu';
  return d.vec3f(color.x * 0.5, 0, 0);
};

it('reads the picture and the glow anywhere from a grade', async () => {
  const { pixels } = await frame(
    [0.4, 0.4, 0.4],
    [{ format: 'rgba8unorm', tone: null, grade: elsewhere, encode: false }],
    { vignette: 0, dither: 0 },
  );

  // Below the threshold, the glow is zero.
  expect(new Set(pixels.filter((_, i) => i % 4 !== 3))).toEqual(new Set([64]));
});

it('blooms what its bright pass keeps', async () => {
  const { middle } = await frame(
    [0.25, 0.25, 0.25],
    [{ format: 'rgba8unorm', tone: null, encode: false }],
    { vignette: 0, dither: 0, glowIntensity: 0.5 },
    redOnly,
  );

  const [r, g, b] = middle;
  expect(r).toBeGreaterThan(g! + 10);
  expect(Math.abs(g! - 64)).toBeLessThanOrEqual(1);
  expect(b).toBe(g);
});

it('blooms only the positive channels of a source that holds negative ones', async () => {
  const root = await gpuRoot();
  const screen = targets.create(
    root,
    { width: WIDTH, height: HEIGHT },
    { hdr: { format: 'rgba16float', sampled: true } },
  );
  const glow = bloom.create(root, { source: () => screen.textures.hdr });
  await glow.initAsync();

  const encoder = root.createCommandEncoder();
  encoder
    .beginRenderPass({ colorAttachments: { view: screen.views.hdr, clearValue: [8, -2, -2, 1] } })
    .end();
  glow.record(encoder);
  encoder.submit();

  // Mip 0 of the glow, half the source's size: 32 × 16 texels of four halves
  const bytesPerRow = (WIDTH / 2) * 8;
  const readback = root.device.createBuffer({
    size: bytesPerRow * (HEIGHT / 2),
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const copy = root.device.createCommandEncoder();
  copy.copyTextureToBuffer(
    { texture: root.unwrap(glow.output) },
    { buffer: readback, bytesPerRow },
    [WIDTH / 2, HEIGHT / 2],
  );
  root.device.queue.submit([copy.finish()]);
  await readback.mapAsync(GPUMapMode.READ);
  const halves = new Float16Array(readback.getMappedRange().slice(0));

  for (let i = 0; i < halves.length; i += 4) {
    expect(halves[i]).toBeGreaterThan(1);
    expect([halves[i + 1], halves[i + 2]]).toEqual([0, 0]);
  }
  root.destroy();
});

it('dithers by a step of what the output stores, on -srgb outputs too', async () => {
  // Linear 0.002 encodes to 6.6 steps of 255, and a step of noise either way stays within 5 to 8.
  // Near black, a step of linear light is about 12 encoded steps.
  for (const format of ['rgba8unorm', 'rgba8unorm-srgb'] as const) {
    const { pixels } = await frame([0.002, 0.002, 0.002], [{ format, tone: null }], {
      vignette: 0,
    });
    const reds = pixels.filter((_, i) => i % 4 === 0);

    expect(Math.min(...reds)).toBeGreaterThanOrEqual(5);
    expect(Math.max(...reds)).toBeLessThanOrEqual(8);
    expect(new Set(reds).size).toBeGreaterThan(1);
  }
});

it('keeps a clear pixel clear with premultiplied alpha, dither on, on -srgb outputs too', async () => {
  for (const format of ['rgba8unorm', 'rgba8unorm-srgb'] as const) {
    const { pixels } = await frame([0, 0, 0, 0], [{ format, alpha: 'premultiplied' }]);

    expect(new Set(pixels)).toEqual(new Set([0]));
  }
});

it('carries the source alpha and writes premultiplied colour, fading the corners out', async () => {
  // Premultiplied 0.2 at alpha 0.5 is 0.4 itself, stored as 0.2 under alpha 0.5.
  const { pixels } = await frame(
    [0.2, 0.2, 0.2, 0.5],
    [{ format: 'rgba8unorm', tone: null, encode: false, alpha: 'premultiplied' }],
    { dither: 0, glowIntensity: 0, vignette: 0.5 },
  );
  const at = ((HEIGHT / 2) * WIDTH + WIDTH / 2) * 4;
  const [red, green, blue, alpha] = pixels.subarray(at, at + 4);
  for (const channel of [red, green, blue]) expect(Math.abs(channel! - 51)).toBeLessThanOrEqual(1);
  expect(Math.abs(alpha! - 127.5)).toBeLessThanOrEqual(1);

  // In a corner, colour and alpha fade together, so the colour itself stays 0.4.
  const corner = [...pixels.subarray(0, 4)];
  expect(corner[3]).toBeLessThan(110);
  expect(Math.abs(corner[0] / corner[3] - 0.4)).toBeLessThan(0.02);
});

it('reads only the region a lower resolution drew, in the bloom, every filter and FXAA', async () => {
  const root = await gpuRoot();
  const screen = targets.create(
    root,
    { width: WIDTH, height: HEIGHT },
    {
      hdr: { format: 'rgba16float', sampled: true },
      output: { format: 'rgba8unorm' },
    },
    { resolution: 0.5 },
  );

  const region = screen.regions.hdr;
  const source = () => screen.textures.hdr;
  const glow = bloom.create(root, { source, region });
  const filters: finish.Filter[] = ['linear', 'sharp'];
  const posts = filters.map((filter) =>
    finish.create(root, {
      format: 'rgba8unorm',
      source,
      region,
      tone: null,
      filter,
      settings: { vignette: 0 },
      glow,
    }),
  );
  const smooth = fxaa.create(root, { format: 'rgba8unorm', source, region });

  // Mid grey in the region, and outside it a bright magenta that any read past the edge shows.
  const grey = root.createRenderPipeline({
    vertex: common.fullScreenTriangle,
    fragment: tgpu.fragmentFn({ out: d.vec4f })(() => {
      'use gpu';
      return d.vec4f(0.18, 0.18, 0.18, 1);
    }),
    targets: { format: 'rgba16float' },
  });

  const encoder = root.createCommandEncoder();
  const fill = encoder.beginRenderPass({
    colorAttachments: { view: screen.views.hdr, clearValue: [50, 0, 50, 1] },
  });
  fill.setViewport(0, 0, region.width, region.height, 0, 1);
  fill.setPipeline(grey);
  fill.draw(3);
  fill.end();

  glow.record(encoder);
  encoder.submit();

  for (const final of [...posts, smooth]) {
    const draw = root.createCommandEncoder();
    const pass = draw.beginRenderPass({ colorAttachments: { view: screen.views.output } });
    final.draw(pass);
    pass.end();
    draw.submit();

    const pixels = await readPixels(root, root.unwrap(screen.textures.output));
    const reds = pixels.filter((_, i) => i % 4 === 0);
    const greens = pixels.filter((_, i) => i % 4 === 1);

    expect(Math.max(...reds)).toBeLessThanOrEqual(120);
    expect(Math.max(...reds.map((red, i) => red - greens[i]))).toBeLessThanOrEqual(1);
  }

  root.destroy();
});

/** Draws `pattern` at each pixel into an `rgba8unorm` source and smooths it into `format`. */
async function smoothed(
  pattern: (at: d.v2f) => d.v4f,
  format: 'rgba8unorm' | 'rgba8unorm-srgb' = 'rgba8unorm',
  settings?: Partial<d.Infer<typeof fxaa.Settings>>,
) {
  const root = await gpuRoot();
  const screen = targets.create(
    root,
    { width: WIDTH, height: HEIGHT },
    { ldr: { format: 'rgba8unorm', sampled: true }, output: { format } },
  );
  const draw = root.createRenderPipeline({
    vertex: common.fullScreenTriangle,
    fragment: tgpu.fragmentFn({ in: { position: d.builtin.position }, out: d.vec4f })(
      ({ position }) => {
        'use gpu';
        return pattern(position.xy);
      },
    ),
    targets: { format: 'rgba8unorm' },
  });
  const smooth = fxaa.create(root, {
    format,
    source: () => screen.textures.ldr,
    ...(settings && { settings }),
  });

  const encoder = root.createCommandEncoder();
  const picture = encoder.beginRenderPass({ colorAttachments: { view: screen.views.ldr } });
  picture.setPipeline(draw);
  picture.draw(3);
  picture.end();
  smooth.record(encoder, screen.views.output);
  encoder.submit();

  const before = await readPixels(root, root.unwrap(screen.textures.ldr));
  const after = await readPixels(root, root.unwrap(screen.textures.output));
  root.destroy();
  return { before, after };
}

const red = (pixels: Uint8Array, x: number, y: number) => pixels[(y * WIDTH + x) * 4];

/** White below a line that falls a pixel every 16 across, so the search runs out along a step. */
const fall = (x: number) => 4 + x / 16;
const stairs = (at: d.v2f) => {
  'use gpu';
  return d.vec4f(d.vec3f(std.step(4 + at.x / 16, at.y)), 1);
};

it('smooths a stair-stepped edge as FXAA 3.11 does, and leaves the rest as it was', async () => {
  const { before, after } = await smoothed(stairs);

  for (let y = 0; y < HEIGHT; y++) {
    for (let x = 0; x < WIDTH; x++) {
      expect([0, 255]).toContain(red(before, x, y));
      if (Math.abs(y + 0.5 - fall(x + 0.5)) > 2) expect(red(after, x, y)).toBe(red(before, x, y));
    }
  }

  // Row 5 from x = 16, as a transcription of the reference code gives it.
  const reference = [242, 202, 202, 173, 173, 155, 155, 139, 116, 100, 100, 82, 82, 53, 53, 13];
  for (const [i, value] of reference.entries()) {
    expect(Math.abs(red(after, 16 + i, 5) - value)).toBeLessThanOrEqual(1);
  }
});

/** Grey with light dots on odd rows and columns: the pixels between dots see no contrast. */
const dots = (at: d.v2f) => {
  'use gpu';
  const odd = std.floor(at) % 2;
  return d.vec4f(d.vec3f(0.25 + odd.x * odd.y * 0.5), 1);
};

it('leaves a pixel without contrast as it was with both thresholds at 0', async () => {
  const { before, after } = await smoothed(dots, 'rgba8unorm', { edge: 0, floor: 0 });

  for (let y = 0; y < HEIGHT; y += 2) {
    for (let x = 0; x < WIDTH; x += 2) expect(red(after, x, y)).toBe(red(before, x, y));
  }
});

it('stores the same picture into an -srgb output', async () => {
  const plain = await smoothed(stairs);
  const srgb = await smoothed(stairs, 'rgba8unorm-srgb');

  const worst = Math.max(...plain.after.map((value, i) => Math.abs(value - srgb.after[i])));
  expect(worst).toBeLessThanOrEqual(1);
});

it('shows depth through a grade reading a shared part, before and after the target is rebuilt', async () => {
  const root = await gpuRoot();
  const size = { width: WIDTH / 2, height: HEIGHT / 2 };
  const screen = targets.create(root, size, {
    hdr: { format: 'rgba16float', sampled: true },
    depth: { format: 'depth24plus', sampled: true },
    output: { format: 'rgba8unorm' },
  });
  const shown = bindings.layout({ depth: { texture: d.textureDepth2d() } });
  const part = shown.bind({ depth: screen.textures.depth });
  const shared = bindings.share(root, { shown: part });

  const depthOf = (_color: d.v3f, uv: d.v2f) => {
    'use gpu';
    const texel = d.vec2i(uv * d.vec2f(std.textureDimensions(shown.$.depth)));
    return d.vec3f(1 - std.textureLoad(shown.$.depth, texel, 0));
  };
  const final = finish.create(root, {
    format: 'rgba8unorm',
    source: () => screen.textures.hdr,
    tone: null,
    encode: false,
    settings: { vignette: 0, dither: 0 },
    grade: tgpu.fn(depthOf).pipe(part.inject()),
  });
  await final.initAsync();

  const draw = (depth: number) => {
    if (screen.fit()) part.replace({ depth: screen.textures.depth });
    const encoder = root.createCommandEncoder();
    const clear = encoder.beginRenderPass({
      colorAttachments: { view: screen.views.hdr },
      depthStencilAttachment: {
        view: screen.views.depth,
        depthClearValue: depth,
        depthLoadOp: 'clear',
        depthStoreOp: 'store',
      },
    });
    clear.end();
    const pass = encoder.beginRenderPass({ colorAttachments: { view: screen.views.output } });
    pass.setBindGroup(shared.layout, shared.group());
    final.draw(pass);
    pass.end();
    encoder.submit();
  };

  draw(0.5);
  size.width = WIDTH;
  size.height = HEIGHT;
  draw(0.25);

  const pixels = await readPixels(root, root.unwrap(screen.textures.output));
  for (const at of [0, (WIDTH * HEIGHT - 1) * 4]) expect(pixels[at]).toBeCloseTo(191, -1);
  expect(await root.takeErrors()).toEqual([]);
});

it('draws FXAA and the finish into a raw pass as their records do', async () => {
  const root = await gpuRoot();
  const size = { width: WIDTH, height: HEIGHT };
  const screen = targets.create(root, size, {
    hdr: { format: 'rgba16float', sampled: true },
    recorded: { format: 'rgba8unorm' },
    drawn: { format: 'rgba8unorm' },
  });
  const source = () => screen.textures.hdr;
  const posts = [
    fxaa.create(root, { format: 'rgba8unorm', source }),
    finish.create(root, { format: 'rgba8unorm', source }),
  ];

  for (const post of posts) {
    const encoder = root.device.createCommandEncoder();
    const clear = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: screen.views.hdr,
          clearValue: [0.8, 0.4, 0.1, 1],
          loadOp: 'clear',
          storeOp: 'store',
        },
      ],
    });
    clear.end();
    post.record(root.createCommandEncoder(encoder), screen.views.recorded);
    const pass = encoder.beginRenderPass({
      colorAttachments: [{ view: screen.views.drawn, loadOp: 'clear', storeOp: 'store' }],
    });
    post.draw(pass);
    pass.end();
    root.device.queue.submit([encoder.finish()]);

    const recorded = await readPixels(root, root.unwrap(screen.textures.recorded));
    const drawn = await readPixels(root, root.unwrap(screen.textures.drawn));
    expect(recorded[0]).toBeGreaterThan(0);
    expect(drawn).toEqual(recorded);
  }
});
