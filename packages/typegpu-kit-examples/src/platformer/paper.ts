import { passes, pipelines, targets, type Frame, type SizeSource } from '@typegpu/kit';
import { memo } from '@typegpu/kit/memo';
import { d, std, tgpu, type TgpuRoot, type TgpuTexture } from 'typegpu';
import { CARDS, cardOf, Seen } from './sprites.ts';

type Picture = TgpuTexture & { usableAsSampled: true };

const Art = tgpu.bindGroupLayout({
  art: { texture: d.texture2d(d.f32) },
  shadows: { texture: d.texture2d(d.f32) },
  smooth: { sampler: 'filtering' },
});
const Source = tgpu.bindGroupLayout({ source: { texture: d.texture2d(d.f32) } });

/** Where the lamp's shadows fall, in texels per texel of height, right and down the picture. */
const FALL = d.vec2f(0.75, 0.9);
/**
 * Texels of height between the cards that cast shadows and the cards that take them, in the
 * channels of the shadow masks: on the sky, on the far hills, on the near hills and on the level.
 * Sprites stand closest to the level, and their feet closer still.
 */
const GAPS = { sky: 4.5, far: 4, near: 4, level: 3, feet: 1 };
/** The penumbra of each mask channel, as a Gaussian's standard deviation in texels. */
const SOFTNESS = [2.4, 2, 2, 1.2];
/** Texels the blur reaches to each side, three of the widest deviations. */
const REACH = 7;
/** The blur's taps: an offset in texels and each mask channel's Gaussian weight, summing to 1. */
const TAPS = (() => {
  const offsets = Array.from({ length: 2 * REACH + 1 }, (_, k) => k - REACH);
  const curves = offsets.map((i) => SOFTNESS.map((s) => Math.exp((i * i * -0.5) / (s * s))));
  const totals = SOFTNESS.map((_, c) => curves.reduce((total, curve) => total + curve[c], 0));
  return offsets.map((offset, k) => ({
    offset,
    weights: d.vec4f(
      ...(curves[k].map((weight, c) => weight / totals[c]) as [number, number, number, number]),
    ),
  }));
})();
/** How dark and how cool a full shadow makes the card. */
const SHADE = d.vec3f(0.52, 0.56, 0.7);
/** Width of a lit cut edge in texels, and its strength on top and left edges. */
const EDGE = { width: 0.33, top: 0.5, left: 0.28 };
/** Paper is white inside, so lit cut edges show pale. */
const CORE = d.vec3f(1, 0.98, 0.93);
/** The lamp's warm pool: centre in UV coordinates, falloff, and light at the rim and centre. */
const LAMP = { at: d.vec2f(0.38, 0.1), falloff: 1.1, rim: 0.93, centre: 1.04 };
const WARM = d.vec3f(1.02, 1, 0.96);

/**
 * Which cards cast shadows on each mask channel's cards: everything in front of the sky, of the
 * far hills, of the near hills and of the level. Clouds take the far hills' channel, since the
 * hills below them cast nothing up onto them.
 */
const casters = (card: number) => {
  'use gpu';
  return std.step(d.vec4f(0.5, 2.5, 3.5, 4.5), d.vec4f(card));
};

/** Blurs `read` along `direction` with each channel's Gaussian. */
const blurred = (screen: d.Infer<typeof passes.ScreenVertex>, direction: d.v2i, art: boolean) => {
  'use gpu';
  const size = d.vec2i(std.textureDimensions(Source.$.source));
  const texel = d.vec2i(std.floor(screen.uv * d.vec2f(size)));
  let sum = d.vec4f();
  for (const tap of tgpu.unroll(TAPS)) {
    const at = std.clamp(texel + direction * tap.offset, d.vec2i(), size - 1);
    const read = std.textureLoad(Source.$.source, at, 0);
    let value = d.vec4f(read);
    if (art) value = casters(std.round(cardOf(read.w)));
    sum += tap.weights * value;
  }
  return Seen({ color: sum });
};

/** Blurs the casters of each channel across the art. */
const castersAcross = (screen: d.Infer<typeof passes.ScreenVertex>) => {
  'use gpu';
  return blurred(screen, d.vec2i(1, 0), true);
};

/** Blurs the masks blurred across down the art, which makes their blur round. */
const castersDown = (screen: d.Infer<typeof passes.ScreenVertex>) => {
  'use gpu';
  return blurred(screen, d.vec2i(0, 1), false);
};

/** The card at texel `at` of the art. */
const cardAt = (at: d.v2f) => {
  'use gpu';
  const texel = d.vec2i(std.floor(at));
  return std.round(cardOf(std.textureLoad(Art.$.art, texel, 0).w));
};

/** The blurred shadow masks around texel position `at`. */
const masksAt = (at: d.v2f) => {
  'use gpu';
  const size = d.vec2f(std.textureDimensions(Art.$.shadows));
  return std.textureSampleLevel(Art.$.shadows, Art.$.smooth, at / size, 0);
};

/** The shadow that the cards in front of `card` cast on it at texel position `at`. */
const shadowOn = (at: d.v2f, card: number) => {
  'use gpu';
  if (card === CARDS.sky) return masksAt(at - FALL * GAPS.sky).x;
  if (card <= CARDS.far) return masksAt(at - FALL * GAPS.far).y;
  if (card === CARDS.near) return masksAt(at - FALL * GAPS.near).z;
  // Where sprites stand on the level, a tighter shadow grounds their feet
  const cast = masksAt(at - FALL * GAPS.level).w;
  return std.max(cast, masksAt(at - FALL * GAPS.feet).w);
};

/**
 * Samples the art with square texels at any scale, blending one-pixel seams, so a fractional
 * upscale keeps every texel the same width.
 */
const crisp = (at: d.v2f) => {
  'use gpu';
  const size = d.vec2f(std.textureDimensions(Art.$.art));
  const seam = std.floor(at + 0.5);
  const blend = std.clamp((at - seam) / std.fwidth(at), d.vec2f(-0.5), d.vec2f(0.5));
  return std.textureSampleLevel(Art.$.art, Art.$.smooth, (seam + blend) / size, 0).xyz;
};

/** The art printed on its card, shaded by the cards in front and lit by the lamp. */
const pixelOf = (screen: d.Infer<typeof passes.ScreenVertex>) => {
  'use gpu';
  const size = d.vec2f(std.textureDimensions(Art.$.art));
  const at = screen.uv * size;
  const card = cardAt(at);
  let color = crisp(at);

  if (card < CARDS.sprites) color *= std.mix(d.vec3f(1), SHADE, shadowOn(at, card));

  // A cut edge catches the lamp where a card lower down lies just above or to the left
  if (card > 0) {
    let lit = d.f32(0);
    if (cardAt(at - d.vec2f(0, EDGE.width)) < card) lit = EDGE.top;
    else if (cardAt(at - d.vec2f(EDGE.width, 0)) < card) lit = EDGE.left;
    color = std.mix(color, CORE, lit);
  }

  const aspect = size.x / size.y;
  const off = (screen.uv - LAMP.at) * d.vec2f(aspect, 1);
  const pool = std.exp(-std.dot(off, off) * LAMP.falloff);
  color *= WARM * std.mix(LAMP.rim, LAMP.centre, pool);
  return Seen({ color: d.vec4f(color, 1) });
};

/**
 * Prints the low-resolution art on paper cards under a desk lamp: each card casts a soft shadow
 * on the cards behind, its cut edges catch the light, and the lamp leaves a warm pool. The
 * shadows blur at the art's resolution, so the print reads two texels a pixel.
 */
export function createPaper(root: TgpuRoot, options: { art: () => Picture; size: SizeSource }) {
  const { art, size } = options;
  const mask = { format: 'rgba16float', sampled: true } as const;
  const across = targets.create(root, size, { color: mask });
  const down = targets.create(root, size, { color: mask });
  const smooth = root.createSampler({ magFilter: 'linear', minFilter: 'linear' });

  const sourceOf = memo((texture: Picture) => root.createBindGroup(Source, { source: texture }));
  const blur = (
    label: string,
    shade: (screen: d.Infer<typeof passes.ScreenVertex>) => Seen,
    read: () => Picture,
  ): passes.Drawing<typeof passes.ScreenVertex, Seen> => ({
    label,
    ...passes.fullScreen,
    pixelOf: shade,
    color: passes.shaded,
    draw(pass) {
      pass.setBindGroup(sourceOf(read()));
      pass.draw(3);
    },
  });
  const blurs = [
    passes.create(root, across, { color: passes.color }),
    passes.create(root, down, { color: passes.color }),
  ];
  const drawn = [
    [blurs[0].add(blur('shadows across', castersAcross, art))],
    [blurs[1].add(blur('shadows down', castersDown, () => across.textures.color))],
  ];

  const printOf = memo((picture: Picture, shadows: Picture) =>
    root.createBindGroup(Art, { art: picture, shadows, smooth }),
  );
  const drawing: passes.Drawing<typeof passes.ScreenVertex, Seen> = {
    label: 'paper',
    ...passes.fullScreen,
    pixelOf,
    color: passes.shaded,
    draw(pass) {
      pass.setBindGroup(printOf(art(), down.textures.color));
      pass.draw(3);
    },
  };

  return {
    /** Add it to the pass that prints the paper, after `record` in each frame. */
    drawing,
    initAsync: () => pipelines.compile(blurs),

    /** Blurs the shadow masks from the art drawn this frame. */
    record(frame: Frame) {
      across.fit();
      down.fit();
      blurs[0].record(frame, drawn[0]);
      blurs[1].record(frame, drawn[1]);
    },

    destroy() {
      across.destroy();
      down.destroy();
    },
  };
}
