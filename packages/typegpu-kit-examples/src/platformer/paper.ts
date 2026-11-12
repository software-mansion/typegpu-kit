import { passes, pipelines, targets, type Frame, type SizeSource } from '@typegpu/kit';
import { memo } from '@typegpu/kit/memo';
import { d, std, tgpu, type TgpuRenderCommands, type TgpuRoot, type TgpuTexture } from 'typegpu';
import { LAYERS, type Layers } from './layers.ts';
import { CARDS, cardOf, Seen, TEXELS } from './sprites.ts';

type Picture = TgpuTexture & { usableAsSampled: true };

const Art = tgpu.bindGroupLayout({
  layers: { texture: d.texture2dArray(d.f32) },
  sprites: { texture: d.texture2d(d.f32) },
  smooth: { sampler: 'filtering' },
});
const Shadows = tgpu.bindGroupLayout({ shadows: { texture: d.texture2d(d.f32) } });
const Source = tgpu.bindGroupLayout({ source: { texture: d.texture2d(d.f32) } });

const SKY = d.vec3f(0.36, 0.6, 0.88);
const HAZE = d.vec3f(0.8, 0.88, 0.92);
/** The layers, back to front. */
const ORDER = [LAYERS.clouds, LAYERS.far, LAYERS.near, LAYERS.level];
/** The texels around a point that bilinear filtering blends. */
const CORNERS = [d.vec2i(0, 0), d.vec2i(1, 0), d.vec2i(0, 1), d.vec2i(1, 1)].map((offset) => ({
  offset,
}));

/** Where four samples across a texel lie, from its centre. */
const QUARTERS = [
  d.vec2f(-0.25, -0.25),
  d.vec2f(0.25, -0.25),
  d.vec2f(-0.25, 0.25),
  d.vec2f(0.25, 0.25),
].map((offset) => ({ offset }));

/** The bilinear weight of the texel at `offset` from the one `t` of a texel before the point. */
const weightOf = (offset: d.v2i, t: d.v2f) => {
  'use gpu';
  const along = std.abs(1 - d.vec2f(offset) - t);
  return along.x * along.y;
};

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

/** Blurs the source along `direction` with each channel's Gaussian. */
const blurred = (screen: d.Infer<typeof passes.ScreenVertex>, direction: d.v2i) => {
  'use gpu';
  const size = d.vec2i(std.textureDimensions(Source.$.source));
  const texel = d.vec2i(std.floor(screen.uv * d.vec2f(size)));
  let sum = d.vec4f();
  for (const tap of tgpu.unroll(TAPS)) {
    const at = std.clamp(texel + direction * tap.offset, d.vec2i(), size - 1);
    sum += tap.weights * std.textureLoad(Source.$.source, at, 0);
  }
  return Seen({ color: sum });
};

/** Blurs the casters across the view. */
const castersAcross = (screen: d.Infer<typeof passes.ScreenVertex>) => {
  'use gpu';
  return blurred(screen, d.vec2i(1, 0));
};

/** Blurs the casters blurred across down the view, which makes their blur round. */
const castersDown = (screen: d.Infer<typeof passes.ScreenVertex>) => {
  'use gpu';
  return blurred(screen, d.vec2i(0, 1));
};

/** The blurred shadow masks around the view's texel position `at`. */
const masksAt = (at: d.v2f) => {
  'use gpu';
  const size = d.vec2f(std.textureDimensions(Shadows.$.shadows));
  return std.textureSampleLevel(Shadows.$.shadows, Art.$.smooth, at / size, 0);
};

/** The shadow that the cards in front of `card` cast on it at the view's texel position `at`. */
const shadowOn = (at: d.v2f, card: number) => {
  'use gpu';
  if (card === CARDS.sky) return masksAt(at - FALL * GAPS.sky).x;
  if (card <= CARDS.far) return masksAt(at - FALL * GAPS.far).y;
  if (card === CARDS.near) return masksAt(at - FALL * GAPS.near).z;
  // Where sprites stand on the level, a tighter shadow grounds their feet
  const cast = masksAt(at - FALL * GAPS.level).w;
  return std.max(cast, masksAt(at - FALL * GAPS.feet).w);
};

/** A layer's texel, clamped to its texture. */
const texelOf = (layer: number, texel: d.v2i) => {
  'use gpu';
  const last = d.vec2i(std.textureDimensions(Art.$.layers)) - 1;
  return std.textureLoad(Art.$.layers, std.clamp(texel, d.vec2i(), last), layer, 0);
};

/** The sky behind every layer, its rows fixed to the view's texels. */
const skyAt = (at: d.v2f, size: d.v2f) => {
  'use gpu';
  const rise = (size.y * 0.5 - std.floor(at.y) - 0.5) / TEXELS;
  return std.mix(HAZE, SKY, std.saturate(rise / 6 + 0.5));
};

/** The view's size in texels, a texel smaller than each layer. */
const viewSize = () => {
  'use gpu';
  return d.vec2f(std.textureDimensions(Art.$.layers)) - 1;
};

/** The sprites' pixel at the view's texel position `at`. */
const spriteAt = (at: d.v2f) => {
  'use gpu';
  const size = d.vec2f(std.textureDimensions(Art.$.sprites));
  return std.textureLoad(Art.$.sprites, d.vec2i(std.floor((at / viewSize()) * size)), 0);
};

/** How much of the view's texel at `at` the sprites cover, from four samples across it. */
const spritesOver = (at: d.v2f) => {
  'use gpu';
  const view = viewSize();
  let sum = d.f32(0);
  for (const corner of tgpu.unroll(QUARTERS)) {
    const uv = (at + corner.offset) / view;
    sum += std.textureSampleLevel(Art.$.sprites, Art.$.smooth, uv, 0).w;
  }
  return sum * 0.25;
};

/**
 * How much each mask channel's casters cover a layer around texel position `at` of its
 * texture, filtered between texels so the masks move smoothly with the layer.
 */
const castersIn = (layer: number, at: d.v2f) => {
  'use gpu';
  const from = at - 0.5;
  const base = d.vec2i(std.floor(from));
  const t = std.fract(from);
  let sum = d.vec4f();
  for (const corner of tgpu.unroll(CORNERS)) {
    const card = std.round(cardOf(texelOf(layer, base + corner.offset).w));
    sum += weightOf(corner.offset, t) * casters(card);
  }
  return sum;
};

/** A full-screen drawing that binds its groups with `bind` before it draws. */
const fullScreen = (
  label: string,
  shade: (screen: d.Infer<typeof passes.ScreenVertex>) => Seen,
  bind: (pass: TgpuRenderCommands) => void,
): passes.Drawing<typeof passes.ScreenVertex, Seen> => ({
  label,
  ...passes.fullScreen,
  pixelOf: shade,
  color: passes.shaded,
  draw(pass) {
    bind(pass);
    pass.draw(3);
  },
});

export interface PaperOptions {
  /** The art in layers, which stores each pixel's card in alpha. */
  readonly layers: Layers;
  /** The sprites at the surface's resolution, premultiplied, all on the front card. */
  readonly sprites: () => Picture;
  /** The view's size in texels, which the shadow masks follow. */
  readonly size: SizeSource;
}

/**
 * Prints the art's layers and the sprites over them on paper cards under a desk lamp at full
 * resolution: each layer moves by whole pixels, each card casts a soft shadow on the cards behind, its cut edges catch the
 * light, and the lamp leaves a warm pool. The shadows blur at the art's resolution.
 */
export function createPaper(root: TgpuRoot, options: PaperOptions) {
  const { layers, sprites, size } = options;
  const { inLayer } = layers;
  const mask = { format: 'rgba16float', sampled: true } as const;
  const cast = targets.create(root, size, { color: mask });
  const across = targets.create(root, size, { color: mask });
  const down = targets.create(root, size, { color: mask });
  const smooth = root.createSampler({ magFilter: 'linear', minFilter: 'linear' });

  /** The card in front at the view's texel position `at`, the sky where nothing covers it. */
  const cardAt = (at: d.v2f) => {
    'use gpu';
    if (spriteAt(at).w > 0.5) return d.f32(CARDS.sprites);
    let card = d.f32(CARDS.sky);
    for (const layer of tgpu.unroll(ORDER)) {
      const alpha = texelOf(layer, d.vec2i(std.floor(inLayer(layer, at)))).w;
      if (alpha > 0) card = std.round(cardOf(alpha));
    }
    return card;
  };

  /** Which cards in front of each mask channel's cards cover each texel of the view. */
  const castersOf = (screen: d.Infer<typeof passes.ScreenVertex>) => {
    'use gpu';
    const at = screen.uv * viewSize();
    // Sprites lie on the front card, so they shadow every card behind
    let sum = d.vec4f(spritesOver(at));
    for (const layer of tgpu.unroll(ORDER)) {
      sum = std.max(sum, castersIn(layer, inLayer(layer, at)));
    }
    return Seen({ color: sum });
  };

  /**
   * Lays a layer over `behind` at the view's texel position `at`, shaded by `shadow` where it
   * lies on a card that takes shadows. Samples the layer with square texels at any scale,
   * blending one-pixel seams, so a fractional upscale keeps every texel the same width.
   */
  const over = (layer: number, at: d.v2f, shadow: number, behind: d.v3f) => {
    'use gpu';
    const inside = inLayer(layer, at);
    const seam = std.floor(inside + 0.5);
    const blend = std.clamp((inside - seam) / std.fwidth(inside), d.vec2f(-0.5), d.vec2f(0.5));
    const from = seam + blend - 0.5;
    const base = d.vec2i(std.floor(from));
    const t = std.fract(from);
    const shade = std.mix(d.vec3f(1), SHADE, shadow);

    // Colours count as premultiplied, so sparkles added where nothing was drawn still glow
    let color = d.vec3f();
    let coverage = d.f32(0);
    for (const corner of tgpu.unroll(CORNERS)) {
      const texel = texelOf(layer, base + corner.offset);
      const card = std.round(cardOf(texel.w));
      const weight = weightOf(corner.offset, t);
      const shaded = card > CARDS.sky && card < CARDS.sprites;
      color += weight * std.select(texel.xyz, texel.xyz * shade, shaded);
      coverage += weight * std.select(d.f32(0), d.f32(1), texel.w > 0);
    }
    return color + (1 - coverage) * behind;
  };

  /** The layers printed on their cards, shaded by the cards in front and lit by the lamp. */
  const pixelOf = (screen: d.Infer<typeof passes.ScreenVertex>) => {
    'use gpu';
    const view = viewSize();
    const at = screen.uv * view;

    let color = skyAt(at, view) * std.mix(d.vec3f(1), SHADE, shadowOn(at, CARDS.sky));
    for (const layer of tgpu.unroll(ORDER)) {
      // Each backdrop layer is one card, behind the level
      const card = std.select(d.f32(layer + 1), d.f32(CARDS.level), layer === LAYERS.level);
      color = over(layer, at, shadowOn(at, card), color);
    }
    const sprite = spriteAt(at);
    color = sprite.xyz + (1 - sprite.w) * color;

    // A cut edge catches the lamp where a card lower down lies just above or to the left
    const card = cardAt(at);
    if (card > 0) {
      let lit = d.f32(0);
      if (cardAt(at - d.vec2f(0, EDGE.width)) < card) lit = EDGE.top;
      else if (cardAt(at - d.vec2f(EDGE.width, 0)) < card) lit = EDGE.left;
      color = std.mix(color, CORE, lit);
    }

    const aspect = view.x / view.y;
    const off = (screen.uv - LAMP.at) * d.vec2f(aspect, 1);
    const pool = std.exp(-std.dot(off, off) * LAMP.falloff);
    color *= WARM * std.mix(LAMP.rim, LAMP.centre, pool);
    return Seen({ color: d.vec4f(color, 1) });
  };

  const artOf = memo((texture: Picture, shown: Picture) =>
    root.createBindGroup(Art, {
      layers: texture.createView(d.texture2dArray(d.f32)),
      sprites: shown,
      smooth,
    }),
  );
  const art = () => artOf(layers.target.textures.color, sprites());
  const sourceOf = memo((texture: Picture) => root.createBindGroup(Source, { source: texture }));
  const shadowsOf = memo((texture: Picture) => root.createBindGroup(Shadows, { shadows: texture }));

  const steps = [
    passes.create(root, cast, { color: passes.color }),
    passes.create(root, across, { color: passes.color }),
    passes.create(root, down, { color: passes.color }),
  ] as const;
  const drawn = [
    [steps[0].add(fullScreen('casters', castersOf, (pass) => pass.setBindGroup(art())))],
    [
      steps[1].add(
        fullScreen('shadows across', castersAcross, (pass) =>
          pass.setBindGroup(sourceOf(cast.textures.color)),
        ),
      ),
    ],
    [
      steps[2].add(
        fullScreen('shadows down', castersDown, (pass) =>
          pass.setBindGroup(sourceOf(across.textures.color)),
        ),
      ),
    ],
  ] as const;

  const drawing = fullScreen('paper', pixelOf, (pass) => {
    pass.setBindGroup(art());
    pass.setBindGroup(shadowsOf(down.textures.color));
  });

  return {
    /** Add it to the pass that prints the paper, after `record` in each frame. */
    drawing,
    initAsync: () => pipelines.compile(steps),

    /** Finds and blurs the shadow masks from the layers drawn this frame. */
    record(frame: Frame) {
      cast.fit();
      across.fit();
      down.fit();
      for (let k = 0; k < steps.length; k++) steps[k].record(frame, drawn[k]);
    },

    destroy() {
      cast.destroy();
      across.destroy();
      down.destroy();
    },
  };
}
