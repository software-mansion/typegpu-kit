import { passes, pipelines, targets, type Frame, type SizeSource } from '@typegpu/kit';
import { memo } from '@typegpu/kit/memo';
import { d, std, tgpu, type TgpuRenderCommands, type TgpuRoot, type TgpuTexture } from 'typegpu';
import { LAYERS, type Layers } from './layers.ts';
import { dithered, nearest, thresholdAt } from './palette.ts';
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
/** Where the lamp's shadows fall, in texels per texel of height, right and down the picture. */
const FALL = d.vec2f(0.75, 0.9);
/**
 * Texels of height between the cards that cast shadows and the cards that take them: the sky, the
 * clouds and far hills, the near hills and the level. Sprites stand closest to the level, and their
 * feet closer still.
 */
const GAPS = { sky: 4.5, far: 4, near: 4, level: 3, feet: 1 };
/**
 * The scenery's shadow masks lie in two halves of one texture, each channel covered by one layer
 * on that layer's own texels, so each shadow dithers on the texels of the layer that casts it and
 * holds still on them as the layer moves. The first half shadows the sky,
 * cast by the clouds, the far hills, the near hills and the level. The second shadows the cards in
 * front of it, cast by the near hills and the level. Sprites cast solid shadows, kept out of the
 * masks so no dither crawls around the figures that move.
 */
const MASKS = { clouds: 0, far: 1, near: 2, level: 3, behind: { near: 0, level: 1 } };
/** The penumbra of each mask channel in each half, as a Gaussian's standard deviation in texels. */
const SOFTNESS = [
  [2.4, 2.4, 2.4, 2.4],
  [2, 2, 2, 2],
];
/** Texels the blur reaches to each side, three of the widest deviations. */
const REACH = 7;
/** The blur's taps: an offset in texels and each half's Gaussian weights, each summing to 1. */
const TAPS = (() => {
  const offsets = Array.from({ length: 2 * REACH + 1 }, (_, k) => k - REACH);
  const [sky, behind] = SOFTNESS.map((softness) => {
    const curves = offsets.map((i) => softness.map((s) => Math.exp((i * i * -0.5) / (s * s))));
    const totals = softness.map((_, c) => curves.reduce((total, curve) => total + curve[c], 0));
    return curves.map((curve) =>
      d.vec4f(
        ...(curve.map((weight, c) => weight / totals[c]) as [number, number, number, number]),
      ),
    );
  });
  return offsets.map((offset, k) => ({ offset, sky: sky[k], behind: behind[k] }));
})();
/** How dark and how cool a full shadow makes the card. */
const SHADE = d.vec3f(0.52, 0.56, 0.7);
/** Width of a lit cut edge in texels, and its strength on top and left edges. */
const EDGE = { width: 0.33, top: 0.5, left: 0.28 };
/** Paper is white inside, so lit cut edges show pale. */
const CORE = d.vec3f(1, 0.98, 0.93);

/** Blurs the source along `direction` with each channel's Gaussian. */
const blurred = (screen: d.Infer<typeof passes.ScreenVertex>, direction: d.v2i) => {
  'use gpu';
  const size = d.vec2i(std.textureDimensions(Source.$.source));
  const texel = d.vec2i(std.floor(screen.uv * d.vec2f(size)));
  // Each half blurs on its own, with its own deviations
  const width = d.i32(size.x / 2);
  const behind = texel.x >= width;
  const first = d.vec2i(std.select(0, width, behind), 0);
  const last = first + d.vec2i(width, size.y) - 1;
  let sum = d.vec4f();
  for (const tap of tgpu.unroll(TAPS)) {
    const at = std.clamp(texel + direction * tap.offset, first, last);
    sum += std.select(tap.sky, tap.behind, behind) * std.textureLoad(Source.$.source, at, 0);
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

/** The blurred shadow masks of half `half` around texel position `at` of the layers' textures. */
const masksAt = (at: d.v2f, half: number) => {
  'use gpu';
  const size = d.vec2f(std.textureDimensions(Shadows.$.shadows));
  const width = size.x * 0.5;
  const x = std.clamp(at.x, 0.5, width - 0.5) + d.f32(half) * width;
  return std.textureSampleLevel(Shadows.$.shadows, Art.$.smooth, d.vec2f(x, at.y) / size, 0);
};

/** A layer's texel, clamped to its texture. */
const texelOf = (layer: number, texel: d.v2i) => {
  'use gpu';
  const last = d.vec2i(std.textureDimensions(Art.$.layers)) - 1;
  return std.textureLoad(Art.$.layers, std.clamp(texel, d.vec2i(), last), layer, 0);
};

/** The far layer's height, in its tiles, where the sky's gradient is halfway from haze to blue. */
const HORIZON = 1.8;

/** The sky at the far layer's texel `cell`, in whole texel rows, so it scrolls with the hills. */
const skyAt = (cell: d.v2i) => {
  'use gpu';
  const rise = (d.f32(cell.y) + 0.5) / TEXELS - HORIZON;
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

/**
 * Whether the sprites, on the front card, shade the view's texel position `at` across `gap`
 * texels of height: solid wherever their outline falls, with no penumbra to dither.
 */
const spritesShade = (at: d.v2f, gap: number) => {
  'use gpu';
  return spriteAt(at - FALL * d.f32(gap)).w > 0.5;
};

/** Whether a layer's cards cover its texel `texel`, 1 where they do. */
const coveredBy = (layer: number, texel: d.v2i) => {
  'use gpu';
  return std.select(d.f32(0), d.f32(1), texelOf(layer, texel).w > 0);
};

/** Which mask channel's casters cover each texel of their layer's texture, in both halves. */
const castersOf = (screen: d.Infer<typeof passes.ScreenVertex>) => {
  'use gpu';
  const twice = screen.uv.x * 2;
  const half = std.floor(twice);
  const texel = d.vec2i(std.floor(d.vec2f(twice - half, screen.uv.y) * (viewSize() + 1)));
  const clouds = coveredBy(LAYERS.clouds, texel);
  const far = coveredBy(LAYERS.far, texel);
  const near = coveredBy(LAYERS.near, texel);
  const level = coveredBy(LAYERS.level, texel);
  const sky = d.vec4f(clouds, far, near, level);
  const behind = d.vec4f(near, level, 0, 0);
  return Seen({ color: std.select(sky, behind, half > 0.5) });
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
 * resolution, in the palette's colours: each layer moves by whole pixels, each card casts a
 * dithered shadow on the cards behind, and its cut edges catch the light.
 */
export function createPaper(root: TgpuRoot, options: PaperOptions) {
  const { layers, sprites, size } = options;
  const { inLayer, cellOf } = layers;
  const mask = { format: 'rgba16float', sampled: true } as const;
  // The masks' two halves side by side, each the size of a layer's texture
  const halves = {
    get width() {
      return (size.width + 1) * 2;
    },
    get height() {
      return size.height + 1;
    },
  };
  const cast = targets.create(root, halves, { color: mask });
  const across = targets.create(root, halves, { color: mask });
  const down = targets.create(root, halves, { color: mask });
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

  /**
   * Whether the shadow in `channel` of the masks' half `half`, falling across `gap` texels of
   * height, darkens the view's texel position `at`. It dithers on the texels of `layer`, the layer
   * that casts it, each of them taking the shadow at its centre as one.
   */
  const darkens = (layer: number, at: d.v2f, half: number, channel: number, gap: number) => {
    'use gpu';
    const inside = inLayer(layer, at);
    const texel = std.floor(inside);
    const shadow = std.dot(
      masksAt(texel + 0.5 - FALL * d.f32(gap), half),
      std.select(d.vec4f(), d.vec4f(1), std.eq(d.vec4i(channel), d.vec4i(0, 1, 2, 3))),
    );
    return shadow > thresholdAt(cellOf(layer, texel));
  };

  /** Whether the cards in front of `card` shade it at the view's texel position `at`. */
  const shadedAt = (at: d.v2f, card: number) => {
    'use gpu';
    if (card === CARDS.sky) {
      return (
        darkens(LAYERS.clouds, at, 0, MASKS.clouds, GAPS.sky) ||
        darkens(LAYERS.far, at, 0, MASKS.far, GAPS.sky) ||
        darkens(LAYERS.near, at, 0, MASKS.near, GAPS.sky) ||
        darkens(LAYERS.level, at, 0, MASKS.level, GAPS.sky) ||
        spritesShade(at, GAPS.sky)
      );
    }
    if (card <= CARDS.far) {
      return (
        darkens(LAYERS.near, at, 1, MASKS.behind.near, GAPS.far) ||
        darkens(LAYERS.level, at, 1, MASKS.behind.level, GAPS.far) ||
        spritesShade(at, GAPS.far)
      );
    }
    if (card === CARDS.near) {
      return (
        darkens(LAYERS.level, at, 1, MASKS.behind.level, GAPS.near) || spritesShade(at, GAPS.near)
      );
    }
    // Where sprites stand on the level, a tighter shadow grounds their feet
    return spritesShade(at, GAPS.level) || spritesShade(at, GAPS.feet);
  };

  /**
   * The layers and sprites printed on their cards in the palette's colours, the sky's gradient
   * dithered on the far layer's texels and each shadow on the texels of the layer that casts it,
   * so each pattern stays put on its own layer as it scrolls.
   */
  const pixelOf = (screen: d.Infer<typeof passes.ScreenVertex>) => {
    'use gpu';
    const at = screen.uv * viewSize();

    // The sky, on the far layer's texels, under whichever layer is in front
    const behind = std.floor(inLayer(LAYERS.far, at));
    const sky = skyAt(cellOf(LAYERS.far, behind));
    let ink = d.vec3f(sky);
    let card = d.f32(CARDS.sky);
    for (const layer of tgpu.unroll(ORDER)) {
      const read = texelOf(layer, d.vec2i(std.floor(inLayer(layer, at))));
      if (read.w > 0) {
        ink = read.xyz;
        card = std.round(cardOf(read.w));
      }
    }
    // Each card's ink is one flat palette colour, and the sky's gradient is dithered between two
    let lit = nearest(ink);
    if (card === CARDS.sky) lit = dithered(ink, thresholdAt(cellOf(LAYERS.far, behind)));
    // Shadows swap the ink for its shaded palette colour, on more texels where they are deeper
    let color = std.select(lit, nearest(lit * SHADE), shadedAt(at, card));

    // Sprites take no shadows, and sparkles add their light where nothing was drawn
    const sprite = spriteAt(at);
    if (sprite.w > 0 || std.dot(sprite.xyz, d.vec3f(1)) > 0) {
      color = nearest(sprite.xyz + (1 - sprite.w) * color);
    }
    if (sprite.w > 0.5) card = CARDS.sprites;

    // A cut edge catches the lamp where a card lower down lies just above or to the left
    if (card > 0) {
      let edge = d.f32(0);
      if (cardAt(at - d.vec2f(0, EDGE.width)) < card) edge = EDGE.top;
      else if (cardAt(at - d.vec2f(EDGE.width, 0)) < card) edge = EDGE.left;
      if (edge > 0) color = nearest(std.mix(color, CORE, edge));
    }
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
