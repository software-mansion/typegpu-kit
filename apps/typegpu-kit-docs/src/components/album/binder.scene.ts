// The binder on the GPU, a texel a dither cell over a dithered shadow: its covers, spine and the
// rings on their plate in the site's five greys, the accent only where an edge faces the light, and
// its sheets, each a strip of quads bent as it turns. Each side holds nine sleeves, each a card as
// the pull draws one: a found card's poster over its caption, or the seal on the back of one still
// hidden. What the pages hold lives in storage sized to the book.

import { d, std, tgpu, type TgpuRenderCommands, type TgpuRoot } from 'typegpu';
import { bayerCell, toneOf } from '../site/dither.ts';
import { print } from 'typegpu-kit-pull/foil.scene.ts';
import { AXIS, CARD, COVER, GAP, HEIGHT, MARGIN, POCKETS, REACH, RINGS, WIDTH } from './book.ts';

/** Quads along a sheet's width and down its height. */
const ALONG = 28;
const DOWN = 4;
/** Quads around a ring's half and around its wire. */
const AROUND = 20;
const WIRE = 6;
/** The posters' size in texels, and the mip levels under it. */
export const POSTER = [96, 72] as const;
const POSTER_LEVELS = 4;
/** How far down a card its poster reaches, 4:3 across its width. */
const ART = CARD.width * 0.75;

/** The camera, the palette, and how big a card prints: its texel and poster level. */
const View = d.struct({
  viewProjection: d.mat4x4f,
  eye: d.vec4f,
  /** The greys from the ground up, the accent and its high, and each rarity's colour. */
  grey: d.arrayOf(d.vec4f, 7),
  accent: d.vec4f,
  high: d.vec4f,
  rarity: d.arrayOf(d.vec4f, 5),
  /** A texel's width over a card's on the page, and the mip level the posters print at. */
  texel: d.f32,
  posterLevel: d.f32,
});

/** The example under the pointer or focus and the one out of its pocket, or -1. */
const Album = d.struct({ hover: d.i32, out: d.i32 });

/** A card: its rarity from 0 common to 4 legendary, and 1 once found. */
const Card = d.struct({ rarity: d.u32, found: d.u32 });

/** A sheet: its turn from the right, the bend that trails it, and its height. */
const Sheet = d.struct({ angle: d.f32, bend: d.f32, height: d.f32 });

const layout = tgpu.bindGroupLayout({
  view: { uniform: View },
  album: { uniform: Album },
  sheets: { storage: d.arrayOf(Sheet), access: 'readonly' },
  /** Each example's card, by gallery index. */
  cards: { storage: d.arrayOf(Card), access: 'readonly' },
  /** The example in each pocket, side by side, -1 for an empty one. */
  pockets: { storage: d.arrayOf(d.i32), access: 'readonly' },
  /** The posters, a layer an example. */
  posters: { texture: d.texture2dArray(d.f32) },
  smooth: { sampler: 'filtering' },
});

const ringsAt = tgpu.const(d.arrayOf(d.f32, 3), [...RINGS.at]);

const TOWARD_LIGHT = std.normalize(d.vec3f(-0.45, 1, 0.55));
const PI = Math.PI;

/** One of the five greys above the ground for a lit `value`, 0 to 1, as the chest prints them. */
const grey = (value: number, threshold: number) => {
  'use gpu';
  return layout.$.view.grey[1 + toneOf(value, threshold)].xyz;
};

/** Whether `p` lies in the box from `low` to `high`. */
const inside = (p: d.v2f, low: d.v2f, high: d.v2f) => {
  'use gpu';
  return p.x >= low.x && p.y >= low.y && p.x < high.x && p.y < high.y;
};

/**
 * Whether `p` lies on the seal's lines, as card.ts stamps them on a card's back: a ring and a
 * diamond within it, lines `width` thick in the seal's radii.
 */
const onSeal = (p: d.v2f, width: number) => {
  'use gpu';
  const r = std.length(p);
  const diamond = std.abs(std.abs(p.x) + std.abs(p.y) - 0.62) < width * 0.75 && r < 0.8;
  return std.abs(r - 0.84) < width || diamond;
};

/**
 * Card `example` in its sleeve at `local`, its own units from its top left: as the pull prints a
 * card, a found one's poster over its caption, else the seal on its back.
 */
const cardColor = (example: number, local: d.v2f, threshold: number) => {
  'use gpu';
  const v = layout.$.view;
  const card = layout.$.cards[example];
  const rarity = v.rarity[card.rarity].xyz;
  const near = std.min(local, d.vec2f(CARD.width, CARD.height) - local);
  const edge = std.min(near.x, near.y);
  // Between the page's ground and the raised one
  const ground = std.mix(v.grey[0].xyz, v.grey[1].xyz, 0.45);
  const lit = example === layout.$.album.hover;
  let color = std.copy(ground);

  if (card.found === 0) {
    // Its back: fine diagonal lines and the seal, quiet in a page of them until it is lit
    if (std.fract((local.x + local.y) * 7) < 0.1) color = std.mix(ground, v.grey[1].xyz, 0.8);
    // At least a texel wide, so the lines hold together on a small page
    if (onSeal((local - d.vec2f(0.5, 0.7)) / 0.26, std.max(0.05, (v.texel * 0.6) / 0.26))) {
      color = std.mix(v.grey[2].xyz, v.accent.xyz, std.select(d.f32(0.25), 0.85, lit));
    }
  } else if (local.y < ART) {
    // The poster, printed as the pull prints it
    const art = local / d.vec2f(CARD.width, ART);
    const level = v.posterLevel;
    const poster = std.textureSampleLevel(layout.$.posters, layout.$.smooth, art, example, level);
    color = print(poster.xyz, threshold);
  } else {
    // The caption as lines of type: the category, the title, the description, and the rarity's gem
    if (inside(local, d.vec2f(0.08, 0.83), d.vec2f(0.32, 0.86))) color = v.accent.xyz;
    if (inside(local, d.vec2f(0.08, 0.9), d.vec2f(0.62, 0.96))) color = v.grey[5].xyz;
    if (inside(local, d.vec2f(0.08, 1.01), d.vec2f(0.9, 1.035))) color = v.grey[3].xyz;
    if (inside(local, d.vec2f(0.08, 1.065), d.vec2f(0.72, 1.09))) color = v.grey[3].xyz;
    const gem = std.abs(local - d.vec2f(0.11, 1.32));
    if (gem.x + gem.y < 0.035) color = std.copy(rarity);
  }

  // The frame: a found card's rarity above common, lit in the accent's high under the pointer
  let frame = v.grey[2].xyz;
  if (card.rarity > 0 && card.found !== 0) frame = std.mix(v.grey[2].xyz, rarity, 0.55);
  if (lit) frame = v.high.xyz;
  if (edge < std.select(0.02, 0.035, lit)) color = std.copy(frame);
  return color;
};

/**
 * The face of side `side` at `across` and `down` on its page: the sheet's board, darker as it
 * turns from the light, and nine sleeves welded round, each over its card.
 */
const pageColor = (side: number, across: number, down: number, n: d.v3f, threshold: number) => {
  'use gpu';
  // Lit so that a level face prints solid in one grey, and a turning one dithers down
  const board = (0.25 * std.max(std.dot(n, TOWARD_LIGHT), 0)) / TOWARD_LIGHT.y;
  const x = across - std.select(MARGIN.inner, MARGIN.outer, side % 2 === 1);
  const y = down - MARGIN.top;
  const column = std.floor(x / (CARD.width + GAP));
  const row = std.floor(y / (CARD.height + GAP));
  if (column < 0 || column > 2 || row < 0 || row > 2) return grey(board, threshold);
  const local = d.vec2f(x - column * (CARD.width + GAP), y - row * (CARD.height + GAP));
  // The weld round each sleeve, raised and catching the light
  const weld = 0.03;
  if (local.x > CARD.width + weld || local.y > CARD.height + weld) return grey(board, threshold);
  if (local.x > CARD.width || local.y > CARD.height) return grey(board + 0.25, threshold);

  const example = layout.$.pockets[side * POCKETS + d.u32(row) * 3 + d.u32(column)];
  if (example < 0 || example === layout.$.album.out) return grey(board - 0.25, threshold);
  return cardColor(example, local, threshold);
};

/** The sheet a point is on, for a point on none. */
const NO_SHEET = 0xffffffff;

/** How much the turning sheets over it shade a point at `world` of sheet `which`, if one. */
const shadowAt = (world: d.v3f, which: number) => {
  'use gpu';
  let dark = d.f32(0);
  const count = std.arrayLength(layout.$.sheets);
  for (let i = d.u32(0); i < count; i++) {
    if (i === which) continue;
    const s = layout.$.sheets[i];
    const lifted = std.sin(s.angle);
    if (lifted < 0.02 || s.height < world.y - 0.001) continue;
    // A turning sheet stands over the stretch of the page between the rings and its free edge
    const edge = REACH * std.cos(s.angle);
    const under = world.x * edge > 0 && std.abs(world.x) < std.abs(edge);
    const soft = 1 - std.smoothstep(0, 0.6 + lifted * 1.6, std.abs(world.x - edge));
    if (under) dark = std.max(dark, 0.5 * lifted);
    else dark = std.max(dark, 0.35 * lifted * soft);
  }
  return dark;
};

/** `color` under a shadow `dark` deep, dithered toward the ground. */
const shaded = (color: d.v3f, dark: number, threshold: number) => {
  'use gpu';
  return std.mix(color, layout.$.view.grey[0].xyz, std.select(0, 0.65, dark * 1.4 > threshold));
};

/** Which of a quad's two triangles' six corners `corner` is, across and down. */
const cornerOf = (corner: number) => {
  'use gpu';
  return d.vec2f(
    d.f32(corner === 1 || corner === 4 || corner === 5),
    d.f32(corner === 2 || corner === 3 || corner === 5),
  );
};

const SheetOut = {
  position: d.builtin.position,
  world: d.vec3f,
  normal: d.vec3f,
  uv: d.vec2f,
  which: d.interpolate('flat', d.u32),
};

const sheetVertex = tgpu.vertexFn({
  in: { vertexIndex: d.builtin.vertexIndex, instanceIndex: d.builtin.instanceIndex },
  out: SheetOut,
})(({ vertexIndex, instanceIndex }) => {
  'use gpu';
  const quad = std.intdiv(vertexIndex, 6);
  const uv =
    (d.vec2f(d.f32(quad % ALONG), d.f32(std.intdiv(quad, ALONG))) + cornerOf(vertexIndex % 6)) /
    d.vec2f(ALONG, DOWN);
  // A strip of constant curvature `k` leaving the rings at the sheet's angle
  const s = layout.$.sheets[instanceIndex];
  const reach = AXIS + uv.x * WIDTH;
  const k = s.bend;
  const end = s.angle + k * reach;
  let along = d.vec2f(std.cos(s.angle), std.sin(s.angle)) * reach;
  if (std.abs(k) > 1e-3) {
    along = d.vec2f(std.sin(end) - std.sin(s.angle), std.cos(s.angle) - std.cos(end)) / k;
  }
  const world = d.vec3f(along.x, s.height + along.y, (uv.y - 0.5) * HEIGHT);
  return {
    position: layout.$.view.viewProjection * d.vec4f(world, 1),
    world,
    normal: d.vec3f(-std.sin(end), std.cos(end), 0),
    uv,
    which: instanceIndex,
  };
});

const sheetFragment = tgpu.fragmentFn({
  in: SheetOut,
  out: d.vec4f,
})(({ position, world, normal, uv, which }) => {
  'use gpu';
  const threshold = bayerCell(d.vec2u(position.xy));
  const toEye = std.normalize(layout.$.view.eye.xyz - world);
  let n = std.normalize(normal);
  let side = which * 2;
  let across = uv.x * WIDTH;
  // From above its back: the page that lies on the left
  if (std.dot(n, toEye) < 0) {
    n = std.neg(n);
    side = side + 1;
    across = WIDTH - across;
  }
  const color = pageColor(side, across, uv.y * HEIGHT, n, threshold);
  return d.vec4f(shaded(color, shadowAt(world, which), threshold), 1);
});

const PlainOut = {
  position: d.builtin.position,
  world: d.vec3f,
  normal: d.vec3f,
  kind: d.interpolate('flat', d.u32),
};

/** The boxes the binder stands on: the covers, the spine, the rings' plate, and its shadow. */
const LEFT = 0;
const SPINE = 2;
const PLATE = 3;
const SHADOW = 4;
const SLABS = 5;
const RING = 5;
const OUTER = REACH + COVER.overhang;
const HALF_Z = HEIGHT / 2 + COVER.overhang;
/** How far the shadow reaches past the covers, and the way the light throws it. */
const SPREAD = 0.45;
const THROWN = d.vec2f(0.1, -0.08);

const Slab = d.struct({ middle: d.vec3f, half: d.vec3f });
const slab = (middle: [number, number, number], half: [number, number, number]) =>
  Slab({ middle: d.vec3f(...middle), half: d.vec3f(...half) });
const COVER_X = (COVER.spine + OUTER) / 2;
const COVER_HALF: [number, number, number] = [(OUTER - COVER.spine) / 2, COVER.thick / 2, HALF_Z];
/**
 * The boxes by middle and half size: the left cover, the right, the spine sunk a little, the plate
 * the rings stand on, and a sheet of shadow under it all.
 */
const boxes = tgpu.const(d.arrayOf(Slab, SLABS), [
  slab([-COVER_X, -COVER.thick / 2, 0], COVER_HALF),
  slab([COVER_X, -COVER.thick / 2, 0], COVER_HALF),
  slab([0, -COVER.thick * 0.75, 0], [COVER.spine, COVER.thick / 2, HALF_Z]),
  slab([0, -0.01, 0], [AXIS * 0.85, 0.02, RINGS.at[2] + 0.3]),
  slab([THROWN.x, -COVER.thick - 0.01, THROWN.y], [OUTER + SPREAD, 0, HALF_Z + SPREAD]),
]);

const slabVertex = tgpu.vertexFn({
  in: { vertexIndex: d.builtin.vertexIndex, instanceIndex: d.builtin.instanceIndex },
  out: PlainOut,
})(({ vertexIndex, instanceIndex }) => {
  'use gpu';
  // A box: six faces, two triangles each
  const face = std.intdiv(vertexIndex, 6);
  const ab = cornerOf(vertexIndex % 6) * 2 - 1;
  const axis = face % 3;
  const sign = std.select(d.f32(-1), d.f32(1), face < 3);
  let p = d.vec3f(sign, ab.x, ab.y);
  let normal = d.vec3f(sign, 0, 0);
  if (axis === 1) {
    p = d.vec3f(ab.y, sign, ab.x);
    normal = d.vec3f(0, sign, 0);
  }
  if (axis === 2) {
    p = d.vec3f(ab.x, ab.y, sign);
    normal = d.vec3f(0, 0, sign);
  }
  const world = boxes.$[instanceIndex].middle + p * boxes.$[instanceIndex].half;
  return {
    position: layout.$.view.viewProjection * d.vec4f(world, 1),
    world,
    normal,
    kind: instanceIndex,
  };
});

const slabFragment = tgpu.fragmentFn({
  in: PlainOut,
  out: d.vec4f,
})(({ position, world, normal, kind }) => {
  'use gpu';
  const v = layout.$.view;
  const threshold = bayerCell(d.vec2u(position.xy));
  // The shadow: dithered, darkest under the covers and fading past them
  if (kind === SHADOW) {
    const q = std.abs(world.xz - THROWN) - d.vec2f(OUTER, HALF_Z);
    const out = std.length(std.max(q, d.vec2f())) / SPREAD;
    return d.vec4f(0, 0, 0, std.select(d.f32(0), 0.6, (1 - out) * 0.9 > threshold));
  }
  const diffuse = std.max(std.dot(std.normalize(normal), TOWARD_LIGHT), 0);
  // A ring: lit, and catching the light where it mirrors it toward the eye
  if (kind === RING) {
    const half = std.normalize(TOWARD_LIGHT + std.normalize(v.eye.xyz - world));
    if (std.pow(std.max(std.dot(std.normalize(normal), half), 0), 30) > threshold) {
      return d.vec4f(v.grey[6].xyz, 1);
    }
    return d.vec4f(grey(0.5 + 0.35 * diffuse, threshold), 1);
  }
  const dark = shadowAt(world, d.u32(NO_SHEET));
  if (kind === PLATE) {
    return d.vec4f(shaded(grey((0.5 * diffuse) / TOWARD_LIGHT.y, threshold), dark, threshold), 1);
  }
  // The covers and the spine: the darkest grey, drawn in lines as the site's figures are
  let color = v.grey[1].xyz;
  if (normal.y < 0.5 || kind === SPINE) return d.vec4f(shaded(color, dark, threshold), 1);

  // A cover's inside: a groove round it like the chest's, on the front cover the seal stamped in,
  // and the outer edges that face the light catching it, as the chest's do; each line at least a
  // texel wide, so it never breaks into dots
  const line = std.max(0.015, v.texel * 0.6);
  const fromX = OUTER - std.abs(world.x);
  const fromZ = HALF_Z - std.abs(world.z);
  const toOuter = std.min(fromX, fromZ);
  const groove = std.abs(std.min(std.abs(world.x) - COVER.spine, toOuter) - 0.09) < line;
  const at = d.vec2f(world.x + COVER_X, world.z) / 0.42;
  const stamp = kind === LEFT && onSeal(at, line / 0.42);
  if (groove || stamp) color = v.grey[2].xyz;
  const light = std.normalize(TOWARD_LIGHT.xz) * std.sign(world.xz);
  const toward = std.max(std.select(light.y, light.x, fromX < fromZ), 0);
  if (toOuter < line * 1.6) {
    color = std.mix(v.grey[3].xyz, std.mix(v.grey[4].xyz, v.accent.xyz, 0.6), toward);
  }
  return d.vec4f(shaded(color, dark, threshold), 1);
});

const ringVertex = tgpu.vertexFn({
  in: { vertexIndex: d.builtin.vertexIndex, instanceIndex: d.builtin.instanceIndex },
  out: PlainOut,
})(({ vertexIndex, instanceIndex }) => {
  'use gpu';
  // Half a torus standing on the plate, in the plane across the spine
  const quad = std.intdiv(vertexIndex, 6);
  const corner = cornerOf(vertexIndex % 6);
  const around = ((d.f32(quad % AROUND) + corner.x) / AROUND) * PI;
  const wire = ((d.f32(std.intdiv(quad, AROUND)) + corner.y) / WIRE) * PI * 2;
  const out = d.vec3f(std.cos(around), std.sin(around), 0);
  const normal = out * std.cos(wire) + d.vec3f(0, 0, std.sin(wire));
  const world =
    d.vec3f(0, 0.01, ringsAt.$[instanceIndex]) + out * RINGS.radius + normal * RINGS.wire;
  return {
    position: layout.$.view.viewProjection * d.vec4f(world, 1),
    world,
    normal,
    kind: d.u32(RING),
  };
});

/**
 * The binder's three pipelines for `format`, and what they read, sized to a book of `sheets`
 * sheets and `cards` cards.
 */
export function createBinderScene(
  root: TgpuRoot,
  format: GPUTextureFormat,
  { sheets, cards }: { readonly sheets: number; readonly cards: number },
) {
  const view = root.createUniform(View);
  const album = root.createUniform(Album, { hover: -1, out: -1 });
  const sheetBuffer = root.createBuffer(d.arrayOf(Sheet, sheets)).$usage('storage');
  const cardBuffer = root.createBuffer(d.arrayOf(Card, Math.max(cards, 1))).$usage('storage');
  const pocketBuffer = root.createBuffer(d.arrayOf(d.i32, sheets * 2 * POCKETS)).$usage('storage');
  const posters = root
    .createTexture({
      size: [POSTER[0], POSTER[1], Math.max(cards, 1)],
      format: 'rgba8unorm',
      mipLevelCount: POSTER_LEVELS,
    })
    .$usage('sampled', 'render');
  const group = root.createBindGroup(layout, {
    view,
    album,
    sheets: sheetBuffer,
    cards: cardBuffer,
    pockets: pocketBuffer,
    posters: posters.createView(d.texture2dArray(d.f32)),
    smooth: root.createSampler({
      magFilter: 'linear',
      minFilter: 'linear',
      mipmapFilter: 'linear',
    }),
  });
  const at = {
    targets: { format },
    depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
  } as const;
  const slabs = root.createRenderPipeline({ ...at, vertex: slabVertex, fragment: slabFragment });
  const rings = root.createRenderPipeline({ ...at, vertex: ringVertex, fragment: slabFragment });
  const pages = root.createRenderPipeline({ ...at, vertex: sheetVertex, fragment: sheetFragment });

  return {
    view,
    album,
    sheets: sheetBuffer,
    cards: cardBuffer,
    pockets: pocketBuffer,
    /** Writes the cards' posters, `POSTER` texels each, a layer a card, and their mip levels. */
    writePosters(layers: readonly (ArrayBuffer | undefined)[]) {
      layers.forEach((pixels, layer) => {
        if (pixels)
          posters.write(pixels, { origin: [0, 0, layer], size: [POSTER[0], POSTER[1], 1] });
      });
      posters.generateMipmaps();
    },
    initAsync: () => Promise.all([slabs.initAsync(), rings.initAsync(), pages.initAsync()]),
    /** Draws the binder into a pass with a depth attachment. */
    draw(pass: TgpuRenderCommands) {
      pass.setBindGroup(group);
      pass.setPipeline(slabs);
      pass.draw(36, SLABS);
      pass.setPipeline(rings);
      pass.draw(AROUND * WIRE * 6, RINGS.at.length);
      pass.setPipeline(pages);
      pass.draw(ALONG * DOWN * 6, sheets);
    },
  };
}
