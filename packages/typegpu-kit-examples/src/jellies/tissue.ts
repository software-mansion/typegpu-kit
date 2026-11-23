import { cameras, lines, passes } from '@typegpu/kit';
import { perlin3d } from '@typegpu/noise';
import { d, std, type TgpuReadonly, type TgpuRoot } from 'typegpu';
import {
  ARMS,
  bellPoint,
  frameOf,
  Jelly,
  MAX,
  NETTLE,
  placed,
  POINTS,
  Strand,
  STRANDS,
  TENTACLES,
} from './jelly.ts';
import { FLASH, hazed, lightAt, lookOf, Seen, SUNLIGHT, timeOf } from './water.ts';

/** Rings from apex to margin, and segments round the axis, of each bell's mesh. */
const RINGS = 28;
const SEGMENTS = 96;
/** Pieces each span between a strand's points is drawn in, along a smooth curve. */
const SMOOTH = 5;
const PIECES = (POINTS - 1) * SMOOTH;

const Skin = d.struct({
  position: d.vec4f,
  world: d.vec3f,
  normal: d.vec3f,
  /** The bell flattened onto a unit disc, the apex at its middle. */
  disc: d.vec2f,
  jelly: d.u32,
});

const Ruffle = d.struct({
  position: d.vec4f,
  world: d.vec3f,
  /** The ribbon's normal, where it folds. */
  fold: d.vec3f,
  across: d.f32,
  along: d.f32,
  strand: d.u32,
});

/**
 * The corner of a quad drawn as two triangles that `vertexIndex` is: x is 1 for the corners one
 * step along, y for those one step across.
 */
export const quadCorner = (vertexIndex: number) => {
  'use gpu';
  const corner = vertexIndex % 6;
  return d.vec2u(
    std.select(d.u32(0), d.u32(1), corner === 1 || corner === 4 || corner === 5),
    std.select(d.u32(0), d.u32(1), corner === 2 || corner === 3 || corner === 5),
  );
};

/**
 * How brightly a jelly startled at `startled` glows at `u` along its body: 0 at its apex, 0.6 at
 * its margin and on down its tentacles to 2. The light runs down from the apex and fades.
 */
const flashOf = (startled: number, u: number) => {
  'use gpu';
  const age = std.max(timeOf.$ - startled, 0);
  const front = age * 0.9;
  const ahead = (u - front) / 0.06;
  const ring = std.exp(-ahead * ahead);
  const behind = 1 - std.smoothstep(front - 0.04, front + 0.04, u);
  return (ring + behind * 0.1) * std.exp(-age * 1.1);
};

/**
 * Light scattered through tissue at `world`, from the surface above, premultiplied by `alpha`,
 * plus what it glows with, then hazed toward the water behind it with distance.
 */
const tissue = (world: d.v3f, tint: d.v3f, up: number, alpha: number, glow: d.v3f) => {
  'use gpu';
  const key = lightAt(world, timeOf.$) * (0.55 + 0.45 * std.saturate(up));
  return hazed(world, tint * SUNLIGHT * (key * 1.15) * alpha + glow, alpha);
};

export interface TissueOptions {
  readonly jellies: TgpuReadonly<d.WgslArray<typeof Jelly>>;
  readonly strands: TgpuReadonly<d.WgslArray<typeof Strand>>;
  /** Each strand's points, `POINTS` a strand. */
  readonly points: TgpuReadonly<d.WgslArray<d.Vec4f>>;
  /** How many jellies to draw at first; `show` changes it. */
  readonly count: number;
}

/** Draws the jellies' bells, tentacles and oral arms as translucent tissue. */
export function createTissue(root: TgpuRoot, options: TissueOptions) {
  const { jellies, strands, points } = options;
  let count = options.count;

  const skinAt = (vertexIndex: number, instanceIndex: number) => {
    'use gpu';
    const quad = std.intdiv(vertexIndex, d.u32(6));
    const corner = quadCorner(vertexIndex);
    const ring = std.intdiv(quad, d.u32(SEGMENTS));
    const segment = quad % SEGMENTS;
    const v = std.max(d.f32(ring + corner.y) / RINGS, 0.01);
    const around = (d.f32(segment + corner.x) / SEGMENTS) * Math.PI * 2;

    const jelly = Jelly(jellies.$[instanceIndex]);
    const time = timeOf.$;
    const here = bellPoint(jelly, time, v, around);
    const down = bellPoint(jelly, time, v + 0.01, around) - here;
    const aside = bellPoint(jelly, time, v, around + 0.01) - here;
    const world = placed(jelly, here);
    return Skin({
      position: cameras.toClip(world),
      world,
      normal: frameOf(jelly.axis) * std.cross(aside, down),
      disc: d.vec2f(std.cos(around), std.sin(around)) * v,
      jelly: instanceIndex,
    });
  };

  const skinOf = (skin: d.Infer<typeof Skin>) => {
    'use gpu';
    const jelly = jellies.$[skin.jelly];
    const nettle = jelly.kind === NETTLE;
    const n = std.normalize(skin.normal);
    const toEye = std.normalize(cameras.current.$.position - skin.world);
    // Positive where the eye sees the outside of the bell, negative where it looks into it
    const facing = std.dot(n, toEye);
    const square = std.saturate(std.abs(facing));
    const v = std.length(skin.disc);
    // Fade the last pixel of the silhouette, and of the hem where the mesh ends, instead of
    // cutting them
    const side = std.saturate(square / std.max(std.fwidth(facing) * 1.5, 0.0001));
    const hem = std.saturate((1 - v) / std.max(std.fwidth(v) * 1.5, 0.0001));
    const edge = side * hem;
    const around = std.atan2(skin.disc.y, skin.disc.x);

    // Seen face-on the tissue is thin and clear; edge-on, and at the margin, the light crosses
    // more of it and it takes more colour
    const graze = 1 - square;
    const graze2 = graze * graze;
    const margin = std.smoothstep(0.8, 1, v);
    const rim = graze2 * graze2 * graze;
    const thin = std.select(d.f32(0.035), d.f32(0.16), nettle);
    const thick = std.select(d.f32(0.32), d.f32(0.45), nettle);
    const body = thin + (thick - thin) * std.saturate(graze2 * graze + margin * 0.4);

    // Four horseshoe gonads round a moon jelly's middle, open toward it, blurred by the tissue
    // over them
    const centre = d.vec2f(std.sign(skin.disc.x), std.sign(skin.disc.y)) * 0.2;
    const offset = skin.disc - centre;
    const wobble = perlin3d.sample(d.vec3f(skin.disc * 9, jelly.seed * 17)) * 0.025;
    const fromRing = std.abs(std.length(offset) - 0.1 - wobble);
    const open = std.smoothstep(
      0.4,
      0.9,
      -std.dot(std.normalize(offset + 0.00001), std.normalize(centre + 0.00001)),
    );
    const blurred = 1 - std.smoothstep(0, 0.06, fromRing);
    const gonads = std.select(blurred * (1 - open) * (0.4 + 0.6 * square), 0, nettle);

    // Faint canals on a moon jelly; pigment banded through a nettle's tissue
    const spokes = std.abs(std.sin(around * 8));
    const canal = (1 - std.smoothstep(0, std.max(std.fwidth(spokes) * 2, 0.06), spokes)) * 0.7;
    const stripe = std.smoothstep(0.7, 0.97, spokes) * std.smoothstep(0.1, 0.3, v);
    const canals = std.select(canal * std.smoothstep(0.3, 0.6, v), 0, nettle);
    const stripes = std.select(0, stripe, nettle);

    const alpha = (body + gonads * 0.1 + canals * 0.02 + stripes * 0.2) * edge;
    // A moon jelly is milky and lit from above; a nettle's tissue is amber and darker for its
    // pigment, and paler where the eye looks into the bell
    const inner = std.select(d.f32(0), d.f32(1), facing < 0);
    const amber = std.mix(jelly.tint, d.vec3f(1, 0.78, 0.5), inner * 0.6);
    const pigment = std.mix(amber, d.vec3f(0.42, 0.1, 0.03), stripes * 0.8);
    const milky = std.mix(jelly.tint, d.vec3f(0.85, 0.62, 0.95), gonads);
    // Thin tissue face-on lets most light through; it glows with colour where it is thick
    const veil = 0.3 + 0.7 * std.saturate(graze2 + margin * 0.5);
    const tint = std.select(
      milky * 1.7,
      pigment * (veil * std.select(0.9, 0.85, inner > 0)),
      nettle,
    );
    const own = lookOf.$.glow;
    const glow =
      jelly.glow * (rim * std.select(0.3, 0.22, nettle) * own * edge) +
      d.vec3f(0.8, 0.45, 0.95) * (gonads * 0.04 * own);
    // A startled jelly's light runs out to its margin and round it
    const shimmer = 0.6 + 0.4 * std.sin(around * 8 - timeOf.$ * 9);
    const flash = flashOf(jelly.startled, v * 0.6) * shimmer;
    const lit =
      glow + FLASH * (flash * (alpha * 0.3 + margin * alpha * 4 + rim * 0.4 * edge) * own);
    const up = n.y * std.sign(facing);
    return Seen({ color: tissue(skin.world, tint, up * 0.6, alpha, lit) });
  };

  /** A point on the Catmull-Rom curve through a strand's points, `s` from 0 to `POINTS - 1`. */
  const curveAt = (base: number, s: number) => {
    'use gpu';
    const i = std.min(d.u32(s), d.u32(POINTS - 2));
    const f = s - d.f32(i);
    const p0 = points.$[base + std.max(i, 1) - 1].xyz;
    const p1 = points.$[base + i].xyz;
    const p2 = points.$[base + i + 1].xyz;
    const p3 = points.$[base + std.min(i + 2, d.u32(POINTS - 1))].xyz;
    const a = p2 - p0;
    const b = p0 * 2 - p1 * 5 + p2 * 4 - p3;
    const c = p1 * 3 - p0 - p2 * 3 + p3;
    return p1 + (a * f + b * (f * f) + c * (f * f * f)) * 0.5;
  };

  /** The unit direction a strand's curve runs at `s`, toward its tip. */
  const tangentAt = (base: number, s: number) => {
    'use gpu';
    return std.normalize(
      curveAt(base, std.min(s + 0.2, POINTS - 1)) - curveAt(base, std.max(s - 0.2, 0)) + 0.00001,
    );
  };

  /** Point `i` of a tentacle, its `SMOOTH` pieces a span along the curve through its points. */
  const tentacleAt = (line: number, i: number) => {
    'use gpu';
    const index = std.intdiv(line, d.u32(TENTACLES)) * STRANDS + (line % TENTACLES);
    const s = d.f32(i) / SMOOTH;
    const along = s / (POINTS - 1);
    return lines.ControlPoint({
      position: curveAt(index * POINTS, s),
      width: strands.$[index].width * (1 - along * 0.75),
      color: d.vec4f(1),
      along,
    });
  };

  const tentacleOf = (fragment: lines.Fragment) => {
    'use gpu';
    const jelly = jellies.$[std.intdiv(fragment.line, d.u32(TENTACLES))];
    const nettle = jelly.kind === NETTLE;
    const profile = 1 - fragment.across * fragment.across;
    const fade = std.saturate(1 - fragment.along);
    const alpha =
      std.select(0.45, 0.5, nettle) * profile * fade * std.sqrt(fade) * fragment.coverage;
    const tint = std.select(jelly.tint, d.vec3f(0.95, 0.42, 0.22), nettle);
    const flash = flashOf(jelly.startled, 0.6 + fragment.along * 1.4);
    const glow = (jelly.glow * (0.25 * fade) + FLASH * (flash * 7)) * (lookOf.$.glow * alpha);
    return tissue(fragment.position, tint, 0.4, alpha, glow);
  };

  const tentacles = lines.create(root, {
    capacity: MAX * TENTACLES,
    length: PIECES + 1,
    point: tentacleAt,
    shade: tentacleOf,
    blend: 'translucent',
    minWidth: { pixels: 1.5 },
    startCap: lines.caps.butt,
    endCap: lines.caps.butt,
    label: 'tentacles',
  });
  tentacles.shown = count * TENTACLES;

  /** Each piece of an arm's ribbon is two quads across, from -1 through 0 to 1. */
  const ruffleAt = (vertexIndex: number, instanceIndex: number) => {
    'use gpu';
    const segment = std.intdiv(vertexIndex, d.u32(12));
    const half = std.intdiv(vertexIndex % 12, d.u32(6));
    const corner = quadCorner(vertexIndex);
    const across = d.f32(half + corner.y) - 1;
    const jellyIndex = std.intdiv(instanceIndex, d.u32(ARMS));
    const index = jellyIndex * STRANDS + TENTACLES + (instanceIndex % ARMS);
    const base = index * POINTS;
    const strand = strands.$[index];
    const time = timeOf.$;

    const s = d.f32(segment + corner.x) / SMOOTH;
    const here = curveAt(base, s);
    const tangent = tangentAt(base, s);
    const toEye = std.normalize(cameras.current.$.position - here);
    const facing = std.normalize(std.cross(tangent, toEye) + 0.00001);
    const front = std.cross(facing, tangent);

    // The ribbon spirals slowly about its length, showing its face in places and its edge in
    // others, and swells and narrows as it goes
    const along = s / (POINTS - 1);
    const turn = along * 4.5 + strand.around + std.sin(time * 0.3 + strand.around) * 0.6;
    const side = facing * std.cos(turn) + front * std.sin(turn);
    const fold = front * std.cos(turn) - facing * std.sin(turn);
    const swell = 0.75 + 0.25 * std.sin(along * 9 + strand.around * 3);
    const width = strand.width * (1 - along * 0.75) * swell;
    // The frill flattens where the ribbon turns edge-on, so its folds never stand out across the
    // screen as a strip of slivers
    const open = std.cos(turn);
    const frill =
      std.sin(along * 22 - time * 0.9 + strand.around) * across * std.abs(across) * open * open;
    const world = here + side * (across * width * 0.5) + fold * (frill * width * 0.3);

    return Ruffle({
      position: cameras.toClip(world),
      world,
      fold,
      across,
      along,
      strand: index,
    });
  };

  const ruffleOf = (ruffle: d.Infer<typeof Ruffle>) => {
    'use gpu';
    const jelly = jellies.$[std.intdiv(ruffle.strand, d.u32(STRANDS))];
    const strand = strands.$[ruffle.strand];
    const nettle = jelly.kind === NETTLE;
    const toEye = std.normalize(cameras.current.$.position - ruffle.world);
    const edgeOn = 1 - std.abs(std.dot(std.normalize(ruffle.fold + 0.00001), toEye));

    // A frilled edge: the ribbon's outline waves gently along its length
    const reach = std.abs(ruffle.across);
    const limit =
      0.82 + 0.18 * std.sin(ruffle.along * 48 + std.sign(ruffle.across) * 1.7 + strand.around);
    const soft = std.max(std.fwidth(reach) * 1.5, 0.04);
    const inside = 1 - std.smoothstep(limit - soft, limit, reach);
    const frill = std.smoothstep(limit - 0.3, limit, reach) * inside;
    const fade =
      std.smoothstep(0, 0.05, ruffle.along) * (1 - std.smoothstep(0.55, 1, ruffle.along));

    const base = std.select(0.03, 0.05, nettle);
    const alpha = (base + edgeOn * edgeOn * base + frill * base) * inside * fade;
    const tint = std.select(jelly.tint, d.vec3f(1, 0.86, 0.64) * 1.5, nettle);
    const flash = flashOf(jelly.startled, 0.4 + ruffle.along * 1.2);
    const glow =
      (jelly.glow * (0.15 + frill * 0.3) + FLASH * (flash * 3)) * (lookOf.$.glow * alpha);
    return Seen({ color: tissue(ruffle.world, tint, 0.3, alpha, glow) });
  };

  return {
    bells: {
      label: 'bells',
      Vertex: Skin,
      at: skinAt,
      pixelOf: skinOf,
      color: passes.shaded,
      translucent: true,
      draw: (pass) => pass.draw(RINGS * SEGMENTS * 6, count),
    } satisfies passes.Drawing<typeof Skin, Seen>,
    tentacles: tentacles.drawing,
    arms: {
      label: 'oral arms',
      Vertex: Ruffle,
      at: ruffleAt,
      pixelOf: ruffleOf,
      color: passes.shaded,
      translucent: true,
      draw: (pass) => pass.draw(PIECES * 12, count * ARMS),
    } satisfies passes.Drawing<typeof Ruffle, Seen>,

    /** Sets how many jellies draw. */
    show(next: number) {
      count = next;
      tentacles.shown = next * TENTACLES;
    },

    destroy() {
      tentacles.destroy();
    },
  };
}
