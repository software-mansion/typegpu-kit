/**
 * Draws polylines as a drawing for pass descriptions, in one draw for a whole set. Points come from
 * the set's buffer, written on the CPU or by kernels, or from a `'use gpu'` function evaluated as
 * the lines draw. Each segment is shaped on the screen by `@typegpu/kit/geometry`'s
 * `polylineVariableWidth`, with joins and caps, and its edges are antialiased from their distance
 * in pixels.
 *
 * @module
 * @injects member `setPoint`, `setCount`, `pointAt` and `held`, which read its own buffers in
 * any pipeline
 * @binds 2 storage, 1 uniform: the drawing, the points and the counts, and `passes.viewport`
 * @binds 2 storage: `setPoint`, `setCount`, `pointAt` or `held`, the points and the counts
 * @owns 1 to 3 buffers: the segment's indices, the points unless `point`, and the counts unless
 * `point` or `count`
 * @records `drawing`, added to passes, after the kernels that write the points.
 */

import * as cameras from '../cameras.ts';
import * as passes from '../passes.ts';
import * as pipelines from '../pipelines.ts';
import { byStage, type Inject } from '../inject.ts';
import {
  caps,
  endCapSlot,
  joins as geometryJoins,
  joinSlot,
  LineControlPoint,
  lineSegmentIndices,
  polylineVariableWidth,
  startCapSlot,
  type JoinInput,
} from '../geometry/index.ts';
import { miterJoinLimitSlot } from '../geometry/lines/joins/miter.ts';
import { d, std, tgpu, type StorageFlag, type TgpuBuffer, type TgpuRoot } from 'typegpu';

/** This schema holds a point of a line. */
export const ControlPoint = d.struct({
  /** Holds the point in world space, or in a flat camera's plane with z as depth. */
  position: d.vec3f,
  /** Sets the full width at this point, in the set's `units`. */
  width: d.f32,
  /** Sets linear colour and opacity, not premultiplied. */
  color: d.vec4f,
  /**
   * Measures distance from the line's start, for dashes and patterns. `write` fills it. A closed
   * line's last segment continues it by the segment's length in world units.
   */
  along: d.f32,
});

export type ControlPoint = d.InferGPU<typeof ControlPoint>;

/** Floats per `ControlPoint` in a set's buffer. */
const FLOATS = d.sizeOf(ControlPoint) / 4;

/** This schema describes a pixel of a line passed to `shade`. */
export const Fragment = d.struct({
  line: d.u32,
  /** Holds the point on the centre line in world space. */
  position: d.vec3f,
  /**
   * Holds the unit normal of a tube along the line: towards the viewer at the centre and sideways
   * at the edges, for lighting.
   */
  normal: d.vec3f,
  /** Holds the colour of the points, blended along the segment. */
  color: d.vec4f,
  /** Holds `along`, blended along the segment. */
  along: d.f32,
  /** Measures how much `along` changes across one pixel, for antialiased dashes. */
  perPixel: d.f32,
  /**
   * Measures across the line from -1 at one edge through 0 at the centre to 1 at the other, as the
   * root mean square over the part of the pixel the line covers. A profile such as `1 - across *
   * across` then averages over the pixel, so a line a pixel or two wide keeps its brightness.
   */
  across: d.f32,
  /** Gives the signed distance to the line's edge in pixels, positive inside. */
  edge: d.f32,
  /**
   * Gives the share of the pixel the line covers, from 0 to 1. A blended line drawn wider than its
   * width to meet `minWidth` covers less by how much narrower it is.
   */
  coverage: d.f32,
});

export type Fragment = d.InferGPU<typeof Fragment>;

/** This schema describes a segment corner for pass outputs that read the vertex. */
export const Corner = d.struct({
  /** Holds the corner position in clip space. */
  position: d.vec4f,
  /** Holds the centre-line point at this end of the segment in world space. */
  world: d.vec3f,
  /** Holds the tube normal: towards the viewer at the centre line and sideways at the rim. */
  normal: d.vec3f,
  color: d.vec4f,
  along: d.f32,
  /**
   * Places the corner on the screen against the segment, in pixels: across it from the centre
   * line, along it from its start, and the segment's length.
   */
  segment: d.vec3f,
  /** Holds the half width in pixels at the segment's start and end. */
  radii: d.vec2f,
  /**
   * Measures out from the end's centre towards the drawn shape's rim, from 0 at the centre to 1 on
   * it, and that times the rim corner's distance in pixels, for joins and caps that are not round.
   * The distance runs along the ray from the centre, so it overstates how far a slanted edge lies.
   * Then 1 inside a join or cap's triangles, and 1 at the segment's end rather than its start.
   */
  rim: d.vec4f,
  /**
   * Says how the ends are measured: 1 for a start and 2 for an end measured from the centre line,
   * round or butt, and 4 for a butt start and 8 for a butt end.
   */
  ends: d.u32,
  line: d.u32,
});

/** This schema provides the surface to pass outputs. */
export const Seen = d.struct({
  /** Holds the point on the centre line in world space. */
  world: d.vec3f,
  /** Holds the tube normal, as `Fragment.normal`. */
  normal: d.vec3f,
  /**
   * Holds the premultiplied colour returned by `shade`. An `opaque` set holds `xyz / w` instead, with
   * alpha the share of samples the line covers, 1 without multisampling.
   */
  color: d.vec4f,
});

/**
 * This type accepts a join or cap from `lines.joins` or `lines.caps`, or a `'use gpu'` function
 * with the same signature.
 */
export type Join = typeof geometryJoins.round;

/** Joins two segments with a straight cut across the outside of the turn. */
const bevel = (join: JoinInput, joinVertexIndex: number, maxJoinCount: number) => {
  'use gpu';
  if (joinVertexIndex === 0) return d.vec2f(join.v);
  const across = std.mix(
    join.d,
    (join.start + join.end) * 0.5,
    d.f32(joinVertexIndex) / d.f32(maxJoinCount),
  );
  return join.C.position + across * join.C.radius;
};

/** Joins segments with `round`, `miter` or `bevel`. */
export const joins = { ...geometryJoins, bevel };

/** Ends lines with `round`, `square`, `butt`, `arrow`, `triangle` or `wedge`. */
export { caps };

/** Shapes that need at most two triangles on each side of an end, where round ones fan out in three. */
const ANGULAR: readonly Join[] = [
  joins.miter,
  bevel,
  caps.butt,
  caps.square,
  caps.arrow,
  caps.triangle,
  caps.wedge,
];

/**
 * Returns the coverage of dashes `on` long with gaps `off` long, measured along the line in the
 * units of `along`, with antialiased ends. Multiply a `shade`'s alpha by it.
 */
export const dash = (fragment: Fragment, on: number, off: number) => {
  'use gpu';
  const length = d.f32(on);
  const period = length + d.f32(off);
  const at = fragment.along - std.floor(fragment.along / period) * period;
  // Signed distance to the nearer end of a dash, positive inside it
  const inside = std.min(at, length - at);
  const outside = -std.min(at - length, period - at);
  const signed = std.select(outside, inside, at <= length);
  return std.saturate(signed / std.max(fragment.perPixel, 1e-6) + 0.5);
};

/** Selects how a set's colour reaches the target. */
export type Blend = 'over' | 'add' | 'opaque' | 'translucent' | GPUBlendState;

export interface Options {
  /** Sets the number of lines to an integer of at least 1. */
  readonly capacity: number;
  /** Sets the most points a line holds to an integer of at least 2. */
  readonly length: number;
  /**
   * Selects `world` for widths in world units, or `points` for widths that keep their size on
   * screen, in points of the pass's `passes.viewport`. Default `world`.
   */
  readonly units?: 'world' | 'points';
  /** Joins each line's last point to its first. Default false. */
  readonly closed?: boolean;
  /**
   * Returns point `i` of `line`, evaluated as the lines draw, in place of the set's point buffer.
   * Default the point buffer.
   * @gpu
   */
  readonly point?: (line: number, i: number) => d.Infer<typeof ControlPoint>;
  /**
   * Returns how many points `line` holds, up to `length`. Default the count buffer, or `length`
   * with `point`.
   * @gpu
   */
  readonly count?: (line: number) => number;
  /** Selects the join between segments, from `lines.joins`. Default `joins.round`. @gpu */
  readonly join?: Join;
  /**
   * Limits a miter join's length, relative to the line's half width, before it bends back towards
   * a bevel. Default 2.
   */
  readonly miterLimit?: number;
  /** Selects the cap at each line's first point, from `lines.caps`. Default `caps.round`. @gpu */
  readonly startCap?: Join;
  /** Selects the cap at each line's last point, from `lines.caps`. Default `caps.round`. @gpu */
  readonly endCap?: Join;
  /**
   * Returns premultiplied colour for a pixel. Default the point colour times the pixel's
   * `coverage`.
   * @gpu
   */
  readonly shade?: (fragment: Fragment) => d.v4f;
  /**
   * Selects `over` or `add` to blend in draw order, `opaque` to write depth, `translucent` for the
   * passes' translucency targets, or a blend state. Opaque lines turn coverage into samples in a
   * multisampled description and cut their edges at half coverage elsewhere. Default `over`.
   */
  readonly blend?: Blend;
  /**
   * Draws lines narrower than this many points, or `{ pixels }` pixels, at this width, so distant
   * lines stay whole. Blended lines fade by how much narrower they are, and a blended segment whose
   * ends both have width 0 draws nothing; opaque lines stay whole. Into depth alone, such as a
   * shadow map, lines keep their own width. Default 1 point.
   */
  readonly minWidth?: number | { readonly pixels: number };
  /** Sets the drawing pipeline label. Default `lines`. */
  readonly label?: string;
}

export interface Lines extends Disposable {
  /** Reports the number of lines, from the `capacity` option. */
  readonly capacity: number;
  /** Reports the most points a line holds, from the `length` option. */
  readonly length: number;
  /**
   * Sets how many lines the drawing draws, lines 0 to `shown - 1`, an integer from 0 to
   * `capacity`. Lines past it cost nothing. Default `capacity`.
   */
  shown: number;
  /** Fills the set's accessors with its own buffers, as their defaults do. */
  inject(): Inject;
  /**
   * Writes point `i` of `line` from a kernel. Requires the point buffer.
   */
  readonly setPoint: (line: number, i: number, point: d.Infer<typeof ControlPoint>) => void;
  /** Sets how many points `line` holds from a kernel. Requires the count buffer. */
  readonly setCount: (line: number, count: number) => void;
  /** Returns point `i` of `line`, from the buffer or from `point`. */
  readonly pointAt: (line: number, i: number) => d.Infer<typeof ControlPoint>;
  /** Returns how many points `line` holds, up to `length`. */
  readonly held: (line: number) => number;
  /** Holds `length` points per line, line `l` from `l * length`. Absent with `point`. */
  readonly points: (TgpuBuffer<d.WgslArray<typeof ControlPoint>> & StorageFlag) | undefined;
  /** Holds each line's point count. Absent with `count`, or with `point` alone. */
  readonly counts: (TgpuBuffer<d.WgslArray<d.U32>> & StorageFlag) | undefined;
  /**
   * Writes `line` from x, y and z coordinates, keeping at most `length` points, and sets its count
   * when the set has a count buffer. Fills `along` with the distance from the first point. `width`
   * and `color` give one value for the line or one per point. Use `write(line, [])` to empty a line.
   */
  write(line: number, positions: ArrayLike<number>, options?: WriteOptions): void;
  /** Draws the `shown` lines in one draw when added to a pass description. */
  readonly drawing: passes.Drawing<typeof Corner, d.Infer<typeof Seen>> & { inject(): Inject };
  /** Destroys the point, count and index buffers. */
  destroy(): void;
}

export interface WriteOptions {
  /** Sets the width, or one width per kept point. Default 1. */
  readonly width?: number | ArrayLike<number>;
  /** Sets r, g, b and a for the line, or four values per point. Default opaque white. */
  readonly color?: ArrayLike<number>;
}

/** Pixels of fringe around each line for its antialiased edge. */
const FRINGE = 1;

const missing = (buffer: string, reason: string) => () => {
  throw new Error(`lines: this set has no ${buffer} buffer for kernels to write: ${reason}.`);
};

/** Pixel coverage of a band `radius` wide each side of the centre, `x` pixels from it. */
const coverageOf = (x: number, radius: number) => {
  'use gpu';
  return std.saturate(std.min(x + radius, 0.5) - std.max(x - radius, -0.5));
};

const plain = (fragment: Fragment) => {
  'use gpu';
  const alpha = fragment.color.w * fragment.coverage;
  return d.vec4f(fragment.color.xyz * alpha, alpha);
};

/** The fringe, which a pass into depth alone leaves out. */
const fringeOf = () => {
  'use gpu';
  if (passes.depthOnly.$) return d.f32(0);
  return d.f32(FRINGE);
};

const onScreen = (clip: d.v4f, radius: number) => {
  'use gpu';
  const half = passes.viewport.$.size * 0.5;
  return LineControlPoint({ position: (clip.xy / clip.w) * half, radius: radius + fringeOf() });
};

/**
 * Moves `from` along to `to` until it enters the depth range, if it is the nearer end and lies
 * outside it. The near plane is z = 0 under standard depth and z = w under reversed depth, and the
 * nearer end of a segment that is not wholly outside can leave the range only through it.
 */
const nearOf = (from: d.v4f, to: d.v4f) => {
  'use gpu';
  if (from.w >= to.w) return d.f32(0);
  let start = from.z;
  let stop = to.z;
  if (from.z > from.w) {
    start = from.w - from.z;
    stop = to.w - to.z;
  }
  if (start >= 0) return d.f32(0);
  return std.min(start / (start - stop) + 1e-4, 1);
};

/** Returns the point `s` of the way from `from` to `to`. */
const between = (
  from: d.Infer<typeof ControlPoint>,
  to: d.Infer<typeof ControlPoint>,
  s: number,
) => {
  'use gpu';
  return ControlPoint({
    position: std.mix(from.position, to.position, s),
    width: std.mix(from.width, to.width, s),
    color: std.mix(from.color, to.color, s),
    along: std.mix(from.along, to.along, s),
  });
};

/**
 * A butt cap that reaches the fringe past the end, square to the segment, so the end's edge fades
 * across the pixel it crosses.
 */
const butt = (join: JoinInput, joinVertexIndex: number, _maxJoinCount: number) => {
  'use gpu';
  const side = join.cross * join.C.radius;
  const back = join.fw * fringeOf();
  if (joinVertexIndex === 0) return join.C.position + side;
  if (joinVertexIndex === 1) return join.C.position + side + back;
  return join.C.position + back;
};

const checked = (name: string, value: number, least: number) => {
  if (!Number.isInteger(value) || value < least) {
    throw new RangeError(
      `lines.create requires \`${name}\` to be an integer of at least ${least}. Received ${value}.`,
    );
  }
};

/**
 * Creates a set of polylines drawn in one draw. They are shaped for the camera of the pass
 * description they are drawn in, which `point`, `count` and `shade` read too. Throws unless
 * `capacity` is an integer of at least 1 and `length` an integer of at least 2.
 *
 * @param options.point a `'use gpu'` function; it, `count` and `shade` carry the fills of what they
 * read, such as `tgpu.fn(curve).pipe(noise.inject())`
 * @returns a set of lines you destroy
 */
export function create(root: TgpuRoot, options: Options): Lines {
  const { capacity, length, units = 'world', closed = false, blend = 'over' } = options;
  const { minWidth = 1, join, startCap, endCap, miterLimit = 2 } = options;
  const minimum = typeof minWidth === 'number' ? minWidth : minWidth.pixels;
  const minimumInPixels = typeof minWidth !== 'number';
  checked('capacity', capacity, 1);
  checked('length', length, 2);

  const owned = options.point === undefined;
  const label = options.label ?? 'lines';
  const pointsBuffer = owned
    ? root
        .createBuffer(d.arrayOf(ControlPoint, capacity * length))
        .$usage('storage')
        .$name(`${label}_points`)
    : undefined;
  const countsBuffer =
    owned && options.count === undefined
      ? root.createBuffer(d.arrayOf(d.u32, capacity)).$usage('storage').$name(`${label}_counts`)
      : undefined;
  // Lines of two points never join
  const shapes = length === 2 && !closed ? [startCap, endCap] : [join, startCap, endCap];
  const fans = shapes.every((shape) => shape !== undefined && ANGULAR.includes(shape)) ? 2 : 3;
  const indices = lineSegmentIndices(fans);
  const indexBuffer = root.createBuffer(d.arrayOf(d.u32, indices.length), indices).$usage('index');

  // Per-set accessors filled with its own buffers, so one kernel can write several sets
  const points = tgpu.accessor(
    d.arrayOf(ControlPoint),
    pointsBuffer ? byStage(pointsBuffer) : missing('point', 'it takes its points from `point`'),
  );
  const counts = tgpu.accessor(
    d.arrayOf(d.u32),
    countsBuffer
      ? byStage(countsBuffer)
      : missing(
          'count',
          options.count
            ? 'it takes its counts from `count`'
            : 'with `point` alone, every line holds `length` points',
        ),
  );
  const inject: Inject = (cfg) => {
    let filled = cfg;
    if (pointsBuffer) filled = filled.with(points, byStage(pointsBuffer));
    if (countsBuffer) filled = filled.with(counts, byStage(countsBuffer));
    return filled;
  };

  const stored = (line: number, i: number) => {
    'use gpu';
    return ControlPoint(points.$[d.u32(line) * length + d.u32(i)]);
  };
  const storedCount = (line: number) => {
    'use gpu';
    return counts.$[line];
  };

  const pointAt = options.point ?? stored;
  const counted = options.count !== undefined || countsBuffer !== undefined;
  const countOf = options.count ?? storedCount;
  const held = (line: number) => {
    'use gpu';
    if (!counted) return d.u32(length);
    return std.min(d.u32(countOf(line)), d.u32(length));
  };

  const setPoint = (line: number, i: number, point: d.Infer<typeof ControlPoint>) => {
    'use gpu';
    if (line >= capacity || i >= length) return;
    points.$[d.u32(line) * length + d.u32(i)] = ControlPoint(point);
  };
  const setCount = (line: number, count: number) => {
    'use gpu';
    if (line >= capacity) return;
    counts.$[line] = std.min(d.u32(count), d.u32(length));
  };

  const shade = options.shade ?? plain;

  const roundJoin = join === undefined || join === joins.round;
  const buttStart = startCap === caps.butt;
  const buttEnd = endCap === caps.butt;
  // Round and butt ends measure from the centre line
  const centredStart = startCap === undefined || startCap === caps.round || buttStart;
  const centredEnd = endCap === undefined || endCap === caps.round || buttEnd;

  const segments = closed ? length : length - 1;
  const opaque = blend === 'opaque';
  const pointed = units === 'points';

  /** The least half width in pixels the lines are drawn at, none into depth alone. */
  const leastOf = () => {
    'use gpu';
    if (passes.depthOnly.$) return d.f32(0);
    if (minimumInPixels) return d.f32(minimum * 0.5);
    return minimum * 0.5 * passes.viewport.$.pixelRatio;
  };

  /** Half a unit of width in pixels: of a point, or of a world unit at a clip w of 1. */
  const scaleOf = () => {
    'use gpu';
    const view = passes.viewport.$;
    if (pointed) return view.pixelRatio * 0.5;
    return cameras.lensOf(cameras.current.$).y * view.size.y * 0.25;
  };

  /** Half the width in pixels of a point at `clip`, at least none. */
  const radiusOf = (point: d.Infer<typeof ControlPoint>, clip: d.v4f, scale: number) => {
    'use gpu';
    const radius = std.max(point.width, 0) * scale;
    if (pointed) return radius;
    return radius / clip.w;
  };

  const hidden = (line: number) => {
    'use gpu';
    const out = Corner();
    out.position = d.vec4f(0, 0, 2, 1);
    out.line = d.u32(line);
    return out;
  };

  const at = (vertexIndex: number, instanceIndex: number) => {
    'use gpu';
    const line = std.intdiv(instanceIndex, d.u32(segments));
    const k = instanceIndex - line * segments;
    const n = held(line);
    let drawn = k + 1 < n;
    if (closed) drawn = k < n && n >= 3;
    if (!drawn) return hidden(line);

    const next = (k + 1) % n;
    const view = cameras.current.$;
    const projection = view.viewProjection;
    let b = ControlPoint(pointAt(line, k));
    let c = ControlPoint(pointAt(line, next));
    // A blended segment of no width covers nothing, so it draws nothing rather than minWidth
    if (!opaque && b.width <= 0 && c.width <= 0) return hidden(line);
    // A closed line's last segment carries `along` on past its last point
    if (closed && next === 0) c.along = b.along + std.distance(b.position, c.position);
    let clipB = projection * d.vec4f(b.position, 1);
    let clipC = projection * d.vec4f(c.position, 1);
    // Both ends lie beyond the same side of the depth range
    if (clipB.z < 0 && clipC.z < 0) return hidden(line);
    if (clipB.z > clipB.w && clipC.z > clipC.w) return hidden(line);

    // Cut the segment where it passes the near plane
    const cutB = nearOf(clipB, clipC);
    const cutC = nearOf(clipC, clipB);
    if (cutB > 0) {
      b = between(b, c, cutB);
      clipB = std.mix(clipB, clipC, cutB);
    }
    if (cutC > 0) {
      c = between(c, b, cutC);
      clipC = std.mix(clipC, clipB, cutC);
    }

    const scale = scaleOf();
    const least = leastOf();
    const rb = radiusOf(b, clipB, scale);
    const rc = radiusOf(c, clipC, scale);
    const pb = onScreen(clipB, std.max(rb, least));
    const pc = onScreen(clipC, std.max(rc, least));

    // A missing neighbour repeats the end to form a cap. A neighbour behind the eye stands in at
    // the near cut of its segment, which lies on that segment, so the end still joins. A cut end
    // continues straight on, so it shows no cap
    let pa = LineControlPoint(pb);
    let joinedB = false;
    if (cutB > 0) {
      pa = LineControlPoint({ position: pb.position * 2 - pc.position, radius: pb.radius });
      joinedB = true;
    } else if (closed || k > 0) {
      const a = ControlPoint(pointAt(line, (k + n - 1) % n));
      const clipA = projection * d.vec4f(a.position, 1);
      const cutA = nearOf(clipA, clipB);
      const nearA = std.mix(clipA, clipB, cutA);
      pa = onScreen(nearA, std.max(radiusOf(between(a, b, cutA), nearA, scale), least));
      joinedB = true;
    }
    let pd = LineControlPoint(pc);
    let joinedC = false;
    if (cutC > 0) {
      pd = LineControlPoint({ position: pc.position * 2 - pb.position, radius: pc.radius });
      joinedC = true;
    } else if (closed || k + 2 < n) {
      const e = ControlPoint(pointAt(line, (k + 2) % n));
      const clipD = projection * d.vec4f(e.position, 1);
      const cutD = nearOf(clipD, clipC);
      const nearD = std.mix(clipD, clipC, cutD);
      pd = onScreen(nearD, std.max(radiusOf(between(e, c, cutD), nearD, scale), least));
      joinedC = true;
    }
    // A neighbour the end's circle holds, such as a repeated point, leaves the geometry a cap
    const ab = pb.position - pa.position;
    const dc = pc.position - pd.position;
    if (std.dot(ab, ab) <= (pa.radius - pb.radius) * (pa.radius - pb.radius)) joinedB = false;
    if (std.dot(dc, dc) <= (pc.radius - pd.radius) * (pc.radius - pd.radius)) joinedC = false;

    const shape = polylineVariableWidth(pa, pb, pc, pd, vertexIndex, fans);

    // Vertices past the first two lie on the rim, round the end their wing names
    let atC = vertexIndex === 1;
    let rim = false;
    if (vertexIndex >= 2) {
      atC = (vertexIndex - 2) % 4 >= 2;
      rim = true;
    }

    const centre = std.select(pb.position, pc.position, atC);

    // Distances across and along the segment are linear on the screen, so they interpolate
    // exactly between corners
    const towardsC = pc.position - pb.position;
    const span = std.length(towardsC);
    let forward = d.vec2f(1, 0);
    if (span > 1e-6) forward = towardsC / span;
    const fromB = shape.vertexPosition - pb.position;

    // Each corner takes the depth and world position of the centre line beside it, carried on
    // past the ends, so the shape lies in one plane through the segment and caps and joins that
    // reach back along it lie at its depth there. A cut end, or one carried so far that it would
    // pass the eye, stays at the end's depth
    let t = std.select(d.f32(0), d.f32(1), atC);
    if (span > 1e-6) t = std.dot(fromB, forward) / span;
    if (cutB > 0) t = std.max(t, 0);
    if (cutC > 0) t = std.min(t, 1);
    if (t * clipB.w + (1 - t) * clipC.w < std.min(clipB.w, clipC.w) * 0.25) t = std.saturate(t);
    const share = (t * clipB.w) / (t * clipB.w + (1 - t) * clipC.w);
    const end = std.mix(clipB, clipC, share);
    const world = std.mix(b.position, c.position, share);

    // The rim's offset on screen, turned into the world for the tube's normal
    const offset = shape.vertexPosition - centre;
    const sideways = view.right * offset.x + view.up * offset.y;
    let normal = cameras.towards(view, world);
    if (rim) normal = std.normalize(sideways + normal * 1e-3);
    let ends = d.u32(0);
    if (std.select(centredStart, roundJoin, joinedB)) ends += 1;
    if (std.select(centredEnd, roundJoin, joinedC)) ends += 2;
    if (buttStart && !joinedB) ends += 4;
    if (buttEnd && !joinedC) ends += 8;

    // Colour holds the end's past it. `along` carries on past it, but stops at the end on a join's
    // or cap's fan, so the halves of a join meet
    const ndc = shape.vertexPosition / (passes.viewport.$.size * 0.5);
    return Corner({
      position: d.vec4f(ndc * end.w, end.z, end.w),
      world,
      normal,
      color: std.mix(b.color, c.color, std.saturate(share)),
      along: std.mix(b.along, c.along, std.select(share, std.saturate(share), vertexIndex >= 6)),
      segment: d.vec3f(
        std.dot(fromB, d.vec2f(-forward.y, forward.x)),
        std.dot(fromB, forward),
        span,
      ),
      radii: d.vec2f(rb, rc),
      rim: d.vec4f(
        std.select(d.vec2f(), d.vec2f(1, std.length(offset)), rim),
        std.select(d.f32(0), d.f32(1), vertexIndex >= 6),
        std.select(d.f32(0), d.f32(1), atC),
      ),
      ends,
      line,
    });
  };

  const pixelOf = (corner: d.Infer<typeof Corner>) => {
    'use gpu';
    const perPixel = std.fwidth(corner.along);
    const segment = corner.segment;
    const span = std.max(segment.z, 1e-6);
    const t = std.saturate(segment.y / span);
    const radius = std.mix(corner.radii.x, corner.radii.y, t);
    const drawn = std.max(radius, leastOf());

    // A butt end stops square at the end: past it, measure across alone
    const ends = corner.ends;
    const squared = ((ends & 4) !== 0 && segment.y < 0) || ((ends & 8) !== 0 && segment.y > span);
    let beyond = segment.y - t * span;
    if (squared) beyond = 0;
    let distance = std.length(d.vec2f(segment.x, beyond));
    let coverage = coverageOf(distance, drawn);

    // Joins and caps of other shapes measure in from the drawn shape's rim, which lies the fringe
    // outside the line's edge, and so does what such a shape adds to the body beyond the line's
    // width
    const rim = corner.rim;
    let round = true;
    if (segment.y < 0) round = (ends & 1) !== 0;
    if (segment.y > span) round = (ends & 2) !== 0;
    if (rim.z > 1e-4) round = (ends & std.select(d.u32(1), d.u32(2), rim.w > 0.5)) !== 0;
    if ((ends & 3) !== 3 && !squared) {
      const out = std.max(rim.x, 1e-4);
      const inside = ((1 - out) * rim.y) / out - FRINGE;
      const outline = std.saturate(inside + 0.5);
      if (!round) {
        coverage = outline;
        distance = std.max(drawn - inside, 0);
      } else if (distance > drawn) {
        coverage = std.max(coverage, outline);
      }
    }
    // A butt end's edge fades over the pixel it crosses
    if ((ends & 4) !== 0) coverage *= std.saturate(segment.y + 0.5);
    if ((ends & 8) !== 0) coverage *= std.saturate(span - segment.y + 0.5);
    if (!opaque) coverage *= radius / std.max(drawn, 1e-6);
    // Across, as the root mean square over the part of the pixel the line covers, so a profile in
    // `shade` keeps a thin line's brightness wherever the line crosses the pixels
    const low = std.max(distance - 0.5, -drawn);
    const high = std.min(distance + 0.5, drawn);
    let spread = distance * distance;
    if (high - low > 1e-4) spread = (high * high * high - low * low * low) / (3 * (high - low));
    const across = std.saturate(std.sqrt(spread) / std.max(drawn, 1e-6));

    // Bend the interpolated normal into a round tube's from how far across the pixel lies
    const eye = cameras.towards(cameras.current.$, corner.world);
    const side = corner.normal - eye * std.dot(corner.normal, eye);
    let normal = d.vec3f(eye);
    if (std.dot(side, side) > 1e-8) {
      normal = std.normalize(side) * across + eye * std.sqrt(1 - across * across);
    }

    let color = shade(
      Fragment({
        line: corner.line,
        position: corner.world,
        normal,
        color: corner.color,
        along: corner.along,
        perPixel,
        across: std.select(d.f32(-1), d.f32(1), segment.x >= 0) * across,
        edge: drawn - distance,
        coverage,
      }),
    );
    if (opaque) {
      // Opaque colour is whole. With several samples, the alpha says how many it covers; with one,
      // the edge is cut at half coverage
      const whole = color.xyz / std.max(color.w, 1e-6);
      if (passes.sampleCount.$ === 1) {
        if (color.w < 0.5) std.discard();
        color = d.vec4f(whole, 1);
      } else {
        color = d.vec4f(whole, color.w);
      }
    }
    return Seen({ world: corner.world, normal, color });
  };

  const limit = tgpu.const(d.f32, miterLimit);
  const drawn: Inject = (cfg) => {
    let shaped = cfg
      .pipe(inject)
      // A whole-number literal would make the join's arithmetic integer
      .with(miterJoinLimitSlot, limit as unknown as number);
    if (join) shaped = shaped.with(joinSlot, join);
    if (startCap) shaped = shaped.with(startCapSlot, buttStart ? butt : startCap);
    if (endCap) shaped = shaped.with(endCapSlot, buttEnd ? butt : endCap);
    return shaped;
  };

  const total = indices.length;
  let shown = capacity;
  const blending: Pick<
    passes.Drawing,
    'blend' | 'translucent' | 'depthStencil' | 'alphaToCoverage'
  > =
    blend === 'opaque'
      ? { alphaToCoverage: true }
      : blend === 'translucent'
        ? { translucent: true }
        : {
            blend: typeof blend === 'string' ? pipelines.blend[blend] : blend,
            depthStencil: { depthWriteEnabled: false },
          };

  const drawing: Lines['drawing'] = {
    label,
    Vertex: Corner,
    at,
    pixelOf,
    color: passes.shaded,
    linear: ['segment', 'rim'],
    flat: ['radii'],
    ...blending,
    inject: () => drawn,
    indices: indexBuffer,
    draw(pass) {
      if (shown > 0) pass.drawIndexed(total, shown * segments);
    },
  };

  const queue = root.device.queue;
  // Made on the first write, so a set whose points come from kernels or `point` stages nothing
  let staging: Float32Array | undefined;
  const count = new Uint32Array(1);

  const destroy = () => {
    pointsBuffer?.destroy();
    countsBuffer?.destroy();
    indexBuffer.destroy();
  };

  return {
    capacity,
    length,
    get shown() {
      return shown;
    },
    set shown(next) {
      if (!Number.isInteger(next) || next < 0 || next > capacity) {
        throw new RangeError(
          `lines: \`shown\` takes an integer from 0 to the capacity, ${capacity}. Received ${next}.`,
        );
      }
      shown = next;
    },
    inject: () => inject,
    setPoint,
    setCount,
    pointAt: (line, i) => {
      'use gpu';
      return ControlPoint(pointAt(line, i));
    },
    held,
    points: pointsBuffer,
    counts: countsBuffer,
    drawing,

    write(line, positions, writeOptions = {}) {
      if (!pointsBuffer) {
        throw new Error(
          'lines: this set takes its points from `point`, so it has no buffer to write.',
        );
      }
      if (!Number.isInteger(line) || line < 0 || line >= capacity) {
        throw new RangeError(`The lines have no line ${line}; there are ${capacity}.`);
      }

      const { width = 1, color = WHITE } = writeOptions;
      const kept = Math.min(Math.floor(positions.length / 3), length);
      if (typeof width !== 'number' && width.length < kept) {
        throw new RangeError(
          `lines: write keeps ${kept} points but has ${width.length} widths; give one per point or one number.`,
        );
      }
      if (color.length !== 4 && color.length < kept * 4) {
        throw new RangeError(
          `lines: write keeps ${kept} points but has ${color.length} colour values; give four per point or four for the line.`,
        );
      }
      const perPoint = color.length >= kept * 4 && kept > 1;
      const staged = (staging ??= new Float32Array(length * FLOATS));
      let along = 0;
      for (let i = 0; i < kept; i++) {
        const offset = i * FLOATS;
        staged[offset] = positions[i * 3];
        staged[offset + 1] = positions[i * 3 + 1];
        staged[offset + 2] = positions[i * 3 + 2];
        // Measure between the f32 positions the GPU reads, so equal points add no distance
        if (i > 0) {
          along += Math.hypot(
            staged[offset] - staged[offset - FLOATS],
            staged[offset + 1] - staged[offset - FLOATS + 1],
            staged[offset + 2] - staged[offset - FLOATS + 2],
          );
        }
        staged[offset + 3] = typeof width === 'number' ? width : width[i];
        const from = perPoint ? i * 4 : 0;
        staged[offset + 4] = color[from];
        staged[offset + 5] = color[from + 1];
        staged[offset + 6] = color[from + 2];
        staged[offset + 7] = color[from + 3];
        staged[offset + 8] = along;
      }

      if (kept > 0) {
        queue.writeBuffer(
          root.unwrap(pointsBuffer),
          line * length * FLOATS * 4,
          staged.buffer,
          0,
          kept * FLOATS * 4,
        );
      }
      if (countsBuffer) {
        count[0] = kept;
        queue.writeBuffer(root.unwrap(countsBuffer), line * 4, count.buffer, 0, 4);
      }
    },

    destroy,
    [Symbol.dispose]: destroy,
  };
}

const WHITE = [1, 1, 1, 1];
