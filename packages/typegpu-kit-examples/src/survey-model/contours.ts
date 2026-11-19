import { cameras, lines, materials, pipelines, shading } from '@typegpu/kit';
import { d, std, tgpu, type TgpuCommandEncoder, type TgpuRoot } from 'typegpu';
import { heightAt, isIndex, LEVELS, levelOf, liftedOf, REACH, SHORE, tints } from './terrain.ts';

/** Points round each ring. */
const POINTS = 192;
const TOTAL = LEVELS * POINTS;
const WORKGROUP = 64;
/** Steps inward from `REACH` that find a ring's last crossing, and halvings that refine it. */
const MARCH = 48;
const HALVINGS = 10;
/** Dash and gap along the soundings, in world units. */
const DASH = 0.14;
const GAP = 0.09;
/** The cyan of the ring under the pointer, linear and bright enough to bloom. */
const CYAN = d.vec3f(0.25, 1.6, 2.2);
/** World units outside the outermost ring that still pick it. */
const NEAR_RING = 0.06;
/** Rays this near level pick no ring. */
const LEVEL = { grazing: 1e-4 };
/** Width of an intermediate contour, as a share of an index contour's. */
const THIN = 0.6;

const Shape = d.struct({
  /** Rings rise from the plinth to their heights as this goes from 0 to 1. */
  lift: d.f32,
  /** Index contours' tube width in world units. */
  width: d.f32,
  /** The ring under the pointer, or `LEVELS` for none. */
  hovered: d.u32,
  /** 1 to dash the soundings. */
  dashed: d.u32,
});

const radiusAt = (direction: d.v2f, level: number) => {
  'use gpu';
  const stride = REACH / MARCH;
  let inner = d.f32(0);
  let outer = d.f32(0);
  for (let step = 0; step < MARCH; step++) {
    const r = REACH - d.f32(step) * stride;
    if (heightAt(direction * r) >= level) {
      inner = r;
      outer = r + stride;
      break;
    }
  }
  for (let k = 0; k < HALVINGS; k++) {
    const middle = (inner + outer) * 0.5;
    if (heightAt(direction * middle) >= level) inner = middle;
    else outer = middle;
  }
  return std.min((inner + outer) * 0.5, REACH);
};

type Ray = Parameters<typeof cameras.hitHeight>[0];

/** Ring `ring`'s radius in `radii` towards `point`, between the two ring points either side. */
const radiusOf = (radii: Float32Array, ring: number, point: readonly number[]) => {
  const turn = ((Math.atan2(point[2], point[0]) / (Math.PI * 2) + 1) % 1) * POINTS;
  const i = Math.floor(turn);
  const r0 = radii[ring * POINTS + (i % POINTS)];
  const r1 = radii[ring * POINTS + ((i + 1) % POINTS)];
  return r0 + (r1 - r0) * (turn - i);
};

/**
 * Traces the island's contour rings into one set of lines and shades them as satin enamel tubes,
 * lit by the lighting of the scene they draw in. A kernel finds each ring point's radius by
 * marching in from the edge to the level's last crossing and halving the step, then a second runs
 * along each ring for its dashes.
 */
export function createContours(root: TgpuRoot, options: { width: number; dashed: boolean }) {
  const { width, dashed } = options;
  const shape = root.createUniform(Shape, {
    lift: 0,
    width,
    hovered: LEVELS,
    dashed: dashed ? 1 : 0,
  });

  const enamel = (fragment: lines.Fragment) => {
    'use gpu';
    const ring = fragment.line;
    const tint = tints.$[ring];
    let material = materials.glossy(tint, 0.38);
    if (ring === shape.$.hovered) {
      material = materials.glossy(CYAN * 0.1, 0.3);
      material.emission = CYAN;
    }
    const lit = shading.shade(material, shading.pointAt(fragment.position, fragment.normal));

    let alpha = fragment.coverage;
    if (ring < SHORE && shape.$.dashed === 1) alpha *= lines.dash(fragment, DASH, GAP);
    return d.vec4f(lit * alpha, alpha);
  };

  const rings = lines.create(root, {
    capacity: LEVELS,
    length: POINTS,
    closed: true,
    blend: 'opaque',
    shade: enamel,
    label: 'contours',
  });

  const trace = tgpu.computeFn({
    workgroupSize: [WORKGROUP],
    in: { gid: d.builtin.globalInvocationId },
  })(({ gid }) => {
    'use gpu';
    if (gid.x >= TOTAL) return;
    const ring = std.intdiv(gid.x, d.u32(POINTS));
    const i = gid.x % POINTS;
    const angle = (d.f32(i) / POINTS) * Math.PI * 2;
    const direction = d.vec2f(std.cos(angle), std.sin(angle));
    const r = radiusAt(direction, levelOf(ring));
    rings.setPoint(
      ring,
      i,
      lines.ControlPoint({
        position: d.vec3f(direction.x * r, liftedOf(ring, shape.$.lift), direction.y * r),
        width: std.select(shape.$.width * THIN, shape.$.width, isIndex(ring)),
        color: d.vec4f(1),
        along: 0,
      }),
    );
    if (i === 0) rings.setCount(ring, POINTS);
  });

  /** Measures each ring's length from its first point, one ring a thread. */
  const measure = tgpu.computeFn({
    workgroupSize: [WORKGROUP],
    in: { gid: d.builtin.globalInvocationId },
  })(({ gid }) => {
    'use gpu';
    if (gid.x >= LEVELS) return;
    let along = d.f32(0);
    let before = rings.pointAt(gid.x, 0).position.xz;
    for (let i = d.u32(1); i < POINTS; i++) {
      const point = lines.ControlPoint(rings.pointAt(gid.x, i));
      along += std.distance(point.position.xz, before);
      before = point.position.xz;
      point.along = along;
      rings.setPoint(gid.x, i, point);
    }
  });

  const tracing = root.createComputePipeline({ compute: trace });
  const measuring = root.createComputePipeline({ compute: measure });

  /** Each ring's radius at each point, read back once for picking. */
  let radii: Float32Array | undefined;
  const values = { lift: 0, width };
  let hovered = LEVELS;
  let changed = true;

  const hit = [0, 0, 0];
  /**
   * The ring under `ray`, or `LEVELS` for none. The rings read as the terraces of a solid model,
   * so the ray lands on the highest terrace whose ring encloses it, and picks the nearer of that
   * terrace's edges. A ring the island hides is never picked.
   */
  const ringUnder = (ray: Ray, measured: Float32Array) => {
    for (let ring = LEVELS - 1; ring >= 0; ring--) {
      if (cameras.hitHeight(ray, liftedOf(ring, values.lift), hit, LEVEL) === undefined) continue;
      const out = Math.hypot(hit[0], hit[2]);
      const edge = radiusOf(measured, ring, hit);
      if (out > edge + (ring === 0 ? NEAR_RING : 0)) continue;
      if (ring === LEVELS - 1) return ring;
      // Higher rings are nested inside lower ones round the one peak
      return out - radiusOf(measured, ring + 1, hit) < edge - out ? ring + 1 : ring;
    }
    return LEVELS;
  };

  return {
    drawing: rings.drawing,
    initAsync: () => pipelines.compile([tracing, measuring]),

    /** Reads back the rings' radii, which stay as they rise. */
    async readRadii() {
      const points = await rings.points!.read();
      radii = Float32Array.from(points, ({ position }) => Math.hypot(position.x, position.z));
    },

    /** Sets how far the rings have risen, from 0 to 1. */
    lift(to: number) {
      if (to === values.lift) return;
      values.lift = to;
      changed = true;
    },

    set width(next: number) {
      if (next === values.width) return;
      values.width = next;
      changed = true;
    },

    set dashed(next: boolean) {
      shape.patch({ dashed: next ? 1 : 0 });
    },

    /** Records a compute pass that traces the rings, when their height or width changed. */
    trace(encoder: TgpuCommandEncoder) {
      if (!changed) return;
      changed = false;
      shape.patch({ lift: values.lift, width: values.width });
      const pass = encoder.beginComputePass();
      pass.setPipeline(tracing);
      pass.dispatchWorkgroups(Math.ceil(TOTAL / WORKGROUP));
      pass.setPipeline(measuring);
      pass.dispatchWorkgroups(1);
      pass.end();
    },

    /**
     * Picks out a ring under `ray`: of the highest terrace whose ring encloses where the ray lands,
     * the nearer of its two edges. Picks none without a ray.
     */
    hover(ray?: Ray) {
      const picked = ray && radii ? ringUnder(ray, radii) : LEVELS;
      if (picked === hovered) return;
      hovered = picked;
      shape.patch({ hovered });
    },

    destroy() {
      rings.destroy();
    },
  };
}
