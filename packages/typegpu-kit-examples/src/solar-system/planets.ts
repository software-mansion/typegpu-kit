import {
  cameras,
  pipelines,
  type CameraBase,
  instances,
  lines,
  materials,
  meshPool,
  picker,
  shading,
  type SizeSource,
} from '@typegpu/kit';
import { byStage, type Inject, unfilled } from '@typegpu/kit/inject';
import { meshes } from '@typegpu/kit/geometry';
import { perlin3d } from '@typegpu/noise';
import { d, std, tgpu, type TgpuComputePass, type TgpuRoot } from 'typegpu';
import {
  BANDED,
  BURNING,
  EARTH,
  EARTHLIKE,
  PLAIN,
  PLANETS,
  SATURN,
  MOON,
  SATURN_RINGS,
  SUN,
  rateOf,
} from './data.ts';

/** `Circuit.around` of a body that circles only the sun. */
const NONE = 0xffffffff;
const WORKGROUP = 64;
/** The sun, the planets in order, then the Moon. */
const BODIES = PLANETS.length + 2;
/** Points round each orbit's closed path. */
const SEGMENTS = 160;
const PATH_COLOR = d.vec3f(0.34, 0.38, 0.6);
/** Path width in points, and its opacity at its middle. */
const PATH_WIDTH = 1.25;
const PATH_OPACITY = 0.16;
/** Linear light a facet facing the sun receives, and light on the night side and at the rim. */
const SUNLIGHT = d.vec3f(1.5, 1.4, 1.22);
const NIGHT = d.vec3f(0.13, 0.13, 0.2);
const RIM = d.vec3f(0.12, 0.16, 0.32);

const Body = d.struct({ position: d.vec3f, spin: d.f32 });
const Focus = d.struct({ planet: d.u32, width: d.f32, grow: d.f32 });

/**
 * World units from the eye at which labels take their size. Nearer or further, they keep that size
 * on screen.
 */
export const SIZE_DISTANCE = 37;

/** A world point in picture heights from the middle of the picture, with y up. */
export const onPicture = (world: d.v3f) => {
  'use gpu';
  const eye = cameras.current.$;
  const clip = cameras.toClip(world);
  const lens = cameras.lensOf(eye);
  return (clip.xy / clip.w) * d.vec2f(lens.y / lens.x, 1) * 0.5;
};

/** World units at `SIZE_DISTANCE` from the top of a planet's details to their bottom. */
const DETAILS_HEIGHT = 1.5;
/** World units from a planet's surface to its name's baseline. */
export const NAME_LIFT = 0.12;
/**
 * World units at `SIZE_DISTANCE` from a name's baseline to the bottom of details above it: the
 * name's height and a gap.
 */
export const DETAILS_ABOVE = 0.6;
/**
 * World units at `SIZE_DISTANCE` the box round a planet's details reaches past their words, and
 * the room below them for a name that moves to their head.
 */
const [FACTS_PAD, NAME_ROOM] = [0.25, 0.62];
/** Picture heights of clear space details need from other bodies before they keep to one side. */
const FACTS_CLEARANCE = 0.01;
/** Each planet's band radius and radius, in world units. */
const MEASURES = tgpu.const(
  d.arrayOf(d.vec2f, PLANETS.length),
  PLANETS.map(({ bandRadius, radius }) => d.vec2f(bandRadius, radius)),
);
/** The sun's radius with its glow, then each planet's extent, rings included, in world units. */
const REACHES = tgpu.const(d.arrayOf(d.f32, PLANETS.length + 1), [
  SUN.radius * 1.2,
  ...PLANETS.map(({ extent }) => extent),
]);

/** How far a point on the picture stands outside a box, its middle then half its size. */
const gapOf = (point: d.v2f, box: d.v4f) => {
  'use gpu';
  const outside = std.abs(point - box.xy) - box.zw;
  return std.length(std.max(outside, d.vec2f())) + std.min(std.max(outside.x, outside.y), 0);
};

/**
 * Planet `planet`'s details and the name beside them as a box on the picture, below its band or
 * `above` it: its middle, then half its width and height. The details are `width` world units
 * wide at `SIZE_DISTANCE`, drawn `grow` times their size.
 */
export const detailsBox = (planet: number, width: number, grow: number, above: boolean) => {
  'use gpu';
  const at = bodies.$[planet + 1].position;
  const eye = cameras.current.$;
  const distance = std.distance(eye.position, at);
  const lens = cameras.lensOf(eye).y * grow;
  // Facts keep their world size up to `SIZE_DISTANCE` and their picture size beyond
  const near = (std.max(distance / SIZE_DISTANCE, 1) * lens) / (2 * distance);
  const far = lens / (2 * SIZE_DISTANCE);
  const measures = MEASURES.$[planet];
  // Upright text looks shorter by the cosine of the eye's height above it, and names lift by
  // world units. Below, the details leave room for the name at their head.
  const rise = std.clamp((eye.position.y - at.y) / distance, -0.9, 0.9);
  const cosine = std.sqrt(1 - rise * rise);
  const upright = cosine * far;
  const named = ((measures.y + NAME_LIFT) * lens) / (2 * distance);
  // The details face the eye from a circle round the planet, nearer the eye and so larger
  const size = distance / SIZE_DISTANCE;
  const radius = std.max(measures.x, (width / 1.8) * size);
  const lift = std.select(-(measures.x + 0.15) * std.max(size, 1), measures.y + NAME_LIFT, above);
  const closer = distance / (distance - (radius + lift * rise) / cosine);
  const under = -(measures.x + 0.15) * near * closer;
  const below = under - (NAME_ROOM + DETAILS_HEIGHT) * upright * closer;
  const bottom = std.select(below, named, above);
  const over = (named + DETAILS_ABOVE * upright) * closer + DETAILS_HEIGHT * upright * closer;
  const top = std.select(under, over, above);
  const half = d.vec2f(width * far, top - bottom) * 0.5 + FACTS_PAD * far;
  const centre = onPicture(at);
  return d.vec4f(centre.x, centre.y + (top + bottom) * 0.5, half);
};

/**
 * How far planet `planet`'s details have moved above its name, from 0 to 1. They keep below its
 * band unless the sun or another planet stands in their way there and less so above.
 */
export const aboveness = (planet: number, width: number, grow: number) => {
  'use gpu';
  const eye = cameras.current.$;
  const lens = cameras.lensOf(eye).y;
  const below = detailsBox(planet, width, grow, false);
  const above = detailsBox(planet, width, grow, true);
  let under = d.f32(1);
  let over = d.f32(1);
  for (let body = d.u32(0); body <= PLANETS.length; body++) {
    if (body !== planet + 1) {
      const at = bodies.$[body].position;
      const disc = (REACHES.$[body] * lens) / (2 * std.distance(eye.position, at));
      under = std.min(under, gapOf(onPicture(at), below) - disc);
      over = std.min(over, gapOf(onPicture(at), above) - disc);
    }
  }
  // Crowded on both sides, they take the clearer
  const clearer = std.smoothstep(-FACTS_CLEARANCE, FACTS_CLEARANCE, (over - under) * 10);
  return clearer * (1 - std.smoothstep(0, FACTS_CLEARANCE, under));
};

/**
 * A circle round the sun, or round body `around` too, at `rate` radians a second from `phase`, a
 * turn about the body's axis at `spin` radians a second, and the world units a pick reaches.
 */
const Circuit = d.struct({
  radius: d.f32,
  rate: d.f32,
  phase: d.f32,
  spin: d.f32,
  around: d.u32,
  reach: d.f32,
});

/** Each body's position and turn about its axis, after `move`. */
export const bodies = tgpu.accessor(
  d.arrayOf(Body, BODIES),
  unfilled('the planets', 'planets', 'createPlanets'),
);

/** Turns about +y by `spin`, then tilts that axis about +z by `tilt`. */
const turnOf = (tilt: number, spin: number) => {
  'use gpu';
  const tiltSin = std.sin(tilt / 2);
  const spinSin = std.sin(spin / 2);
  const tiltCos = std.cos(tilt / 2);
  const spinCos = std.cos(spin / 2);
  return d.vec4f(-tiltSin * spinSin, tiltCos * spinSin, spinCos * tiltSin, tiltCos * spinCos);
};

/** The rotation that undoes the unit quaternion `turn`. */
export const inverse = (turn: d.v4f) => {
  'use gpu';
  return d.vec4f(turn.xyz * -1, turn.w);
};

/** A unit normal constant over each triangle, facing the eye. */
const facetOf = (point: shading.Point) => {
  'use gpu';
  const normal = std.normalize(std.cross(std.dpdx(point.position), std.dpdy(point.position)));
  return std.select(normal, normal * -1, std.dot(normal, point.view) < 0);
};

/** A low-poly body whose colour patterns stay constant over each facet. */
const Faceted = instances.define('Faceted', {
  host: { body: d.u32, radius: d.f32, tilt: d.f32, accent: d.vec3f, pattern: d.u32 },
  place: (row) => {
    'use gpu';
    const body = bodies.$[row.body];
    return instances.Placement({
      rotation: turnOf(row.tilt, body.spin),
      position: body.position,
      scale: d.vec3f(row.radius),
    });
  },
  material: (pixel, base) => {
    'use gpu';
    const row = pixel.instance;
    const turn = turnOf(row.tilt, bodies.$[row.body].spin);
    const local = instances.rotate(inverse(turn), facetOf(pixel.point));
    const grain = perlin3d.sample(local * 2.5);
    let color = std.mix(base.color, row.accent, std.saturate(grain * 1.5 + 0.3));
    let emission = d.vec3f(base.emission);

    if (row.pattern === BANDED) {
      const band = std.sin(local.y * 9 + grain * 1.5);
      color = std.mix(base.color, row.accent, std.smoothstep(-0.2, 0.2, band));
    } else if (row.pattern === EARTHLIKE) {
      const land = perlin3d.sample(local * 1.6 + d.vec3f(3, 1, 2)) > 0.08;
      color = std.select(base.color, row.accent, land);
      color = std.select(color, d.vec3f(0.92, 0.95, 1), std.abs(local.y) > 0.8);
    } else if (row.pattern === BURNING) {
      emission = base.emission * (0.75 + grain * 0.5);
    }

    const made = materials.matte(color);
    made.emission = d.vec3f(emission);
    return made;
  },
});

/** Lights each facet from the sun at the origin. Emissive facets dim towards their edges. */
const sunlit = (material: shading.Material, point: shading.Point) => {
  'use gpu';
  const normal = facetOf(point);
  const day = std.saturate(std.dot(normal, std.normalize(point.position * -1)));
  const facing = std.saturate(std.dot(normal, point.view));
  const rim = (1 - facing) * (1 - facing) * (1 - facing);
  const lit = material.color * (SUNLIGHT * day + NIGHT + RIM * rim);
  return lit + material.emission * (0.65 + 0.35 * facing);
};

/**
 * Draws the sun, the planets, Saturn's ring, the Moon and each planet's orbit. One kernel a frame
 * moves the bodies, which other modules read through `inject()`.
 */
export function createPlanets(root: TgpuRoot, view: CameraBase, size: SizeSource) {
  const placed = root.createBuffer(d.arrayOf(Body, BODIES)).$usage('storage');
  const placement = byStage(placed);
  const inject: Inject = (cfg) => cfg.with(bodies, placement);

  const positions = placed.as('readonly');
  /**
   * The chosen planet, whose band and details the orbit paths leave clear, or `NONE`, its details'
   * width at `SIZE_DISTANCE` and how many times their size labels are drawn.
   */
  const focus = root.createUniform(Focus, { planet: NONE, width: 0, grow: 1 });

  /** How far a point is clear on the picture of the chosen planet's band and details, 0 to 1. */
  const clearOfFacts = (world: d.v3f) => {
    'use gpu';
    const chosen = focus.$;
    if (chosen.planet === NONE) return d.f32(1);
    const at = bodies.$[chosen.planet + 1].position;
    const eye = cameras.current.$;
    const distance = std.distance(eye.position, at);
    const lens = cameras.lensOf(eye).y * chosen.grow;
    const band =
      (MEASURES.$[chosen.planet].x * std.max(distance / SIZE_DISTANCE, 1) * lens) / (2 * distance);
    const picture = onPicture(world);
    const margin = band * 0.3;
    let clear = std.smoothstep(0, margin, std.distance(picture, onPicture(at)) - band);
    const above = aboveness(chosen.planet, chosen.width, chosen.grow);
    const under = detailsBox(chosen.planet, chosen.width, chosen.grow, false);
    const over = detailsBox(chosen.planet, chosen.width, chosen.grow, true);
    clear = std.min(clear, std.mix(1, std.smoothstep(0, margin, gapOf(picture, under)), 1 - above));
    return std.min(clear, std.mix(1, std.smoothstep(0, margin, gapOf(picture, over)), above));
  };

  const pool = meshPool.create(root, { capacity: 2 });
  const encoder = root.createCommandEncoder();
  const ball = pool.add(meshes.icosphere({ radius: 1, segments: 2 }), encoder);
  const torus = meshes.torus({ radius: 1, tube: 0.12, ringSegments: 48, tubeSegments: 4 });
  const ring = pool.add(meshes.transform(torus, std.scaling4(d.vec3f(1, 0.25, 1))), encoder);
  encoder.submit();

  const capacity = BODIES + SATURN_RINGS.length;
  const faceted = instances.create(root, pool, { kind: Faceted, capacity });
  const renderer = faceted.renderer({ shade: sunlit }).pipe(inject);

  const time = root.createUniform(d.f32);
  const orbits = root.createReadonly(d.arrayOf(Circuit, BODIES), [
    { radius: 0, rate: 0, phase: 0, spin: SUN.spin, around: NONE, reach: 0 },
    ...PLANETS.map((planet) => ({
      radius: planet.orbit,
      rate: rateOf(planet.orbit),
      phase: planet.phase,
      spin: planet.spin,
      around: NONE,
      reach: planet.extent,
    })),
    { radius: MOON.orbit, rate: MOON.rate, phase: 0, spin: 0.3, around: EARTH + 1, reach: 0 },
  ]);

  const circle = (orbit: d.Infer<typeof Circuit>, t: number) => {
    'use gpu';
    const angle = orbit.phase + orbit.rate * t;
    return d.vec3f(std.cos(angle), 0, -std.sin(angle)) * orbit.radius;
  };

  const move = tgpu.computeFn({
    workgroupSize: [WORKGROUP],
    in: { gid: d.builtin.globalInvocationId },
  })(({ gid }) => {
    'use gpu';
    const i = gid.x;
    if (i >= BODIES) return;
    const orbit = Circuit(orbits.$[i]);
    let position = circle(orbit, time.$);
    if (orbit.around !== NONE) position += circle(Circuit(orbits.$[orbit.around]), time.$);
    bodies.$[i] = Body({ position, spin: orbit.spin * time.$ });
  });
  const moving = root.pipe(inject).createComputePipeline({ compute: move });

  const picks = picker.create(root, view, {
    capacity: PLANETS.length,
    size,
    position: (i) => {
      'use gpu';
      return d.vec3f(positions.$[i + 1].position);
    },
    radius: (i) => {
      'use gpu';
      return orbits.$[i + 1].reach;
    },
  });

  const paths = lines.create(root, {
    capacity: PLANETS.length,
    length: SEGMENTS,
    closed: true,
    units: 'points',
    label: 'orbits',
    shade: tgpu
      .fn((fragment: lines.Fragment) => {
        'use gpu';
        const alpha =
          (1 - fragment.across * fragment.across) *
          PATH_OPACITY *
          fragment.coverage *
          clearOfFacts(fragment.position);
        return d.vec4f(PATH_COLOR * alpha, alpha);
      })
      .pipe(inject),
  });
  const loop = new Float32Array(SEGMENTS * 3);
  for (const [i, planet] of PLANETS.entries()) {
    for (let k = 0; k < SEGMENTS; k++) {
      const angle = (k / SEGMENTS) * Math.PI * 2;
      loop[k * 3] = Math.cos(angle) * planet.orbit;
      loop[k * 3 + 2] = -Math.sin(angle) * planet.orbit;
    }
    paths.write(i, loop, { width: PATH_WIDTH });
  }

  const sun = { radius: SUN.radius, tilt: 0.1, color: d.vec3f(), emission: SUN.emission };
  faceted.add(ball, { body: 0, ...sun, accent: d.vec3f(), pattern: BURNING });
  for (const [i, { radius, tilt, color, accent, pattern }] of PLANETS.entries()) {
    faceted.add(ball, { body: i + 1, radius, tilt, color, accent, pattern });
  }
  const saturn = PLANETS[SATURN];
  for (const { radius, color } of SATURN_RINGS) {
    const band = { radius: saturn.radius * radius, tilt: saturn.tilt, color, accent: color };
    faceted.add(ring, { body: SATURN + 1, ...band, pattern: PLAIN });
  }
  const { radius, color, accent } = MOON;
  faceted.add(ball, { body: BODIES - 1, radius, tilt: 0, color, accent, pattern: PLAIN });
  faceted.flush();

  let chosen = { planet: NONE, width: 0, grow: 1 };

  return {
    renderer,
    /** Each planet's orbit path. Add after `renderer`, which hides the paths behind the bodies. */
    paths: paths.drawing,
    /** Picks the planet nearest a point in points, which reaches the CPU a frame or more later. */
    picks,
    inject: () => inject,
    initAsync: () => pipelines.compile([moving, picks]),

    /**
     * Clears the orbit paths from round `planet`'s band and its details, `width` world units wide
     * at `SIZE_DISTANCE` and drawn `grow` times their size. No planet clears nothing.
     */
    focusOn(planet: number | undefined, facts: { width: number; grow: number }) {
      const next = { planet: planet ?? NONE, ...facts };
      const { width, grow } = next;
      if (next.planet === chosen.planet && width === chosen.width && grow === chosen.grow) return;
      chosen = next;
      focus.write(next);
    },

    /** Moves every body to `seconds` of simulated time. */
    move(pass: TgpuComputePass, seconds: number) {
      time.write(seconds);
      pass.setPipeline(moving);
      pass.dispatchWorkgroups(Math.ceil(BODIES / WORKGROUP));
    },

    destroy() {
      paths.destroy();
      picks.destroy();
      faceted.destroy();
      pool.destroy();
      placed.destroy();
      orbits.buffer.destroy();
    },
  };
}
