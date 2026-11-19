import { cameras, passes } from '@typegpu/kit';
import { meshes } from '@typegpu/kit/geometry';
import { perlin3d, randf } from '@typegpu/noise';
import { d, std, type TgpuRoot } from 'typegpu';
import { rateOf } from './data.ts';

const TAU = Math.PI * 2;
/** Radians a second round the sun of an orbit one world unit out. Wider orbits are slower. */
const RATE = rateOf(1);
/** Semi-major axes in world units, between Mars's orbit and Jupiter's. */
const INNER = 7;
const OUTER = 7.6;
const MOST_ECCENTRIC = 0.05;
/** Spread of inclinations in radians, which sets how thick the belt is. */
const INCLINATION = 0.035;
/** Radii in world units of the smallest rock and the largest, which few come near. */
const SMALLEST = 0.014;
const LARGEST = 0.05;
/** How far facets stand out from or sink into a rock, as a fraction of its radius. */
const BUMPS = 0.3;
/** Radians a second the fastest rock tumbles. */
const TUMBLE = 1.6;
/** Newton steps solving Kepler's equation, from a guess already close at low eccentricity. */
const NEWTON_STEPS = 3;
/** Linear colours of the darkest rock and the palest. */
const DARK = d.vec3f(0.04, 0.037, 0.034);
const PALE = d.vec3f(0.11, 0.1, 0.09);
const SUNLIGHT = d.vec3f(1.5, 1.4, 1.22);
const SHADE = d.vec3f(0.05, 0.05, 0.08);
/** The share of rocks with an icy facet that glints when it turns to the sun, and how brightly. */
const GLINTING = 0.015;
const GLINT = d.vec3f(4, 3.8, 3.5);
const GLINT_SHARPNESS = 40;

const rock = meshes.unindexed(meshes.icosphere({ radius: 1, segments: 1 }));

const Corner = d.struct({
  position: d.vec4f,
  world: d.vec3f,
  color: d.vec3f,
  glints: d.f32,
});
const Seen = d.struct({ color: d.vec4f });

/** Where a rock is round the sun at `time`, from orbital elements seeded by its index. */
const orbitAt = (time: number) => {
  'use gpu';
  // Rocks bunch towards the middle of the belt
  const axis = std.mix(INNER, OUTER, (randf.sample() + randf.sample()) / 2);
  const eccentricity = randf.sample() * MOST_ECCENTRIC;
  const inclination = randf.normal(0, INCLINATION);
  const node = randf.sample() * TAU;
  const perihelion = randf.sample() * TAU;

  // Kepler's equation, M = E - e sin E, from the mean anomaly round to the eccentric one
  let mean = randf.sample() * TAU + (RATE / (axis * std.sqrt(axis))) * time;
  mean -= TAU * std.floor(mean / TAU + 0.5);
  let eccentric = mean + eccentricity * std.sin(mean);
  for (let step = 0; step < NEWTON_STEPS; step++) {
    const miss = eccentric - eccentricity * std.sin(eccentric) - mean;
    eccentric -= miss / (1 - eccentricity * std.cos(eccentric));
  }

  const along = axis * (std.cos(eccentric) - eccentricity);
  const beside = axis * std.sqrt(1 - eccentricity * eccentricity) * std.sin(eccentric);
  const u = along * std.cos(perihelion) - beside * std.sin(perihelion);
  const v = along * std.sin(perihelion) + beside * std.cos(perihelion);
  const tilted = v * std.cos(inclination);
  return d.vec3f(
    u * std.cos(node) - tilted * std.sin(node),
    v * std.sin(inclination),
    -(u * std.sin(node) + tilted * std.cos(node)),
  );
};

/** Turns `point` about the unit `axis` by `angle`. */
const turn = (point: d.v3f, axis: d.v3f, angle: number) => {
  'use gpu';
  const cos = std.cos(angle);
  return (
    point * cos +
    std.cross(axis, point) * std.sin(angle) +
    axis * (std.dot(axis, point) * (1 - cos))
  );
};

/**
 * Draws a main belt of `count` tumbling rocks between Mars and Jupiter. Each rock's orbit, shape
 * and tumble follow from its index and the time, and nothing is stored between frames, so any
 * moment draws the same. It writes no depth, so add it after the sky.
 */
export function createBelt(root: TgpuRoot, options: { count: number }) {
  const { count } = options;
  const time = root.createUniform(d.f32);

  const drawing: passes.Drawing<typeof Corner, d.Infer<typeof Seen>> = {
    label: 'belt',
    Vertex: Corner,
    flat: ['color', 'glints'],
    at: (vertexIndex: number, instanceIndex: number) => {
      'use gpu';
      randf.seed(d.f32(instanceIndex) + 1);
      const center = orbitAt(time.$);

      const radius = std.mix(SMALLEST, LARGEST, std.pow(randf.sample(), 4));
      const stretch = d.vec3f(randf.sample(), randf.sample(), randf.sample()) * 0.5 + 0.6;
      const lumps = randf.inUnitCube() * 50;
      const axis = randf.onUnitSphere();
      const tumble = randf.sample() * TAU + (randf.sample() * 2 - 1) * TUMBLE * time.$;
      const color = std.mix(DARK, PALE, randf.sample());
      const glints = std.select(d.f32(0), d.f32(1), randf.sample() < GLINTING);

      const unit = rock.vertexAt(vertexIndex).position;
      const bump = 1 + perlin3d.sample(unit * 1.4 + lumps) * BUMPS;
      const world = center + turn(unit * stretch * (radius * bump), axis, tumble);
      return Corner({
        position: cameras.toClip(world),
        world,
        color,
        glints,
      });
    },
    pixelOf: (corner) => {
      'use gpu';
      const view = cameras.towards(cameras.current.$, corner.world);
      const facet = std.normalize(std.cross(std.dpdx(corner.world), std.dpdy(corner.world)));
      const normal = std.select(facet, facet * -1, std.dot(facet, view) < 0);
      const sun = std.normalize(corner.world * -1);
      const day = std.saturate(std.dot(normal, sun));
      const glint = std.pow(
        std.saturate(std.dot(normal, std.normalize(sun + view))),
        GLINT_SHARPNESS,
      );
      const lit = corner.color * (SUNLIGHT * day + SHADE) + GLINT * (glint * day * corner.glints);
      return Seen({ color: d.vec4f(lit, 1) });
    },
    color: passes.shaded,
    // Hidden by the bodies, the belt leaves depth alone so orbit paths and labels read over it
    depthStencil: { depthWriteEnabled: false, depthCompare: 'less' },
    primitive: { cullMode: 'back' },
    draw: (pass) => pass.draw(rock.vertexCount, count),
  };

  return {
    drawing,
    /** Moves every rock to `seconds` of simulated time. */
    move(seconds: number) {
      time.write(seconds);
    },
  };
}
