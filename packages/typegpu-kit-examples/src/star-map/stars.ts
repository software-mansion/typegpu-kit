import { cameras, pipelines, type CameraBase } from '@typegpu/kit';
import { d, std, tgpu, type TgpuReadonly, type TgpuRenderCommands, type TgpuRoot } from 'typegpu';

export const Star = d.struct({
  position: d.vec3f,
  /** World units from its middle to its edge. */
  radius: d.f32,
  /** Linear HDR colour. */
  color: d.vec3f,
});

type Vec3 = [number, number, number];

export interface StarData {
  readonly position: Vec3;
  readonly radius: number;
  readonly color: Vec3;
}

const ARMS = 3;
/**
 * Galaxy shape in world units and radians. Stars span radii `core` to `core + disc`. Arms wind
 * by `winding` radians per unit radius. Stars deviate by `spread` radians and up to `height`
 * above or below the disc, tapering toward the rim.
 */
const SHAPE = { core: 0.3, disc: 6, winding: 0.75, spread: 0.3, height: 0.18 };
/** Minimum and maximum star radius in world units, biased toward small stars. */
const RADIUS = [0.02, 0.1];
/** Minimum and maximum brightness, biased toward dim stars. */
const SHINE = [1.5, 7.5];

/** How sharply a star's light falls off from its middle. */
const FALLOFF = 9;
/** Radius growth per unit of `glow`. */
const SWELL = 3;
/** Highlight-ring radius, width and brightness. Radius and width use star-radius units. */
const RING = { radius: 0.8, width: 0.08, brightness: 0.6 };

/** A standard normal sample, by the Box–Muller transform. */
const normal = () =>
  Math.sqrt(-2 * Math.log(1 - Math.random())) * Math.cos(Math.PI * 2 * Math.random());

/**
 * Generates `count` stars in a spiral around the origin on the CPU. Outer arms favour hot blue
 * stars, and the core favours cooler yellow and orange stars.
 */
export function galaxy(count: number): StarData[] {
  const blue: Vec3 = [0.62, 0.74, 1];
  const yellow: Vec3 = [1, 0.86, 0.66];
  const orange: Vec3 = [1, 0.5, 0.28];

  return Array.from({ length: count }, (_, i) => {
    const out = Math.random() ** 0.7;
    const radius = SHAPE.core + out * SHAPE.disc;
    const arm = ((i % ARMS) / ARMS) * Math.PI * 2;
    const angle = arm + radius * SHAPE.winding + normal() * SHAPE.spread;
    const height = normal() * SHAPE.height * (1.1 - out);

    // Bias outer stars toward blue while retaining all colours at every radius
    const heat = Math.min(1, Math.max(0, out + normal() * 0.25));
    const [from, to, t] = heat < 0.5 ? [orange, yellow, heat * 2] : [yellow, blue, heat * 2 - 1];
    const shine = SHINE[0] + Math.random() ** 3 * (SHINE[1] - SHINE[0]);

    return {
      position: [Math.cos(angle) * radius, height, Math.sin(angle) * radius],
      radius: RADIUS[0] + Math.random() ** 6 * (RADIUS[1] - RADIUS[0]),
      color: from.map((c, k) => (c + (to[k] - c) * t) * shine) as Vec3,
    };
  });
}

/**
 * Draws additive camera-facing star discs into the caller's pass. `glow` increases size and
 * brightness and adds a ring.
 */
export function createStars(
  root: TgpuRoot,
  view: CameraBase,
  options: {
    stars: TgpuReadonly<d.WgslArray<typeof Star>>;
    format: GPUTextureFormat;
    glow: (i: number) => number;
  },
) {
  const { stars, format, glow } = options;
  const count = stars.buffer.dataType.elementCount;

  const billboard = tgpu.vertexFn({
    in: { vertex: d.builtin.vertexIndex, instance: d.builtin.instanceIndex },
    out: { position: d.builtin.position, at: d.vec2f, color: d.vec3f, lift: d.f32 },
  })(({ vertex, instance }) => {
    'use gpu';
    const star = stars.$[instance];
    const lift = glow(instance);
    const corner = d.vec2f(std.select(-1, 1, vertex % 2 === 1), std.select(-1, 1, vertex > 1));
    const reach = star.radius * (1 + lift * SWELL);
    const world =
      star.position +
      (cameras.current.$.right * corner.x + cameras.current.$.up * corner.y) * reach;
    return {
      position: cameras.toClip(world),
      at: corner,
      color: star.color,
      lift,
    };
  });

  const disc = tgpu.fragmentFn({
    in: { at: d.vec2f, color: d.vec3f, lift: d.f32 },
    out: d.vec4f,
  })(({ at, color, lift }) => {
    'use gpu';
    const r = std.length(at);
    const core = std.exp(-r * r * FALLOFF);
    const ring = std.smoothstep(RING.width, 0, std.abs(r - RING.radius)) * std.min(lift, 1);
    return d.vec4f(color * (core * (1 + lift) + ring * RING.brightness), 0);
  });

  const pipeline = root.pipe(view.inject()).createRenderPipeline({
    vertex: billboard,
    fragment: disc,
    targets: { format, blend: pipelines.blend.add },
    primitive: { topology: 'triangle-strip' },
  });

  return {
    initAsync: () => pipeline.initAsync(),

    draw(pass: TgpuRenderCommands) {
      pass.setPipeline(pipeline);
      pass.draw(4, count);
    },
  };
}
