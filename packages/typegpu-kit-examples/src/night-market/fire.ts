import { cameras, Particle, particles } from '@typegpu/kit';
import { d, std, type TgpuRoot } from 'typegpu';

type Vec3 = readonly [number, number, number];

/** Flames a second, and embers a second among them. */
const FLAMES = 70;
const EMBERS = 9;
/** Where the flames are hottest, white-yellow, then orange, then a dull red as they fade. */
const CORE = d.vec3f(5, 3.2, 1.2);
const BODY = d.vec3f(3.2, 0.9, 0.15);
const SMOKE = d.vec3f(0.5, 0.06, 0.01);
const SPARK = d.vec3f(7, 2.6, 0.6);
/** Metres across a flame particle and an ember. Anything smaller than `EMBER_BELOW` is an ember. */
const FLAME_SIZE = 0.3;
const EMBER_SIZE = 0.025;
const EMBER_BELOW = 0.05;

const rise = (flame: Particle) => {
  'use gpu';
  let next = Particle(flame);
  const ember = flame.size < EMBER_BELOW;
  const flutter = flame.age * 9 + flame.seed * 40;
  if (ember) {
    next.velocity += d.vec3f(std.sin(flutter) * 0.06, 0.02, std.cos(flutter) * 0.06);
  } else {
    // Flames draw in towards the middle as they climb
    next.velocity = d.vec3f(next.velocity.x * 0.9, next.velocity.y + 0.035, next.velocity.z * 0.9);
  }
  return next;
};

const burn = (flame: Particle, local: d.v2f) => {
  'use gpu';
  const age = flame.age / flame.life;
  const edge = std.saturate(1 - std.length(local));
  if (flame.size < EMBER_BELOW) {
    const glow = edge * edge * (1 - age) * (0.6 + 0.4 * std.sin(flame.age * 30 + flame.seed * 20));
    return d.vec4f(SPARK * glow, 0);
  }
  const heat = std.mix(CORE, BODY, std.smoothstep(0, 0.45, age));
  const color = std.mix(heat, SMOKE, std.smoothstep(0.45, 1, age));
  const fade = edge * edge * std.smoothstep(0, 0.1, age) * (1 - age);
  return d.vec4f(color * (fade * 0.5), 0);
};

const shrink = (flame: Particle, corner: d.v2f) => {
  'use gpu';
  const across = cameras.current.$.right * corner.x + cameras.current.$.up * corner.y;
  const size = flame.size * (1 - (flame.age / flame.life) * 0.6);
  return flame.position + across * (size * 0.5);
};

/** Flames licking up from the logs at `at` and embers drifting up into the rain. */
export function createFire(
  root: TgpuRoot,
  view: cameras.CameraBase,
  options: { step: number; at: Vec3 },
) {
  const { step, at } = options;
  const fire = particles.create(root, view, {
    capacity: 512,
    step,
    update: rise,
    place: shrink,
    shade: burn,
    label: 'fire',
    settings: { gravity: [0, 0, 0], drag: 0.6 },
  });

  const flames = {
    position: [at[0], at[1] + 0.12, at[2]] as const,
    area: [0.18, 0.04, 0.18] as const,
    velocity: [0, 0.75, 0] as const,
    scatter: 0.12,
    size: FLAME_SIZE,
    life: 0.75,
    variance: 0.4,
  };
  const flaming = fire.stream(FLAMES, flames);
  const sparking = fire.stream(EMBERS, {
    position: [at[0], at[1] + 0.3, at[2]],
    area: [0.15, 0.05, 0.15],
    velocity: [0, 1.6, 0],
    scatter: 0.5,
    size: EMBER_SIZE,
    life: 2.6,
    variance: 0.5,
  });

  return {
    drawing: fire.drawing,
    spawn: fire.spawn,
    step: fire.step,
    compact: fire.compact,
    initAsync: () => fire.initAsync(),
    destroy() {
      fire.destroy();
    },

    /** Burns at `strength`, 1 for the full fire: more and larger flames, and more embers. */
    burn(strength: number) {
      flaming.rate = FLAMES * strength;
      sparking.rate = EMBERS * strength;
      flames.size = FLAME_SIZE * Math.sqrt(strength);
    },
  };
}
