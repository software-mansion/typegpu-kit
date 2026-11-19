import { cameras, Particle, particles, shading } from '@typegpu/kit';
import { d, std, type TgpuRoot } from 'typegpu';

const DROPS = 7000;
/** The box the rain falls through, which a drop leaves at the bottom and enters at the top. */
const LOW = [-6, 0, -8] as const;
const HIGH = [6, 7, 8] as const;
const FALL = d.vec3f(0.7, -7.5, 0.35);
/** Metres a streak spans along its fall, and across. */
const STREAK = 0.28;
const WIDTH = 0.01;
/** The faint grey a drop shows over the moon and sky light it glints. */
const SKY = d.vec3f(0.0105, 0.012, 0.0145);
/** The share of the light arriving at a drop that its streak glints. */
const GLINT = 0.012;

const fall = (drop: Particle) => {
  'use gpu';
  let next = Particle(drop);
  next.velocity = FALL * (0.85 + drop.seed * 0.3);
  if (next.position.y < LOW[1]) next.position.y += HIGH[1] - LOW[1];
  if (next.position.x > HIGH[0]) next.position.x -= HIGH[0] - LOW[0];
  if (next.position.z > HIGH[2]) next.position.z -= HIGH[2] - LOW[2];
  return next;
};

const streak = (drop: Particle, corner: d.v2f) => {
  'use gpu';
  const down = std.normalize(drop.velocity);
  const toEye = std.normalize(cameras.current.$.position - drop.position);
  const across = std.normalize(std.cross(down, toEye));
  return drop.position + across * (corner.x * drop.size * 0.5) + down * (corner.y * STREAK * 0.5);
};

/** A streak glints in the light arriving at its drop, from the lighting the scene's pass fills. */
const shine = (drop: Particle, local: d.v2f) => {
  'use gpu';
  const thin = 1 - std.abs(local.x);
  const along = 1 - local.y * local.y;
  const alpha = thin * along * drop.color.w;
  const lit = SKY + shading.incoming(drop.position) * GLINT;
  return d.vec4f(lit * alpha, 0);
};

/** Rain falling through the market, streaks that glint in the lamps' and the fire's light. */
export function createRain(root: TgpuRoot, view: cameras.CameraBase, options: { step: number }) {
  const { step } = options;
  const rain = particles.create(root, view, {
    capacity: DROPS,
    step,
    update: fall,
    place: streak,
    shade: shine,
    label: 'rain',
    settings: { gravity: [0, 0, 0], drag: 0 },
  });
  rain.burst({
    count: DROPS,
    position: [(LOW[0] + HIGH[0]) / 2, (LOW[1] + HIGH[1]) / 2, (LOW[2] + HIGH[2]) / 2],
    area: [(HIGH[0] - LOW[0]) / 2, (HIGH[1] - LOW[1]) / 2, (HIGH[2] - LOW[2]) / 2],
    color: [1, 1, 1, 0.5],
    size: WIDTH,
    life: 1e9,
    variance: 0.5,
  });

  return rain;
}
