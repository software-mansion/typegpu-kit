import { cameras, passes } from '@typegpu/kit';
import { randf } from '@typegpu/noise';
import { d, std } from 'typegpu';
import { quadCorner } from './tissue.ts';
import { hazed, lightAt, Seen, stirAt, SUNLIGHT, timeOf } from './water.ts';

/** Motes of marine snow, and the box they drift down through. */
const MOTES = 700;
const LOW = d.vec3f(-4, -2.6, -7);
const HIGH = d.vec3f(4, 2.4, 1.6);

const Mote = d.struct({
  position: d.vec4f,
  world: d.vec3f,
  /** The quad's corner, from -1 to 1 on each axis. */
  corner: d.vec2f,
  /** The share of the drawn size the mote fills, and how far through its fall it is. */
  coverage: d.f32,
  fall: d.f32,
});

const moteAt = (vertexIndex: number, instanceIndex: number) => {
  'use gpu';
  randf.seed(d.f32(instanceIndex) + 0.5);
  const start = d.vec3f(randf.sample(), randf.sample(), randf.sample());
  const size = 0.002 + randf.sample() * 0.003;
  const time = timeOf.$;

  // Each mote sinks slowly, the larger ones faster, and wraps round to the top of the box
  const fall = std.fract(start.y - (time * (0.01 + size * 4)) / (HIGH.y - LOW.y));
  const sway = std.sin(time * 0.25 + start.z * 40) * 0.04;
  const resting = LOW + (HIGH - LOW) * d.vec3f(start.x, fall, start.z) + d.vec3f(sway, 0, 0);
  // A stir carries the motes along while it lasts
  const centre = resting + stirAt(resting) * 0.15;

  const corner = d.vec2f(quadCorner(vertexIndex)) * 2 - 1;
  const toEye = std.normalize(cameras.current.$.position - centre);
  const right = std.normalize(std.cross(d.vec3f(0, 1, 0), toEye));
  const up = std.cross(toEye, right);

  const clip = cameras.toClip(centre);
  const pixel = (2 * clip.w) / (cameras.lensOf(cameras.current.$).y * passes.viewport.$.size.y);
  // Motes very near the eye are out of focus: larger and fainter
  const blur = std.max(1.5 - std.length(cameras.current.$.position - centre), 0) * 0.008;
  const drawn = std.max(size + blur, pixel * 1.5);
  const world = centre + (right * corner.x + up * corner.y) * drawn;
  const share = size / drawn;

  return Mote({
    position: cameras.toClip(world),
    world,
    corner,
    coverage: share * share,
    fall,
  });
};

const moteOf = (mote: d.Infer<typeof Mote>) => {
  'use gpu';
  const r = std.length(mote.corner);
  // Fade in at the top of the box and out at its floor, so the wrap never shows
  const ends = std.smoothstep(0, 0.08, mote.fall) * (1 - std.smoothstep(0.9, 1, mote.fall));
  const alpha = (1 - std.smoothstep(0.2, 1, r)) * 0.45 * mote.coverage * ends;
  const lit = SUNLIGHT * (lightAt(mote.world, timeOf.$) * 0.9);
  return Seen({ color: hazed(mote.world, lit * alpha, alpha) });
};

/** Marine snow drifting down through the tank, for scale and depth. */
export const snow: passes.Drawing<typeof Mote, Seen> = {
  label: 'marine snow',
  Vertex: Mote,
  at: moteAt,
  pixelOf: moteOf,
  color: passes.shaded,
  translucent: true,
  draw: (pass) => pass.draw(6, MOTES),
};
