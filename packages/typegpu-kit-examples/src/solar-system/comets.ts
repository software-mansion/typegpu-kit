import { instances, trails } from '@typegpu/kit';
import { quat } from 'math';
import { d, std, tgpu, type TgpuComputePass, type TgpuRoot } from 'typegpu';
import { rateOf } from './data.ts';

/** Seconds a step moves each comet and adds a point to its tail. */
export const STEP = 1 / 60;

/** Gravitational parameter in world units cubed per second squared, as the planets orbit. */
const GM = rateOf(1) ** 2;
const WORKGROUP = 64;
const COMETS = 2;
/** Points along each comet's tail, one a step. */
const TAIL_POINTS = 160;
/** Linear colours of the first comet's tail and the second's. */
const COOL = d.vec3f(0.55, 0.85, 1);
const WARM = d.vec3f(1, 0.7, 0.45);
/** How much brighter than its tail a comet's head glows. */
const HEAD = 7;

/** Turns about +x by `tilt`, then about +y by `turn`. */
const orientation = (tilt: number, turn: number) =>
  quat.fromEuler(quat.create(), [tilt, turn, 0, 'yxz']);

const Ellipse = d.struct({
  /** Semi-major axis in world units, and eccentricity from 0 for a circle towards 1. */
  axis: d.f32,
  eccentricity: d.f32,
  /** Radians round the sun from perihelion, which the kernel advances. */
  angle: d.f32,
  orientation: d.vec4f,
});

/** Moves two comets round the sun a fixed step at a time, pushing a point to each tail. */
export function createComets(root: TgpuRoot, options: { tailWidth: number }) {
  const { tailWidth } = options;
  const orbits = root.createMutable(d.arrayOf(Ellipse, COMETS), [
    { axis: 8.6, eccentricity: 0.8, angle: -1.3, orientation: orientation(0.3, 0.4) },
    { axis: 5.2, eccentricity: 0.72, angle: 2.2, orientation: orientation(-0.22, 2.6) },
  ]);

  const tails = trails.create(root, {
    capacity: COMETS,
    length: TAIL_POINTS,
    settings: { width: tailWidth },
    shade: (point, across) => {
      'use gpu';
      const alpha = (1 - across * across) * (1 - point.along);
      const glow = 1 + HEAD * std.exp(-point.along * 60);
      return d.vec4f(std.mix(COOL, WARM, d.f32(point.trail)) * alpha * glow, alpha);
    },
  });

  const move = tgpu.computeFn({
    workgroupSize: [WORKGROUP],
    in: { gid: d.builtin.globalInvocationId },
  })(({ gid }) => {
    'use gpu';
    const i = gid.x;
    if (i >= COMETS) return;

    // Equal areas in equal times: a comet moves faster near the sun
    const orbit = Ellipse(orbits.$[i]);
    const span = orbit.axis * (1 - orbit.eccentricity * orbit.eccentricity);
    const before = span / (1 + orbit.eccentricity * std.cos(orbit.angle));
    const angle = (orbit.angle + (std.sqrt(GM * span) / (before * before)) * STEP) % (Math.PI * 2);
    orbits.$[i].angle = angle;

    const distance = span / (1 + orbit.eccentricity * std.cos(angle));
    const flat = d.vec3f(std.cos(angle), 0, std.sin(angle)) * distance;
    tails.push(i, instances.rotate(orbit.orientation, flat));
  });
  const moving = root.createComputePipeline({ compute: move });

  return {
    tails,
    initAsync: () => moving.initAsync(),

    /** Records one step in the caller's compute pass. */
    step(pass: TgpuComputePass) {
      pass.setPipeline(moving);
      pass.dispatchWorkgroups(Math.ceil(COMETS / WORKGROUP));
    },

    destroy() {
      tails.destroy();
      orbits.buffer.destroy();
    },
  };
}
