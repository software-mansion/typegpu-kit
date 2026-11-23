import { d, std } from 'typegpu';
import type { Frame as Placed } from './vessel.ts';

/** A frame as the shaders read it: its rotation's columns, then its translation. */
export const Frame = d.struct({ x: d.vec4f, y: d.vec4f, z: d.vec4f, t: d.vec4f });

const AXES = ['x', 'y', 'z', 't'] as const;

/** `frame` as the shaders read it, written into `into` if given, which allocates nothing. */
export const gpuFrame = (frame: Placed, into = Frame()) => {
  for (const axis of AXES) {
    into[axis].x = frame[axis][0];
    into[axis].y = frame[axis][1];
    into[axis].z = frame[axis][2];
  }
  return into;
};

/** World vector `v` in `frame`'s local space; subtract `frame.t` first for a point. */
export const toLocal = (frame: d.Infer<typeof Frame>, v: d.v3f) => {
  'use gpu';
  return d.vec3f(std.dot(frame.x.xyz, v), std.dot(frame.y.xyz, v), std.dot(frame.z.xyz, v));
};

/** `frame`'s local vector `v` in world space. */
export const fromLocal = (frame: d.Infer<typeof Frame>, v: d.v3f) => {
  'use gpu';
  return frame.x.xyz * v.x + frame.y.xyz * v.y + frame.z.xyz * v.z;
};
