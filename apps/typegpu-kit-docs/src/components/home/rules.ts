// The birds' rules, which the home page shows as written here: each from its comment to its end

import { d, std } from 'typegpu';
import { airAt, tuned, type Near, type Seen } from './birds.ts';

// Separate from nearby birds, align and move towards the flock
export const flock = (me: Seen, near: Near) => {
  'use gpu';
  const along = near.heading - me.velocity;
  const toward = near.centre * tuned.$.toward;
  return near.away * 1.1 + along + toward + airAt(me) * 1.6;
};

// Follow a ring and avoid nearby birds
export const orbit = (me: Seen, near: Near) => {
  'use gpu';
  const out = d.vec3f(me.position.x, 0, me.position.z);
  const from = std.max(std.length(out), 1e-3);
  const around = d.vec3f(-out.z, 0, out.x) / from;
  const back = (out / from) * ((tuned.$.radius - from) * 3);
  return (around * 4 - me.velocity) * 2 + back + near.away;
};

// Slow down near neighbours and move towards them
export const huddle = (me: Seen, near: Near) => {
  'use gpu';
  const crowded = std.saturate(near.mates / tuned.$.crowd);
  const wanted = airAt(me) * (4 * (1 - crowded));
  const closer = near.centre * (crowded * 3);
  return (wanted - me.velocity) * 2 + closer + near.away * 0.6;
};

export const RULES = { flock, orbit, huddle };
export type RuleName = keyof typeof RULES;
