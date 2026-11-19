import { d, std } from 'typegpu';

// Octahedral encoding maps unit vectors to the square [-1, 1]²

const signs = (v: d.v2f) => {
  'use gpu';
  return std.select(d.vec2f(-1), d.vec2f(1), std.ge(v, d.vec2f()));
};

/** Folds a unit vector onto the octahedron, then the lower half over the upper. */
export const encode = (n: d.v3f) => {
  'use gpu';
  const flat = n.xy / (std.abs(n.x) + std.abs(n.y) + std.abs(n.z));
  const folded = (1 - std.abs(flat.yx)) * signs(flat);
  return std.select(flat, folded, n.z < 0);
};

/** Unfolds a point of the square back into a unit vector. */
export const decode = (e: d.v2f) => {
  'use gpu';
  const z = 1 - std.abs(e.x) - std.abs(e.y);
  const fold = std.saturate(-z);
  return std.normalize(d.vec3f(e - signs(e) * fold, z));
};
