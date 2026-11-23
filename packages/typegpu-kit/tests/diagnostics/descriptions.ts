import { d, type TgpuRoot } from 'typegpu';
import { passes, targets } from '../../src/index.ts';

declare const root: TgpuRoot;
const screen = targets.create(
  root,
  { width: 1, height: 1 },
  {
    hdr: { format: 'rgba16float' },
    id: { format: 'r32uint' },
    depth: { format: 'depth24plus' },
  },
);

const id = passes.output(d.u32, {
  write: (pixel: { id: number }) => {
    'use gpu';
    return pixel.id;
  },
});

export const unknown = passes.create(root, screen, { hdr: passes.color, deep: passes.depth });

const scene = passes.create(root, screen, { hdr: passes.color, id, depth: passes.depth });

// The description keeps exactly the writes it was given
export const writes: {
  readonly hdr: typeof passes.color;
  readonly id: typeof id;
  readonly depth: typeof passes.depth;
} = scene.writes;

const Corner = d.struct({ position: d.vec4f });
const Glow = d.struct({ color: d.vec4f });

// `passes.drawing` infers the vertex and the pixel, so its callbacks need no annotations
const embers = passes.drawing({
  Vertex: Corner,
  at: () => {
    'use gpu';
    return Corner();
  },
  pixelOf: (vertex) => {
    'use gpu';
    return Glow({ color: d.vec4f(vertex.position.z) });
  },
  color: passes.shaded,
  draw: (pass) => pass.draw(3),
});

export const any: passes.AnyDrawing = embers;

// Embers carry no id for the `id` target
export const missing = scene.add(embers);
export const left = scene.without('id').add(embers);

// A field the vertex lacks is an error where it is read
export const lacking = passes.drawing({
  Vertex: Corner,
  at: () => {
    'use gpu';
    return Corner();
  },
  pixelOf: (vertex) => {
    'use gpu';
    return Glow({ color: d.vec4f(vertex.uv.x) });
  },
  color: passes.shaded,
  draw: (pass) => pass.draw(3),
});
