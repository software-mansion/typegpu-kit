import { cameras, history, passes } from '@typegpu/kit';
import { meshes } from '@typegpu/kit/geometry';
import { d, std, tgpu, type TgpuRoot } from 'typegpu';

/** Nodes along each side of the lattice, a unit apart. */
const SIDE = 6;
/** Half-extents measured from the centre to the outer faces. */
export const HALF = SIDE / 2;
/** Beams along each axis. */
const ROW = SIDE * SIDE * (SIDE - 1);
const BEAMS = 3 * ROW;
/** Beam width falls below one pixel after a few nodes, causing shimmer without TAA. */
const THICKNESS = 0.035;
/** Radians the lattice tips towards the camera, so its top shows. */
const TILT = 0.35;

const SUN = std.normalize(d.vec3f(0.5, 0.8, 0.35));
// Steel and ember-glow colours are in linear light
const STEEL = d.vec3f(0.55, 0.6, 0.7);
const EMBER = d.vec3f(2.4, 0.7, 0.15);

const beam = meshes.unindexed(meshes.box());

const Corner = d.struct({ position: d.vec4f, world: d.vec3f, normal: d.vec3f });
const Seen = d.struct({ world: d.vec3f, color: d.vec4f });
type Point = d.Infer<typeof Corner>;
type Lit = d.Infer<typeof Seen>;

/** Rotation about y in radians. History provides the last-frame value. */
const turn = tgpu.accessor(d.f32);

const placed = (point: d.v3f) => {
  'use gpu';
  const tilted = d.vec3f(
    point.x,
    point.y * std.cos(TILT) - point.z * std.sin(TILT),
    point.y * std.sin(TILT) + point.z * std.cos(TILT),
  );
  const c = std.cos(turn.$);
  const s = std.sin(turn.$);
  return d.vec3f(c * tilted.x + s * tilted.z, tilted.y, c * tilted.z - s * tilted.x);
};

const at = (vertexIndex: number, instanceIndex: number) => {
  'use gpu';
  const axis = std.intdiv(instanceIndex, d.u32(ROW));
  const k = instanceIndex % ROW;
  const along = d.f32(k % (SIDE - 1)) + 0.5;
  const a = d.f32(std.intdiv(k, d.u32(SIDE - 1)) % SIDE);
  const b = d.f32(std.intdiv(k, d.u32((SIDE - 1) * SIDE)));

  let middle = d.vec3f(along, a, b);
  let size = d.vec3f(1 + THICKNESS, THICKNESS, THICKNESS);
  if (axis === 1) {
    middle = d.vec3f(a, along, b);
    size = d.vec3f(THICKNESS, 1 + THICKNESS, THICKNESS);
  }
  if (axis === 2) {
    middle = d.vec3f(a, b, along);
    size = d.vec3f(THICKNESS, THICKNESS, 1 + THICKNESS);
  }

  const corner = beam.vertexAt(vertexIndex);
  const world = placed(middle - (SIDE - 1) / 2 + corner.position * size);
  return Corner({
    position: cameras.toClip(world),
    world,
    normal: placed(corner.normal),
  });
};

const pixelOf = (corner: Point) => {
  'use gpu';
  const light = std.max(std.dot(std.normalize(corner.normal), SUN), 0);
  const glow = std.exp((-HALF - corner.world.y) * 0.9);
  return Seen({ world: corner.world, color: d.vec4f(STEEL * (0.06 + light) + EMBER * glow, 1) });
};

/**
 * Draws a rotating lattice into scene passes. Rotation uses history so last-frame vertex
 * evaluation reads the last-frame angle.
 */
export function createLattice(root: TgpuRoot) {
  const turned = history.create(root, d.f32, { usage: 'uniform' });
  const drawing: passes.Drawing<typeof Corner, Lit> = {
    Vertex: Corner,
    at,
    pixelOf,
    color: passes.shaded,
    primitive: { cullMode: 'back' },
    inject: () => (cfg) => cfg.with(turn, turned.reading),
    draw: (pass) => pass.draw(beam.vertexCount, BEAMS),
  };
  const plain: passes.Drawing<typeof Corner, Lit> = {
    ...drawing,
    inject: () => (cfg) => cfg.with(turn, turned.current.as('uniform')),
  };

  return {
    drawing,
    /**
     * The same beams using a plain rotation uniform, so motion vectors contain only camera
     * motion.
     */
    plain,
    turned,
  };
}
