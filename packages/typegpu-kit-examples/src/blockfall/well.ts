import { cameras, passes } from '@typegpu/kit';
import { meshes } from '@typegpu/kit/geometry';
import { srgbToLinear } from '@typegpu/color';
import { d, std } from 'typegpu';

/** The gold of the well's edges and cell lines, a little brighter than white so the edges glow. */
const GOLD = srgbToLinear(d.vec3f(1, 0.78, 0.48)).mul(1.4);
/** The faint cool tint of the glass walls. */
const GLASS = srgbToLinear(d.vec3f(0.45, 0.55, 0.75));
/** The walls' fill, their cell lines and their edges, as opacities. */
const OPACITY = { walls: 0.045, lines: 0.11, edges: 0.75 };
/** Cell line and edge widths in pixels. */
const STROKE = { lines: 1.1, edges: 2.2 };
/** The fewest world units a pixel spans in the grid's distances. */
const SMALLEST_PIXEL = 1e-6;

const cube = meshes.unindexed(meshes.box());

const Corner = d.struct({ position: d.vec4f, world: d.vec3f, normal: d.vec3f });
const Seen = d.struct({ color: d.vec4f });

/**
 * Draws the well's far glass walls with a faint grid of its cells and gold edges, so its limits
 * read from any side without standing between the camera and the play. The plinth carries the
 * floor's grid. Add it after the solid blocks: it blends over them and writes no depth.
 */
export function createWell(
  size: readonly [width: number, height: number, depth: number],
): passes.Drawing<typeof Corner, d.Infer<typeof Seen>> {
  const extent = d.vec3f(...size);

  return {
    Vertex: Corner,

    at: (vertexIndex) => {
      'use gpu';
      const corner = cube.vertexAt(vertexIndex);
      const world = (corner.position + 0.5) * extent;
      return Corner({
        position: cameras.toClip(world),
        world,
        normal: corner.normal,
      });
    },

    pixelOf: (corner) => {
      'use gpu';
      const world = corner.world;
      // World units per pixel. The axis along the normal is constant across a face, so the
      // smallest pixel keeps it from dividing by zero
      const pixel = std.max(std.fwidth(world), d.vec3f(SMALLEST_PIXEL));
      const across = 1 - std.abs(corner.normal);

      // Pixel distance to the cell lines and to the face's edges, along the face's two axes
      const cells = std.abs(std.fract(world + 0.5) - 0.5) / pixel;
      const reach = across * STROKE.lines - cells;
      // The plinth's brass border is the walls' foot, so neither a cell line nor an edge doubles it
      const foot = std.select(reach.y, d.f32(-1), world.y < 0.5);
      const lines = std.saturate(std.max(std.max(reach.x, foot), reach.z));
      // Interpolation can land a hair outside a wall, so distances to the sides start at zero
      const sides = std.min(world, extent - world);
      const inside = std.max(d.vec3f(sides.x, extent.y - world.y, sides.z), d.vec3f(0));
      const rim = across * STROKE.edges - inside / pixel;
      const edges = std.saturate(std.max(std.max(rim.x, rim.y), rim.z));

      // The plinth carries the floor and the open top has no glass, and the walls fade upward so the pile carries the light
      const wall = std.select(d.f32(1), d.f32(0), std.abs(corner.normal.y) > 0.5);
      const fade = std.mix(1, 0.35, std.saturate(world.y / extent.y));
      const glow = std.max(lines * OPACITY.lines * fade, edges * OPACITY.edges);
      const alpha = std.saturate(OPACITY.walls * fade + glow) * wall;
      const tint = std.mix(GLASS, GOLD, glow / std.max(alpha, 1e-4));
      return Seen({ color: d.vec4f(tint * alpha, alpha) });
    },

    color: passes.shaded,
    translucent: true,
    depthStencil: { depthWriteEnabled: false, depthCompare: 'less' },
    // Front-face culling shows the inside of the far walls
    primitive: { cullMode: 'front' },
    draw: (pass) => pass.draw(cube.vertexCount),
  };
}
