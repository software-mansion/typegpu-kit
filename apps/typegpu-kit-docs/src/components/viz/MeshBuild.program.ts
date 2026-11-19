import { meshes } from '@typegpu/kit/geometry';

type Vec3 = { readonly x: number; readonly y: number; readonly z: number };

export interface Vertex {
  readonly position: Vec3;
  readonly normal: Vec3;
  readonly uv: { readonly x: number; readonly y: number };
}

/** The shapes the figure offers, each as its code reads, coarser than the defaults but the box's. */
const shapes = {
  box: { code: 'box()', make: () => meshes.box() },
  plane: { code: 'plane()', make: () => meshes.plane() },
  sphere: {
    code: 'sphere({ segments: 10, rings: 6 })',
    make: () => meshes.sphere({ segments: 10, rings: 6 }),
  },
  cylinder: {
    code: 'cylinder({ radialSegments: 10 })',
    make: () => meshes.cylinder({ radialSegments: 10 }),
  },
  torus: {
    code: 'torus({ ringSegments: 14, tubeSegments: 6 })',
    make: () => meshes.torus({ ringSegments: 14, tubeSegments: 6 }),
  },
};
export type Shape = keyof typeof shapes;

export const SHAPES = Object.keys(shapes) as Shape[];

export interface Built {
  readonly code: string;
  /** `vertexAt(i)` of the unindexed shape: a vertex per corner, three corners a triangle. */
  readonly corners: readonly Vertex[];
  /** `indexAt(i)` of the indexed shape: which of its vertices corner `i` uses. */
  readonly indices: readonly number[];
  /** How many corners use each vertex of the indexed shape. */
  readonly uses: readonly number[];
}

export function createProgram() {
  const built = new Map<Shape, Built>();

  return {
    /** The shape's corners, evaluated as a vertex shader would, and its indices. */
    build(name: Shape): Built {
      const found = built.get(name);
      if (found) return found;

      const { code, make } = shapes[name];
      const shape = make();
      const flat = meshes.unindexed(shape);
      const corners = Array.from({ length: flat.vertexCount }, (_, i) => flat.vertexAt(i));
      const indices = Array.from({ length: shape.indexCount }, (_, i) => shape.indexAt(i));
      const uses = Array.from({ length: shape.vertexCount }, () => 0);
      for (const index of indices) uses[index] += 1;

      const shown = { code, corners, indices, uses };
      built.set(name, shown);
      return shown;
    },
  };
}
