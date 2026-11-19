import { recordingRoot } from '@typegpu/kit/testing';
import { meshPool, type Mesh } from '@typegpu/kit';
import { meshes } from '@typegpu/kit/geometry';

export const VERTICES = 2048;
export const INDICES = 8192;

const shapes = {
  box: () => meshes.box(),
  sphere: () => meshes.sphere(),
  roundedBox: () => meshes.roundedBox(),
  icosphere: () => meshes.icosphere(),
};
export type Shape = keyof typeof shapes;

export const SHAPES = Object.keys(shapes) as Shape[];

export function createProgram() {
  const { root } = recordingRoot();
  const pool = meshPool.create(root, { vertices: VERTICES, indices: INDICES, capacity: 8 });
  const names: Shape[] = [];

  return {
    get meshes(): readonly Mesh[] {
      return pool.meshes;
    },

    names: names as readonly Shape[],

    /** Adds a shape, or returns why the pool refused it. */
    add(name: Shape): string | undefined {
      try {
        pool.add(shapes[name]());
        names.push(name);
        return undefined;
      } catch (error) {
        return (error as Error).message;
      }
    },

    destroy() {
      pool.destroy();
    },
  };
}
