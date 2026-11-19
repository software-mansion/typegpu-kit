import { instances, meshPool } from '@typegpu/kit';
import type { TgpuRoot } from 'typegpu';

/** A piece of the scene: a model by name, and its row's values. */
export interface Placement {
  readonly model: string;
  /** Its position, rotation, scale and any material field, such as `emission`. */
  readonly values: instances.Values;
}

/** One pool of every model and shape, and one set of instances placing them. */
export function createProps(
  root: TgpuRoot,
  models: Readonly<Record<string, meshPool.ModelData | meshPool.Shape>>,
  placements: readonly Placement[],
) {
  const pool = meshPool.create(root, meshPool.roomFor(Object.values(models)));
  // A shape, such as `meshes.cylinder()`, is one mesh with nothing to decode
  const add = (item: meshPool.ModelData | meshPool.Shape) =>
    'meshes' in item ? pool.add(item) : { meshes: [pool.add(item)], ready: Promise.resolve() };
  const added = new Map(Object.entries(models).map(([name, item]) => [name, add(item)]));
  const modelOf = (name: string) => added.get(name)!;

  // A model takes a row for each of its meshes
  const rows = placements.reduce((sum, { model }) => sum + modelOf(model).meshes.length, 0);
  const props = instances.create(root, pool, { capacity: rows });
  for (const { model, values } of placements) props.add(modelOf(model), values);

  return {
    renderer: props.renderer(),
    /** Settles once every model's textures are on the GPU. */
    ready: Promise.all([...added.values()].map((model) => model.ready)),
    flush: () => props.flush(),
    destroy() {
      props.destroy();
      pool.destroy();
    },
  };
}
