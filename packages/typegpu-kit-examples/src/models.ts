import type { gltf, images, meshPool } from '@typegpu/kit';

/** `gltf.read` result, with decoded images when a decoder is available. */
export type Model = Omit<gltf.ModelData, 'images'> & Pick<meshPool.ModelData, 'images'>;

/** Decodes `model` images when `decode` is provided. Otherwise retains encoded images. */
export async function decoded(model: gltf.ModelData, decode?: images.Decode): Promise<Model> {
  if (!decode) return model;
  return { ...model, images: await Promise.all(model.images.map(decode)) };
}
