// The native index over the examples package's files: Metro bundles every model, texture and
// poster as an asset, and `loading` reads and decodes them on the JS thread

import { gltf, images } from '@typegpu/kit';
import { Asset } from 'expo-asset';
import { File } from 'expo-file-system';
import type { Loading, Say } from 'typegpu-kit-examples/contract';

const files = require.context(
  '../../../packages/typegpu-kit-examples/src',
  true,
  /\.(glb|webp|hdr)$/,
);

/** Asset module ids by path from the package's `src`, such as `blockfall/poster.webp`. */
const ids = new Map(files.keys().map((key) => [key.replace(/^\.\//, ''), files(key) as number]));

/** The poster of example `key`, as an image source. */
export function posterOf(key: string): number | undefined {
  return ids.get(`${key}/poster.webp`);
}

/** The poster of example `key`, decoded for a texture, or undefined when it has none. */
export async function posterBitmapOf(key: string): Promise<ImageBitmap | undefined> {
  const id = posterOf(key);
  if (id === undefined) return undefined;
  return createImageBitmap(await bytesOf(id));
}

async function bytesOf(id: number): Promise<ArrayBuffer> {
  const asset = await Asset.fromModule(id).downloadAsync();
  if (!asset.localUri) throw new Error(`${asset.name} has no local copy.`);
  return new File(asset.localUri).arrayBuffer();
}

const read: Loading['read'] = (path) => {
  const id = ids.get(path);
  if (id === undefined) throw new Error(`The app bundles no ${path}.`);
  return bytesOf(id);
};

const font: Loading['font'] = async (name) => {
  if (name !== 'inter') throw new Error(`The kit ships no font ${String(name)}.`);
  const [json, atlas] = await Promise.all([
    Promise.resolve<unknown>(require('@typegpu/kit/fonts/inter.json')),
    bytesOf(require('@typegpu/kit/fonts/inter.rgba') as number),
  ]);
  return { json, atlas };
};

let cardFonts: ReturnType<typeof readCardFonts> | undefined;

const readCardFonts = async () => {
  const [display, atlas] = await Promise.all([
    font('inter'),
    bytesOf(require('../assets/fonts/mono.rgba') as number),
  ]);
  return { display, mono: { json: require('../assets/fonts/mono.json') as unknown, atlas } };
};

/** The chest's card's fonts, read once: the kit's Inter Display and the site's JetBrains Mono. */
export const cardFontsOf = () => (cardFonts ??= readCardFonts());

let decoder: Promise<gltf.Draco> | undefined;

/**
 * Makes a Draco decoder from Google's JavaScript build of it, once, since Hermes has no
 * WebAssembly. Requiring it here leaves its code unrun until the first Draco model.
 */
const draco = () =>
  (decoder ??= (
    require('../vendor/draco/draco_decoder_gltf.js') as () => Promise<gltf.DracoModule>
  )().then((module) => gltf.draco(module)));

/**
 * Reads the package's files and the kit's fonts, saying what it loads through `say`. Images decode
 * to bitmaps here, as the scene's worker has no `createImageBitmap`.
 */
export const loading = (say?: Say): Loading => ({
  read,
  font,
  decode: images.decodeBitmap,
  draco,
  ...(say && { say }),
});
