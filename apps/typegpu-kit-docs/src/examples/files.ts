import interAtlas from '@typegpu/kit/fonts/inter.rgba?url';
import interJson from '@typegpu/kit/fonts/inter.json?url';
import type { gltf } from '@typegpu/kit';
import type { FontName, Loading, Read, Say } from 'typegpu-kit-examples/contract';

const PACKAGE = '../../../../packages/typegpu-kit-examples/src/';

/** Vite's URL of every model, texture, HDR image and sound the examples ship, by path from the package's `src/`. */
const urls = Object.fromEntries(
  Object.entries(
    import.meta.glob<string>(
      '../../../../packages/typegpu-kit-examples/src/*/{models,textures,sounds}/*.{glb,webp,hdr,wav}',
      { query: '?url', import: 'default', eager: true },
    ),
  ).map(([path, url]) => [path.slice(PACKAGE.length), url]),
);

const FONTS: Readonly<Record<FontName, { readonly json: string; readonly atlas: string }>> = {
  inter: { json: interJson, atlas: interAtlas },
};

/** Reads a file by the URL Vite gives it. */
const fetchFile: Read = async (url) => {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: ${response.status} ${response.statusText}`);
  return response.arrayBuffer();
};

/** Reads a file the examples package ships, by path from its `src/`. */
export const fetchExampleFile: Read = (path) => {
  const url = urls[path];
  if (!url) throw new Error(`The examples ship no ${path}.`);
  return fetchFile(url);
};

/** Fetches a font the kit ships: its JSON parsed, and its atlas. */
const fetchFont: Loading['font'] = async (name) => {
  const { json, atlas } = FONTS[name];
  const [parsed, pixels] = await Promise.all([
    fetch(json).then((response) => response.json() as Promise<unknown>),
    fetchFile(atlas),
  ]);
  return { json: parsed, atlas: pixels };
};

let decoder: Promise<gltf.Draco> | undefined;

/**
 * Loads the Draco decoder's code and WebAssembly once, when the first Draco model needs them, and
 * again after a failed load.
 */
const draco = () =>
  (decoder ??= import('./draco.ts')
    .then((module) => module.decoder())
    .catch((error: unknown) => {
      decoder = undefined;
      throw error;
    }));

/** Loads examples' files on a page, saying their status through `say`. */
export const loading = (say?: Say): Loading => ({
  read: fetchExampleFile,
  font: fetchFont,
  draco,
  ...(say && { say }),
});
