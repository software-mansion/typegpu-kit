// The files an example ships, named by path from its directory, and `resolve`, which reads and
// decodes them through a host's `Loading`. Each `@uses` tag names the kit modules that loading
// what the export describes runs, which the docs count for the examples that import it

import { font as fonts, gltf, images, type Baked } from '@typegpu/kit';
import type { FontName, Loading, Picture } from './contract.ts';
import { decoded, type Model } from './models.ts';

/** A glTF binary, read with its images decoded where the scene can't decode them. */
export interface Glb {
  readonly kind: 'glb';
  readonly path: string;
  /** Whether its meshes are Draco compressed, read with the host's decoder. */
  readonly draco: boolean;
}

/** A glTF binary split into named parts that share its images. */
export interface Parts<P extends string> {
  readonly kind: 'parts';
  readonly path: string;
  readonly parts: readonly P[];
  readonly draco: boolean;
}

export interface GlbOptions<P extends string> {
  /** Reads these named nodes as parts instead of one model. */
  readonly parts?: readonly P[];
  /** Marks a file whose meshes are Draco compressed. Default false. */
  readonly draco?: boolean;
}

/** An image, decoded unpremultiplied and without colour conversion, as height in alpha needs. */
export interface Image {
  readonly kind: 'image';
  readonly path: string;
}

/** A Radiance `.hdr` image, decoded to half floats on any thread. */
export interface Hdr {
  readonly kind: 'hdr';
  readonly path: string;
}

/** A font the kit ships. */
export interface Font {
  readonly kind: 'font';
  readonly name: FontName;
}

export type Asset = Glb | Parts<string> | Image | Hdr | Font;

export interface Manifest {
  readonly [name: string]: Asset | Manifest;
}

/** @uses kit/gltf */
export function glb(path: string, options?: GlbOptions<never>): Glb;
export function glb<const P extends string>(
  path: string,
  options: GlbOptions<P> & { readonly parts: readonly P[] },
): Parts<P>;
export function glb(path: string, { parts, draco = false }: GlbOptions<string> = {}) {
  return parts ? { kind: 'parts', path, parts, draco } : { kind: 'glb', path, draco };
}

/** @uses kit/images */
export function image(path: string): Image {
  return { kind: 'image', path };
}

/** @uses kit/images */
export function hdr(path: string): Hdr {
  return { kind: 'hdr', path };
}

/** @uses kit/font */
export function font(name: FontName): Font {
  return { kind: 'font', name };
}

/** One asset per name, such as a model per piece of a castle. */
export function each<N extends string, A extends Asset>(
  names: readonly N[],
  make: (name: N) => A,
): Record<N, A> {
  return Object.fromEntries(names.map((name) => [name, make(name)])) as Record<N, A>;
}

/**
 * What `resolve` makes of a manifest: a model per glb, a record of parts, a picture, half floats
 * per HDR image, a baked font.
 */
export type Loaded<M> = {
  readonly [K in keyof M]: M[K] extends Glb
    ? Model
    : M[K] extends Parts<infer P>
      ? Readonly<Record<P, Pick<Model, 'meshes' | 'images'>>>
      : M[K] extends Image
        ? Picture
        : M[K] extends Hdr
          ? images.HdrImage
          : M[K] extends Font
            ? Baked
            : Loaded<M[K]>;
};

const MIME_TYPES: Readonly<Record<string, string>> = {
  webp: 'image/webp',
  png: 'image/png',
  jpg: 'image/jpeg',
  jpeg: 'image/jpeg',
};

const isAsset = (value: Asset | Manifest): value is Asset => typeof value.kind === 'string';

const equal = (a: Uint8Array, b: Uint8Array) => {
  if (a.byteLength !== b.byteLength) return false;
  for (let i = 0; i < a.byteLength; i++) if (a[i] !== b[i]) return false;
  return true;
};

/**
 * `decode`, run once for each distinct image, so that files sharing an image give the mesh pool
 * one bitmap, which it uploads once.
 */
function once(decode: images.Decode): images.Decode {
  const seen: { bytes: Uint8Array; result: ReturnType<images.Decode> }[] = [];
  return (encoded) => {
    const found = seen.find(({ bytes }) => equal(bytes, encoded.bytes));
    if (found) return found.result;
    const result = decode(encoded);
    seen.push({ bytes: encoded.bytes, result });
    return result;
  };
}

/**
 * Reads and decodes every asset in `manifest`, the example `key`'s, through `loading`.
 *
 * @uses kit/font kit/gltf kit/images
 */
export async function resolve<M extends Manifest>(
  key: string,
  manifest: M,
  loading: Loading,
): Promise<Loaded<M>> {
  const { read } = loading;
  const decode = loading.decode && once(loading.decode);
  /** The read options for a model, with the host's Draco decoder when it needs one. */
  const decoding = async (asset: Glb | Parts<string>): Promise<{ readonly draco?: gltf.Draco }> => {
    if (!asset.draco) return {};
    if (!loading.draco)
      throw new Error(`${key}/${asset.path} is Draco, and the host has no decoder.`);
    return { draco: await loading.draco() };
  };
  const leaves = (part: Manifest): Asset[] =>
    Object.values(part).flatMap((value) => (isAsset(value) ? [value] : leaves(value)));
  const kinds = new Set(leaves(manifest).map((asset) => asset.kind));
  loading.say?.(
    kinds.has('glb') || kinds.has('parts')
      ? 'loading models'
      : kinds.has('image') || kinds.has('hdr')
        ? 'loading textures'
        : 'loading the font',
  );

  const load = async (asset: Asset): Promise<unknown> => {
    if (asset.kind === 'font') {
      const { json, atlas } = await loading.font(asset.name);
      return fonts.read(json, atlas);
    }

    const bytes = await read(`${key}/${asset.path}`);
    if (asset.kind === 'hdr') return images.decodeHdr(new Uint8Array(bytes));
    if (asset.kind === 'glb') {
      return decoded(gltf.read(bytes, await decoding(asset)), decode);
    }
    if (asset.kind === 'parts') {
      const file = gltf.read(bytes, { ...(await decoding(asset)), parts: asset.parts });
      const { images: pictures } = await decoded(file, decode);
      return Object.fromEntries(
        asset.parts.map((name) => [name, { meshes: file.parts[name].meshes, images: pictures }]),
      );
    }

    const extension = asset.path.slice(asset.path.lastIndexOf('.') + 1).toLowerCase();
    const mimeType = MIME_TYPES[extension];
    if (!mimeType) throw new Error(`${key}/${asset.path} is not an image the examples read.`);
    // Alpha can hold data such as height rather than coverage, so it stays unpremultiplied
    return (decode ?? images.decodeBitmap)({ bytes: new Uint8Array(bytes), mimeType });
  };

  const walk = async (part: Manifest): Promise<object> => {
    const entries = Object.entries(part).map(
      async ([name, value]) =>
        [name, isAsset(value) ? await load(value) : await walk(value)] as const,
    );
    return Object.fromEntries(await Promise.all(entries));
  };

  return (await walk(manifest)) as Loaded<M>;
}
