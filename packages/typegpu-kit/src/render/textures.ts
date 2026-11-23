import { d, tgpu, type TgpuBindGroup, type TgpuRoot, type TgpuUniform } from 'typegpu';
import * as images from '../images.ts';
import type { Decode, EncodedImage, ImageSource } from '../images.ts';

// Bind a group per mesh because textures change between draws; bindings parts hold resources
// shared across a pass

/**
 * This schema holds which uvs a texture reads and how they move: uv set `uvSet`, 0 or 1, through
 * the affine map `u' = dot(u.xy, uv) + u.z`, and `v'` from `v` the same way. The identity has `u`
 * (1, 0, 0) and `v` (0, 1, 0).
 */
export const TextureUvs = d.struct({ u: d.vec3f, v: d.vec3f, uvSet: d.u32 });

/**
 * Tells shaders whether the pool keeps a second uv set. `meshPool.create` fills it with `uvs1`, and
 * without it shaders read the first set only.
 */
export const keepsSecondSet = tgpu.slot(false);

/**
 * This schema holds texture sampling factors. Shapes use normal scale 0, occlusion strength 1,
 * alpha cutoff 0 and the first uv set, unmoved. A uniform made from it starts zeroed, which reads
 * every texture at (0, 0), so fill each row of `uvs` before drawing with it.
 */
export const Reading = d.struct({
  /**
   * Scales the x and y components of the sampled normal. It is 0 for a material without a normal
   * texture, so `materials.mappedNormal` skips the texture.
   */
  normalScale: d.f32,
  /** Sets how much the occlusion texture darkens, from 0 for none to 1 for all of it. */
  occlusionStrength: d.f32,
  /** Sets the alpha threshold for discarding a pixel. A value of 0 discards nothing. */
  alphaCutoff: d.f32,
  /** Holds the uvs each texture reads. */
  uvs: d.struct({
    color: TextureUvs,
    metallicRoughness: TextureUvs,
    normal: TextureUvs,
    emission: TextureUvs,
    occlusion: TextureUvs,
  }),
});

/**
 * This layout binds mesh textures and sampling factors before each draw, for the fragment stage
 * alone. Missing textures use 1×1 defaults with white colour, a flat normal and no occlusion, so
 * material factors stay unchanged.
 */
export const layout = tgpu.bindGroupLayout({
  color: { texture: d.texture2d(d.f32), visibility: ['fragment'] },
  metallicRoughness: { texture: d.texture2d(d.f32), visibility: ['fragment'] },
  normal: { texture: d.texture2d(d.f32), visibility: ['fragment'] },
  emission: { texture: d.texture2d(d.f32), visibility: ['fragment'] },
  occlusion: { texture: d.texture2d(d.f32), visibility: ['fragment'] },
  filtering: { sampler: 'filtering', visibility: ['fragment'] },
  reading: { uniform: Reading, visibility: ['fragment'] },
});

/**
 * These sampler fields use WebGPU names. Defaults use linear filtering and repeating
 * coordinates.
 */
export interface SamplerData {
  readonly addressModeU?: GPUAddressMode;
  readonly addressModeV?: GPUAddressMode;
  readonly magFilter?: GPUFilterMode;
  readonly minFilter?: GPUFilterMode;
  readonly mipmapFilter?: GPUMipmapFilterMode;
}

/**
 * This transform moves a texture's coordinates as glTF's `KHR_texture_transform` does: scale,
 * then rotate, then offset.
 */
export interface TextureTransformData {
  /** Moves coordinates after scaling and rotating. Default `[0, 0]`. */
  readonly offset?: readonly [u: number, v: number];
  /** Turns coordinates by this many radians, counter-clockwise in uv space. Default 0. */
  readonly rotation?: number;
  /** Scales coordinates, such as `[4, 4]` to repeat a texture four times. Default `[1, 1]`. */
  readonly scale?: readonly [u: number, v: number];
}

export interface TextureData {
  /** Selects an image by its index in the model. */
  readonly image: number;
  readonly sampler?: SamplerData;
  /** Picks the mesh's uv set the texture reads: 0 for `uvs`, 1 for `uvs1`. Default 0. */
  readonly uvSet?: 0 | 1;
  /** Moves the coordinates this texture reads, such as to tile it. Default none. */
  readonly transform?: TextureTransformData;
}

export interface TexturesData {
  /** Multiplies material colour with sRGB colour and alpha. */
  readonly color?: TextureData;
  /** Multiplies roughness by the green channel and metallic by the blue channel. */
  readonly metallicRoughness?: TextureData;
  /** Supplies a tangent-space normal. */
  readonly normal?: TextureData;
  /** Multiplies emission with sRGB colour. */
  readonly emission?: TextureData;
  /** Multiplies occlusion by the red channel, scaled by the material's `occlusionStrength`. */
  readonly occlusion?: TextureData;
}

type Texture = images.Texture;
type Slot = keyof TexturesData;
/** Holds one value per texture slot, by name, as the layout binds them. */
type Slots<T> = { readonly [S in Slot]: T };

/** Builds a value for each texture slot. */
const perSlot = <T>(of: (slot: Slot) => T): Slots<T> => ({
  color: of('color'),
  metallicRoughness: of('metallicRoughness'),
  normal: of('normal'),
  emission: of('emission'),
  occlusion: of('occlusion'),
});

/** Waits for each slot's texture, rejecting with the first that fails. */
const settled = (pending: Slots<Texture | Promise<Texture>>) =>
  Promise.all([
    pending.color,
    pending.metallicRoughness,
    pending.normal,
    pending.emission,
    pending.occlusion,
  ]).then(([color, metallicRoughness, normal, emission, occlusion]): Slots<Texture> => ({
    color,
    metallicRoughness,
    normal,
    emission,
    occlusion,
  }));

/** The sampling factors of a material, beside the uvs its textures read. */
type Factors = Omit<d.Infer<typeof Reading>, 'uvs'>;

/** The uvs a texture reads, with its transform as rows: scale, then rotate, then offset. */
function uvsOf(data?: TextureData): d.Infer<typeof TextureUvs> {
  const [x, y] = data?.transform?.offset ?? [0, 0];
  const [su, sv] = data?.transform?.scale ?? [1, 1];
  const angle = data?.transform?.rotation ?? 0;
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);
  return {
    u: d.vec3f(cos * su, sin * sv, x),
    v: d.vec3f(-sin * su, cos * sv, y),
    uvSet: data?.uvSet ?? 0,
  };
}

type Color = 'srgb' | 'linear';

/** Colour and emission hold sRGB colour; the others hold data. */
const COLORS: Slots<Color> = {
  color: 'srgb',
  metallicRoughness: 'linear',
  normal: 'linear',
  emission: 'srgb',
  occlusion: 'linear',
};

const SLOTS = Object.keys(COLORS) as Slot[];

const DESTROYED = 'The mesh pool was destroyed before its images were uploaded.';

const sameBytes = (a: Uint8Array, b: Uint8Array) => {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
};

/**
 * Creates texture storage for a mesh pool. It decodes each image once per format and shares it
 * across meshes and models that use the same bytes, or the same pixels or bitmap object.
 */
export function createTextures(root: TgpuRoot, custom?: Decode, anisotropy = 1) {
  const made: Texture[] = [];

  const solid = (picture: images.Picture, color: Color = 'linear') => {
    const texture = images.create(root, picture, { color, mips: false });
    made.push(texture);
    return texture;
  };

  const whitePixel = { width: 1, height: 1, pixels: new Uint8Array([255, 255, 255, 255]) };
  const white = solid(whitePixel);
  const whiteSrgb = solid(whitePixel, 'srgb');
  // Use half floats for the flat normal because a byte cannot represent 0.5 exactly
  const flatNormal = solid({
    width: 1,
    height: 1,
    pixels: new Uint16Array([0x3800, 0x3800, 0x3c00, 0x3c00]),
  });

  const defaults: Slots<Texture> = {
    color: whiteSrgb,
    metallicRoughness: white,
    normal: flatNormal,
    emission: whiteSrgb,
    occlusion: white,
  };
  const samplers = new Map<string, ReturnType<TgpuRoot['createSampler']>>();
  const samplerOf = (data: SamplerData = {}) => {
    const filters = {
      magFilter: data.magFilter ?? 'linear',
      minFilter: data.minFilter ?? 'linear',
      mipmapFilter: data.mipmapFilter ?? 'linear',
    } as const;
    // Enable anisotropy only when all sampler filters are linear
    const linear = Object.values(filters).every((filter) => filter === 'linear');
    const descriptor = {
      addressModeU: data.addressModeU ?? 'repeat',
      addressModeV: data.addressModeV ?? 'repeat',
      ...filters,
      maxAnisotropy: linear ? anisotropy : 1,
    } as const;
    const key = Object.values(descriptor).join();

    let sampler = samplers.get(key);
    if (!sampler) {
      sampler = root.createSampler(descriptor);
      samplers.set(key, sampler);
    }
    return { key, sampler };
  };

  const readings = new Map<string, TgpuUniform<typeof Reading>>();
  const readingOf = (factors: Factors, slots: TexturesData = {}) => {
    const reading = {
      ...factors,
      normalScale: slots.normal ? factors.normalScale : 0,
      uvs: perSlot((slot) => uvsOf(slots[slot])),
    };
    const key = JSON.stringify(reading);
    let uniform = readings.get(key);
    if (!uniform) {
      uniform = root.createUniform(Reading);
      // Write under pool ownership before the first draw uses this buffer
      uniform.write(reading);
      readings.set(key, uniform);
    }
    return { key, uniform };
  };

  const fallback = root.createBindGroup(layout, {
    ...defaults,
    filtering: samplerOf().sampler,
    reading: readingOf({ normalScale: 0, occlusionStrength: 1, alphaCutoff: 0 }).uniform,
  });

  let destroyed = false;

  const created = (picture: images.Picture, color: Color) => {
    const texture = images.create(root, picture, { color });
    made.push(texture);
    return texture;
  };

  /** Encoded images by their bytes, which the pool copies to decode and to match later images. */
  const encoded: (EncodedImage & { color: Color; texture: Promise<Texture> })[] = [];
  /**
   * Pixels and bitmaps by the caller's image, read when first added and never kept, so a changed
   * image needs a new object.
   */
  const pictured = new WeakMap<images.Picture, Partial<Record<Color, Promise<Texture>>>>();

  const decoded = (image: EncodedImage, color: Color) => {
    const shared = encoded.find(
      (each) =>
        each.color === color &&
        each.mimeType === image.mimeType &&
        sameBytes(each.bytes, image.bytes),
    );
    if (shared) return shared.texture;

    // Copy into an array of its own: a Node Buffer's `slice` shares the caller's memory
    const owned = { bytes: new Uint8Array(image.bytes), mimeType: image.mimeType };
    const texture = images.decode(owned, custom).then((picture) => {
      if (destroyed) throw new Error(DESTROYED);
      return created(picture, color);
    });
    const entry = { ...owned, color, texture };
    encoded.push(entry);

    // Remove failed decodes from the cache so later models can retry
    texture.catch(() => encoded.splice(encoded.indexOf(entry), 1));
    return texture;
  };

  const upload = (image: ImageSource, color: Color) => {
    if ('bytes' in image) return decoded(image, color);
    let byColor = pictured.get(image);
    if (!byColor) pictured.set(image, (byColor = {}));
    // Upload at once, so the caller may change the image as soon as `add` returns. The executor
    // runs now, and a throw rejects the promise
    return (byColor[color] ??= new Promise<Texture>((resolve) => resolve(created(image, color))));
  };

  const groups = new Map<string, TgpuBindGroup<(typeof layout)['entries']>>();

  const groupFor = (
    textures: Slots<Texture>,
    filtering: ReturnType<typeof samplerOf>,
    read: ReturnType<typeof readingOf>,
  ) => {
    const key = [
      ...SLOTS.map((slot) => made.indexOf(textures[slot])),
      filtering.key,
      read.key,
    ].join();

    let group = groups.get(key);
    if (!group) {
      group = root.createBindGroup(layout, {
        ...textures,
        filtering: filtering.sampler,
        reading: read.uniform,
      });
      groups.set(key, group);
    }
    return group;
  };

  return {
    /** Provides default textures for shapes and meshes awaiting image upload. */
    fallback,

    /**
     * Returns a mesh texture bind group, at once for a material that names no texture, otherwise
     * after decoding and uploading its images.
     */
    groupOf(slots: TexturesData, pictures: readonly ImageSource[], factors: Factors) {
      // Use the first texture sampler, preferring colour, because each mesh binds one sampler
      const first = SLOTS.map((slot) => slots[slot]).find((data) => data !== undefined);
      const filtering = samplerOf(first?.sampler);
      const read = readingOf(factors, slots);
      if (!first) return groupFor(defaults, filtering, read);

      const pending = perSlot((slot) => {
        const data = slots[slot];
        if (!data) return defaults[slot];

        const image = pictures[data.image];
        if (!image) return Promise.reject(new Error(`The model has no image ${data.image}.`));
        return upload(image, COLORS[slot]);
      });

      return settled(pending).then((loaded) => {
        if (destroyed) throw new Error(DESTROYED);
        return groupFor(loaded, filtering, read);
      });
    },

    destroy() {
      destroyed = true;
      for (const texture of made) texture.destroy();
    },
  };
}
