import { mipmaps, pipelines } from '@typegpu/kit';
import { perlin2d } from '@typegpu/noise';
import { d, std, tgpu, type TgpuComputePass, type TgpuRoot } from 'typegpu';

const TILE = 16;
/** The foam tile; a level `t` keeps about 1 − t of its red field. Lengths are in metres. */
const PATTERN = {
  texels: 1024,
  mips: 11,
  cells: 8,
  size: 9,
  streaks: 2.5,
  stretch: 16,
  drawn: 0.5,
  detail: 4.3,
  holes: 0.35,
  blur: d.vec2f(0.03, 0.25),
};
/** Foam within `FILM` of its level is a film that hides `FILM_OPACITY` of the water. */
const FILM = 0.25;
const FILM_OPACITY = 0.45;
/** The share of the water below which foam fades out instead of breaking into specks. */
const SPARSE = 0.2;

/** Octaves at whole multiples of the lattice, which the cache wraps, keep the tile seamless. */
const fbm = (at: d.v2f, octaves: number, roughness: number) => {
  'use gpu';
  let p = d.vec2f(at);
  let sum = d.f32(0);
  let weight = d.f32(0.5);
  for (let octave = 0; octave < octaves; octave++) {
    sum += perlin2d.sample(p) * weight;
    p = p * 2 + 0.37;
    weight *= roughness;
  }
  return sum;
};

/** The pixel's share a `field` covers above `level`: crisp up close, averaged when `blurred`. */
const above = (field: number, level: number, blurred: number) => {
  'use gpu';
  const margin = field - level;
  const pixel = std.max(std.fwidth(margin), 1e-4);
  const crisp = std.smoothstep(-pixel, pixel, margin);
  const share = std.saturate(1 - level);
  return std.mix(crisp, share, blurred) * std.smoothstep(0, SPARSE, share);
};

export type Foam = ReturnType<typeof createFoam>;

/**
 * Whitecap lace: a tile of torn foam that `layer` cuts to the share of the water the breaking
 * waves call for, so a fresh cap closes over and an older one erodes into lace along the wind.
 */
export function createFoam(root: TgpuRoot, options: { direction: number }) {
  const { direction } = options;
  const wind = d.vec2f(Math.cos(direction), Math.sin(direction));
  const tile = root
    .createTexture({
      size: [PATTERN.texels, PATTERN.texels],
      format: 'rgba16float',
      mipLevelCount: PATTERN.mips,
    })
    .$usage('sampled', 'storage');
  const tileWrite = tile.createView(d.textureStorage2d('rgba16float', 'write-only'), {
    baseMipLevel: 0,
    mipLevelCount: 1,
  });
  const tileRead = tile.createView(d.texture2d(d.f32));
  const levels = mipmaps.create(root, tile, { combine: mipmaps.mean });
  const tiling = root.createSampler({
    magFilter: 'linear',
    minFilter: 'linear',
    mipmapFilter: 'linear',
    addressModeU: 'repeat',
    addressModeV: 'repeat',
    maxAnisotropy: 4,
  });

  const lattice = perlin2d.staticCache({ root, size: d.vec2u(PATTERN.cells, PATTERN.cells) });
  const bake = root.pipe(lattice.inject()).createComputePipeline({
    compute: tgpu.computeFn({
      workgroupSize: [TILE, TILE],
      in: { gid: d.builtin.globalInvocationId },
    })(({ gid }) => {
      'use gpu';
      if (gid.x >= PATTERN.texels || gid.y >= PATTERN.texels) return;
      const at = ((d.vec2f(gid.xy) + 0.5) / PATTERN.texels) * PATTERN.cells;
      // Warped by its own low octaves, the clumps and holes come out irregular and torn
      const warp = d.vec2f(fbm(at + 5.2, 2, 0.5), fbm(at + 11.7, 2, 0.5));
      const field = 0.5 + 0.5 * std.tanh(fbm(at + warp * 1.6, 7, 0.62) * 3.4);
      const bubbles = 0.5 + 0.5 * std.tanh(fbm(at * 16 + 7.3, 3, 0.6) * 5);
      std.textureStore(tileWrite.$, gid.xy, d.vec4f(field, bubbles, 0, 1));
    }),
  });

  return {
    initAsync: () => pipelines.compile([bake, levels]),
    /**
     * Foam at an undisplaced point: the share of the pixel it hides, and the bubbles' brightness.
     * `age` runs from 0 for a fresh cap to 1 for spent foam; `footprint` is the pixel in metres.
     */
    layer: (xz: d.v2f, share: number, age: number, footprint: number) => {
      'use gpu';
      const caps = std.textureSample(tileRead.$, tiling.$, xz / PATTERN.size);
      const along = d.vec2f(
        std.dot(xz, d.vec2f(-wind.y, wind.x)),
        std.dot(xz, wind) / PATTERN.stretch,
      );
      const streaks = std.textureSample(tileRead.$, tiling.$, along / PATTERN.streaks + 0.61);
      const blurred = std.smoothstep(PATTERN.blur.x, PATTERN.blur.y, footprint);
      const fine = std.textureSample(
        tileRead.$,
        tiling.$,
        (xz * PATTERN.detail) / PATTERN.size + 0.37,
      ).x;
      const field = std.mix(caps.x, streaks.x, PATTERN.drawn * age);
      const dense = std.mix(std.smoothstep(0, FILM, field - (1 - share)), 1, blurred);
      // Inside, the foam thins over the finer octave into lace and bright clumps
      const body = std.mix(FILM_OPACITY, 1, dense) * (1 - PATTERN.holes * (1 - fine));
      return d.vec2f(above(field, 1 - share, blurred) * body, fine);
    },
    /** Records the tile and its mips, once, before the first frame. */
    bake(pass: TgpuComputePass) {
      bake.with(pass).dispatchWorkgroups(PATTERN.texels / TILE, PATTERN.texels / TILE);
      levels.record(pass);
    },
    destroy() {
      tile.destroy();
      lattice.destroy();
    },
  };
}
