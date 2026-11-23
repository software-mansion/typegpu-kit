import { baker } from '@typegpu/kit';
import { d, std, type TgpuCommandEncoder, type TgpuRoot } from 'typegpu';

/** Maximum flake count. Adding a flake at capacity replaces the oldest. */
const MOST = 32;
/** How fast a flake's core glow falls off from its centre. */
const CORE = 400;
/** How fast a flake's halo glow falls off from its centre. */
const HALO = 0.6;
/** Radians each flake turns from the one laid before it. */
const TURN = 2.4;
/** Half-length and half-width of a dropped flake in sheet units. */
const DROPPED = d.vec2f(0.055, 0.045);

/** Flake centre in sheet coordinates from -1 to 1, half-length, half-width and angle in radians. */
const Oat = d.struct({ at: d.vec2f, size: d.vec2f, angle: d.f32 });

/**
 * Flake count, attraction strength and attraction range in sheet units. The sheet spans -1 to
 * 1.
 */
export const Food = d.struct({ count: d.u32, strength: d.f32, reach: d.f32 });

/** Initial flakes surround the starting blob and stay visible in a wide picture. */
const LAID: [number, number][] = [
  [0.4, -0.12],
  [0.2, 0.34],
  [-0.14, 0.42],
  [-0.46, 0.22],
  [-0.62, -0.2],
  [-0.28, -0.4],
  [0.08, -0.46],
  [0.7, 0.2],
  [-0.84, 0.18],
  [0.62, -0.38],
  [0.48, 0.48],
  [-0.66, 0.46],
  [0.88, -0.1],
];

const rotate = (v: d.v2f, angle: number) => {
  'use gpu';
  const c = std.cos(angle);
  const s = std.sin(angle);
  return d.vec2f(c * v.x - s * v.y, s * v.x + c * v.y);
};

/**
 * Paints food into a texture matching the trail grid when flakes change. Each cell stores
 * coverage, attraction, core glow and halo glow.
 */
export function createOats(root: TgpuRoot, options: { size: number }) {
  const { size } = options;
  const oats = root.createReadonly(d.arrayOf(Oat, MOST));
  const food = root.createUniform(Food);
  const texture = root
    .createTexture({ size: [size, size], format: 'rgba16float' })
    .$usage('storage', 'sampled');
  const reading = texture.createView(d.texture2d(d.f32));
  const smooth = root.createSampler({ magFilter: 'linear', minFilter: 'linear' });

  const painting = baker.create(root, texture, {
    texel: (uv) => {
      'use gpu';
      const at = uv * 2 - 1;
      const cell = 2 / size;
      let cover = d.f32(0);
      let smell = d.f32(0);
      let core = d.f32(0);
      let halo = d.f32(0);

      for (let i = d.u32(0); i < food.$.count; i++) {
        const oat = oats.$[i];
        // Transform the ellipse into a unit circle before measuring distance
        const local = rotate(at - oat.at, -oat.angle) / oat.size;
        const r = std.length(local);
        const edge = (r - 1) * std.min(oat.size.x, oat.size.y);
        cover = std.max(cover, std.smoothstep(cell, -cell, edge));
        const reach = std.max(edge, 0) / food.$.reach;
        smell += std.exp(-reach * reach);
        core += std.exp(-r * r * CORE);
        halo += std.exp(-r * r * HALO);
      }

      return d.vec4f(cover, smell, core, halo);
    },
  });
  const written = { count: LAID.length, strength: 1, reach: 0.2 };
  let laid = LAID.length;
  let changed = true;

  oats.write(
    LAID.map(([x, y], i) => ({
      at: d.vec2f(x, y),
      size: d.vec2f(0.05 + 0.01 * (i % 3), 0.04 + 0.006 * ((i * 2) % 3)),
      angle: i * TURN,
    })),
  );

  return {
    /**
     * Samples food at grid coordinates `uv` from 0 to 1. Returns attraction in `x` and flake
     * coverage in `y`.
     */
    foodAt: (uv: d.v2f) => {
      'use gpu';
      const found = std.textureSampleLevel(reading.$, smooth.$, uv, 0).xy;
      return d.vec2f(found.y * food.$.strength, found.x);
    },
    /** Returns core glow in `x` and halo glow in `y` at grid coordinates `uv`. */
    glowAt: (uv: d.v2f) => {
      'use gpu';
      return std.textureSampleLevel(reading.$, smooth.$, uv, 0).zw;
    },
    initAsync: () => painting.initAsync(),

    set(strength: number) {
      written.strength = strength;
      food.write(written);
    },

    /**
     * Adds a flake at sheet coordinates `at`, from -1 to 1. Replaces the oldest flake at
     * capacity.
     */
    drop(at: d.v2f) {
      oats.patch({
        [laid % MOST]: { at, size: DROPPED, angle: laid * TURN },
      });
      laid++;
      written.count = Math.min(laid, MOST);
      food.write(written);
      changed = true;
    },

    /** Repaints only when flakes have changed. */
    paint(encoder: TgpuCommandEncoder) {
      if (!changed) return;
      changed = false;
      painting.record(encoder);
    },

    destroy() {
      texture.destroy();
      oats.buffer.destroy();
    },
  };
}

export type Oats = ReturnType<typeof createOats>;
