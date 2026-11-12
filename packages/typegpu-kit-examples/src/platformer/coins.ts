import {
  cameras,
  coroutines,
  liveLists,
  passes,
  table,
  type Coroutines,
  type RowId,
} from '@typegpu/kit';
import { d, std, type TgpuRoot } from 'typegpu';
import { HALF } from './hero.ts';
import type { Level } from './level.ts';
import { pixelated, quad, snapped, sprite, TEXELS, type ArtOptions, type Seen } from './sprites.ts';

const Coin = table.define('Coin', {
  host: {
    position: d.vec2f,
    /** Collection time in seconds, or -1 before collection. */
    takenAt: d.f32,
  },
});

/** Seconds a taken coin flies up before it goes. */
export const VANISH = 0.4;
/** Tiles a second a taken coin flies up. */
const RISE = 4;

/** A coin's radius, in tiles. */
const RADIUS = 0.3;
/** How far a coin's centre may be from the hero's for the hero to take it, in tiles. */
const REACH = [HALF[0] + RADIUS, HALF[1] + RADIUS] as const;
/**
 * Spin rate in radians per second, phase change per tile in radians, and minimum width as a
 * fraction of face-on width.
 */
const SPIN = { rate: 3, stagger: 0.7, edge: 0.25 };
const GOLD = d.vec3f(1, 0.8, 0.22);
const RIM = d.vec3f(0.78, 0.47, 0.1);
const SHINE = d.vec3f(1, 0.97, 0.8);

/** A corner of a coin's quad, in quad units and in texels from the coin's snapped centre. */
const Corner = d.struct({ position: d.vec4f, local: d.vec2f, texel: d.vec2f });

export interface CoinsOptions extends ArtOptions {
  readonly capacity: number;
  /** Runs taken coins' flights. */
  readonly scheduler: Coroutines;
}

/** Stores coins in a CPU-written table and lists them with live lists. */
export function createCoins(root: TgpuRoot, options: CoinsOptions) {
  const { capacity, time, scheduler } = options;

  const coins = table.create(root, Coin, { capacity, defaults: { takenAt: -1 } });
  const live = liveLists.create(root, { ...coins.live(), draws: [quad] });
  const coin = coins.gpu;
  const taken = coins.fields.takenAt;

  /** Coins not yet taken, bottom row first and left to right in a row. */
  let waiting: { id: RowId<typeof Coin>; x: number; y: number }[] = [];

  const drawing: passes.Drawing<typeof Corner, Seen> = {
    label: 'coins',
    Vertex: Corner,

    at: (vertexIndex, instanceIndex) => {
      'use gpu';
      const slot = live.slot(0, instanceIndex);
      const corner = quad.vertexAt(vertexIndex).position.xz;
      const center = coin.position(slot);

      const takenAt = coin.takenAt(slot);
      const age = std.select(0, time.$ - takenAt, takenAt >= 0);
      const size = RADIUS * std.saturate(1 - age / VANISH);

      // Retain some width so the coin remains visible edge-on
      const turn = std.abs(std.cos(time.$ * SPIN.rate + center.x * SPIN.stagger));
      const across = std.max(turn, SPIN.edge);
      const offset = corner * d.vec2f(across, 1) * size;
      const world = snapped(center + d.vec2f(0, age * RISE)) + offset;

      return Corner({
        position: cameras.toClip(d.vec3f(world, 0)),
        local: corner,
        texel: offset * TEXELS,
      });
    },

    cutout: (corner) => {
      'use gpu';
      return std.length(pixelated(corner.local, corner.texel)) > 1;
    },

    pixelOf: (corner) => {
      'use gpu';
      const local = pixelated(corner.local, corner.texel);
      if (std.length(local) > 0.72) return sprite(RIM);
      if (std.abs(local.x + local.y * 0.5 + 0.2) < 0.18) return sprite(SHINE);
      return sprite(GOLD);
    },

    color: passes.shaded,
    draw: (pass) => live.draw(pass, 0),
  };

  function* vanish(id: RowId<typeof Coin>) {
    yield* coroutines.seconds(VANISH);

    // A reset can remove the coin before its animation ends
    if (coins.has(id)) coins.remove(id);
  }

  return {
    table: coins,
    compact: live.compact,
    initAsync: () => live.initAsync(),
    drawing,

    reset(level: Level) {
      coins.clear();
      waiting = level.coins.map(([x, y]) => ({ id: coins.add({ position: [x, y] }), x, y }));
      waiting.sort((a, b) => a.y - b.y || a.x - b.x);
    },

    /**
     * Collects coins touching the hero at `position`. Calls `onTake` with each coin position
     * and returns the count.
     */
    collect(
      position: readonly [number, number],
      now: number,
      onTake?: (x: number, y: number) => void,
    ) {
      const [x, y] = position;
      let count = 0;

      for (let i = 0; i < waiting.length; i++) {
        const near = waiting[i];
        if (Math.abs(near.x - x) >= REACH[0] || Math.abs(near.y - y) >= REACH[1]) continue;

        waiting.splice(i--, 1);
        taken.set(near.id, now);
        scheduler.start(vanish(near.id));
        onTake?.(near.x, near.y);
        count++;
      }

      return count;
    },

    destroy() {
      coins.destroy();
      live.destroy();
    },
  };
}

export type Coins = ReturnType<typeof createCoins>;
