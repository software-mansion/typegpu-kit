import { d } from 'typegpu';
import { expect, it } from 'vitest';
import { bayer, bayerCell } from '../src/components/site/dither.ts';
import { band, SETTLE, strengthAt } from '../src/components/site/glow.scene.ts';

const edges = (v: d.v4f) => [v.x, v.y, v.z, v.w];

it('thresholds cells by one 8×8 Bayer matrix on the CPU and the GPU', () => {
  const thresholds = new Set<number>();
  for (let y = 0; y < 16; y++) {
    for (let x = 0; x < 16; x++) {
      expect(bayerCell(d.vec2u(x, y))).toBe(bayer(x, y));
      thresholds.add(bayer(x, y));
    }
  }
  expect(thresholds.size).toBe(64);
});

it('stretches the light toward the next entry, edge ahead first, and gathers it there', () => {
  const from = d.vec4f(0, 0, 200, 28);
  const to = d.vec4f(0, 112, 200, 140);

  expect(edges(band(from, to, 0))).toEqual(edges(from));
  expect(edges(band(from, to, SETTLE))).toEqual(edges(to));

  // Going down, the bottom edge leads and the top trails; going up, the other way round.
  const down = band(from, to, SETTLE / 3);
  expect(down.w - from.w).toBeGreaterThan(down.y - from.y);
  expect(down.w - down.y).toBeGreaterThan(28);

  const up = band(to, from, SETTLE / 3);
  expect(to.y - up.y).toBeGreaterThan(to.w - up.w);
});

it('fades the light over the move', () => {
  expect(strengthAt(d.vec2f(0, 1), 0)).toBe(0);
  expect(strengthAt(d.vec2f(0, 1), SETTLE)).toBe(1);
  expect(strengthAt(d.vec2f(1, 0), SETTLE / 2)).toBeLessThan(1);
});
