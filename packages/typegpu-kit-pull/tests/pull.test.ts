import { describe, expect, it } from 'vitest';
import { boxStage, pulledBox } from '../src/box.ts';
import { cardAt, cardSize, emblem } from '../src/card.ts';
import { COLORS } from '../src/dither.ts';
import { draw, LINES, raritiesOf, RARITIES } from '../src/rarity.ts';
import { FLIGHT, LAUNCH } from '../src/timeline.ts';
import { countLines } from '../scripts/lines.ts';

describe('rarity', () => {
  it('counts every example as the sources stand; `pnpm fix` writes them', () => {
    expect(LINES).toEqual(countLines());
  });

  it('gives each tier its share, the shortest examples common and the longest legendary', () => {
    const lines = Array.from({ length: 50 }, (_, i) => 1000 - i);
    const tiers = raritiesOf(lines);
    const counts = RARITIES.map(({ name }) => tiers.filter((tier) => tier === name).length);
    expect(counts).toEqual([20, 14, 9, 5, 2]);
    expect(tiers[49]).toBe('common');
    expect(tiers[0]).toBe('legendary');
  });

  it('draws from the pool with a turn and a shudder', () => {
    const rolls = [0.99, 0.2, 0.7];
    expect(draw(['a', 'b', 'c'], () => rolls.shift()!)).toEqual({
      item: 'c',
      direction: 1,
      phase: 0.7,
    });
  });
});

describe('the card', () => {
  const [width, height] = cardSize(390, 844, 92, 360);
  const { slot, size, ground } = boxStage(390, 844, height, 92);
  const stage = { width, height, slot, size };
  const exitAt = (t: number) => pulledBox(t, 0.5, ground, size, 0, COLORS.accent[0], [1, 0, 0]);

  it('waits hidden in the box, then lands face up in its slot', () => {
    expect(cardAt('box', 0, exitAt(0), stage, 1, 0).visible).toBe(false);
    const end = LAUNCH.box + FLIGHT;
    const landed = cardAt('box', end, exitAt(end), stage, 1, 0.5);
    expect(landed).toMatchObject({ x: slot[0], y: slot[1], scale: 1, ry: 0, rim: undefined });
  });

  it('prints its seal from the core out to the ring', () => {
    expect(emblem(0, 0)).toBeCloseTo(0.92);
    expect(emblem(0.84, 0)).toBeGreaterThan(1);
    expect(emblem(0.95, 0.95)).toBeLessThan(0.01);
  });
});
