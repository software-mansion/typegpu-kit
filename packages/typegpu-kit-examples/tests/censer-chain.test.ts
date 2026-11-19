import { expect, it } from 'vitest';
import { createChain, CROWN, LENGTH, PIVOT, type Chain } from '../src/censer/chain.ts';

/** The chain's length from the pivot to the crown, along its links. */
const along = (chain: Chain) => {
  let total = 0;
  for (let n = 1; n < chain.chain.length; n++) {
    const a = chain.positions[chain.chain[n - 1]];
    const b = chain.positions[chain.chain[n]];
    total += Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
  }
  return total;
};
const fromPivot = (point: ArrayLike<number>) =>
  Math.hypot(point[0] - PIVOT[0], point[1] - PIVOT[1], point[2] - PIVOT[2]);

it('swings with the period of its length and settles to its idle swing', () => {
  const chain = createChain(1.2);
  const crossings: number[] = [];
  let last = chain.middle[0];
  for (let i = 0; i < 60 * 40; i++) {
    chain.advance(1 / 60);
    if (last < 0 && chain.middle[0] >= 0) crossings.push(i / 60);
    last = chain.middle[0];
  }
  const period = crossings.at(-1)! - crossings.at(-2)!;
  expect(period).toBeCloseTo(2 * Math.PI * Math.sqrt(LENGTH / 9.81), 1);
  expect(chain.amplitude()).toBeCloseTo(0.5, 1);
});

it('falls slack when lifted over its pivot and snaps taut when dropped', () => {
  const chain = createChain(0);
  chain.hold([0, 0, 0]);
  for (let i = 0; i < 90; i++) {
    const t = Math.min(1, i / 60);
    chain.pull([0.6 * t, 1.2 + 3.1 * t, 0]);
    chain.advance(1 / 60);
  }
  // Held above the pivot, upright, its chain drapes far shorter than its length
  expect(chain.middle[1]).toBeGreaterThan(PIVOT[1]);
  expect(fromPivot(chain.positions[chain.chain.at(-1)!])).toBeLessThan((LENGTH - CROWN) * 0.6);
  expect(chain.localOf([chain.middle[0], chain.middle[1] + 1, chain.middle[2]])[1]).toBeGreaterThan(
    0.9,
  );

  chain.release();
  let longest = 0;
  let farthest = 0;
  for (let i = 0; i < 180; i++) {
    chain.advance(1 / 60);
    longest = Math.max(longest, along(chain));
    farthest = Math.max(farthest, fromPivot(chain.middle));
  }
  expect(farthest).toBeGreaterThan(LENGTH - 0.1);
  expect(longest).toBeLessThan((LENGTH - CROWN) * 1.06);
});

it('flies on as fast as a hand throws it', () => {
  const chain = createChain(0);
  // A hand sweeping it sideways at 5 m/s leaves it moving about that fast
  chain.hold([0, 0, 0]);
  const from = [...chain.middle];
  for (let i = 1; i <= 15; i++) {
    chain.pull([from[0] + (5 * i) / 60, from[1], from[2]]);
    chain.advance(1 / 60);
  }
  const thrown = Math.hypot(...chain.velocity);
  chain.release();
  chain.advance(1 / 60);
  expect(thrown).toBeGreaterThan(3.5);
  expect(Math.hypot(...chain.velocity)).toBeGreaterThan(thrown * 0.7);
});
