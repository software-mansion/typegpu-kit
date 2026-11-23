import { describe, expect, it } from 'vitest';
import { bookOf, sidesOf, spreadOf } from '../src/components/album/book.ts';
import { between, boxOf, onto } from '../src/components/album/flight.ts';
import { createTurning } from '../src/components/album/turning.ts';

/** Steps `turning` a sixtieth of a second at a time for `seconds`. */
const run = (turning: ReturnType<typeof createTurning>, seconds = 3) => {
  for (let t = 0; t < seconds; t += 1 / 60) turning.step(1 / 60);
};

describe('the book', () => {
  // Ten of one kind take two sides; three of another start a side of their own
  const book = bookOf([...Array<string>(10).fill('Worlds'), ...Array<string>(3).fill('Games')]);

  it('gives each category its own sides, nine pockets a side, two sides a sheet', () => {
    expect(book.sides.map(({ title, members }) => [title, members.length])).toEqual([
      ['Worlds', 9],
      ['Worlds', 1],
      ['Games', 3],
    ]);
    expect(book.sheets).toBe(2);
    expect(sidesOf(spreadOf(2), book.sheets)).toEqual({ left: 1, right: 2 });
  });
});

describe('the turning', () => {
  it('lies open at a spread, and turns a sheet at a time from there', () => {
    const turning = createTurning(3, false);
    turning.turnTo(2);
    expect(turning.landed).toBe(true);
    expect(turning.sheets.map(({ angle }) => angle / Math.PI)).toEqual([1, 1, 0]);
    // Stacked from the cover up on both sides
    const [a, b, c] = turning.sheets.map(({ height }) => height);
    expect(b).toBeGreaterThan(a!);
    expect(c).toBeGreaterThan(0);
    expect(turning.turn(false)).toBe(true);
    turning.step(0.3);
    expect(turning.sheets[1].angle).toBeLessThan(Math.PI);
    expect(turning.landed).toBe(false);
    run(turning);
    expect(turning.sheets.map(({ angle }) => Math.round(angle / Math.PI))).toEqual([1, 0, 0]);
    turning.turnTo(0);
    expect(turning.turn(false)).toBe(false);
  });
});

describe('the flight', () => {
  it('starts on the pocket and ends facing the viewer', () => {
    const pocket = boxOf(10, 20, 50, 70);
    const facing = boxOf(300, 100, 200, 280);
    expect(between(pocket, facing, 0)).toEqual(pocket);
    expect(between(pocket, facing, 1)).toEqual(facing);
  });

  it('lays a box over the four corners it is given', () => {
    const corners = [
      [10, 20],
      [130, 32],
      [118, 190],
      [4, 170],
    ] as const;
    const m = onto(corners, 200, 280)
      .match(/matrix3d\((.*)\)/)![1]
      .split(',')
      .map(Number);
    [
      [0, 0],
      [200, 0],
      [200, 280],
      [0, 280],
    ].forEach(([x, y], i) => {
      // Column-major, as CSS reads it, then divided by w
      const w = m[3] * x! + m[7] * y! + m[15];
      const point = [(m[0] * x! + m[4] * y! + m[12]) / w, (m[1] * x! + m[5] * y! + m[13]) / w];
      expect(point[0]).toBeCloseTo(corners[i][0], 2);
      expect(point[1]).toBeCloseTo(corners[i][1], 2);
    });
  });
});
