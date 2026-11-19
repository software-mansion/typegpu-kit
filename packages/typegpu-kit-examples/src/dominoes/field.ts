// Lay out the portrait once on the CPU. A lead line along the far edge triggers columns toward
// the viewer, producing a diagonal wave

import { NONE, SPACING, type Placed } from './layout.ts';

/** Image width and height in dominoes, one pixel per domino. */
export const COLUMNS = 100;
export const ROWS = 60;

/** The field's centre in world x and z. */
export const MIDDLE = [0, -60] as const;

/** Standing dominoes use eight-wide diagonal colour bands. The falling wave crosses these bands. */
const bandOf = (column: number, row: number) => Math.floor((column - row + ROWS) / 8) % 6;

/**
 * Orders the far-edge lead line before its columns of `ROWS` dominoes. The lead falls right and
 * columns fall toward +z. Each `cell` holds image coordinates from the top left.
 */
export function layField(): Placed[] {
  const placed: Placed[] = [];
  const left = MIDDLE[0] - ((COLUMNS - 1) * SPACING) / 2;
  const far = MIDDLE[1] - ((ROWS - 1) * SPACING) / 2;

  for (let column = 0; column < COLUMNS; column++) {
    placed.push({
      x: left + column * SPACING,
      z: far - SPACING,
      yaw: Math.PI / 2,
      depth: column,
      prev: column === 0 ? NONE : column - 1,
      next: [column + 1 < COLUMNS ? column + 1 : NONE, NONE],
      color: 0,
    });
  }

  for (let column = 0; column < COLUMNS; column++) {
    const lead = placed[column];
    for (let row = 0; row < ROWS; row++) {
      const index = placed.length;
      const prev = row === 0 ? column : index - 1;
      placed.push({
        x: lead.x,
        z: far + row * SPACING,
        yaw: 0,
        depth: column + 1 + row,
        prev,
        next: [NONE, NONE],
        color: bandOf(column, row),
        cell: [column, row],
      });
      const pushes = placed[prev].next;
      pushes[pushes[0] === NONE ? 0 : 1] = index;
    }
  }

  return placed;
}
