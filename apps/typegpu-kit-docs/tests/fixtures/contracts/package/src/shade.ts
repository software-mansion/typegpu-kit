/**
 * Shading for the fixture.
 *
 * @module
 */

/**
 * A second block, which the contracts report.
 *
 * @module
 */

export const brighten = (x: number) => {
  'use gpu';
  return x + 1;
};
