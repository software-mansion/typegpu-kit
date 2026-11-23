// A card's way between its pocket and the viewer, on screen: four corners from its top left
// clockwise, in CSS pixels, eased from the pocket's corners as the camera sees them to a box facing
// the viewer, and the transform that lays an element over them.

type Point = readonly [number, number];
export type Quad = readonly [Point, Point, Point, Point];

/** The corners of the box `x`, `y`, `width`, `height`. */
export const boxOf = (x: number, y: number, width: number, height: number): Quad => [
  [x, y],
  [x + width, y],
  [x + width, y + height],
  [x, y + height],
];

/** Between `pocket`, at `k` 0, and `facing`, at `k` 1, lifted clear of the page on the way. */
export const between = (pocket: Quad, facing: Quad, k: number): Quad => {
  const lift = k * (1 - k) * (facing[3][1] - facing[0][1]);
  return pocket.map(([x, y], i) => {
    const [fx, fy] = facing[i];
    return [x + (fx - x) * k, y + (fy - y) * k - lift] as const;
  }) as unknown as Quad;
};

/**
 * The CSS `matrix3d` that lays a `width` × `height` box, its origin at its top left, onto `quad`:
 * the projective map of the unit square onto a quad, after Heckbert, scaled to the box.
 */
export function onto(quad: Quad, width: number, height: number) {
  const [[x0, y0], [x1, y1], [x2, y2], [x3, y3]] = quad;
  const dx1 = x1 - x2;
  const dx2 = x3 - x2;
  const dx3 = x0 - x1 + x2 - x3;
  const dy1 = y1 - y2;
  const dy2 = y3 - y2;
  const dy3 = y0 - y1 + y2 - y3;
  const det = dx1 * dy2 - dx2 * dy1;
  const g = (dx3 * dy2 - dx2 * dy3) / det;
  const h = (dx1 * dy3 - dx3 * dy1) / det;
  // Column by column: what x, y, z and 1 add to the screen's x, y, z and w
  const m = [
    [(x1 - x0 + g * x1) / width, (y1 - y0 + g * y1) / width, 0, g / width],
    [(x3 - x0 + h * x3) / height, (y3 - y0 + h * y3) / height, 0, h / height],
    [0, 0, 1, 0],
    [x0, y0, 0, 1],
  ];
  return `matrix3d(${m
    .flat()
    .map((n) => +n.toFixed(7))
    .join(',')})`;
}
