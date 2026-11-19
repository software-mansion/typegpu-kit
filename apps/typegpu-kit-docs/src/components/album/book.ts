// The binder's book: how the examples fill its sides, and its shape in world units with y up and
// the spine along z. Each category takes a side of nine pockets, and the next side again for every
// nine more. The sides are the sheets' fronts and backs in turn: a sheet lies on the right with its
// front up, and turns about the rings at x = 0 to lie on the left with its back up. Imports
// nothing, so the scene, the pointer and the tests read the same numbers.

/** A pocketed card, and the room between pockets. */
export const CARD = { width: 1, height: 1.4 } as const;
export const GAP = 0.1;
/** Pockets a side holds, three by three. */
export const POCKETS = 9;

/** A page's margins: by the rings, where its holes are punched; at its free edge; above; below. */
export const MARGIN = { inner: 0.36, outer: 0.14, top: 0.16, bottom: 0.16 } as const;

/** A sheet's width from its inner edge, and its height. */
export const WIDTH = MARGIN.inner + MARGIN.outer + 3 * CARD.width + 2 * GAP;
export const HEIGHT = MARGIN.top + MARGIN.bottom + 3 * CARD.height + 2 * GAP;

/** From the rings to a sheet's inner edge, and to its free edge. */
export const AXIS = 0.1;
export const REACH = AXIS + WIDTH;

/** The rise of each sheet over the one below it. */
export const STACK = 0.016;

/** The covers' overhang past the sheets, their thickness, and the spine's half width. */
export const COVER = { overhang: 0.16, thick: 0.08, spine: 0.24 } as const;

/** The rings along the spine: where each stands, its radius, and the wire's. */
export const RINGS = { at: [-1.55, 0, 1.55], radius: 0.28, wire: 0.035 } as const;

/**
 * The book for examples in gallery order, given by their categories: its sides, each a category
 * and the examples in its pockets by gallery index, its sheets, and where each example lies.
 */
export function bookOf(categories: readonly string[]) {
  const sides: { title: string; members: number[] }[] = [];
  const places: { side: number; pocket: number }[] = [];
  const groups = Map.groupBy(categories.keys(), (index) => categories[index]);
  for (const [title, members] of groups) {
    for (let start = 0; start < members.length; start += POCKETS) {
      const side = sides.length;
      const page = members.slice(start, start + POCKETS);
      page.forEach((example, pocket) => (places[example] = { side, pocket }));
      sides.push({ title, members: page });
    }
  }
  return { sides, sheets: Math.max(1, Math.ceil(sides.length / 2)), places };
}

/** Sheet `index` of `sheets`, at rest on the right, where the first lies on top, and on the left. */
export const restRight = (index: number, sheets: number) => STACK * (sheets - index);
export const restLeft = (index: number) => STACK * (index + 1);

/**
 * The sides spread `spread` shows, left and right: a sheet's back and the next sheet's front, or -1
 * for the inside of a cover.
 */
export const sidesOf = (spread: number, sheets: number) => ({
  left: spread > 0 ? spread * 2 - 1 : -1,
  right: spread < sheets ? spread * 2 : -1,
});

/** The spread that shows `side`. */
export const spreadOf = (side: number) => (side + 1) >> 1;

/** The middle of pocket `pocket` on side `side`, lying open, as x and z. */
export function pocketMiddle(side: number, pocket: number) {
  // From its page's left edge as it lies face up
  const from = side % 2 === 1 ? MARGIN.outer : MARGIN.inner;
  const across = from + (pocket % 3) * (CARD.width + GAP) + CARD.width / 2;
  const down = MARGIN.top + Math.floor(pocket / 3) * (CARD.height + GAP) + CARD.height / 2;
  const x = side % 2 === 1 ? -(AXIS + WIDTH - across) : AXIS + across;
  return [x, down - HEIGHT / 2] as const;
}
