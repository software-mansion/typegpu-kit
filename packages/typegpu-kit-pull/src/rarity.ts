// The tiers each example's meta.json names, the examples' lines of code, and the draw of a card.

import counted from './lines.json';

/** The tiers, commonest first, with each one's colour, the site's token `--<token>`. */
export const RARITIES = [
  { name: 'common', title: 'Common', token: 'f-kind' },
  { name: 'uncommon', title: 'Uncommon', token: 'f-needs' },
  { name: 'rare', title: 'Rare', token: 'f-records' },
  { name: 'epic', title: 'Epic', token: 'f-injects' },
  { name: 'legendary', title: 'Legendary', token: 'f-requires' },
] as const;

export type Rarity = (typeof RARITIES)[number]['name'];

/** The tier named `rarity`, as an example's `meta.json` names it, with its title and colour. */
export function rarityOf(rarity: string) {
  const tier = RARITIES.find(({ name }) => name === rarity);
  if (!tier) throw new Error(`'${rarity}' is not a rarity`);
  return tier;
}

/** Non-blank lines of TypeScript in each example's directory, by key, from `pnpm fix`. */
export const LINES: Readonly<Record<string, number>> = counted;

/**
 * Draws one of `pool` at random, with the way the card turns over, 1 or -1, and the vessel's
 * shudder, 0 to 1.
 */
export function draw<T>(pool: readonly T[], random: () => number = Math.random) {
  const item = pool[Math.floor(random() * pool.length)];
  const direction = random() < 0.5 ? 1 : -1;
  return { item, direction, phase: random() };
}
