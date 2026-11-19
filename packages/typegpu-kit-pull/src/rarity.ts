// The examples' rarities, set by the size of their source, and the draw of a card.

import counted from './lines.json';

/**
 * The tiers, commonest first, with how many of the examples each takes and its colour, the site's
 * token `--<token>`.
 */
export const RARITIES = [
  { name: 'common', title: 'Common', share: 0.4, token: 'f-kind' },
  { name: 'uncommon', title: 'Uncommon', share: 0.27, token: 'f-needs' },
  { name: 'rare', title: 'Rare', share: 0.17, token: 'f-records' },
  { name: 'epic', title: 'Epic', share: 0.1, token: 'f-injects' },
  { name: 'legendary', title: 'Legendary', share: 0.06, token: 'f-requires' },
] as const;

export type Rarity = (typeof RARITIES)[number]['name'];

/** The tier of `rarity`, with its title and colour. */
export const rarityOf = (rarity: Rarity) => RARITIES.find(({ name }) => name === rarity)!;

/** Non-blank lines of TypeScript in each example's directory, by key, from `pnpm fix`. */
export const LINES: Readonly<Record<string, number>> = counted;

/** Each example's tier from its `lines`: the shortest 40 % common, up to the longest 6 % legendary. */
export function raritiesOf(lines: readonly number[]): Rarity[] {
  const ranked = lines.map((count, index) => ({ count, index }));
  // Hermes lacks `toSorted`
  ranked.sort((a, b) => a.count - b.count || a.index - b.index);
  const tiers: Rarity[] = Array.from(lines, () => 'common');
  let start = 0;
  RARITIES.forEach(({ name, share }, i) => {
    const end =
      i === RARITIES.length - 1 ? ranked.length : start + Math.round(share * ranked.length);
    for (const { index } of ranked.slice(start, end)) tiers[index] = name;
    start = end;
  });
  return tiers;
}

/**
 * Draws one of `pool` at random, with the way the card turns over, 1 or -1, and the vessel's
 * shudder, 0 to 1.
 */
export function draw<T>(pool: readonly T[], random: () => number = Math.random) {
  const item = pool[Math.floor(random() * pool.length)];
  const direction = random() < 0.5 ? 1 : -1;
  return { item, direction, phase: random() };
}
