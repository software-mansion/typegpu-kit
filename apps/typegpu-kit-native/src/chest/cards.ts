// Every example as a card the chest can hold, with the site's rarities and colours.

import { categories, examples } from 'typegpu-kit-examples';
import { FINISHES, type Finish } from 'typegpu-kit-pull/card.ts';
import { rgbOf, type Rgb01 } from 'typegpu-kit-pull/dither.ts';
import { LINES, raritiesOf, rarityOf, type Rarity } from 'typegpu-kit-pull/rarity.ts';
import { tokens } from '../theme.ts';

/** One example as its card shows it. */
export interface Entry {
  readonly key: string;
  readonly title: string;
  readonly line: string;
  /** The category's title, `Worlds`, and the example's place in the gallery, from 1. */
  readonly category: string;
  readonly number: number;
  /** Non-blank lines of TypeScript in the example's directory, which set its rarity. */
  readonly lines: number;
  readonly rarity: Rarity;
  /** The rarity's title, colour and finish. */
  readonly tier: string;
  readonly tint: Rgb01;
  readonly finish: Finish;
}

const lines = examples.map((example) => LINES[example.key] ?? 0);
const rarities = raritiesOf(lines);

/** The cards, in gallery order. */
export const entries: readonly Entry[] = examples.map((example, index) => {
  const rarity = rarities[index];
  const tier = rarityOf(rarity);
  return {
    key: example.key,
    title: example.title,
    line: example.line,
    category: categories.find(({ name }) => name === example.category)?.title ?? '',
    number: index + 1,
    lines: lines[index],
    rarity,
    tier: tier.title,
    tint: rgbOf(tokens[tier.token]),
    finish: FINISHES[rarity],
  };
});
