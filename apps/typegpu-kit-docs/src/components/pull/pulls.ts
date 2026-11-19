// Client side: the examples as cards and their rarities. Imports only the site's
// addresses, so it runs anywhere.

import type { Rarity } from 'typegpu-kit-pull/rarity.ts';
import { BASE } from '../site/urls.ts';

export { RARITIES, rarityOf, type Rarity } from 'typegpu-kit-pull/rarity.ts';

/** One example as a card shows it. */
export interface Entry {
  readonly key: string;
  readonly title: string;
  readonly line: string;
  /** The category's title, `Worlds`. */
  readonly category: string;
  /** The poster's address and the example's full-screen page. */
  readonly poster: string;
  readonly href: string;
  /** Its place in the gallery, from 1. */
  readonly number: number;
  /** The modules it uses that few others do, as the gallery tags it. */
  readonly modules: readonly string[];
  /** Non-blank lines of TypeScript in the example's directory, which set its rarity. */
  readonly lines: number;
  readonly rarity: Rarity;
}

/** A tier's colour, its site token. */
export const colorOf = ({ token }: { readonly token: string }) => `var(--${token})`;

/** Whether the page should skip travel and show results at once. */
export const stillMotion = () => matchMedia('(prefers-reduced-motion: reduce)').matches;

let entries: Promise<Entry[]> | undefined;
/** The cards, fetched once a page, or again after a fetch that failed. */
export const loadEntries = () =>
  (entries ??= fetch(`${BASE}examples/pulls.json`)
    .then((response) => {
      if (!response.ok) throw new Error(`The cards did not load: ${response.status}`);
      return response.json() as Promise<Entry[]>;
    })
    .catch((error: unknown) => {
      entries = undefined;
      throw error;
    }));
