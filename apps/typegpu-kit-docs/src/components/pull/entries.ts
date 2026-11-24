// Server side: every gallery example as a card, with the rarity its meta.json names.

import { categories } from 'typegpu-kit-examples';
import { examples } from '../../examples/examples.ts';
import { loadModules } from '../site/load.ts';
import { distinctOf } from '../site/modules.ts';
import { smallPosterOf } from '../site/posters.ts';
import { exampleHref } from '../site/urls.ts';
import { LINES, rarityOf } from 'typegpu-kit-pull/rarity.ts';
import type { Entry } from './pulls.ts';

/** Every gallery example as a card, in gallery order. */
export async function loadEntries(): Promise<Entry[]> {
  const modules = await loadModules();
  const every = examples.map((example) => example.key);
  return examples.map((example, index) => ({
    key: example.key,
    title: example.title,
    line: example.line,
    category: categories.find(({ name }) => name === example.category)?.title ?? '',
    poster: smallPosterOf(example.key) ?? '',
    href: exampleHref(example.key),
    number: index + 1,
    modules: distinctOf(modules, example.key, every).map((module) => module.namespace),
    lines: LINES[example.key] ?? 0,
    rarity: rarityOf(example.rarity).name,
  }));
}
