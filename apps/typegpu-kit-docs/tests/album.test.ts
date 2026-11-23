import { readdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { examples } from 'typegpu-kit-examples';
import { describe, expect, it } from 'vitest';
import { HIDDEN } from '../src/components/album/hidden.ts';

const DOCS = join(import.meta.dirname, '../src/content/docs');

/** Every docs page by its address, `/kit/grid/`, with its source. */
const pages = new Map(
  readdirSync(DOCS, { recursive: true, encoding: 'utf8' })
    .filter((file) => file.endsWith('.mdx'))
    .map((file) => {
      const slug = file.replace(/\.mdx$/, '').replace(/(^|\/)index$/, '');
      return [slug ? `/${slug}/` : '/', readFileSync(join(DOCS, file), 'utf8')] as const;
    }),
);

/** The id Starlight gives a heading. */
const idOf = (heading: string) =>
  heading
    .toLowerCase()
    .replace(/[^a-z0-9 -]/g, '')
    .trim()
    .replace(/\s+/g, '-');

describe('hidden cards', () => {
  it('hide every card once, on the page and in the section their hints name', () => {
    const keys = new Set(examples.map(({ key }) => key));
    expect(Object.keys(HIDDEN).toSorted()).toEqual([...keys].toSorted());
    const placed = [...pages].flatMap(([page, source]) =>
      [...source.matchAll(/<Hidden card="([^"]+)"/g)].map((match) => ({
        page,
        card: match[1],
        before: source.slice(0, match.index),
      })),
    );
    expect(placed.map(({ card }) => card).toSorted()).toEqual(Object.keys(HIDDEN).toSorted());
    for (const { page, card, before } of placed) {
      const hiding = HIDDEN[card];
      expect(keys.has(card), card).toBe(true);
      expect(hiding.page, card).toBe(page);
      expect(pages.get(page), card).toMatch(new RegExp(`^title: ${hiding.title}$`, 'm'));
      const headings = [...before.matchAll(/^##+ (.+)$/gm)].map((match) => idOf(match[1]));
      expect(headings.at(-1), card).toBe(hiding.section);
    }
  });
});
