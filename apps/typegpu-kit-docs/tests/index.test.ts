import { readdirSync, readFileSync } from 'node:fs';
import { join, sep } from 'node:path';
import { expect, it } from 'vitest';
import { buildIndex, linksOf } from '../src/components/site/nav.ts';
import { extractContracts, repoRoot } from '../src/contracts/extract.ts';
import { modulesOf } from '../src/contracts/markdown.ts';

const app = join(repoRoot, 'apps/typegpu-kit-docs/src');
const slugOf = (file: string) =>
  file
    .split(sep)
    .join('/')
    .replace(/(^|\/)index\.\w+$/, '')
    .replace(/\.\w+$/, '');
const filesIn = (dir: string) => readdirSync(join(app, dir), { recursive: true, encoding: 'utf8' });

const files = filesIn('content/docs').filter((file) => /\.mdx?$/.test(file));
/** Every docs page as an index slug: `kit/frames`, or `''` for the overview. */
const docs = files.map(slugOf);
/** Pages built outside the docs collection that the index may list, such as `examples`. */
const custom = filesIn('pages')
  .filter((file) => file.endsWith('.astro') && !file.includes('['))
  .map(slugOf);

const pages = files.map((file) => ({
  id: slugOf(file),
  title: slugOf(file),
  modules: modulesOf(readFileSync(join(app, 'content/docs', file), 'utf8')).map((id) => ({ id })),
}));
const index = linksOf(buildIndex(extractContracts().contracts, pages));

it('links only pages that exist', () => {
  const missing = index.filter((link) => !docs.includes(link.page) && !custom.includes(link.page));
  expect(missing.map((link) => `${link.label} → ${link.page}`)).toEqual([]);
});

/** Pages the header links instead of the index. */
const inHeader = ['map'];

it('lists every docs page but the 404 and those the header links', () => {
  const unlisted = docs.filter(
    (page) =>
      page !== '404' && !inHeader.includes(page) && !index.some((link) => link.page === page),
  );
  expect(unlisted).toEqual([]);
});
