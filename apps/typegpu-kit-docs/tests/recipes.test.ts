import { existsSync, readdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';

const recipes = join(import.meta.dirname, '../src/content/docs/recipes');
const viz = join(import.meta.dirname, '../src/components/viz');

/** What a twoslash block shows, without the parts its cut markers and `@filename` lines hide. */
function visibleOf(block: string) {
  let lines = block.split('\n');
  const cut = lines.findLastIndex((line) => line.trim() === '// ---cut---');
  if (cut >= 0) lines = lines.slice(cut + 1);
  const after = lines.findIndex((line) => line.trim() === '// ---cut-after---');
  if (after >= 0) lines = lines.slice(0, after);
  const shown: string[] = [];
  let hidden = false;
  for (const line of lines) {
    if (line.trim() === '// ---cut-start---') hidden = true;
    else if (line.trim() === '// ---cut-end---') hidden = false;
    else if (!hidden && !line.startsWith('// @filename:')) shown.push(line);
  }
  return shown;
}

/** The lines of `excerpt` that `source` lacks in that order, compared without indentation. */
function missingFrom(excerpt: readonly string[], source: string) {
  const lines = source.split('\n').map((line) => line.trim());
  let at = 0;
  return excerpt
    .map((line) => line.trim())
    .filter((line) => line !== '')
    .filter((line) => {
      const found = lines.indexOf(line, at);
      if (found < 0) return true;
      at = found + 1;
      return false;
    });
}

describe('recipes', () => {
  it('reads what a block shows', () => {
    const block =
      'hidden\n// ---cut---\na\n// ---cut-start---\nb\n// ---cut-end---\nc\n// ---cut-after---\nd';
    expect(visibleOf(block)).toEqual(['a', 'c']);
    expect(missingFrom(['  a', 'c'], 'a\nb\nc')).toEqual([]);
    expect(missingFrom(['c', 'a'], 'a\nb\nc')).toEqual(['a']);
  });

  // A recipe with a figure keeps its files in viz/<page>/, which the figure runs, so a titled
  // block must be an excerpt of the file it names there
  const paged = readdirSync(recipes)
    .map((file) => file.replace(/\.mdx?$/, ''))
    .filter((id) => existsSync(join(viz, id)));

  it.each(paged)('%s shows its figure’s code', (id) => {
    const page = readFileSync(join(recipes, `${id}.mdx`), 'utf8');
    const titled = [...page.matchAll(/```ts twoslash title="([^"]+)"\n([\s\S]*?)\n```/g)].map(
      ([, title, body]) => ({ title: title!, file: join(viz, id, title!), body: body! }),
    );
    const blocks = titled.filter((block) => existsSync(block.file));
    // Only a page's scene.ts may be its own: the figure's scene adds its buttons and models
    const unmatched = titled.filter((block) => !existsSync(block.file)).map((block) => block.title);

    expect(unmatched.filter((title) => title !== 'scene.ts')).toEqual([]);
    expect(blocks.length).toBeGreaterThan(0);
    for (const block of blocks) {
      const missing = missingFrom(visibleOf(block.body), readFileSync(block.file, 'utf8'));
      expect(missing, `${id}: ${block.title} shows lines its file lacks`).toEqual([]);
    }
  });
});
