import { createSatteriMarkdownProcessor } from '@astrojs/markdown-satteri';
import { expect, it } from 'vitest';
import { tableLabels } from '../src/build/tables.ts';

const render = async (markdown: string) => {
  const processor = await createSatteriMarkdownProcessor({ hastPlugins: [tableLabels] });
  return (await processor.render(markdown)).code;
};
const starts = (html: string, tag: string) =>
  [...html.matchAll(new RegExp(`<${tag}\\b[^>]*>`, 'g'))].map(([start]) => start);

it("labels each body cell with its column's heading", async () => {
  const html = await render(`
| Option | \`default\` | |
| --- | --- | --- |
| size | 1 | a note |
`);

  expect(starts(html, 'table')).toEqual(['<table>']);
  expect(starts(html, 'td')).toEqual([
    '<td data-label="Option">',
    '<td data-label="default">',
    '<td>',
  ]);
});

it('marks a table grouped when a row leaves its first cell empty', async () => {
  const html = await render(`
| Group | Row | Value |
| --- | --- | --- |
| render | meshes | 2 |
| | lights | 1 |
`);

  expect(starts(html, 'table')).toEqual(['<table data-grouped="">']);
});
