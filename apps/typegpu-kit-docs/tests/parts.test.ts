import { expect, it } from 'vitest';
import { scan } from '../src/build/links.ts';
import { missingParts } from '../src/build/parts.ts';

it('names the tags, classes, ids and attributes a page holds', () => {
  const { parts } = scan(`
    <html data-has-sidebar>
    <sl-sidebar-pane popover id="starlight__sidebar" class="sidebar-pane  astro-6vmk2rg5">
    <script>'<b class="scripted">'</script>
  `);

  expect([...parts]).toEqual([
    'html',
    '[data-has-sidebar]',
    'sl-sidebar-pane',
    '[popover]',
    '[id]',
    '#starlight__sidebar',
    '[class]',
    '.sidebar-pane',
    '.astro-6vmk2rg5',
  ]);
});

it('says which parts no page has, and what relies on them', () => {
  const expected = {
    '.header': 'theme.css',
    '#starlight__search': 'theme.css',
    'site-search': 'Header.astro',
  };
  const pages = ['<header class="header">', '<site-search><dialog></dialog></site-search>'];

  expect(missingParts(pages, expected)).toEqual([
    'No page has #starlight__search, which theme.css relies on.',
  ]);
});
