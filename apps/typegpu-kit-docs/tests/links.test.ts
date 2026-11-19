import { expect, it } from 'vitest';
import {
  addressesOf,
  checkAddresses,
  checkDescriptions,
  addressOf,
  checkLinks,
  describe,
  scan,
} from '../src/build/links.ts';

it('reads links and ids from markup, not from scripts, styles or comments', () => {
  const { hrefs, ids } = scan(`
    <h2 id="setting-up">Setting up</h2>
    <a class="pill" aria-label="a > b" href="/kit/frames/#module-frames&amp;x">frames</a>
    <svg><a href='../map/'><text id=map-label>map</text></a></svg>
    <link rel="stylesheet" href="/_astro/page.css">
    <!-- <a href="/commented/">old</a> -->
    <script>document.body.innerHTML = '<a href="/scripted/">x</a>';</script>
    <style>a[href="/styled/"] { color: red }</style>
  `);

  expect(hrefs).toEqual(['/kit/frames/#module-frames&x', '../map/']);
  expect([...ids]).toEqual(['setting-up', 'map-label']);
});

it('serves a built file at its address', () => {
  expect(addressOf('index.html')).toBe('/');
  expect(addressOf('kit/frames/index.html')).toBe('/kit/frames/');
  expect(addressOf('404.html')).toBe('/404.html');
});

it('reports links to missing pages and anchors, and pages no link reaches', () => {
  const pages = new Map([
    ['/', '<a href="/kit/frames/">frames</a><a href="https://example.com/">out</a>'],
    [
      '/kit/frames/',
      `<h1 id="_top">Frames</h1>
       <a href="#_top">top</a>
       <a href="../surface/#resizing">surface</a>
       <a href="/kit/frames/#missing">nowhere</a>
       <a href="/kit/framez/">typo</a>
       <a href="/favicon.svg">icon</a>
       <a href="/examples/blockfall/?debug">blockfall</a>`,
    ],
    ['/kit/surface/', '<h2 id="sizing">Sizing</h2>'],
    ['/examples/blockfall/', '<a href="/">back</a>'],
    ['/orphan/', '<a href="/">back</a>'],
    ['/404.html', '<a href="/">back</a>'],
  ]);

  const report = checkLinks(pages, {
    isFile: (address) => address === '/favicon.svg',
    exempt: ['/404.html'],
  });

  expect(report.problems).toEqual([
    { kind: 'anchor', href: '../surface/#resizing', from: ['/kit/frames/'] },
    { kind: 'anchor', href: '/kit/frames/#missing', from: ['/kit/frames/'] },
    { kind: 'page', href: '/kit/framez/', from: ['/kit/frames/'] },
  ]);
  expect(report.unreachable).toEqual(['/orphan/']);
  expect(describe(report)).toEqual([
    'No such anchor: ../surface/#resizing, linked from /kit/frames/',
    'No such anchor: /kit/frames/#missing, linked from /kit/frames/',
    'No page at /kit/framez/, linked from /kit/frames/',
    'No link reaches /orphan/',
  ]);
});

it('keeps every published page and module card, and allows new addresses', () => {
  const pages = new Map([
    ['/kit/', '<h1 id="_top">Home</h1>'],
    [
      '/kit/kit/finish/',
      '<section id="module-finish"></section><section id="module-bloom"></section>',
    ],
    ['/kit/kit/frames/', '<h2 id="every-frame">Every frame</h2>'],
  ]);
  const served = addressesOf(pages, '/kit/');

  expect(served).toEqual([
    '/',
    '/kit/finish/',
    '/kit/finish/#module-bloom',
    '/kit/finish/#module-finish',
    '/kit/frames/',
  ]);
  expect(checkAddresses(served, served)).toEqual([]);
  expect(checkAddresses([...served, '/kit/tone/#module-tone'], served.slice(1))).toEqual([
    'The published address / is gone; redirect it, keep its anchor or accept the removal with ADDRESSES=record pnpm build',
    'The published address /kit/tone/#module-tone is gone; redirect it, keep its anchor or accept the removal with ADDRESSES=record pnpm build',
  ]);
  expect(checkAddresses(served.slice(1), served)).toEqual([]);
  // A page that redirects answers for the cards it held
  const moved = ['/', '/kit/finish/', '/kit/frames/'];
  expect(checkAddresses(served, moved, ['/kit/finish/'])).toEqual([]);
});

it('asks every page for one description, but those it is told to skip', () => {
  const described = '<meta name="description" content="Draws a frame." />';
  const pages = new Map([
    ['/kit/frames/', `<head>${described}<meta property="og:description" content="x" /></head>`],
    ['/source/forest/', '<head><meta property="og:description" content="x" /></head>'],
    ['/examples/forest/', `<head>${described}${described}</head>`],
    ['/404.html', '<head></head>'],
  ]);

  expect(checkDescriptions(pages, ['/404.html'])).toEqual([
    '/source/forest/ has 0 descriptions; give it one',
    '/examples/forest/ has 2 descriptions; give it one',
  ]);
});
