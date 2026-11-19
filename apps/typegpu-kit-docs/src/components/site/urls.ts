// Addresses on the site, under astro.config.mjs's `base`: `/` locally, `/typegpu-kit/` on GitHub
// Pages. Pages write their links from the root (`/kit/frames/`), and links.ts prefixes them

/** The site's base, ending in a slash; `/` in Node, where the kit's guide is generated. */
export const BASE = (import.meta.env?.BASE_URL ?? '/').replace(/\/?$/, '/');

/** The id of the page's title, which Starlight's table of contents links as its first entry. */
export const PAGE_TITLE_ID = '_top';

/** A link written from the root (`/kit/frames/`) under the base; others as they are. */
export const rooted = (href: string) =>
  href.startsWith('/') && !href.startsWith('//') ? BASE + href.slice(1) : href;

/** The slug of the page that says how to read the docs, which every strip and card links. */
export const READING = 'how-to-read';

/** A docs page's address from its slug: `kit/frames`, or `''` for the overview. */
export const pageHref = (slug: string) => (slug ? `${BASE}${slug}/` : BASE);

/** An example's full-screen page, from its key (`blockfall`), or the gallery without one. */
export const exampleHref = (key?: string) => pageHref(key ? `examples/${key}` : 'examples');

/** A page's Markdown, from its slug: `kit/frames.md`, `examples/lattice.md`; the map's is the module index. */
export const markdownHref = (slug: string) =>
  `${BASE}${slug === 'map' ? 'modules' : slug || 'index'}.md`;

/** An example's source page, from its key. */
export const sourceHref = (key: string) => pageHref(`source/${key}`);

/** The slug of the page at `pathname`: `/kit/frames/` is `kit/frames`, the base is `''`. */
export const slugOf = (pathname: string) =>
  (pathname.startsWith(BASE) ? pathname.slice(BASE.length) : pathname)
    .replace(/^\/+|\/+$/g, '')
    .replace(/(^|\/)index\.html$|\.html$/, '');
