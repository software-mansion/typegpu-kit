// Fails the build on a link to a page or an anchor that isn't there, on a page no link reaches,
// and on a published address that no longer resolves. It reads the built pages, so it sees the
// links components make (the index, the module headers, the map, the example cards) as well as
// those written in the pages.

import type { SatteriProcessorOptions } from '@astrojs/markdown-satteri';
import type { AstroIntegration } from 'astro';
import { existsSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

type HastPlugin = Exclude<
  NonNullable<SatteriProcessorOptions['hastPlugins']>[number],
  ((...args: never[]) => unknown) | readonly unknown[] | false | null | undefined
>;
type Context = Parameters<
  Extract<NonNullable<HastPlugin['element']>, { filter: string[] }>['visit']
>[1];
type Rebase = (href: unknown, node: Parameters<Context['setProperty']>[0], ctx: Context) => void;

/**
 * The Sätteri plugin that serves the pages' links from the root (`/kit/frames/`) under `base`
 * (`/typegpu-kit/`), for `markdown.processor`.
 */
export const baseLinks = (base: string): HastPlugin => {
  const rebase: Rebase = (href, node, ctx) => {
    if (typeof href === 'string' && href.startsWith('/') && !href.startsWith('//')) {
      ctx.setProperty(node, 'href', base + href.slice(1));
    }
  };
  return {
    name: 'typegpu-kit-base-links',
    element: { filter: ['a'], visit: (link, ctx) => rebase(link.properties?.href, link, ctx) },
    mdxJsxFlowElement: {
      filter: ['LinkCard'],
      visit(card, ctx) {
        const href = card.attributes.find(
          (each) => each.type === 'mdxJsxAttribute' && each.name === 'href',
        );
        rebase(href?.value, card, ctx);
      },
    },
  };
};

export interface LinkProblem {
  /** `page` when nothing is built at the address, `anchor` when the page has no such id. */
  readonly kind: 'page' | 'anchor';
  readonly href: string;
  /** The pages that link it, as addresses: `/kit/frames/`. */
  readonly from: readonly string[];
}

export interface LinkReport {
  readonly problems: readonly LinkProblem[];
  /** Pages no chain of links from the site's home reaches. */
  readonly unreachable: readonly string[];
}

const origin = 'https://typegpu-kit.invalid';

/** Comments, and elements whose text isn't markup: a link written in a script isn't one. */
const opaque = /<!--[\s\S]*?-->|<(script|style|template|textarea)\b[^>]*>[\s\S]*?<\/\1\s*>/gi;
const value = `(?:"[^"]*"|'[^']*'|[^\\s"'=<>\`]+)`;
const startTag = new RegExp(
  `<([a-z][\\w:-]*)((?:\\s+[^\\s"'>/=]+(?:\\s*=\\s*${value})?)*)\\s*/?>`,
  'gi',
);
const attribute = new RegExp(`([^\\s"'>/=]+)(?:\\s*=\\s*(${value}))?`, 'g');

const entities: Record<string, string> = { amp: '&', lt: '<', gt: '>', quot: '"', apos: "'" };
const decode = (text: string) =>
  text.replace(/&(#x?[\da-f]+|\w+);/gi, (whole, name: string) => {
    if (name.startsWith('#')) {
      const hex = name[1] === 'x' || name[1] === 'X';
      return String.fromCodePoint(hex ? parseInt(name.slice(2), 16) : Number(name.slice(1)));
    }
    return entities[name] ?? whole;
  });
const unquote = (raw: string) => decode(/^["']/.test(raw) ? raw.slice(1, -1) : raw);

/**
 * The `href` of every `<a>` and every `id` in a page, and its parts as selectors: each tag
 * (`site-search`), class (`.header`), id (`#_top`) and attribute name (`[data-has-sidebar]`).
 */
export function scan(html: string) {
  const hrefs: string[] = [];
  const ids = new Set<string>();
  const parts = new Set<string>();

  for (const [, tag, attributes = ''] of html.replace(opaque, '').matchAll(startTag)) {
    const element = tag!.toLowerCase();
    parts.add(element);

    for (const [, name, raw] of attributes.matchAll(attribute)) {
      const lower = name!.toLowerCase();
      parts.add(`[${lower}]`);
      if (raw === undefined) continue;

      if (lower === 'id') {
        ids.add(unquote(raw));
        parts.add(`#${unquote(raw)}`);
      } else if (lower === 'class') {
        for (const each of unquote(raw).split(/\s+/)) if (each) parts.add(`.${each}`);
      } else if (lower === 'href' && element === 'a') {
        hrefs.push(unquote(raw));
      }
    }
  }

  return { hrefs, ids, parts };
}

/** The address a built file is served at under `base`: `kit/frames/index.html` is `/kit/frames/`. */
export const addressOf = (file: string, base = '/') =>
  `${base}${file.split(sep).join('/')}`.replace(/(^|\/)index\.html$/, '$1');

export interface LinkCheck {
  /** Whether a file is built at an address that isn't a page, such as `/favicon.svg`. */
  readonly isFile?: (address: string) => boolean;
  /** Pages that needn't be reachable, such as the 404. */
  readonly exempt?: readonly string[];
  /** The page every other must be reachable from. Default `'/'`. */
  readonly home?: string;
}

/**
 * Checks every internal link between `pages`, each an address and its HTML, against the pages and
 * the files beside them.
 */
export function checkLinks(
  pages: ReadonlyMap<string, string>,
  { isFile = () => false, exempt = [], home = '/' }: LinkCheck = {},
): LinkReport {
  const scanned = new Map([...pages].map(([address, html]) => [address, scan(html)]));
  const pageAt = (pathname: string) =>
    [pathname, `${pathname}/`, pathname.replace(/(^|\/)index\.html$/, '$1')].find((each) =>
      scanned.has(each),
    );

  const problems = new Map<
    string,
    { kind: LinkProblem['kind']; href: string; from: Set<string> }
  >();
  const links = new Map<string, Set<string>>();

  for (const [address, { hrefs }] of scanned) {
    const out = new Set<string>();
    links.set(address, out);

    for (const href of hrefs) {
      if (href === '' || href === '#') continue;
      const url = new URL(href, origin + address);
      if (url.origin !== origin) continue;

      const pathname = decodeURIComponent(url.pathname);
      const target = pageAt(pathname);
      const fragment = decodeURIComponent(url.hash.slice(1));

      let kind: LinkProblem['kind'] | undefined;
      if (target === undefined) kind = isFile(pathname) ? undefined : 'page';
      else if (fragment && !scanned.get(target)!.ids.has(fragment)) kind = 'anchor';
      if (target !== undefined) out.add(target);
      if (!kind) continue;

      const key = `${kind} ${href}`;
      const problem = problems.get(key) ?? { kind, href, from: new Set<string>() };
      problem.from.add(address);
      problems.set(key, problem);
    }
  }

  const reached = new Set<string>();
  const queue = scanned.has(home) ? [home] : [];
  for (let next = queue.pop(); next !== undefined; next = queue.pop()) {
    if (reached.has(next)) continue;
    reached.add(next);
    queue.push(...(links.get(next) ?? []));
  }

  return {
    problems: [...problems.values()].map(({ from, ...problem }) => ({
      ...problem,
      from: [...from].toSorted(),
    })),
    unreachable: [...scanned.keys()]
      .filter((address) => !reached.has(address) && !exempt.includes(address))
      .toSorted(),
  };
}

const listOf = (from: readonly string[]) =>
  from.length > 3 ? `${from.slice(0, 3).join(', ')} and ${from.length - 3} more` : from.join(', ');

const problemOf: Record<LinkProblem['kind'], string> = {
  page: 'No page at',
  anchor: 'No such anchor:',
};

/** The report as lines for the build's log, or none when there's nothing to fix. */
export function describe({ problems, unreachable }: LinkReport) {
  return [
    ...problems.map(
      ({ kind, href, from }) => `${problemOf[kind]} ${href}, linked from ${listOf(from)}`,
    ),
    ...unreachable.map((address) => `No link reaches ${address}`),
  ];
}

/**
 * The addresses a built site serves that links from outside it rely on, without the base: every
 * page, and each module card's anchor. The kit's skill, its declarations and copied links point
 * at them.
 */
export function addressesOf(pages: ReadonlyMap<string, string>, base = '/'): string[] {
  return [...pages]
    .flatMap(([address, html]) => {
      const page = `/${address.slice(base.length)}`;
      const cards = [...scan(html).ids].filter((id) => id.startsWith('module-'));
      return [page, ...cards.map((id) => `${page}#${id}`)];
    })
    .toSorted();
}

/**
 * Compares the addresses a site serves with those it has published, as lines for the build's log.
 * A published address stays: a page that moves leaves a redirect, which also answers for the cards
 * it held, and a card that moves leaves its anchor. New addresses are recorded by local builds.
 * `redirects` are the served pages that send their reader on.
 */
export function checkAddresses(
  published: readonly string[],
  served: readonly string[],
  redirects: readonly string[] = [],
) {
  const now = new Set(served);
  const moved = new Set(redirects);
  return published
    .filter((address) => !now.has(address) && !moved.has(address.replace(/#.*/, '')))
    .map(
      (address) =>
        `The published address ${address} is gone; redirect it, keep its anchor or accept the removal with ADDRESSES=record pnpm build`,
    );
}

/** Names the pages without exactly one description, which link cards and search results show. */
export function checkDescriptions(
  pages: ReadonlyMap<string, string>,
  exempt: readonly string[] = [],
) {
  return [...pages]
    .filter(([address]) => !exempt.includes(address))
    .flatMap(([address, html]) => {
      const count = html.match(/<meta[^>]+name=["']?description["']?[\s/>]/gi)?.length ?? 0;
      return count === 1 ? [] : [`${address} has ${count} descriptions; give it one`];
    });
}

/** The addresses the site has published, which `ADDRESSES=record pnpm build` rewrites. */
const addressesFile = fileURLToPath(new URL('./addresses.json', import.meta.url));

/** The built pages under `dir`, as paths relative to it. */
export function htmlFiles(dir: string): string[] {
  return readdirSync(dir, { recursive: true, encoding: 'utf8' }).filter((file) =>
    file.endsWith('.html'),
  );
}

/** The check, run on the built site. */
export function linksIntegration(): AstroIntegration {
  let base = '/';
  return {
    name: 'typegpu-kit:links',
    hooks: {
      'astro:config:done': ({ config }) => {
        base = config.base.replace(/\/?$/, '/');
      },
      'astro:build:done': ({ dir, logger }) => {
        const root = fileURLToPath(dir);
        const pages = new Map(
          htmlFiles(root).map((file) => [
            addressOf(file, base),
            readFileSync(join(root, file), 'utf8'),
          ]),
        );
        const isFile = (address: string) =>
          address.startsWith(base) && existsSync(join(root, address.slice(base.length)));
        // A page that moved leaves a redirect, which no link reaches
        const redirects = [...pages]
          .filter(([, html]) => /<meta[^>]+http-equiv=["']?refresh/i.test(html))
          .map(([address]) => address);
        const lines = describe(
          checkLinks(pages, {
            isFile,
            exempt: [`${base}404.html`, `${base}404/`, ...redirects],
            home: base,
          }),
        );

        const served = addressesOf(pages, base).filter((address) => !address.startsWith('/404'));
        const published: string[] = JSON.parse(readFileSync(addressesFile, 'utf8'));
        const moved = redirects.map((address) => `/${address.slice(base.length)}`);
        if (!process.env.CI) {
          const added = served.filter((address) => !published.includes(address));
          if (process.env.ADDRESSES === 'record' || added.length > 0) {
            const kept =
              process.env.ADDRESSES === 'record'
                ? served
                : [...new Set([...published, ...served])].toSorted();
            writeFileSync(addressesFile, `${JSON.stringify(kept, null, 2)}\n`);
            if (added.length > 0) logger.info(`Added ${added.length} addresses.`);
          }
        }
        if (process.env.ADDRESSES !== 'record' || process.env.CI)
          lines.push(...checkAddresses(published, served, moved));
        lines.push(...checkDescriptions(pages, [`${base}404.html`, `${base}404/`, ...redirects]));

        if (lines.length === 0) {
          logger.info(`Every link between ${pages.size} pages resolves.`);
          return;
        }
        for (const line of lines) logger.error(line);
        throw new Error(`${lines.length} broken links or missing addresses; see above.`);
      },
    },
  };
}
