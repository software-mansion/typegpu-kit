// Fails the build when Starlight's markup that the site restyles or reaches into is gone from
// every page. Starlight's markup isn't its public API and changes between versions; without this,
// an upgrade would leave the frame unstyled or a script finding nothing, and the build passing.

import type { AstroIntegration } from 'astro';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { htmlFiles, scan } from './links.ts';

/**
 * Starlight's markup, as `scan` names parts, and what relies on each. Only parts the site always
 * has belong here: the frame, headings' anchors, code blocks. Those of components a page may leave
 * out (asides, tabs, steps, cards, file trees, badges) and the search results, which Pagefind
 * builds in the browser, are checked by eye after an upgrade.
 */
export const starlightParts: Readonly<Record<string, string>> = {
  '[data-has-sidebar]': 'theme.css and Header.astro',
  '[data-has-hero]': 'theme.css',
  '.header': 'theme.css and HomeHero.astro',
  '.main-frame': 'HomeHero.astro',
  '.main-pane': 'HomeHero.astro',
  '.sidebar-pane': 'theme.css',
  '.sidebar-content': 'theme.css',
  '#starlight__sidebar': 'theme.css, Sidebar.astro and MobileMenuToggle.astro',
  '[popover]': 'MobileMenuToggle.astro, which opens #starlight__sidebar as a popover',
  '.right-sidebar': 'theme.css',
  '.content-panel': 'theme.css and HomeHero.astro',
  '.sl-markdown-content': 'theme.css',
  '.sl-anchor-link': 'theme.css',
  '.sl-skip-link': 'theme.css',
  'site-search': 'theme.css',
  '#starlight__search': 'theme.css',
  '.expressive-code': 'theme.css',
};

/** A line for each part of `expected` that none of `pages` holds. */
export function missingParts(
  pages: Iterable<string>,
  expected: Readonly<Record<string, string>> = starlightParts,
) {
  const found = new Set<string>();
  for (const html of pages) for (const part of scan(html).parts) found.add(part);

  return Object.entries(expected)
    .filter(([part]) => !found.has(part))
    .map(([part, users]) => `No page has ${part}, which ${users} relies on.`);
}

/** The check, run on the built site. */
export function partsIntegration(): AstroIntegration {
  return {
    name: 'typegpu-kit:parts',
    hooks: {
      'astro:build:done': ({ dir, logger }) => {
        const root = fileURLToPath(dir);
        const lines = missingParts(
          htmlFiles(root).map((file) => readFileSync(join(root, file), 'utf8')),
        );

        if (lines.length === 0) {
          logger.info(
            `Starlight's markup has the ${Object.keys(starlightParts).length} parts the site relies on.`,
          );
          return;
        }
        for (const line of lines) logger.error(line);
        throw new Error(
          "Starlight's markup changed; see above. Restyle what replaced each part, and update starlightParts in src/build/parts.ts.",
        );
      },
    },
  };
}
