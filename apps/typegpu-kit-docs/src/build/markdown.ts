// Writes each page as Markdown at its address with `.md` for the trailing slash, `/kit/frames.md`,
// each example's description and files at `/examples/<name>.md`, the gallery's list at
// `/examples.md`, and `/llms.txt`, which lists them, for agents reading the site. Each names the
// kit's version. The package's docs/ ships the same pages, unstamped, since they match the installed
// code, and the examples' own files in place of their Markdown.

import type { AstroIntegration } from 'astro';
import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, join, posix } from 'node:path';
import { fileURLToPath } from 'node:url';
import { categories, examples as everyExample, type Example } from 'typegpu-kit-examples';
import {
  checkedContracts,
  kitVersion,
  llmsGroups,
  renderLlms,
  renderPages,
  stamped,
} from '../contracts/agents.ts';
import type { ModuleContract } from '../contracts/contract.ts';
import { read, type Source } from '../source/read.ts';

export function markdownIntegration(): AstroIntegration {
  let start = '/';

  return {
    name: 'typegpu-kit:markdown',
    hooks: {
      'astro:config:done': ({ config }) => {
        const base = config.base.replace(/\/?$/, '/');
        start = config.site ? new URL(base, config.site).href : base;
      },
      'astro:build:done': ({ dir, logger }) => {
        const root = fileURLToPath(dir);
        const contracts = checkedContracts();
        const version = kitVersion();
        const { files, sections } = renderPages(contracts, exampleMarkdown);
        const all = new Map(files);
        all.set(exampleMarkdown('examples'), renderGallery());
        for (const example of everyExample) {
          all.set(
            exampleMarkdown(`examples/${example.key}`),
            renderExample(example, read(example.key), contracts, sections),
          );
        }
        for (const [path, text] of all) {
          mkdirSync(dirname(join(root, path)), { recursive: true });
          writeFileSync(join(root, path), path.endsWith('.md') ? stamped(text, version) : text);
        }
        writeFileSync(
          join(root, 'llms.txt'),
          renderLlms(
            [...llmsGroups(contracts), exampleListing()],
            (path) => `${start}${path}`,
            version,
          ),
        );
        logger.info(
          `Wrote ${files.size} pages and ${everyExample.length} examples as Markdown, and llms.txt, for ${version}.`,
        );
      },
    },
  };
}

/** An example's Markdown on the site, by its slug: `examples/lattice.md`, and the gallery's. */
export const exampleMarkdown = (slug: string) => `${slug}.md`;

/** The examples, as /llms.txt lists them after the pages, each with its gallery line. */
export function exampleListing() {
  return {
    label: 'Examples',
    pages: galleryOrder().map((example) => ({
      id: `examples/${example.key}`,
      title: example.title,
      description: example.line,
    })),
  };
}

/** The examples in the gallery's order: by category, then by their own order. */
const galleryOrder = () =>
  categories.flatMap(({ name }) => everyExample.filter((example) => example.category === name));

/** The site's examples.md: every example under the gallery's categories, linked as Markdown. */
export function renderGallery() {
  return [
    '# Examples',
    '',
    'Each example runs full screen in a WebGPU browser at `examples/<name>/`. Its Markdown, `examples/<name>.md`, holds its description and every file, in reading order.',
    ...categories.flatMap(({ name, title, line }) => [
      '',
      `## ${title}`,
      '',
      line,
      '',
      ...everyExample
        .filter((example) => example.category === name)
        .map((example) => `- [${example.title}](examples/${example.key}.md): ${example.line}`),
    ]),
    '',
  ].join('\n');
}

/** A code fence longer than any run of backticks in `code`. */
const fence = (code: string) =>
  '`'.repeat(Math.max(3, ...[...code.matchAll(/`+/g)].map(([run]) => run.length + 1)));

/**
 * The site's `examples/<name>.md`: an example's description, the kit modules it imports, linked
 * where `sections` has them, and its files in reading order, whole.
 */
export function renderExample(
  example: Example,
  source: Source,
  contracts: readonly ModuleContract[],
  sections: ReadonlyMap<string, string>,
) {
  // Of what it imports from the kit, the modules: `particles` and `memo`, not `Particle`
  const modules = source.kit.flatMap((name) => {
    const contract = contracts.find((each) => each.namespace === name);
    if (!contract) return [];
    const section = sections.get(contract.id);
    return [section ? `[\`${name}\`](${posix.relative('examples', section)})` : `\`${name}\``];
  });
  return [
    `# ${example.title}`,
    '',
    example.description,
    '',
    `It runs full screen at [${example.key}/](${example.key}/) in a WebGPU browser.${
      modules.length > 0 ? ` Kit modules: ${modules.join(', ')}.` : ''
    }`,
    '',
    '## Files, in reading order',
    '',
    ...source.files.map((file) => `- \`${file.name}\`${file.summary ? `: ${file.summary}` : ''}`),
    ...source.files.flatMap((file) => [
      '',
      `## ${file.name}`,
      '',
      `${fence(file.code)}ts title="${file.name}"`,
      file.code.replace(/\n$/, ''),
      fence(file.code),
    ]),
    '',
  ].join('\n');
}
