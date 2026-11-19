import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { pageFiles } from '../src/contracts/agents.ts';
import { moduleById } from '../src/contracts/contract.ts';
import { extractContracts, repo } from '../src/contracts/extract.ts';
import { frontmatterOf } from '../src/contracts/markdown.ts';

/** Every public page that lists modules, with the parts of it these checks read. */
const modulePages = pageFiles()
  .map(({ id, file }) => {
    const { fields, body } = frontmatterOf(readFileSync(file, 'utf8'));
    return {
      slug: id,
      title: fields.get('title') ?? '',
      description: fields.get('description') ?? '',
      modules: fields.get('modules'),
      body,
    };
  })
  .filter((page) => page.modules !== undefined);

/** A page's body before its first section, each code block cut to its opening fence. */
const opening = (body: string) =>
  body.replace(/^(`{3,}|~{3,})[\s\S]*?^\1/gm, (block) => block.split('\n')[0]).split(/^## /m)[0];
const usage = /^```(ts twoslash|tsx)/m;
/** The components a page imports from the figures in `components/viz/`. */
const figuresOf = (body: string) =>
  [...body.matchAll(/^import (\w+) from '[^']*\/components\/viz\/[^']+';$/gm)].map((m) => m[1]);
const showsFigure = (text: string, names: readonly string[]) =>
  names.some((name) => new RegExp(`^<${name}(?:[\\s/>]|$)`, 'm').test(text));

describe('module pages', () => {
  it('list their modules inline, as the agents docs read them', () => {
    const block = modulePages.filter((page) => !/^\[[^\]]+\]$/.test(page.modules!));
    expect(block.map((page) => page.slug)).toEqual([]);
  });

  it('take a noun phrase for a title and one sentence for a description', () => {
    const odd = modulePages.filter(
      (page) =>
        /[.!?:]$|@/.test(page.title) ||
        !/^[A-Z`].*\.$/.test(page.description) ||
        /[.!?] /.test(page.description) ||
        page.description.includes('@typegpu/'),
    );
    expect(odd.map((page) => `${page.slug}: ${page.title} / ${page.description}`)).toEqual([]);
  });

  it('show a figure or their usage before their first section', () => {
    const bare = modulePages.filter((page) => {
      const text = opening(page.body);
      return !showsFigure(text, figuresOf(page.body)) && !usage.test(text);
    });
    expect(bare.map((page) => page.slug)).toEqual([]);
  });

  it('say the problem in a paragraph before their first figure', () => {
    const cold = modulePages.filter((page) => {
      const names = figuresOf(page.body);
      const lines = opening(page.body)
        .split('\n')
        .filter((line) => line.trim() !== '' && !/^import |^<\/?(Module|Hidden)\b/.test(line));
      const figure = lines.findIndex((line) => showsFigure(line.trim(), names));
      return figure !== -1 && !lines.slice(0, figure).some((line) => /^[A-Z`[*]/.test(line));
    });
    expect(cold.map((page) => page.slug)).toEqual([]);
  });

  it('keep a list to five items and a bullet to 40 words, outside <Rules>', () => {
    const long = modulePages.flatMap((page) =>
      listsOf(page.body).flatMap((items) => {
        const words = Math.max(...items.map(wordsIn));
        if (items.length <= 5 && words <= 40) return [];
        const start = items[0].slice(0, 40);
        return [`${page.slug}: ${items.length} items, longest ${words} words, at "${start}"`];
      }),
    );
    expect(long).toEqual([]);
  });

  it('quote only messages their modules throw in their Errors sections', () => {
    const { contracts } = extractContracts();
    const drifted = modulePages.flatMap((page) => {
      const errors = /^## Errors\n([\s\S]*?)(?=^## |(?![\s\S]))/m.exec(page.body)?.[1];
      if (!errors) return [];
      const source = page
        .modules!.slice(1, -1)
        .split(',')
        .flatMap((id) => sourcesOf(join(repo.package, moduleById(contracts, id.trim()).file)))
        .map(thrownText)
        .join('\n');
      return messagesIn(errors)
        .filter((message) => !said(message, source))
        .map((message) => `${page.slug}: ${message}`);
    });
    expect(drifted).toEqual([]);
  });
});

/** A page's lists outside code and `<Rules>`, each as its top-level items with what nests in them. */
function listsOf(body: string) {
  const text = body
    .replace(/^(`{3,}|~{3,})[\s\S]*?^\1/gm, '')
    .replace(/^<Rules>[\s\S]*?^<\/Rules>/gm, '');
  const lists: string[][] = [];
  let open: string[] | undefined;
  for (const line of text.split('\n')) {
    if (/^(?:[-*]|\d+\.) /.test(line)) {
      if (!open) lists.push((open = []));
      open.push(line.replace(/^(?:[-*]|\d+\.) /, ''));
    } else if (open && /^\s+\S/.test(line)) {
      open[open.length - 1] += ` ${line.trim()}`;
    } else if (line.trim() !== '') {
      open = undefined;
    }
  }
  return lists;
}

/** The words a reader reads in a bullet: a link counts by its text. */
const wordsIn = (item: string) =>
  item
    .replace(/\[([^\]]*)\]\([^)]*\)/g, '$1')
    .split(/\s+/)
    .filter(Boolean).length;

/** A module's file and the files it imports by relative path, which hold what it throws. */
function sourcesOf(file: string, seen = new Set<string>()): string[] {
  if (!seen.has(file)) {
    seen.add(file);
    for (const [, specifier] of readFileSync(file, 'utf8').matchAll(/from '(\.[^']+\.ts)'/g)) {
      sourcesOf(join(dirname(file), specifier!), seen);
    }
  }
  return [...seen];
}

/**
 * A file's text as its messages read: a string split over lines joined, escaped quotes as
 * written, a local string constant in place of its name, and `§` for anything else interpolated.
 */
function thrownText(file: string) {
  const text = readFileSync(file, 'utf8')
    .replace(/['`]\s*\+\s*['`]/g, '')
    .replace(/\\([`'])/g, '$1');
  const constants = new Map(
    [...text.matchAll(/const (\w+) = (?:`([^`$]*(?:\$\{\w+\}[^`$]*)*)`|'([^']*)');/g)].map(
      (match) => [match[1], match[2] ?? match[3]] as const,
    ),
  );
  return text
    .replace(/\$\{(\w+)\}/g, (whole, name: string) => constants.get(name) ?? whole)
    .replace(/\$\{[^}]*\}/g, '§');
}

/** The code spans in Markdown text, double-backticked ones too. */
const spans = (text: string) =>
  [...text.matchAll(/(`{1,2})(?!`)(.+?)\1(?!`)/g)].map((match) => match[2].trim());

const escape = (text: string) => text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

/** Whether a word's run of letters is a value the code fills in: a number, a quoted name or a type. */
const value = (runs: readonly string[], i: number) =>
  /\d|[a-z][A-Z]/.test(runs[i]) || (runs[i - 1].endsWith("'") && runs[i + 1][0] === "'");

/**
 * The messages an Errors section quotes: a table's first column, or a code span in a list that
 * reads as a sentence rather than an identifier.
 */
function messagesIn(section: string) {
  return section.split('\n').flatMap((line) => {
    if (line.startsWith('|')) {
      const cell = line.split('|')[1];
      return /^\s*-+\s*$|^\s*Message\s*$/.test(cell) ? [] : spans(cell);
    }
    return spans(line).filter((span) => / /.test(span) && /^[A-Z'"]|[.…]$/.test(span));
  });
}

/**
 * Whether the source says a message: each part between its `…` and bracketed values appears word
 * for word. A value the code interpolates, `§`, stands for a number, a quoted name or a type, or
 * for up to four words between two the source has, or before an elision.
 */
function said(message: string, source: string) {
  const parts = message.split(/(…|\[[^\]]*\])/);
  return parts
    .map((part, i) => ({
      words: part.trim().split(/\s+/).filter(Boolean),
      elided: i % 2 === 0 && i + 1 < parts.length,
    }))
    .filter(({ words }, i) => i % 2 === 0 && words.join(' ').length > 2)
    .every(({ words, elided }) => {
      const tokens = words.map((word) => {
        const runs = word.split(/(\w+)/);
        const pattern = runs.map((run, i) =>
          i % 2 === 0 ? escape(run) : value(runs, i) ? `(?:${run}|§)` : run,
        );
        return new RegExp(pattern.join(''), 'y');
      });
      const at = (i: number, p: number): boolean => {
        if (i === tokens.length) return true;
        if (i > 0 && source[p] === '§') {
          const most = Math.min(4, tokens.length - i - (elided ? 0 : 1));
          for (let k = 1; k <= most; k++) if (after(i + k, p + 1)) return true;
        }
        const token = tokens[i];
        token.lastIndex = p;
        return token.test(source) && after(i + 1, token.lastIndex);
      };
      const after = (i: number, p: number) => {
        if (i === tokens.length) return true;
        const space = /\s+/y;
        space.lastIndex = p;
        if (space.test(source)) return at(i, space.lastIndex);
        return (source[p - 1] === '§' || source[p] === '§') && at(i, p);
      };
      for (let p = 0; p < source.length; p++) if (at(0, p)) return true;
      return false;
    });
}
