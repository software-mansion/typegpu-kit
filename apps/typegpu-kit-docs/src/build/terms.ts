// Marks the first use of each of the docs' terms (components/site/terms.ts) in a page's prose as a
// button, and ends the page with one popover per term it marked: the term, its sentence and a link
// to the section that teaches it. The page that teaches a term takes no mark for it, and the
// reading page, which lists them all, takes none.

import type { SatteriProcessorOptions } from '@astrojs/markdown-satteri';
import { readFileSync } from 'node:fs';
import {
  inWords,
  plain as allPlain,
  terms as allTerms,
  termId,
  type Term,
} from '../components/site/terms.ts';
import { READING } from '../components/site/urls.ts';
import { frontmatterOf, headingsOf } from '../contracts/markdown.ts';

type Entry = NonNullable<SatteriProcessorOptions['hastPlugins']>[number];
type Factory = Extract<Entry, (...args: never[]) => unknown>;
type Plugin = Exclude<Entry, Factory | readonly unknown[] | false | null | undefined>;
type Context = Parameters<NonNullable<Plugin['before']>>[1];
type Node = Parameters<Context['replaceNode']>[0];
type Content = Exclude<Parameters<Context['replaceNode']>[1], readonly unknown[]>;

const docs = new URL('../content/docs/', import.meta.url);

/** The docs page a file holds, as a slug: `kit/frames`, or `''` for the overview. */
function pageOf(file: URL | undefined) {
  if (!file?.href.startsWith(docs.href)) return undefined;
  const id = file.href.slice(docs.href.length).replace(/\.mdx?$/, '');
  return id === 'index' ? '' : id;
}

/** The page a term's home is on, as a slug. */
const homePage = ({ home }: Term) => home.replace(/#.*$/, '').replace(/^\/|\/$/g, '');

/** A page's title and the source of its body, by its slug. */
type Read = (page: string) => { title: string; body: string } | undefined;

const fromSource: Read = (page) => {
  const source = readFileSync(new URL(`${page || 'index'}.mdx`, docs), 'utf8');
  const { fields, body } = frontmatterOf(source);
  return { title: fields.get('title') ?? '', body };
};

/**
 * Where a term's link goes and what it says: the heading of the section that teaches it, the title
 * of the page when its opening does, or `Words` for its row in the Words section. `read` gives a
 * page where its source file is out of reach, as in a component the build bundles.
 */
export function homeOf(term: Term, read: Read = fromSource) {
  if (inWords(term)) return { href: term.home, label: 'Words' };

  const anchor = term.home.split('#')[1];
  const source = read(homePage(term));
  const label = anchor ? headingsOf(source?.body ?? '').get(anchor) : source?.title;
  if (!label) throw new Error(`No section at ${term.home}, the home of the term "${term.term}".`);
  return { href: term.home, label };
}

const escaped = (text: string) => text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

/**
 * Finds the first use of each term in a page's prose, text by text in reading order. A use is one
 * of a term's forms as a whole word, in any case; where phrases overlap, the longest is the one
 * read, and a phrase in `plain` is no use. `page` is the page's slug, whose own terms it leaves
 * alone.
 */
export function marker(terms: readonly Term[], page?: string, plain: readonly string[] = []) {
  const byForm = new Map<string, Term | undefined>(
    plain.map((form) => [form.toLowerCase(), undefined]),
  );
  for (const term of terms) {
    const forms = term.alone === false ? (term.also ?? []) : [term.term, ...(term.also ?? [])];
    for (const form of forms) byForm.set(form.toLowerCase(), term);
  }
  // A page's text holds the apostrophe as typed or as smart punctuation sets it
  const forms = [...byForm.keys()]
    .toSorted((a, b) => b.length - a.length)
    .map((form) => escaped(form).replaceAll(' ', '\\s+').replaceAll("'", "['’]"));
  const use = new RegExp(`(?<![\\p{L}\\p{N}_-])(?:${forms.join('|')})(?![\\p{L}\\p{N}_-])`, 'giu');
  const marked = new Set<Term>();

  return {
    /** The terms marked so far, in the order the page uses them. */
    marked,
    /** `text` in pieces: what stays text, and each term used here first on the page, as written. */
    split(text: string) {
      const pieces: (string | { term: Term; text: string })[] = [];
      let from = 0;
      for (const match of text.matchAll(use)) {
        const term = byForm.get(match[0].toLowerCase().replace(/\s+/g, ' ').replace('’', "'"));
        if (!term || marked.has(term) || homePage(term) === page) continue;
        marked.add(term);
        if (match.index > from) pieces.push(text.slice(from, match.index));
        pieces.push({ term, text: match[0] });
        from = match.index + match[0].length;
      }
      if (from < text.length) pieces.push(text.slice(from));
      return pieces;
    },
  };
}

/** Elements that hold prose, and those whose text is no prose: names, code, links and captions. */
const PROSE = /^(p|li|td)$/;
const NEVER =
  /^(h[1-6]|th|caption|figure|figcaption|summary|a|button|label|strong|b|code|pre|kbd|samp|var|script|style|svg|FileTree)$/;
const NEVER_CLASS = /(^|\s)(not-content|starlight-aside__title)(\s|$)/;

const nameOf = (node: Node) =>
  node.type === 'element'
    ? node.tagName
    : node.type === 'mdxJsxFlowElement' || node.type === 'mdxJsxTextElement'
      ? (node.name ?? '')
      : '';

function classOf(node: Node) {
  if (node.type === 'element') return [node.properties.className ?? ''].flat().join(' ');
  if (node.type !== 'mdxJsxFlowElement' && node.type !== 'mdxJsxTextElement') return '';
  const attribute = node.attributes.find(
    (each) => each.type === 'mdxJsxAttribute' && each.name === 'class',
  );
  return typeof attribute?.value === 'string' ? attribute.value : '';
}

const text = (value: string): Content => ({ type: 'text', value });
const element = (
  tagName: string,
  properties: Record<string, string | string[]>,
  children: Content[],
): Content => ({ type: 'element', tagName, properties, children }) as Content;

/** A term's button and its popover name each other, and share an anchor name for placing it. */
const button = (term: Term, written: string) =>
  element(
    'button',
    {
      type: 'button',
      className: ['term'],
      popoverTarget: termId(term),
      style: `anchor-name: --${termId(term)}`,
    },
    [text(written)],
  );

function popover(term: Term) {
  const { href, label } = homeOf(term);
  return element(
    'div',
    {
      id: termId(term),
      popover: '',
      className: ['term-pop', 'not-content'],
      style: `position-anchor: --${termId(term)}`,
      'data-pagefind-ignore': '',
    },
    [
      element('b', {}, [text(term.term)]),
      element('span', {}, [text(term.means)]),
      element('a', { href }, [text(`${label} →`)]),
    ],
  );
}

/** The Sätteri plugin, for `markdown.processor`, before the plugin that rebases links. */
export const termMarks =
  (terms: readonly Term[] = allTerms, plain: readonly string[] = allPlain): Factory =>
  ({ fileURL }) => {
    const page = pageOf(fileURL);
    if (page === READING) return null;

    return {
      name: 'typegpu-kit-terms',
      before(root, ctx) {
        const { marked, split } = marker(terms, page, plain);

        const walk = (node: Node, prose: boolean) => {
          if (node.type === 'text') {
            const pieces = prose ? split(node.value) : [];
            if (pieces.some((piece) => typeof piece !== 'string')) {
              ctx.replaceNode(
                node,
                pieces.map((piece) =>
                  typeof piece === 'string' ? text(piece) : button(piece.term, piece.text),
                ),
              );
            }
            return;
          }
          if (!('children' in node)) return;
          const name = nameOf(node);
          if (NEVER.test(name) || NEVER_CLASS.test(classOf(node))) return;
          for (const child of node.children) walk(child, prose || PROSE.test(name));
        };

        walk(root, false);
        if (marked.size > 0) ctx.appendChild(root, [...marked].map(popover));
      },
    };
  };
