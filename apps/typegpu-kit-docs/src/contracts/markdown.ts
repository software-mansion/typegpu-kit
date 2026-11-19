import { dirname, posix } from 'node:path';
import { linesOf, MANAGERS } from '../components/site/managers.ts';
import { terms } from '../components/site/terms.ts';

export interface PageOptions {
  /** A module's card where the page shows one, as Markdown. */
  readonly module: (id: string) => string;
  /** Where every module is listed, from the docs' root. */
  readonly index: string;
  /** An example's file from the docs' root, by its slug: `examples/lattice`, or `examples` for the
   * gallery. Default its folder, `examples/lattice/`. */
  readonly example?: ((slug: string) => string) | undefined;
}

export interface ConvertedPage {
  /** The file, from the docs' root: `kit/frames.md`. */
  readonly path: string;
  readonly text: string;
  /** The modules it shows, each with the anchor of its section, or `''` for the top. */
  readonly modules: ReadonlyMap<string, string>;
}

/** The Markdown file of the page with collection id `id`: `kit/frames.md`, `index.md`. */
export const markdownPath = (id: string) => `${id}.md`;

/** A heading's anchor as GitHub and Astro make it, with `seen` counting repeats. */
export function slugger() {
  const seen = new Map<string, number>();

  return (heading: string) => {
    const base = plainText(heading)
      .toLowerCase()
      .replace(/[^\p{L}\p{M}\p{N}\p{Pc}\- ]/gu, '')
      .replace(/ /g, '-');
    const count = seen.get(base) ?? 0;
    seen.set(base, count + 1);
    return count === 0 ? base : `${base}-${count}`;
  };
}

/** A heading's text as it renders: no code marks, emphasis or link targets. */
const plainText = (markdown: string) =>
  markdown
    .replace(/\[([^\]]*)\]\([^)]*\)/g, '$1')
    .replace(/[`*]/g, '')
    .trim();

/**
 * A page's frontmatter keys on one line each, unquoted, and its body. A key holding nested keys or
 * a block list reads as empty.
 */
export function frontmatterOf(source: string) {
  const match = /^---\n([\s\S]*?)\n---\n/.exec(source);
  const fields = new Map<string, string>();
  if (!match) return { fields, body: source };

  for (const [, key, raw] of match[1].matchAll(/^(\w+):[ \t]*(.*)$/gm)) {
    const value = raw!.trim();
    fields.set(key!, /^(["']).*\1$/.test(value) ? value.slice(1, -1) : value);
  }
  return { fields, body: source.slice(match[0].length) };
}

/** The module ids a page's frontmatter lists. */
export function modulesOf(source: string) {
  const listed = frontmatterOf(source).fields.get('modules');
  if (!listed) return [];
  return listed
    .replace(/^\[|\]$/g, '')
    .split(',')
    .map((id) => id.trim())
    .filter(Boolean);
}

/**
 * A site link as a link from `from` (a Markdown file under the docs' root) to another file there:
 * `/kit/frames/#x` is `../kit/frames.md#x`, an example's page what `example` gives, by default its
 * directory, and the map `index`, which stands for it.
 */
export function relink(
  href: string,
  from: string,
  index?: string,
  example = (slug: string) => `${slug}/`,
) {
  if (!href.startsWith('/')) return href;

  const [, path = '', anchor = ''] = /^([^?#]*)(?:\?[^#]*)?(#.*)?$/.exec(href)!;
  const slug = path.replace(/^\/+|\/+$/g, '');

  const target =
    slug === 'examples' || slug.startsWith('examples/')
      ? example(slug)
      : slug === 'map' && index
        ? index
        : markdownPath(slug === '' ? 'index' : slug);
  const relative = posix.relative(dirname(from), target) || target;
  return `${relative}${target.endsWith('/') && !relative.endsWith('/') ? '/' : ''}${anchor}`;
}

/** Links in a line of prose, relinked from `from`. */
function relinkLine(line: string, from: string, options: PageOptions) {
  return line.replace(/\[((?:[^\]`]|`[^`]*`)*)\]\((\/[^)\s]*)\)/g, (_, text: string, href) => {
    const target = relink(href, from, options.index, options.example);
    return `[${text}](${target})`;
  });
}

/** An opening tag's name and attributes, the raw text of each value, from `<` to its `>`. */
interface Tag {
  readonly name: string;
  readonly attributes: ReadonlyMap<string, string>;
  readonly selfClosing: boolean;
  /** Where in the text the tag ends, after its `>`. */
  readonly end: number;
}

function readTag(text: string, start: number): Tag {
  const name = /^<([A-Z][\w.]*)/.exec(text.slice(start))![1];
  let quote = '';
  let depth = 0;
  let i = start + name.length + 1;

  for (; i < text.length; i++) {
    const char = text[i];
    if (quote) {
      if (char === quote) quote = '';
    } else if (char === '"' || char === "'" || char === '`') {
      quote = char;
    } else if (char === '{') {
      depth++;
    } else if (char === '}') {
      depth--;
    } else if (char === '>' && depth === 0) {
      break;
    }
  }
  if (i >= text.length) throw new Error(`<${name}> has no end: ${text.slice(start, start + 80)}`);

  const inner = text.slice(start + name.length + 1, i);
  const attributes = new Map<string, string>();
  for (const [, key, quoted, braced] of inner.matchAll(
    /([\w-]+)(?:=(?:"([^"]*)"|\{((?:[^{}]|\{[^{}]*\})*)\}))?/g,
  )) {
    attributes.set(key!, quoted ?? braced ?? 'true');
  }

  return { name, attributes, selfClosing: inner.trimEnd().endsWith('/'), end: i + 1 };
}

const asides: Record<string, string> = {
  note: 'Note',
  tip: 'Tip',
  caution: 'Caution',
  danger: 'Danger',
};

/** Components whose children are page content: their tags go, what they hold stays. */
const containers = new Set([
  'Tabs',
  'TabItem',
  'Steps',
  'FileTree',
  'Aside',
  'Rules',
  'Highlights',
]);

/** Components that show every module, standing for the index beside the pages. */
const indexes = new Set(['ModuleMap', 'ModuleTable']);

const FENCE = /^(\s*)(`{3,}|~{3,})(.*)$/;

/**
 * A code block's opening line and lines as Markdown shows them: twoslash's hidden lines and
 * directives gone, and of expressive-code's meta only the title. A block that marks deleted lines
 * becomes a diff.
 */
function codeBlock(open: string, lines: readonly string[]) {
  const [, indent = '', fence = '', info = ''] = FENCE.exec(open)!;
  const [lang = '', ...rest] = info.trim().split(/\s+/);
  const meta = rest.join(' ');
  const title = /\btitle="([^"]*)"/.exec(meta)?.[1];
  const marks = (key: string) => {
    const ranges = new RegExp(`\\b${key}=\\{([^}]*)\\}`).exec(meta)?.[1];
    const marked = new Set<number>();
    for (const range of ranges?.split(',') ?? []) {
      const [from, to = from] = range.split('-').map(Number);
      for (let line = from!; line <= to!; line++) marked.add(line);
    }
    return marked;
  };

  let body = [...lines];
  if (/\btwoslash\b/.test(meta)) {
    const cut = body.findLastIndex((line) => /^\s*\/\/ ---cut(-before)?---\s*$/.test(line));
    if (cut >= 0) body = body.slice(cut + 1);
    const after = body.findIndex((line) => /^\s*\/\/ ---cut-after---\s*$/.test(line));
    if (after >= 0) body = body.slice(0, after);

    const kept: string[] = [];
    let hidden = false;
    for (const line of body) {
      if (/^\s*\/\/ ---cut-start---\s*$/.test(line)) hidden = true;
      else if (/^\s*\/\/ ---cut-end---\s*$/.test(line)) hidden = false;
      else if (!hidden && !/^\s*\/\/\s*(@\w+|\^[?|!^]*\s*$)/.test(line)) kept.push(line);
    }
    body = kept;
  }

  const deleted = marks('del');
  let language = lang;
  if (deleted.size > 0) {
    const inserted = marks('ins');
    language = 'diff';
    body = body.map((line, i) => {
      const sign = deleted.has(i + 1) ? '-' : inserted.has(i + 1) ? '+' : ' ';
      return `${indent}${sign}${line.slice(indent.length)}`;
    });
  }

  const head = `${indent}${fence}${language}${title ? ` title="${title}"` : ''}`;
  return [head, ...body];
}

/** Convert MDX to Markdown, without figures, and reject malformed components. */
export function convertPage(id: string, source: string, options: PageOptions): ConvertedPage {
  const path = markdownPath(id);
  const { fields, body } = frontmatterOf(source);
  const listed = modulesOf(source);
  const lines = body.split('\n');
  const out: string[] = [];
  const modules = new Map<string, string>();
  const slug = slugger();

  const title = fields.get('title') ?? id;
  slug(title);

  let anchor = '';
  let indexed = false;
  /** Open blockquotes, `:::` asides and `<Aside>`s, each with the indent of its content. */
  const quotes: { indent: string; strip: number | undefined; directive: boolean }[] = [];

  const emit = (line: string) => {
    const quote = quotes.at(-1);
    if (!quote) {
      out.push(line);
      return;
    }
    quote.strip ??= line.trim() ? line.length - line.trimStart().length : undefined;
    const rest = line.slice(Math.min(quote.strip ?? 0, line.length - line.trimStart().length));
    out.push(rest.trim() ? `${quote.indent}> ${rest}` : `${quote.indent}>`);
  };
  const paragraph = (indent: string, text: string) => {
    emit('');
    emit(`${indent}${text}`);
    emit('');
  };
  /** A module's card, quoted, its links relinked. */
  const card = (indent: string, module: string) =>
    options
      .module(module)
      .split('\n')
      .map((line) => `${indent}>${line ? ` ${relinkLine(line, path, options)}` : ''}`);
  const openQuote = (indent: string, label: string, directive: boolean) => {
    emit('');
    emit(`${indent}> **${label}**`);
    emit(`${indent}>`);
    quotes.push({ indent, strip: undefined, directive });
  };

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const indent = /^\s*/.exec(line)![0];
    const trimmed = line.trim();

    const fence = FENCE.exec(line);
    if (fence) {
      const close = new RegExp(`^\\s*${fence[2][0]}{${fence[2].length},}\\s*$`);
      const end = lines.findIndex((each, j) => j > i && close.test(each));
      if (end < 0) throw new Error(`${id}: a code block at line ${i + 1} has no end.`);
      for (const each of codeBlock(line, lines.slice(i + 1, end))) emit(each);
      emit(lines[end]);
      i = end;
      continue;
    }

    if (/^import\s/.test(line)) {
      while (!/(^import\s+['"]|\bfrom\s+['"][^'"]+['"]);?\s*$/.test(lines[i])) i++;
      continue;
    }

    if (trimmed.startsWith('{/*')) {
      while (!lines[i].includes('*/}')) i++;
      continue;
    }

    const directive = /^:::(\w+)(?:\[(.*)\])?\s*$/.exec(trimmed);
    if (directive) {
      const kind = asides[directive[1]] ?? directive[1];
      openQuote(indent, directive[2] ? `${kind}: ${directive[2]}` : kind, true);
      continue;
    }
    if (trimmed === ':::') {
      if (!quotes.at(-1)?.directive)
        throw new Error(`${id}: a ::: at line ${i + 1} closes nothing.`);
      quotes.pop();
      emit('');
      continue;
    }

    const closing = /^<\/([A-Z][\w.]*)>$/.exec(trimmed);
    if (closing) {
      if (closing[1] === 'Aside') quotes.pop();
      emit('');
      continue;
    }

    if (/^<[A-Z]/.test(trimmed)) {
      const rest = lines.slice(i).join('\n');
      const start = line.indexOf('<');
      const tag = readTag(rest, start);
      const used = rest.slice(0, tag.end).split('\n').length - 1;
      const after = rest.slice(tag.end).split('\n')[0].trim();
      if (after) throw new Error(`${id}: text after <${tag.name}> on its line: ${after}`);
      i += used;

      const attribute = (key: string) => tag.attributes.get(key);

      if (containers.has(tag.name)) {
        // A tab's label is no heading on the site, so it takes no anchor here either
        if (tag.name === 'TabItem') {
          paragraph(indent, `**${attribute('label') ?? ''}**`);
        } else if (tag.name === 'Aside') {
          const kind = asides[attribute('type') ?? 'note'] ?? 'Note';
          const named = attribute('title');
          openQuote(indent, named ? `${kind}: ${named}` : kind, false);
        }
        continue;
      }

      if (!tag.selfClosing) {
        const close = lines.findIndex((each, j) => j > i && each.trim() === `</${tag.name}>`);
        if (close < 0) throw new Error(`${id}: <${tag.name}> at line ${i + 1} has no end.`);
        i = close;
      }

      if (tag.name === 'Install') {
        const commands = Object.fromEntries(tag.attributes);
        emit('');
        for (const each of ['```sh', ...linesOf(MANAGERS[0], commands), '```'])
          emit(`${indent}${each}`);
        emit('');
      } else if (tag.name === 'Module') {
        const module = attribute('id')!;
        modules.set(module, anchor);
        emit('');
        for (const each of card(indent, module)) emit(each);
        emit('');
      } else if (tag.name === 'LinkCard') {
        const href = relink(attribute('href')!, path, options.index, options.example);
        const label = attribute('title')!;
        const link = `[${label}](${href})`;
        paragraph(indent, `${link}: ${attribute('description') ?? ''}`.replace(/: $/, '.'));
      } else if (tag.name === 'Words') {
        emit('');
        for (const { term, means } of terms) emit(`${indent}- **${term}** ${means}`);
        emit('');
      } else if (indexes.has(tag.name)) {
        if (!indexed) {
          const href = posix.relative(dirname(path), options.index);
          paragraph(
            indent,
            `Every module, with its import and what it is for: [${href}](${href}).`,
          );
        }
        indexed = true;
      }
      // Figures and other components go with their captions, which address someone watching
      continue;
    }

    const section = /^#{1,6}\s+(.*)$/.exec(line);
    if (section) anchor = slug(section[1]);

    emit(relinkLine(line, path, options));
  }

  if (quotes.length > 0) throw new Error(`${id}: an aside has no end.`);

  const head = [`# ${title}`];
  const description = fields.get('description');
  if (description) head.push('', description);
  if (listed.length === 1) {
    head.push('', ...card('', listed[0]));
    modules.set(listed[0], '');
  }

  // A quote's blank lines go at its end and where they repeat
  const kept = [...head, '', ...out].filter(
    (line, i, all) => !(/^\s*>$/.test(line) && (!all[i + 1]?.trim() || /^\s*>$/.test(all[i + 1]))),
  );
  const text = `${kept
    .join('\n')
    .replace(/[ \t]+$/gm, '')
    .replace(/\n{3,}/g, '\n\n')
    .trim()}\n`;

  const empty = emptyHeading(text);
  if (empty) throw new Error(`${id}: "${empty}" has nothing under it in Markdown.`);
  return { path, text, modules };
}

/** The first heading in `markdown` with nothing before the next heading at its level or above. */
function emptyHeading(markdown: string) {
  // Each line with text, and its level if it is a heading outside a code block
  const lines: { readonly text: string; readonly level?: number }[] = [];
  let fence: RegExp | undefined;
  for (const text of markdown.split('\n')) {
    if (!text.trim()) continue;
    if (fence) {
      if (fence.test(text)) fence = undefined;
      lines.push({ text });
      continue;
    }
    const open = FENCE.exec(text);
    if (open) fence = new RegExp(`^\\s*${open[2][0]}{${open[2].length},}\\s*$`);
    const level = open ? undefined : /^(#{1,6})\s/.exec(text)?.[1].length;
    lines.push(level ? { text, level } : { text });
  }
  return lines.find(({ level }, i) => {
    if (level === undefined) return false;
    const next = lines[i + 1];
    return !next || (next.level !== undefined && next.level <= level);
  })?.text;
}

/** The lines of `markdown` outside its code blocks, with inline code blanked. */
export function proseOf(markdown: string) {
  const prose: string[] = [];
  let fence: RegExp | undefined;

  for (const line of markdown.split('\n')) {
    const quoted = line.replace(/^(\s*>)+/, '');
    if (fence) {
      if (fence.test(quoted)) fence = undefined;
      continue;
    }
    const open = FENCE.exec(quoted);
    if (open) {
      fence = new RegExp(`^\\s*${open[2][0]}{${open[2].length},}\\s*$`);
      continue;
    }
    prose.push(line.replace(/(`+)[\s\S]*?\1/g, (code) => ' '.repeat(code.length)));
  }
  return prose;
}

/** The headings in `markdown` by their anchors, as GitHub makes them, each with its text. */
export function headingsOf(markdown: string) {
  const slug = slugger();
  const headings = new Map<string, string>();
  let fence: RegExp | undefined;

  for (const line of markdown.split('\n')) {
    if (fence) {
      if (fence.test(line)) fence = undefined;
      continue;
    }
    const open = FENCE.exec(line);
    if (open) fence = new RegExp(`^\\s*${open[2][0]}{${open[2].length},}\\s*$`);
    const heading = /^#{1,6}\s+(.*)$/.exec(line);
    if (!open && heading) headings.set(slug(heading[1]), plainText(heading[1]));
  }
  return headings;
}

/** The anchors of the headings in `markdown`, as GitHub makes them. */
export const anchorsOf = (markdown: string) => new Set(headingsOf(markdown).keys());

/** The targets of the links in `markdown`'s prose. */
export const linksOf = (markdown: string) =>
  proseOf(markdown).flatMap((line) =>
    [...line.matchAll(/\]\(([^)\s]+)\)/g)].map(([, href]) => href!),
  );
