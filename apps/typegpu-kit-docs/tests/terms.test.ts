import { createSatteriMarkdownProcessor } from '@astrojs/markdown-satteri';
import { describe, expect, it } from 'vitest';
import { homeOf, termMarks } from '../src/build/terms.ts';
import { plain, terms, type Term } from '../src/components/site/terms.ts';
import { pageFiles } from '../src/contracts/agents.ts';

describe('the terms', () => {
  it('are told apart by every form', () => {
    const forms = [...terms.flatMap((term) => [term.term, ...(term.also ?? [])]), ...plain];
    const lower = forms.map((form) => form.toLowerCase());
    expect(forms.filter((form, at) => lower.indexOf(form.toLowerCase()) !== at)).toEqual([]);
  });

  it('are in alphabetical order', () => {
    const words = terms.map(({ term }) => term);
    const sorted = words.toSorted((a, b) => a.localeCompare(b, 'en', { sensitivity: 'base' }));
    expect(words).toEqual(sorted);
  });

  it('each mean one sentence', () => {
    const odd = terms.filter(
      ({ means }) => !/^[A-Za-z].*\.$/.test(means) || /[.!?] /.test(means) || /[`—–]/.test(means),
    );
    expect(odd.map(({ term }) => term)).toEqual([]);
  });

  it('each have a home on a page: a section, its opening or their row among the words', () => {
    const pages = new Set(pageFiles().map(({ id }) => (id === 'index' ? '/' : `/${id}/`)));
    expect(terms.filter(({ home }) => !pages.has(home.replace(/#.*$/, '')))).toEqual([]);
    for (const term of terms) expect(homeOf(term).label, term.term).not.toBe('');
  });
});

/** The terms a page marks, each with its text as written. */
const marks = (html: string) =>
  [
    ...html.matchAll(
      /<button type="button" class="term" popovertarget="term-([\w-]+)"[^>]*>([^<]*)</g,
    ),
  ].map(([, id, written]) => `${id}: ${written}`);

describe('marking a page', () => {
  const sample: readonly Term[] = [
    {
      term: 'kernel',
      also: ['kernels'],
      means: 'A kernel runs.',
      home: '/how-to-read/#word-kernel',
    },
    { term: 'pass', means: 'A pass holds work.', home: '/how-to-read/#word-pass' },
    {
      term: 'pass description',
      means: 'It describes.',
      home: '/how-to-read/#word-pass-description',
    },
    { term: 'table', means: 'A table holds rows.', home: '/kit/table/#row-ids' },
    {
      term: 'part',
      also: ['shared part'],
      alone: false,
      means: 'A part is shared.',
      home: '/how-to-read/#word-part',
    },
  ];
  const render = async (markdown: string, file?: string) => {
    const processor = await createSatteriMarkdownProcessor({
      hastPlugins: [termMarks(sample, ['lookup table'])],
    });
    const fileURL = file && new URL(`../src/content/docs/${file}`, import.meta.url);
    return (await processor.render(markdown, fileURL ? { fileURL } : undefined)).code;
  };

  it('marks the first use of each term, as written, and ends the page with its popover', async () => {
    const html = await render('Kernels move rows. A kernel reads a table.\n\n- Another kernel.');

    expect(marks(html)).toEqual(['kernel: Kernels', 'table: table']);
    expect(html).toContain(
      '<div id="term-kernel" popover="" class="term-pop not-content" style="position-anchor: --term-kernel" data-pagefind-ignore="">' +
        '<b>kernel</b><span>A kernel runs.</span><a href="/how-to-read/#word-kernel">Words →</a></div>',
    );
    expect(html).toContain('<a href="/kit/table/#row-ids">Row ids →</a>');
  });

  it('takes the longest term and whole words only', async () => {
    const html = await render('A pass description names each pass, bypass and sub-pass.');

    expect(marks(html)).toEqual(['pass-description: pass description', 'pass: pass']);
  });

  it('reads a word in its everyday sense where its entry or a longer phrase says so', async () => {
    const html = await render('Part of a lookup table goes in a shared part of a table.');

    expect(marks(html)).toEqual(['part: shared part', 'table: table']);
  });

  it('leaves headings, bold names, code, links and unmarked content alone', async () => {
    const html = await render(`
# The kernel

See [a kernel](/x/), a **kernel** and \`kernel\`.

| kernel | |
| --- | --- |
| a table | |

<div class="not-content"><p>A pass.</p></div>
`);

    expect(marks(html)).toEqual(['table: table']);
  });

  it('leaves a term alone on the page that teaches it', async () => {
    const html = await render('A table of kernels.', 'kit/table.mdx');

    expect(marks(html)).toEqual(['kernel: kernels']);
    expect(marks(await render('A kernel.', 'how-to-read.mdx'))).toEqual([]);
  });
});
