// The source page: names marked everywhere they appear, and a card for the one clicked

import { reducedMotion } from '../components/viz/figure.ts';
import type { Captures, Local, Traced } from './read.ts';

/** A traced name with, for the kit, the module it belongs to. */
export type Shown = Traced & {
  readonly module?: { readonly name: string; readonly summary: string; readonly href?: string };
};

interface Data {
  /** The example page with its GPU debug panel running. */
  readonly profile: string;
  readonly traced: Readonly<Record<string, Shown>>;
}

/** Uses listed per file before the rest are counted. */
const PER_FILE = 8;

/** Lines of a declaration a card shows before the rest are counted, and a peek beside a name. */
const SHOWN = { card: 40, peek: 16 };

/** How long the pointer rests on a name before its declaration shows beside it. */
const PEEK_DELAY = 350;

const CAPTURES: Record<keyof Captures, string> = {
  bound: 'Bound to its pipeline, read from GPU memory as it runs',
  filled: 'Filled where its pipeline is created',
  baked: 'Written into the WGSL as constants',
};

const element = <K extends keyof HTMLElementTagNameMap>(
  tag: K,
  className = '',
  ...children: (Node | string)[]
) => {
  const made = document.createElement(tag);
  if (className) made.className = className;
  made.append(...children);
  return made;
};

const button = (className: string, text: string) => {
  const made = element('button', className, text);
  made.type = 'button';
  return made;
};

/** `1 draw`, `2 draws`, `1 pass`, `3 passes`, `1 dispatch`. */
const plural = (count: number, word: string) =>
  `${count} ${count === 1 ? word.replace(/(?<=ss|ch)es$|s$/, '') : word}`;

const middle = (span: HTMLElement) => {
  const rect = span.getBoundingClientRect();
  return rect.top + rect.height / 2;
};

const idAt = (target: EventTarget | null) =>
  (target as HTMLElement | null)?.closest<HTMLElement>('[data-n]')?.dataset.n;

/** Glides a jump, or makes it at once under reduced motion. */
const glide = (): ScrollBehavior => (reducedMotion() ? 'auto' : 'smooth');

/** Scrolls to a line and flashes it. */
const go = (id: string) => {
  const line = document.getElementById(id);
  if (!line) return;
  line.scrollIntoView({ block: 'center', behavior: glide() });
  line.classList.remove('flash');
  void line.offsetWidth;
  line.classList.add('flash');
};

const lineText = (id: string) => document.getElementById(id)?.textContent?.trim() ?? '';

/** Removes up to `count` leading spaces from a line's highlighted tokens. */
const dedent = (line: Node, count: number) => {
  const walker = document.createTreeWalker(line, NodeFilter.SHOW_TEXT);
  let left = count;
  for (let text = walker.nextNode() as Text | null; text && left > 0;) {
    const spaces = /^ */.exec(text.data)![0].length;
    const cut = Math.min(spaces, left);
    text.data = text.data.slice(cut);
    left -= cut;
    if (text.data.length > 0) break;
    text = walker.nextNode() as Text | null;
  }
};

/**
 * A local's declaration as the page highlights it, from its name's line to its end, its
 * indentation removed and its marks left behind.
 */
const declarationOf = (entry: Local, shown: number) => {
  const last = Math.min(entry.to ?? entry.line, entry.line + shown - 1);
  const lines: HTMLElement[] = [];
  for (let n = entry.line; n <= last; n++) {
    const line = document.getElementById(`${entry.file}:${n}`)?.cloneNode(true) as
      | HTMLElement
      | undefined;
    if (line) lines.push(line);
  }
  const indent = Math.min(
    ...lines.flatMap((line) => {
      const text = line.textContent ?? '';
      return text.trim() ? [/^ */.exec(text)![0].length] : [];
    }),
  );
  const pre = element('pre', 'declaration');
  for (const line of lines) {
    for (const node of line.querySelectorAll('[data-n]')) {
      node.removeAttribute('data-n');
      node.removeAttribute('class');
    }
    dedent(line, indent);
    pre.append(...(pre.firstChild ? ['\n'] : []), ...line.childNodes);
  }
  const rest = (entry.to ?? entry.line) - last;
  if (rest > 0) pre.append('\n', element('span', 'rest', `… ${plural(rest, 'lines')} more`));
  return pre;
};

/** A line of code with every whole word in `names` marked. */
const withNames = (text: string, names: readonly string[]) =>
  text
    .split(new RegExp(`\\b(${names.map((name) => name.replace(/\$/g, '\\$')).join('|')})\\b`))
    .map((part, i) => (i % 2 === 1 ? element('mark', '', part) : part));

/**
 * A doc comment's paragraphs, with `code` set as code; source comments wrap at the line limit, so
 * single breaks join.
 */
const paragraphs = (doc: string) =>
  doc.split(/\n\s*\n/).map((paragraph) =>
    element(
      'p',
      '',
      ...paragraph
        .replace(/\s*\n\s*/g, ' ')
        .split('`')
        .map((part, i) => (i % 2 === 1 ? element('code', '', part) : part)),
    ),
  );

/**
 * The part of the viewport the code shows in: below the bar and a file's sticky heading, and above
 * `sheet`, the card where it covers the foot of the screen.
 */
const codeViewport = (code: HTMLElement, sheet: HTMLElement | undefined) => {
  const bar = document.querySelector('.bar')?.getBoundingClientRect().bottom ?? 0;
  const heading = code.querySelector<HTMLElement>('.file h2')?.offsetHeight ?? 0;
  return { top: bar + heading, bottom: sheet?.getBoundingClientRect().top ?? innerHeight };
};

/** Hover marks names; touch screens mark them on a tap instead. */
const HOVER = '(hover: hover)';

/** The layout that stacks the preview above the code and opens cards as a sheet. */
const NARROW = '(max-width: 900px)';

/** The line a local name is declared on, as `file:line`, the id of that line. */
const definitionOf = (entry: Shown | undefined) =>
  entry?.kind === 'local' ? `${entry.file}:${entry.line}` : undefined;

/** What a peek beside a name shows: its declaration or parameters, and its doc's first paragraph. */
const peekOf = (entry: Shown) => {
  const lead = entry.kind === 'module' ? entry.module?.summary : entry.doc;
  return [
    ...(entry.kind === 'local' ? [declarationOf(entry, SHOWN.peek)] : []),
    ...(entry.kind === 'api' ? [element('pre', '', entry.call)] : []),
    ...(lead ? paragraphs(lead).slice(0, 1) : []),
  ];
};

/** A paragraph ending with a link to a kit module's docs. */
const moduleLine = (module: NonNullable<Shown['module']>, lead: string) => {
  const docs = element('a', '', `${module.name} docs`);
  if (module.href) docs.href = module.href;
  return element('p', '', `${lead} `, ...(module.href ? [docs] : []));
};

export function start() {
  const data = JSON.parse(document.getElementById('source')!.textContent!) as Data;
  const code = document.getElementById('code')!;
  const card = document.getElementById('card')!;
  if (!matchMedia(HOVER).matches) {
    document.getElementById('hint')!.textContent =
      'Tap a name to mark it everywhere it appears, read its docs and list its uses.';
  }

  for (const pre of code.querySelectorAll('pre')) pre.setAttribute('aria-describedby', 'hint');

  const spansOf = (id: string) => [
    ...code.querySelectorAll<HTMLElement>(`[data-n="${CSS.escape(id)}"]`),
  ];

  // Marking a name: every token of it, the line numbers of their lines, and a chip at each edge
  // counting those off screen, which scrolls to the nearest
  let marked: string | undefined;
  let pinned = false;
  const hovers = matchMedia(HOVER).matches;
  const narrow = matchMedia(NARROW);
  const sheet = () => (pinned && narrow.matches ? card : undefined);

  const chip = (edge: 'top' | 'bottom', at: number, spans: HTMLElement[]) => {
    const made = button('beyond', `${edge === 'top' ? '↑' : '↓'} ${spans.length} more`);
    made.style.left = `${code.getBoundingClientRect().left + 8}px`;
    made.style.top = `${edge === 'top' ? at + 6 : at - 28}px`;
    const nearest = edge === 'top' ? spans.at(-1)! : spans[0];
    made.addEventListener('click', () => go(nearest.closest('.line')!.id));
    // Inside the code, so moving onto it doesn't count as leaving
    code.append(made);
  };

  const placeChips = () => {
    for (const each of code.querySelectorAll('.beyond')) each.remove();
    if (!marked) return;
    const spans = spansOf(marked);
    const view = codeViewport(code, sheet());
    // A file off screen answers for all its spans, so files the browser skips stay unlaid
    const files = new Map<Element, DOMRect>();
    const sideOf = (span: HTMLElement) => {
      const file = span.closest('.file')!;
      const rect = files.get(file) ?? file.getBoundingClientRect();
      files.set(file, rect);
      if (rect.bottom < view.top) return 'above';
      if (rect.top > view.bottom) return 'below';
      const at = middle(span);
      return at < view.top ? 'above' : at > view.bottom ? 'below' : '';
    };
    const sides = spans.map(sideOf);
    const above = spans.filter((_, i) => sides[i] === 'above');
    const below = spans.filter((_, i) => sides[i] === 'below');
    if (above.length > 0) chip('top', view.top, above);
    if (below.length > 0) chip('bottom', view.bottom, below);
  };

  const unmark = () => {
    marked = undefined;
    for (const node of code.querySelectorAll('.def, .use, .used')) {
      node.classList.remove('def', 'use', 'used');
    }
    placeChips();
  };

  const mark = (id: string) => {
    unmark();
    marked = id;
    const definition = definitionOf(data.traced[id]);
    let defined = false;
    for (const span of spansOf(id)) {
      const line = span.closest('.line')!;
      const isDefinition: boolean = !defined && line.id === definition;
      defined ||= isDefinition;
      span.classList.add(isDefinition ? 'def' : 'use');
      line.classList.add('used');
    }
    placeChips();
  };

  /** What a GPU function's card says becomes of it, linking the panel that shows its WGSL. */
  const profiled = () => {
    const link = element('a', '', 'GPU debug panel');
    link.href = data.profile;
    return element(
      'p',
      'aside',
      'TypeGPU compiles it into the WGSL of each pipeline that calls it. Capture a frame in the ',
      link,
      ' to read the WGSL of each pass.',
    );
  };

  // A clicked name's card: what it is, its docs, and its uses
  const lineLink = (id: string, ...children: (Node | string)[]) => {
    const a = element('a', '', ...children);
    a.href = `#${id}`;
    a.addEventListener('click', (event) => {
      event.preventDefault();
      history.replaceState(null, '', a.href);
      go(id);
    });
    return a;
  };

  /** A link that marks `id` on hover and pins it on click, or its name as text if untraced. */
  const nameLink = (id: string) => {
    const entry = data.traced[id];
    if (!entry) return element('span', '', id.replace(/~\d+$/, ''));
    const a = element('a', 'name', entry.kind === 'api' ? entry.title : entry.name);
    const definition = definitionOf(entry);
    if (definition) a.href = `#${definition}`;
    a.addEventListener('click', (event) => {
      event.preventDefault();
      if (definition) go(definition);
      pin(id);
    });
    if (hovers) {
      a.addEventListener('mouseenter', () => {
        if (!pinned) mark(id);
      });
      a.addEventListener('mouseleave', () => {
        if (!pinned) unmark();
      });
    }
    return a;
  };

  const usesOf = (id: string, definition: string | undefined) => {
    const spans = spansOf(id);
    // As written, which renamed imports change
    const names = [...new Set(spans.map((span) => span.textContent!))];
    const lines = [...new Set(spans.map((span) => span.closest('.line')!.id))].filter(
      (line) => line !== definition,
    );
    if (lines.length === 0) return [element('h4', '', 'Not used elsewhere')];
    const files = [...new Set(lines.map((line) => line.split(':')[0]))];
    const nodes: HTMLElement[] = [element('h4', '', `Used on ${plural(lines.length, 'lines')}`)];
    for (const file of files) {
      const here = lines.filter((line) => line.startsWith(`${file}:`));
      const list = element('ul');
      for (const line of here.slice(0, PER_FILE)) {
        const where = element('span', 'where', `${files.length > 1 ? line : line.split(':')[1]}  `);
        list.append(element('li', '', lineLink(line, where, ...withNames(lineText(line), names))));
      }
      if (here.length > PER_FILE) {
        list.append(element('li', 'where', `and ${here.length - PER_FILE} more in ${file}`));
      }
      nodes.push(list);
    }
    return nodes;
  };

  const capturesOf = (captures: Captures) =>
    (Object.keys(CAPTURES) as (keyof Captures)[]).flatMap((group) => {
      const ids = captures[group];
      if (!ids) return [];
      return [
        element('h4', '', CAPTURES[group]),
        element(
          'p',
          'names',
          ...ids.flatMap((id, i) => (i === 0 ? [nameLink(id)] : [', ', nameLink(id)])),
        ),
      ];
    });

  const showCard = (id: string) => {
    const entry = data.traced[id];
    if (!entry) return;
    const definition = definitionOf(entry);
    const close = button('', hovers ? 'close · Esc' : 'close');
    close.addEventListener('click', unpin);
    const title = element('h3', '', entry.kind === 'api' ? entry.title : entry.name);
    const nodes: HTMLElement[] = [title];

    if (entry.kind === 'local') {
      title.append(lineLink(definition!, definition!));
      if (entry.gpu) title.append(element('span', 'tag gpu', 'runs on the GPU'));
      nodes.push(declarationOf(entry, SHOWN.card));
      if (entry.doc) nodes.push(...paragraphs(entry.doc));
      if (entry.gpu) {
        nodes.push(profiled(), ...(entry.captures ? capturesOf(entry.captures) : []));
      }
    } else if (entry.kind === 'module') {
      title.append(element('span', 'tag', 'kit module'));
      if (entry.module) nodes.push(moduleLine(entry.module, entry.module.summary));
    } else {
      title.append(element('span', 'tag', entry.from));
      nodes.push(element('pre', '', entry.call));
      if (entry.doc) nodes.push(...paragraphs(entry.doc));
      if (entry.module) {
        nodes.push(moduleLine(entry.module, `From the ${entry.module.name} module.`));
      }
    }
    title.append(close);
    card.replaceChildren(...nodes, ...usesOf(id, definition));
  };

  // On a narrow screen the card opens as a sheet over the foot of the screen
  function pin(id: string) {
    pinned = true;
    showCard(id);
    card.classList.add('open');
    card.scrollTop = 0;
    mark(id);
  }

  function unpin() {
    pinned = false;
    card.classList.remove('open');
    unmark();
    card.replaceChildren();
  }

  // A name the pointer rests on shows its declaration beside it, away from where it's declared
  const peek = element('div', 'peek');
  peek.hidden = true;
  document.body.append(peek);
  let peeking: ReturnType<typeof setTimeout> | undefined;
  let peeked: Element | undefined;
  const unpeek = () => {
    clearTimeout(peeking);
    peeked = undefined;
    peek.hidden = true;
  };
  const showPeek = (span: HTMLElement) => {
    const entry = data.traced[span.dataset.n!];
    if (!entry || span.closest('.line')!.id === definitionOf(entry)) return;
    const shown = peekOf(entry);
    if (shown.length === 0) return;
    peek.replaceChildren(...shown);
    peek.hidden = false;
    const at = span.getBoundingClientRect();
    const size = peek.getBoundingClientRect();
    const below = at.bottom + 6 + size.height <= innerHeight;
    peek.style.top = `${below ? at.bottom + 6 : Math.max(8, at.top - 6 - size.height)}px`;
    peek.style.left = `${Math.max(8, Math.min(at.left, innerWidth - size.width - 8))}px`;
  };

  if (hovers) {
    code.addEventListener('mouseover', (event) => {
      const id = idAt(event.target);
      if (id && !pinned && id !== marked) mark(id);
      const span = (event.target as HTMLElement).closest<HTMLElement>('[data-n]') ?? undefined;
      if (span === peeked) return;
      unpeek();
      if (!span) return;
      peeked = span;
      peeking = setTimeout(() => showPeek(span), PEEK_DELAY);
    });
    code.addEventListener('mouseleave', () => {
      unpeek();
      if (!pinned) unmark();
    });
    addEventListener('scroll', unpeek, { passive: true });
  }
  code.addEventListener('click', (event) => {
    unpeek();
    const copy = (event.target as HTMLElement).closest<HTMLElement>('.copy');
    if (copy) {
      const lines = document.getElementById(copy.dataset.file!)!.querySelectorAll('.line');
      const say = (text: string) => {
        copy.textContent = text;
        setTimeout(() => (copy.textContent = 'Copy'), 1200);
      };
      navigator.clipboard.writeText([...lines].map((line) => line.textContent).join('\n')).then(
        () => say('Copied'),
        () => say("Couldn't copy"),
      );
      return;
    }
    const id = idAt(event.target);
    if (!id) return;
    if (pinned && id === marked) {
      unpin();
      return;
    }
    pin(id);
    // Lines keep clear of the sheet as they scroll into view, so this lifts a tapped one above it
    if (sheet()) {
      (event.target as HTMLElement)
        .closest('.line')
        ?.scrollIntoView({ block: 'nearest', behavior: glide() });
    }
  });
  addEventListener('keydown', (event) => {
    // Stepping through names peeks at the one it reaches
    if (!event.defaultPrevented) unpeek();
    if (event.key === 'Escape' && pinned) unpin();
  });

  // From a focused file, arrow keys step through its names and Enter pins one, as a click does
  const STEPS: Record<string, number> = { ArrowRight: 1, ArrowLeft: -1, ArrowDown: 1, ArrowUp: -1 };
  code.addEventListener('keydown', (event) => {
    const at = event.target as HTMLElement;
    const pre = at.closest('pre');
    if (!pre) return;
    const names = [...pre.querySelectorAll<HTMLElement>('[data-n]')];
    const step = STEPS[event.key];
    let next: HTMLElement | undefined;
    if (at === pre) {
      if (step === undefined || names.length === 0) return;
      // Entering a file starts at its first name on screen
      const top = codeViewport(code, sheet()).top;
      next = names.find((span) => span.getBoundingClientRect().top >= top) ?? names[0];
    } else if (event.key === 'Enter') {
      at.click();
    } else if (event.key === 'Escape') {
      pre.focus();
      return;
    } else if (step !== undefined) {
      const index = names.indexOf(at);
      if (event.key === 'ArrowRight' || event.key === 'ArrowLeft') next = names[index + step];
      else {
        const line = at.closest('.line');
        const elsewhere = (span: HTMLElement) => span.closest('.line') !== line;
        next =
          step > 0
            ? names.slice(index + 1).find(elsewhere)
            : names.slice(0, index).findLast(elsewhere);
      }
    } else return;
    event.preventDefault();
    if (!next) return;
    next.tabIndex = -1;
    next.focus();
  });
  code.addEventListener('focusin', (event) => {
    const span = event.target as HTMLElement;
    if (!span.matches('[data-n]')) return;
    if (!pinned) mark(span.dataset.n!);
    unpeek();
    peeked = span;
    showPeek(span);
  });
  code.addEventListener('focusout', (event) => {
    const span = event.target as HTMLElement;
    if (!span.matches('[data-n]')) return;
    span.removeAttribute('tabindex');
    unpeek();
    if (!pinned) unmark();
  });
  // Scrolling moves only the chips counting what's off screen; the marks stay
  let placing = 0;
  const replaceChips = () => {
    if (!marked || placing) return;
    placing = requestAnimationFrame(() => {
      placing = 0;
      placeChips();
    });
  };
  addEventListener('scroll', replaceChips, { passive: true });
  addEventListener('resize', replaceChips);

  if (location.hash) go(decodeURIComponent(location.hash.slice(1)));
}
