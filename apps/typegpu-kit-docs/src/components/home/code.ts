// The frame's code as the page shows it, cut from frame.code.ts and rules.ts as they run.
// In the shown code, `{{name}}` marks a draggable number and `[module] ` a line that needs it.

import frameSource from './frame.code.ts?raw';
import { KNOBS, shown, type Knobs } from './knobs.ts';
import rulesSource from './rules.ts?raw';
import type { RuleName } from './rules.ts';
import type { Removable } from './subject.ts';

/** A plate of the stack, and the block of code that makes it. */
export type Part = 'final' | 'bloom' | 'normals' | 'lit' | 'shadow' | 'grid' | 'birds';

export interface Block {
  readonly part: Part;
  readonly module: string;
  readonly says: string;
  /** What the plate holds, in a sentence, for the block's step of the build. */
  readonly tells: string;
  /** The module its name takes out. */
  readonly removable?: Removable;
  readonly code: string;
}

/** A knob the code reads, which the page shows as its number to drag. */
const KNOB = /\b(?:knob|tuned\.\$)\.(\w+)\b/g;

/** A line's closing `// [module, …]`: the modules it needs. */
const NEEDS = / \/\/ \[([\w,]+)\]$/;

/** The lines of frame.code.ts from `// #region <part>` to its end, as the page shows them. */
export function regionOf(part: Part) {
  const lines = frameSource.split('\n');
  const from = lines.findIndex((line) => line.trim() === `// #region ${part}`);
  const to = lines.findIndex((line, i) => i > from && line.trim() === '// #endregion');
  if (from < 0 || to < 0) throw new Error(`frame.code.ts has no region ${part}.`);
  const shownLines = lines.slice(from + 1, to).filter((line) => !line.endsWith('// hide'));
  const indent = Math.min(...shownLines.map((line) => /^ */.exec(line)![0].length));
  return shownLines
    .map((line) => {
      const needs = NEEDS.exec(line);
      const code = (needs ? line.slice(0, needs.index) : line)
        .slice(indent)
        .replaceAll(KNOB, '{{$1}}');
      return needs ? `[${needs[1]}] ${code}` : code;
    })
    .join('\n');
}

const SAYS: readonly Omit<Block, 'code'>[] = [
  {
    part: 'birds',
    module: 'kit.table',
    says: 'the birds, a row each',
    tells: "Each block holds one bird's nine words for position, heading and force.",
  },
  {
    part: 'grid',
    module: 'kit.grid',
    says: 'who is near whom',
    tells:
      'Each cell shows its bird count. Birds look for neighbours in their own cell and nearby cells.',
    removable: 'grid',
  },
  {
    part: 'shadow',
    module: 'kit.lighting',
    says: "the sun's shadow",
    tells: 'The shadow map stores depth viewed from the sun.',
    removable: 'shadow',
  },
  {
    part: 'lit',
    module: 'kit.lighting',
    says: 'a sun, and a lamp in one bird of eight',
    tells:
      'Sunlight, sky light and lamps light the scene. One bird in eight is a lamp, and the lighting finds the lamps near each point through the grid.',
    removable: 'lit',
  },
  {
    part: 'normals',
    module: 'kit.passes',
    says: 'lit colour, normals and depth',
    tells:
      'The scene pass draws the ground and chevron birds into colour, normals and depth. This plate shows normals.',
  },
  {
    part: 'bloom',
    module: 'kit.bloom',
    says: "the lamps' glow",
    tells: "Bloom blurs the lamps' light through a chain of smaller textures.",
    removable: 'bloom',
  },
  {
    part: 'final',
    module: 'kit.finish',
    says: 'the graded frame',
    tells: 'The final pass adds glow, tone mapping, a vignette and dithered colour levels.',
  },
];

/** In the order the frame builds them: the stack's plates from the bottom up. */
export const BLOCKS: readonly Block[] = SAYS.map((block) => ({
  ...block,
  code: regionOf(block.part),
}));

/** The birds' rules as rules.ts has them, each from its comment, with its number to drag. */
export const RULE_CODE = Object.fromEntries(
  [...rulesSource.matchAll(/^(\/\/ [^\n]*)\nexport const (\w+) = ([\s\S]*?\n\};)$/gm)].map(
    ([, comment, name, body]) => [
      name,
      `${comment}\nconst ${name} = ${body}`.replaceAll(KNOB, '{{$1}}'),
    ],
  ),
) as Record<RuleName, string>;

/** What each rule is for, under its name. */
export const RULE_SAYS: Record<RuleName, string> = {
  flock: 'the frame above',
  orbit: 'a ring',
  huddle: 'counts its neighbours',
};

/** How the steering takes `name`. */
export const handIn = (name: RuleName) =>
  `const steering = createSteering(root, birds, cells, { rules: { ${name} } });`;

const KEYWORDS = new Set([
  'const',
  'let',
  'return',
  'import',
  'from',
  'export',
  'if',
  'for',
  'while',
  'function',
  'new',
  'await',
  'async',
  'of',
  'as',
]);
const NAMES = new Set(['flock', 'orbit', 'huddle']);

const escape = (text: string) =>
  text.replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');

const TOKEN =
  /(\/\*[\s\S]*?\*\/|\/\/[^\n]*)|('[^'\n]*')|(\b\d[\d_]*(?:\.\d+)?(?:e-?\d+)?\b)|([A-Za-z_$][\w$]*)(?=\s*\()|([A-Za-z_$][\w$]*)|([^\w\s'])/g;

/** TypeScript as HTML, with spans for keywords, strings, numbers, calls, punctuation and comments. */
export function highlight(code: string) {
  let html = '';
  let last = 0;
  for (const match of code.matchAll(TOKEN)) {
    html += escape(code.slice(last, match.index));
    last = match.index + match[0].length;
    const [text, comment, string, number, call, word, punctuation] = match;
    const at = escape(text);
    if (comment) html += `<span class="c">${at}</span>`;
    else if (string) html += `<span class="s">${at}</span>`;
    else if (number) html += `<span class="n">${at}</span>`;
    else if (call)
      html += KEYWORDS.has(call)
        ? `<span class="k">${at}</span>`
        : NAMES.has(call)
          ? `<span class="y">${at}</span>`
          : `<span class="f">${at}</span>`;
    else if (word)
      html += KEYWORDS.has(word)
        ? `<span class="k">${at}</span>`
        : NAMES.has(word)
          ? `<span class="y">${at}</span>`
          : at;
    else if (punctuation) html += `<span class="p">${at}</span>`;
  }
  return html + escape(code.slice(last));
}

/**
 * A knob as HTML: a number to drag, or step with the arrow keys, once the runtime enables its range
 * input (input.ts).
 */
const knobHtml = (name: string) => {
  const knob = KNOBS[name as keyof Knobs];
  if (!knob) throw new Error(`No knob ${name}.`);
  const value = shown(name as keyof Knobs, knob.value);
  const range = `min="${knob.min}" max="${knob.max}" step="${knob.step}" value="${knob.value}"`;
  return (
    `<label class="knob viz-number" data-knob="${name}"><span class="viz-range">` +
    `<input type="range" ${range} aria-label="${name}" data-control="${name}" autocomplete="off" disabled>` +
    `<output aria-hidden="true">${value}</output></span></label>`
  );
};

/** A line of code as HTML, highlighted, with each `{{knob}}` a number to drag. */
const lineHtml = (text: string) =>
  text
    .split(/(\{\{\w+\}\})/)
    .map((part) => {
      const name = /^\{\{(\w+)\}\}$/.exec(part)?.[1];
      return name ? knobHtml(name) : highlight(part);
    })
    .join('');

/**
 * Code as HTML, a span a line: highlighted, its knobs to drag, and each line that needs a module
 * marked with it, so it strikes through while the module is out.
 */
export function render(code: string) {
  return code
    .split('\n')
    .map((line) => {
      const needs = /^\[([\w,]+)\] /.exec(line);
      const text = needs ? line.slice(needs[0].length) : line;
      const data = needs ? ` data-needs="${needs[1].replaceAll(',', ' ')}"` : '';
      return `<span class="line"${data}>${lineHtml(text) || ' '}</span>`;
    })
    .join('\n');
}
