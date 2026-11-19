import {
  injections,
  kinds,
  modes,
  platforms,
  submissions,
  type Injection,
  type Kind,
  type Mode,
  type Platform,
  type Submission,
  type TagFacts,
} from './contract.ts';

/**
 * Contract tags belong in a module's `@module` block or on a factory. The gate rejects them
 * elsewhere. Quote identifiers in backticks and explain only facts the types and values omit.
 *
 * - `@module` marks the block whose description supplies the summary.
 * - `@kind <platform|tool|module|cpu|shader>` sets a kind the code cannot infer.
 * - `@platform <web|native>` limits a module to one platform. Omit it for both.
 * - `@flat` marks named imports such as `import { useScene } from ...`.
 * - `@injects <hook|definition|member> [note]` names an accessor kind and its fill.
 *   Use one tag per kind.
 * - `@binds <n> storage[, <n> uniform]: <what>` counts buffers used by each set of accessors.
 *   A module with injection needs this tag. WebGPU guarantees 8 storage bindings per stage.
 * - `@requires <feature> <required|optional|option> [note]` names a checked device feature.
 *   `option` means an option requires it.
 * - `@submits <setup|always> [note]` declares independent submissions.
 * - `@allocates <when>[: what]` declares allocations during recording.
 * - `@owns <item>, <item>...[: note]` lists created GPU objects and modules, such as
 *   `5 buffers, 3 pipelines, live lists`. The gate checks each against the code.
 * - `@records <note>` gives recording order or other per-frame instructions.
 * - `@unstable` marks a module that may change in any way in a minor release. Its card says so.
 *
 * On factories, `@param <name> <words>` and `@returns <words>` describe arguments and results.
 * On any export, `@hidden` omits it from cards and docs/modules.md. On a field of an options type
 * that takes a function, `@gpu` says it takes a `'use gpu'` one, which runs in a shader, and `@cpu`
 * that it runs on the CPU. The gate requires one of them.
 */
export const tags = [
  'module',
  'kind',
  'platform',
  'flat',
  'injects',
  'binds',
  'requires',
  'submits',
  'allocates',
  'owns',
  'records',
  'unstable',
] as const;

/** Tags any declaration may carry, which say nothing about the module. */
export const declarationTags = ['param', 'returns', 'hidden'] as const;

export interface Doc {
  readonly description: string;
  readonly tags: readonly { readonly name: string; readonly text: string }[];
}

/** Splits a `/** … *\/` comment into its description and its block tags. */
export function parseDoc(comment: string): Doc {
  const lines = comment
    .replace(/^\/\*\*/, '')
    .replace(/\*\/$/, '')
    .split('\n')
    .map((line) => line.replace(/^\s*\* ?/, '').trimEnd());

  const description: string[] = [];
  const found: { name: string; text: string }[] = [];

  for (const line of lines) {
    const tag = /^@(\w[\w-]*)\s*(.*)$/.exec(line.trim());
    if (tag) found.push({ name: tag[1], text: tag[2] });
    else if (found.length > 0) found[found.length - 1].text += ` ${line.trim()}`;
    else description.push(line);
  }

  return {
    description: description.join('\n').trim(),
    tags: found.map(({ name, text }) => ({ name, text: text.trim() })),
  };
}

/** The first sentence of a description, on one line. */
export function firstSentence(text: string): string {
  const flat = text.replace(/\s+/g, ' ').trim();
  let quoted = false;

  for (let i = 0; i < flat.length; i++) {
    if (flat[i] === '`') quoted = !quoted;
    const ends = !quoted && flat[i] === '.' && (i + 1 === flat.length || flat[i + 1] === ' ');
    if (ends && !/\b(?:e\.g|i\.e)$/.test(flat.slice(0, i))) return flat.slice(0, i + 1);
  }

  return flat;
}

/** A description's sentences, on one line each: a sentence ends at a full stop outside code. */
export function sentencesOf(text: string): string[] {
  const sentences: string[] = [];
  let rest = text.replace(/\s+/g, ' ').trim();

  while (rest) {
    const sentence = firstSentence(rest);
    sentences.push(sentence);
    rest = rest.slice(sentence.length).trim();
  }

  return sentences;
}

/**
 * A field's TSDoc split into its default, from a sentence `Default <value>.` or `Defaults to
 * <value>.`, and its first other sentence.
 */
export function splitDefault(text: string): { doc?: string; default?: string } {
  const sentences = sentencesOf(text);
  const at = sentences.findIndex((sentence) => /^Defaults?\b/.test(sentence));
  const fallback =
    at < 0
      ? undefined
      : sentences[at].replace(/^Defaults?(?: to| is)?:?\s+/, '').replace(/\.$/, '');
  const doc = sentences.find((_, i) => i !== at);

  return { ...(doc ? { doc } : {}), ...(fallback ? { default: fallback } : {}) };
}

const known = (list: readonly string[], value: string) => list.includes(value);

/** A tag's text split after its first `count` words: the value, then the note. */
function words(text: string, count: number) {
  const parts = text.split(/\s+/).filter(Boolean);
  const note = parts.slice(count).join(' ');
  return { values: parts.slice(0, count), note: note || undefined };
}

/** A tag's text split at its first colon: `on resize: the levels` is `on resize`, then `the levels`. */
function colon(text: string) {
  const at = text.indexOf(':');
  if (at < 0) return { value: text.trim(), note: undefined };
  return { value: text.slice(0, at).trim(), note: text.slice(at + 1).trim() || undefined };
}

/** A `@param` or `@returns` text without TSDoc's optional hyphen. */
export const unhyphened = (text: string) => text.replace(/^-\s*/, '').trim();

/** `2 storage`, `1 uniform` or `1 storage, 1 uniform`: what a `@binds` costs a stage. */
function counts(text: string) {
  const found = /^(?:(\d+) storage)?(?:(?:,\s*|\s+and\s+)?(\d+) uniform)?$/.exec(text);
  if (!found || (found[1] === undefined && found[2] === undefined)) return undefined;
  return { storage: Number(found[1] ?? 0), uniform: Number(found[2] ?? 0) };
}

const withNote = <T extends object>(value: T, note: string | undefined) =>
  note ? { ...value, note } : value;

/**
 * What a module's tags say, and what is wrong with them. `where` names the module in issues.
 * `@param`, `@returns` and `@hidden` belong to a declaration; the caller reads those.
 */
export function readTags(found: Doc['tags'], where: string, issues: string[]): TagFacts {
  const facts: TagFacts = {
    injects: [],
    binds: [],
    requires: [],
    flat: false,
    unstable: false,
  };
  const once = new Set<string>();

  for (const { name, text } of found) {
    if (known(declarationTags, name)) continue;
    if (!known(tags, name)) {
      issues.push(`${where}: unknown tag @${name}.`);
      continue;
    }

    if (['kind', 'platform', 'submits', 'allocates', 'owns', 'records'].includes(name)) {
      if (once.has(name)) issues.push(`${where}: @${name} twice; give it one tag.`);
      once.add(name);
    }

    if (name === 'flat') facts.flat = true;
    if (name === 'unstable') facts.unstable = true;

    if (name === 'kind') {
      if (known(kinds, text)) facts.kind = text as Kind;
      else issues.push(`${where}: @kind takes ${kinds.join(', ')}, got '${text}'.`);
    }

    if (name === 'platform') {
      if (known(platforms, text)) facts.platform = text as Platform;
      else
        issues.push(
          `${where}: @platform takes ${platforms.join(' or ')}, got '${text}'; leave it out for both.`,
        );
    }

    if (name === 'injects') {
      const { values, note } = words(text, 1);
      const kind = values[0] ?? '';
      if (known(injections, kind)) facts.injects.push(withNote({ kind: kind as Injection }, note));
      else issues.push(`${where}: @injects takes ${injections.join(', ')} first, got '${text}'.`);
    }

    if (name === 'binds') {
      const { value, note } = colon(text);
      const cost = counts(value);
      if (cost && note) facts.binds.push({ ...cost, note });
      else issues.push(`${where}: @binds takes '<n> storage, <n> uniform: what', got '${text}'.`);
    }

    if (name === 'requires') {
      const { values, note } = words(text, 2);
      const [feature, mode] = values;
      if (!feature || !mode || !known(modes, mode)) {
        issues.push(`${where}: @requires takes a feature and ${modes.join(', ')}, got '${text}'.`);
      } else {
        facts.requires.push(withNote({ feature, mode: mode as Mode }, note));
      }
    }

    if (name === 'submits') {
      const { values, note } = words(text, 1);
      const when = values[0] ?? '';
      if (known(submissions, when)) facts.submits = withNote({ when: when as Submission }, note);
      else
        issues.push(`${where}: @submits takes ${submissions.join(' or ')} first, got '${text}'.`);
    }

    if (name === 'allocates') {
      const { value, note } = colon(text);
      if (value) facts.allocates = withNote({ when: value }, note);
      else issues.push(`${where}: @allocates needs to say when.`);
    }

    if (name === 'owns') {
      const { value, note } = colon(text);
      const items = value
        .split(',')
        .map((item) => item.trim())
        .filter(Boolean);
      if (items.length > 0) facts.owns = withNote({ items }, note);
      else issues.push(`${where}: @owns needs what it owns, such as '2 buffers, a pipeline'.`);
    }

    if (name === 'records') {
      if (text) facts.order = text;
      else issues.push(`${where}: @records needs a note on how to call them.`);
    }
  }

  return facts;
}
