import type { GpuObject } from './contract.ts';

/** The words `@owns` counts GPU objects with. */
const nouns: Record<string, GpuObject> = {
  buffer: 'buffer',
  uniform: 'buffer',
  texture: 'texture',
  target: 'texture',
  sampler: 'sampler',
  'query set': 'query set',
  pipeline: 'pipeline',
  'bind group': 'bind group',
};
const noun = Object.keys(nouns).join('|');

/**
 * `3 buffers`, `3 or 4 buffers`, `2 to 7 buffers`, `up to 16 query sets`, `a texture per target`,
 * `one per added function`.
 */
const counting = new RegExp(
  `^(up to )?(a|an|one|\\d+)(?: (?:or|to) (\\d+))?(?: (${noun})s?)?( per .+| a .+)?$`,
);
/** `targets`, `culling pipelines`: some, uncounted. */
const some = new RegExp(`^(?:\\w+ )?(${noun})s$`);
/** `a readback`, `a passes description`, `liveLists with \`lists\``. */
const naming = /^(?:an? )?([\w.]+)(?: description)?(?: ((?:when|unless|with) .+))?$/;

/** One part of an `@owns` item, read. */
export interface Owned {
  readonly text: string;
  /** The GPU object it counts. */
  readonly object?: GpuObject;
  /** How many it counts: at least `min`, and at most `max` if it says. */
  readonly min: number;
  readonly max?: number;
  /** A word that may name a module whose own `@owns` it adds: `readback`, `cameras`, `images`. */
  readonly module?: string | undefined;
  /** A condition after the module, `unless switched off`, makes what it owns optional. */
  readonly optional: boolean;
  /** The GPU objects it mentions anywhere: `a uniform per way of reading textures` mentions two. */
  readonly mentions: readonly GpuObject[];
  /** Every word it says, each of which may name a module. */
  readonly words: readonly string[];
}

const count = (word: string) =>
  word === 'a' || word === 'an' || word === 'one' ? 1 : Number(word);

const mentioned = (text: string) =>
  Object.entries(nouns)
    .filter(([word]) => new RegExp(`\\b${word.replace(' ', '\\s+')}s?\\b`, 'i').test(text))
    .map(([, object]) => object);

/**
 * Reads `@owns` items, part by part: `3 buffers`, `a sampler`, `3 or 4 buffers`, `2 to 7 buffers`.
 * `a texture per target`, `one per …`, `2 bind groups and a buffer per level`, `up to …` and bare
 * plurals (`targets`) leave the count unbounded or start it at none. A part that names a module,
 * `a readback` or `` what `images` makes ``, counts nothing itself.
 */
export function readOwns(items: readonly string[]): Owned[] {
  return items.flatMap((item) => {
    const parts = item.split(' and ').filter((part) => part.length > 0);
    // `2 bind groups and a buffer per level`: a qualifier on a part that names its own object
    // counts the parts before it the same way; `5 pipelines and one per each` adds any number
    const qualifies = parts.map((part) => {
      const match = counting.exec(part);
      return match?.[4] !== undefined && match[5] !== undefined;
    });

    let last: GpuObject | undefined;
    return parts.map((text, index): Owned => {
      const words = [...text.matchAll(/[A-Za-z]\w*/g)].map((match) => match[0]);
      const quoted = /`([\w.]+)`/.exec(text)?.[1]?.split('.')[0];
      const base = { text, optional: false, mentions: mentioned(text), words };

      const match = counting.exec(text);
      if (match && (match[4] ?? last)) {
        const object: GpuObject = match[4] ? nouns[match[4]] : last!;
        last = object;
        const per = match[5] !== undefined || qualifies.slice(index + 1).includes(true);
        if (per) return { ...base, object, min: 0 };
        const min = match[1] ? 0 : count(match[2]);
        const max = match[3] ? Number(match[3]) : count(match[2]);
        return { ...base, object, min, max };
      }

      const plural = some.exec(text);
      if (plural) {
        last = nouns[plural[1]];
        return { ...base, object: last, min: 0, module: words.at(-1) };
      }

      const named = naming.exec(text);
      const module = named?.[1]?.split('.')[0] ?? quoted;
      return { ...base, min: 0, optional: !!named?.[2], module };
    });
  });
}
