import {
  exportGroups,
  moduleById,
  type ExportGroup,
  type Injection,
  type Kind,
  type Mode,
  type ModuleContract,
  type Handle,
  type Options,
} from './contract.ts';
import { signatureOf, type Signature } from './signature.ts';

/** The colour a fact's dot and pills take. */
export type Facet =
  | 'records'
  | 'injects'
  | 'needs'
  | 'requires'
  | 'platform'
  | 'kind'
  | 'exception';

export interface Pill {
  readonly text: string;
  readonly facet: Facet;
  /** The module it names, by id. */
  readonly module?: string | undefined;
  /** An identifier, which Markdown quotes: a module's namespace, a feature. */
  readonly code?: boolean | undefined;
  /** Dashed outline for an optional value. */
  readonly dashed?: boolean | undefined;
  readonly title?: string | undefined;
}

export type Part =
  | { readonly type: 'pill'; readonly pill: Pill }
  | { readonly type: 'code'; readonly text: string }
  | { readonly type: 'text'; readonly text: string }
  /** Words beside a value that only qualify it, muted: `only`, `optional`. */
  | { readonly type: 'aside'; readonly text: string }
  /** An example that uses the module, by key. */
  | { readonly type: 'example'; readonly key: string }
  | { readonly type: 'check'; readonly text: string; readonly on: boolean };

/** How a renderer resolves a card's references. */
export interface Links {
  /** A module's page, or its card on a page of several. */
  readonly module: (id: string) => string | undefined;
  /** An example's page and title. */
  readonly example: (key: string) => { readonly href: string; readonly title: string };
  /** Where a method a check names sits on the card, for a renderer that shows the card. */
  readonly member?: ((method: string) => string | undefined) | undefined;
}

export type FactKey =
  | 'stability'
  | 'kind'
  | 'platform'
  | 'records'
  | 'submits'
  | 'owns'
  | 'allocates'
  | 'lifecycle'
  | 'injects'
  | 'binds'
  | 'needs'
  | 'reads'
  | 'requires'
  | ExportGroup
  | 'types'
  | 'usedBy';

export interface Entry {
  readonly value: readonly Part[];
  /** Words on the value, as the source writes them: the card makes them a sentence. */
  readonly note?: readonly Part[] | undefined;
}

export interface Fact {
  readonly key: FactKey;
  readonly label: string;
  /** The dot's colour; none for a fact that is no facet, such as what it owns. */
  readonly facet?: Facet | undefined;
  /** The value breaks the frame's rules, such as allocating on resize. */
  readonly exception?: boolean | undefined;
  readonly value?: readonly Part[] | undefined;
  /** Methods and what each records into. */
  readonly pairs?: readonly { readonly method: string; readonly pill: Pill }[] | undefined;
  /** Items on lines of their own, each with a note. */
  readonly entries?: readonly Entry[] | undefined;
  readonly note?: readonly Part[] | undefined;
}

/** The order cards and modules.md list facts in. */
export const factKeys: readonly FactKey[] = [
  'stability',
  'kind',
  'records',
  'submits',
  'owns',
  'allocates',
  'lifecycle',
  'injects',
  'binds',
  'needs',
  'reads',
  'requires',
  'platform',
  'functions',
  'accessors',
  'outputs',
  'schemas',
  'constants',
  'helpers',
  'types',
  'usedBy',
];

/** The facts a card lists under its signature, as exports of the module. */
export const exportKeys: readonly FactKey[] = [
  'functions',
  'accessors',
  'outputs',
  'schemas',
  'constants',
  'helpers',
  'types',
];

const labels: Partial<Record<FactKey, string>> = {
  functions: 'GPU functions',
  usedBy: 'used by',
};

export const modes: Record<Mode, string> = {
  required: 'required',
  optional: 'used when the device has it',
  option: 'needed by an option',
};

const submissions = { setup: 'while setting up', always: 'every frame, on its own' } as const;

/**
 * What a card's labels state, each in a sentence: their tooltips, and the key on the page that
 * says how to read a card, in the key's order: the card's groups, then stability and examples.
 */
export const meanings: Readonly<Record<Exclude<FactKey, ExportGroup | 'types' | 'kind'>, string>> =
  {
    records: 'The methods that record into your frame, and what each one takes.',
    submits: 'It submits commands on its own, which no other module does.',
    owns: 'The GPU resources it allocates and frees in destroy().',
    allocates: 'It creates GPU objects after setup, which a steady frame otherwise never does.',
    binds: 'The buffers a shader stage binds when it reads the module.',
    lifecycle: 'What to await before the first frame, and what frees its resources.',
    needs: 'Other modules its factories take.',
    reads: 'Modules whose inject() must fill the pipeline that calls its functions.',
    injects: 'What its inject() fills in a pipeline that reads it.',
    requires: 'Device features it needs or uses.',
    platform: 'It runs on one platform only.',
    stability: 'Its names or shape may still change in a minor release.',
    usedBy: 'Examples that use it.',
  };

/** The facet of each fact that has one: the colour of its dot and its pills. */
export const facetOf: Readonly<Partial<Record<FactKey, Facet>>> = {
  stability: 'exception',
  kind: 'kind',
  records: 'records',
  submits: 'exception',
  allocates: 'exception',
  injects: 'injects',
  needs: 'needs',
  reads: 'needs',
  requires: 'requires',
  platform: 'platform',
};

/** What an argument's underline says the factory does with it. */
export const roles = {
  borrows: 'borrows, never destroys',
  takes: 'takes',
  needs: 'needs a module',
  returns: 'returns',
} as const;

/** What each kind means, beside its pill and in its tooltip. */
export const kinds: Record<Kind, string> = {
  module: 'borrows the root and owns GPU resources',
  cpu: 'no GPU',
  shader: "'use gpu' functions only",
  platform: 'platform helpers',
  tool: 'for development and tests',
};

/** What `unstable` means, beside the word on a card. */
export const UNSTABLE = 'may change in any way in a minor release';

export const capital = (text: string) => text[0].toUpperCase() + text.slice(1);

/** A count and what it counts: `1 field`, `3 members`. */
export const counted = (n: number, one: string) => `${n} ${one}${n === 1 ? '' : 's'}`;

/** What a `@binds` costs a stage: `2 storage`, `1 storage · 1 uniform`, or `none` of its own. */
export function costOf({ storage, uniform }: { storage: number; uniform: number }) {
  const parts = [
    storage > 0 ? `${storage} storage` : undefined,
    uniform > 0 ? `${uniform} uniform` : undefined,
  ].filter(Boolean);
  return parts.length > 0 ? parts.join(' · ') : 'none';
}

/** Describe the dependencies' hooks and accessors. */
function fills(injects: readonly { kind: Injection }[]) {
  const hook = injects.some((each) => each.kind === 'hook');
  const accessors = injects.some((each) => each.kind !== 'hook');
  if (hook && accessors) return 'a hook and accessors';
  if (hook) return 'a hook';
  return accessors ? 'accessors' : undefined;
}

/** Words with `code` in backticks, as a tag's note or a label writes them. */
export function prose(text: string | undefined): Part[] | undefined {
  if (!text) return undefined;
  return text
    .split(/(`[^`]+`)/)
    .filter(Boolean)
    .map((part) =>
      part.startsWith('`')
        ? { type: 'code', text: part.slice(1, -1) }
        : { type: 'text', text: part },
    );
}

/** Words as a sentence: a capital, unless they start with code, and a full stop. */
export function sentence(parts: readonly Part[]): Part[] {
  const out = [...parts];
  const first = out[0];
  if (first?.type === 'text') out[0] = { ...first, text: capital(first.text) };

  const last = out.at(-1);
  if (last?.type === 'text') {
    if (!/[.!?]$/.test(last.text)) out[out.length - 1] = { ...last, text: `${last.text}.` };
  } else {
    out.push({ type: 'text', text: '.' });
  }
  return out;
}

const plain = (text: string): Part[] => [{ type: 'text', text }];
const codes = (names: readonly string[]): Part[] =>
  names.map((name): Part => ({ type: 'code', text: name }));

/** Every fact of a module that isn't its default, in `factKeys` order. */
export function factsOf(module: ModuleContract, modules: readonly ModuleContract[]): Fact[] {
  const linked = (id: string): Pill => {
    const other = moduleById(modules, id);
    return { text: other.namespace, facet: 'needs', module: id, code: true };
  };

  const fact = (key: FactKey): Fact | undefined => {
    const label = labels[key] ?? key;
    const facet = facetOf[key];

    switch (key) {
      case 'stability': {
        if (!module.unstable) return undefined;
        return {
          key,
          label,
          facet,
          exception: true,
          value: plain('unstable'),
          note: plain(UNSTABLE),
        };
      }

      case 'kind': {
        const pill: Pill = { text: module.kind, facet: 'kind', title: capital(kinds[module.kind]) };
        if (module.kind === 'module') return undefined;
        return {
          key,
          label,
          facet,
          value: [{ type: 'pill', pill }],
        };
      }

      case 'platform': {
        if (!module.platform) {
          return undefined;
        }
        const pill: Pill = { text: module.platform, facet: 'platform' };
        return {
          key,
          label,
          facet,
          value: [
            { type: 'pill', pill },
            { type: 'aside', text: 'only' },
          ],
        };
      }

      case 'records': {
        const note = prose(module.order);
        if (module.records.length === 0 && !note) {
          return undefined;
        }
        // A module that records nothing may still say what to call in a frame
        if (module.records.length === 0) return { key, label, facet, note };
        return {
          key,
          label,
          facet,
          pairs: module.records.map((record) => ({
            method: record.method,
            pill: { text: record.into, facet: 'records' },
          })),
          note,
        };
      }

      case 'submits': {
        if (!module.submits) return undefined;
        return {
          key,
          label,
          facet,
          exception: true,
          value: plain(submissions[module.submits.when]),
          note: prose(module.submits.note),
        };
      }

      case 'owns': {
        if (!module.owns) return undefined;
        return {
          key,
          label,
          // Keep separators with their items so wrapping occurs between items
          value: module.owns.items.map((item, i, items) => ({
            type: 'text',
            text: i < items.length - 1 ? `${item} ·` : item,
          })),
          note: prose(module.owns.note),
        };
      }

      case 'allocates': {
        if (!module.allocates) {
          return undefined;
        }
        return {
          key,
          label,
          facet,
          exception: true,
          value: plain(module.allocates.when),
          note: prose(module.allocates.note),
        };
      }

      case 'lifecycle': {
        const { initAsync, destroy, dispose } = module.lifecycle;
        if (!initAsync && !destroy && !dispose) {
          return undefined;
        }
        // Only list initAsync when the caller has compilation to await
        return {
          key,
          label,
          value: [
            ...(initAsync ? [{ type: 'check', text: 'initAsync()', on: true } as const] : []),
            { type: 'check', text: 'destroy()', on: destroy },
            { type: 'check', text: '[Symbol.dispose]', on: dispose },
          ],
        };
      }

      case 'injects': {
        if (module.injects.length === 0) {
          return undefined;
        }
        return {
          key,
          label,
          facet,
          entries: module.injects.map((each) => ({
            value: [{ type: 'pill', pill: { text: each.kind, facet: 'injects' } }],
            note: prose(each.note),
          })),
        };
      }

      case 'binds': {
        if (module.binds.length === 0) {
          return undefined;
        }
        return {
          key,
          label,
          entries: module.binds.map((each) => ({
            value: plain(costOf(each)),
            note: prose(each.note),
          })),
        };
      }

      case 'needs': {
        if (module.needs.length === 0) {
          return undefined;
        }
        return {
          key,
          label,
          facet,
          entries: module.needs.map((id) => {
            const filled = fills(moduleById(modules, id).injects);
            const note: Part[] = [{ type: 'text', text: 'as ' }];
            module.via[id].split(' or ').forEach((at, i) => {
              if (i > 0) note.push({ type: 'text', text: ' or ' });
              note.push({ type: 'code', text: at });
            });
            if (filled) note.push({ type: 'text', text: `; fills ${filled} in its pipelines` });
            return { value: [{ type: 'pill', pill: linked(id) }], note };
          }),
        };
      }

      case 'reads': {
        if (module.reads.length === 0) {
          return undefined;
        }
        return {
          key,
          label,
          facet,
          entries: module.reads.map((id) => ({
            value: [{ type: 'pill', pill: linked(id) }],
            note: [
              { type: 'text', text: 'its ' },
              { type: 'code', text: 'inject()' },
              { type: 'text', text: ' fills them in the calling pipeline' },
            ],
          })),
        };
      }

      case 'requires': {
        if (module.requires.length === 0) {
          return undefined;
        }
        const pillOf = (each: ModuleContract['requires'][number]): Pill => ({
          text: each.feature,
          facet: 'requires',
          code: true,
          dashed: each.mode !== 'required',
          title: modes[each.mode],
        });
        return {
          key,
          label,
          facet,
          entries: module.requires.map((each) => ({
            value: [
              { type: 'pill', pill: pillOf(each) },
              { type: 'aside', text: each.mode },
            ],
            note: prose(each.note ?? modes[each.mode]),
          })),
        };
      }

      case 'functions':
      case 'accessors':
      case 'outputs':
      case 'schemas':
      case 'constants':
      case 'helpers':
      case 'types': {
        const names = module[key];
        if (names.length === 0) return undefined;
        return { key, label, value: codes(names) };
      }

      case 'usedBy': {
        if (module.usedBy.length === 0) return undefined;
        return {
          key,
          label,
          value: module.usedBy.map((example): Part => ({ type: 'example', key: example })),
        };
      }
    }
  };

  return factKeys.flatMap((key) => fact(key) ?? []);
}

/** The line that imports a module; a flat module's by its factories when `short`. */
export function importOf(contract: ModuleContract, short = false) {
  if (!contract.flat) {
    return contract.specifier === contract.package || contract.namespaceExport
      ? `import { ${contract.namespace} } from '${contract.specifier}'`
      : `import * as ${contract.namespace} from '${contract.specifier}'`;
  }

  const factories = [...new Set(contract.factories.map((factory) => factory.name))];
  const names =
    short && factories.length > 0
      ? factories
      : [...factories, ...exportGroups.flatMap((group) => contract[group])];
  return `import { ${names.join(', ')} } from '${contract.specifier}'`;
}

/**
 * The facts a strip names above a page's prose, each by its shortest value: where it records, what
 * it needs and where it breaks a rule. The card keeps their notes, and what the module owns and
 * binds.
 */
export function glanceOf(facts: readonly Fact[]): Fact[] {
  return facts.flatMap((fact): Fact[] => {
    const { key, label, facet, exception } = fact;
    const short = (value: readonly Part[]) =>
      value.length > 0 ? [{ key, label, facet, exception, value }] : [];

    switch (key) {
      case 'stability':
      case 'platform':
      case 'submits':
      case 'allocates':
        return short(fact.value ?? []);
      case 'records': {
        const into = [...new Set((fact.pairs ?? []).map((pair) => pair.pill.text))];
        return short(
          into.map((text): Part => ({ type: 'pill', pill: { text, facet: 'records' } })),
        );
      }
      case 'lifecycle':
        return short(
          (fact.value ?? []).flatMap((part): Part[] =>
            part.type === 'check' && part.on && part.text !== '[Symbol.dispose]'
              ? [{ type: 'code', text: part.text }]
              : [],
          ),
        );
      case 'injects':
      case 'needs':
      case 'reads':
      case 'requires':
        return short(
          (fact.entries ?? []).flatMap((entry) =>
            entry.value.filter((part) => part.type === 'pill'),
          ),
        );
      default:
        return [];
    }
  });
}

export interface Card {
  /** Every fact that isn't its default, in `factKeys` order. */
  readonly facts: readonly Fact[];
  readonly signature: Signature;
  /** The facts by the question each answers; stability sits in the title and examples close it. */
  readonly groups: readonly { readonly label: string; readonly facts: readonly Fact[] }[];
  /** What the module exports beside its factories, each group that has any. */
  readonly exports: readonly Fact[];
  /** Each options type its factories and their handles' methods take, with every field. */
  readonly options: readonly Options[];
  /** What its factories return, with every member. */
  readonly handles: readonly Handle[];
}

const cardGroups: readonly { label: string; keys: readonly FactKey[] }[] = [
  { label: 'In a frame', keys: ['records', 'submits'] },
  { label: 'Holds', keys: ['owns', 'allocates', 'binds', 'lifecycle'] },
  { label: 'Connects to', keys: ['needs', 'reads', 'injects', 'requires', 'platform'] },
];

/** A module's card: the facts that are set, each once. */
export function cardOf(module: ModuleContract, modules: readonly ModuleContract[]): Card {
  const facts = factsOf(module, modules);
  return {
    facts,
    signature: signatureOf(module, modules),
    groups: cardGroups
      .map(({ label, keys }) => ({ label, facts: facts.filter((fact) => keys.includes(fact.key)) }))
      .filter((group) => group.facts.length > 0),
    exports: facts.filter((fact) => exportKeys.includes(fact.key)),
    options: module.options,
    handles: module.handles,
  };
}

// Markdown, for the kit's docs/modules.md and the review printout

/** A part as Markdown: code and identifiers in backticks, an example by its key. */
function partMarkdown(part: Part) {
  switch (part.type) {
    case 'code':
      return `\`${part.text}\``;
    case 'pill':
      return part.pill.code ? `\`${part.pill.text}\`` : part.pill.text;
    case 'example':
      return part.key;
    case 'check':
      return part.on ? `\`${part.text}\`` : '';
    case 'text':
      // An item of a list the card breaks between, such as what a module owns
      return part.text.replace(/ ·$/, ', ');
    case 'aside':
      return part.text;
  }
}

/** Parts as Markdown. Words keep their own spacing; values beside each other take a comma. */
export function markdownOf(parts: readonly Part[] | undefined): string {
  let out = '';
  let previous: Part | undefined;

  for (const part of parts ?? []) {
    const text = partMarkdown(part);
    if (!text) continue;

    const words = previous?.type === 'text' || part.type === 'text';
    out += !previous || words ? text : `${part.type === 'aside' ? ' ' : ', '}${text}`;
    previous = part;
  }

  return out.trim();
}

const noted = (value: string, note: readonly Part[] | undefined) =>
  note ? `${value}: ${markdownOf(note)}` : value;

/** A fact as modules.md writes it: `**label** value: note.` */
export function factMarkdown(fact: Fact): string {
  const pairs = fact.pairs?.map((pair) => `\`${pair.method}\` ${pair.pill.text}`).join(', ');
  const entries = fact.entries
    ?.map((entry) => noted(markdownOf(entry.value), entry.note))
    .join('; ');
  const value = pairs ?? entries ?? markdownOf(fact.value);
  const text = `**${fact.label}** ${value ? noted(value, fact.note) : markdownOf(fact.note)}`;
  return /[.!?]$/.test(text) ? text : `${text}.`;
}

/** The field of the options object that holds an options type: `shadow` of `options.shadow`. */
export const fieldOf = (table: Options) => table.parameter.split('.').slice(1).join('.');

/** Name the factories that take this options type and its enclosing field, if any. */
function takenText(table: Options): string {
  const field = fieldOf(table);
  const factories = table.takenBy.join(', ');
  return field ? `${field} in ${factories}` : factories;
}

/** Each options type as modules.md writes it: its name, then its fields, `?` on optional ones. */
export function optionsMarkdown(options: readonly Options[]): string | undefined {
  if (options.length === 0) return undefined;
  const tables = options.map((table) => {
    const fields = table.fields.map((field) => `\`${field.name}${field.optional ? '?' : ''}\``);
    return `\`${table.name}\` (${takenText(table)}) ${fields.join(', ')}`;
  });
  return `**options** ${tables.join('; ')}.`;
}
