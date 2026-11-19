import { moduleById, type Factory, type ModuleContract, type Parameter } from './contract.ts';

export type Role = 'borrows' | 'needs' | 'takes' | 'returns';

/** What a labelled argument is, as its label and the key show it. */
export interface Label {
  readonly role: Role;
  /** The argument as the line shows it, such as `camera` or `Particles`. */
  readonly text: string;
  /** A few words on what the factory does with it. */
  readonly words?: string | undefined;
  /** The module it needs, by id. */
  readonly module?: string | undefined;
}

export type Token =
  | { readonly text: string; readonly label?: undefined }
  | { readonly text: string; readonly label: Label };

export interface Line {
  readonly name: string;
  readonly tokens: readonly Token[];
}

export interface Signature {
  readonly lines: readonly Line[];
  /** Each labelled argument once, in the order the lines name them. */
  readonly labels: readonly Label[];
  /** Factories past the lines shown, by name. */
  readonly rest: readonly string[];
}

/** Lines past which the rest are listed by name. */
const SHOWN = 5;

/** A parameter by its name, and an options object by the fields it requires too. */
const shape = (parameter: Parameter) => {
  const required = (parameter.fields ?? []).filter((field) => !field.optional);
  return [parameter.name, ...required.map((field) => field.name)].join(' ');
};

const key = (factory: Factory) => `${factory.name}(${factory.parameters.map(shape).join(',')})`;

/**
 * Overloads that differ only in which parameters or fields are optional become one line. An
 * options object whose fields differ between them shows as its name.
 */
function merged(factories: readonly Factory[]) {
  const lines: Factory[] = [];

  for (const factory of factories) {
    const twin = lines.findIndex((line) => key(line) === key(factory));
    if (twin === -1) {
      lines.push(factory);
      continue;
    }

    const parameters = lines[twin].parameters.map((parameter, i): Parameter => {
      const other = factory.parameters[i];
      const same = JSON.stringify(parameter.fields) === JSON.stringify(other?.fields);
      const { fields: _fields, more: _more, ...plain } = parameter;
      const kept = same ? parameter : plain;
      return other?.optional ? { ...kept, optional: true } : kept;
    });
    lines[twin] = { ...lines[twin], parameters };
  }

  return lines;
}

/** The prefix a module's names are called with: `frames.`, or none for a flat module. */
export const prefixOf = (module: ModuleContract) => (module.flat ? '' : `${module.namespace}.`);

export function signatureOf(module: ModuleContract, modules: readonly ModuleContract[]): Signature {
  const factories = merged(module.factories);
  const shown = factories.length > SHOWN ? factories.slice(0, SHOWN - 1) : factories;
  const rest = [...new Set(factories.slice(shown.length).map((factory) => factory.name))];
  const prefix = prefixOf(module);

  /** An argument, labelled when the factory does something worth saying with it. */
  const argument = (
    text: string,
    each: { root?: boolean | undefined; module?: string | undefined; note?: string | undefined },
  ): Token => {
    if (each.root)
      return { text, label: { role: 'borrows', text, words: each.note ?? 'the root' } };
    if (each.module) {
      moduleById(modules, each.module);
      return { text, label: { role: 'needs', text, words: each.note, module: each.module } };
    }
    if (each.note) return { text, label: { role: 'takes', text, words: each.note } };
    return { text };
  };

  const parameter = (each: Parameter): Token[] => {
    const optional = each.optional ? '?' : '';
    if (!each.fields) return [argument(`${each.name}${optional}`, each)];

    // An options object: the fields worth naming, then the rest
    const tokens: Token[] = [{ text: '{ ' }];
    each.fields.forEach((field, i) => {
      if (i > 0) tokens.push({ text: ', ' });
      tokens.push(argument(`${field.name}${field.optional ? '?' : ''}`, field));
    });
    tokens.push({ text: `${each.more ? ', …' : ''} }${optional}` });
    return tokens;
  };

  const lines: Line[] = shown.map((factory) => {
    const tokens: Token[] = [{ text: `${prefix}${factory.name}(` }];
    factory.parameters.forEach((each, i) => {
      if (i > 0) tokens.push({ text: ', ' });
      tokens.push(...parameter(each));
    });
    tokens.push({ text: ')' });

    if (factory.returns) {
      tokens.push({ text: ' → ' });
      tokens.push(
        factory.result
          ? {
              text: factory.returns,
              label: { role: 'returns', text: factory.returns, words: factory.result },
            }
          : { text: factory.returns },
      );
    }
    return { name: factory.name, tokens };
  });

  const labels: Label[] = [];
  for (const line of lines) {
    for (const token of line.tokens) {
      const label = token.label;
      if (!label) continue;
      const seen = labels.some(
        (each) =>
          each.role === label.role && each.text === label.text && each.words === label.words,
      );
      if (!seen) labels.push(label);
    }
  }

  return { lines, labels, rest: rest.map((name) => `${prefix}${name}`) };
}

/** A line as plain text: `frames.create(root, record) → Frames`. */
export const textOf = (line: Line) => line.tokens.map((token) => token.text).join('');

/** Wrap after parameter commas or before the return arrow. Keep the result with its arrow and never start a line with whitespace, a comma or a closing bracket. */
export function runsOf(tokens: readonly Token[]) {
  const runs: Token[][] = [];

  tokens.forEach((token, i) => {
    if (token.text === ' → ') {
      runs.at(-1)!.push({ text: ' ' });
      runs.push([{ text: '→ ' }]);
      return;
    }

    const wraps = i === 0 || tokens[i - 1].text.endsWith(', ');
    if (wraps) runs.push([token]);
    else runs.at(-1)!.push(token);
  });

  return runs;
}
