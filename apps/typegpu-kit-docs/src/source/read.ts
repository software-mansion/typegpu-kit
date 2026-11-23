// Reads an example's files at build time: their reading order and summaries, the lines that run on
// the GPU, and what each name refers to, resolved by TypeScript's checker

import { readdirSync, statSync } from 'node:fs';
import { createRequire } from 'node:module';
import { dirname, join } from 'node:path';
import ts from 'typescript';

// Through the docs' dependencies, which resolve from wherever the build bundles this file
const resolveFrom = createRequire(import.meta.url).resolve;
const SRC = dirname(resolveFrom('typegpu-kit-examples'));
const EXAMPLES = dirname(SRC);
const KIT = `${dirname(dirname(resolveFrom('@typegpu/kit')))}/`;

/** Ids of what a GPU function reads from outside itself, by how TypeGPU passes it to the shader. */
export interface Captures {
  /** Buffers, textures and samplers, bound to the pipeline and read each frame. */
  readonly bound?: readonly string[];
  /** Accessors and slots, filled where a pipeline is created. */
  readonly filled?: readonly string[];
  /** Numbers, vectors and matrices, written into the WGSL as constants. */
  readonly baked?: readonly string[];
}

/** A name the example declares. */
export interface Local {
  readonly kind: 'local';
  readonly name: string;
  readonly file: string;
  readonly line: number;
  /** The last line of its declaration worth showing, when past `line`: a function's signature, or
   * all of a type, an interface or a value. */
  readonly to?: number;
  /** Its doc comment, or for a parameter its `@param` text. */
  readonly doc?: string;
  /** It declares a `'use gpu'` function. */
  readonly gpu?: true;
  readonly captures?: Captures;
}

/** A kit module the example imports as a namespace, such as `bloom`. */
export interface Module {
  readonly kind: 'module';
  readonly name: string;
}

/** A function or method from a package, such as `bloom.create` or `root.createUniform`. */
export interface Api {
  readonly kind: 'api';
  readonly name: string;
  /** Its name with its owner: `bloom.create`, `TgpuRoot.createUniform`. */
  readonly title: string;
  /** Its parameters: `create(root, options)`. */
  readonly call: string;
  readonly doc?: string;
  readonly from: string;
  /** For the kit, its file in the package, such as `src/post/bloom.ts`. */
  readonly file?: string;
}

export type Traced = Local | Module | Api;

export interface SourceFile {
  readonly name: string;
  readonly code: string;
  /** The first sentence of the doc comment on its exported `create*` function, or another. */
  readonly summary: string;
  /** First and last lines of each outermost `'use gpu'` function, from 1. */
  readonly gpu: readonly (readonly [number, number])[];
  /** Each traced identifier's offset and the id it refers to. */
  readonly marks: readonly (readonly [number, string])[];
}

export interface Source {
  /** `scene.ts`, then what it imports, breadth first, then the rest. */
  readonly files: readonly SourceFile[];
  /** What each mark refers to, by id. */
  readonly traced: Readonly<Record<string, Traced>>;
  /** Values imported from `@typegpu/kit`, such as the `bloom` namespace. */
  readonly kit: readonly string[];
  /** Other packages imported. */
  readonly packages: readonly string[];
}

/** Example keys, such as `slime-mould`. */
export const exampleKeys = () =>
  readdirSync(SRC).filter((name) =>
    statSync(join(SRC, name, 'scene.ts'), { throwIfNoEntry: false }),
  );

// One program for every example; rebuilt when a file changes, as under `pnpm dev`
let cached: { program: ts.Program; stamp: string } | undefined;

function program() {
  const config = ts.getParsedCommandLineOfConfigFile(
    join(EXAMPLES, 'tsconfig.json'),
    {},
    { ...ts.sys, onUnRecoverableConfigFileDiagnostic: () => {} },
  )!;
  const roots = config.fileNames.filter((file) => file.startsWith(`${SRC}/`));
  const stamp = roots.map((file) => statSync(file).mtimeMs).join();
  if (cached?.stamp !== stamp) {
    cached = {
      program: ts.createProgram({
        rootNames: roots,
        options: config.options,
        ...(cached ? { oldProgram: cached.program } : {}),
      }),
      stamp,
    };
  }
  return cached.program;
}

const isExported = (node: ts.Node) =>
  ts.canHaveModifiers(node) &&
  (ts.getModifiers(node) ?? []).some((modifier) => modifier.kind === ts.SyntaxKind.ExportKeyword);

/** Every `scene.ts`'s summary, in place of its `createScene` doc comment, which describes the example. */
const SCENE = 'Creates the scene: its resources, its settings and what each frame records.';

function summaryOf(source: ts.SourceFile) {
  const functions = source.statements.filter(
    (statement): statement is ts.FunctionDeclaration =>
      ts.isFunctionDeclaration(statement) && isExported(statement),
  );
  const created = functions.filter((fn) => fn.name?.text.startsWith('create'));
  for (const fn of [...created, ...functions.filter((other) => !created.includes(other))]) {
    const doc = ts.getJSDocCommentsAndTags(fn).find(ts.isJSDoc);
    const text = doc && ts.getTextOfJSDocComment(doc.comment);
    if (text) return text.replace(/\s+/g, ' ').split(/(?<=\.)\s/)[0];
  }
  return '';
}

const isGpu = (node: ts.Node): node is ts.FunctionLikeDeclaration & { body: ts.Block } => {
  if (
    !ts.isArrowFunction(node) &&
    !ts.isFunctionExpression(node) &&
    !ts.isFunctionDeclaration(node)
  ) {
    return false;
  }
  const first = node.body && ts.isBlock(node.body) ? node.body.statements[0] : undefined;
  return (
    first !== undefined &&
    ts.isExpressionStatement(first) &&
    ts.isStringLiteral(first.expression) &&
    first.expression.text === 'use gpu'
  );
};

const lineOf = (source: ts.SourceFile, position: number) =>
  source.getLineAndCharacterOfPosition(position).line + 1;

/**
 * Outermost GPU functions, with lines widened to the statement declaring them, as a shader's
 * header, by the line they start on.
 */
function gpuOf(source: ts.SourceFile) {
  const found = new Map<number, { to: number; fn: ts.FunctionLikeDeclaration }>();
  const visit = (node: ts.Node) => {
    if (!isGpu(node)) {
      ts.forEachChild(node, visit);
      return;
    }
    let from: ts.Node = node;
    while (
      ts.isCallExpression(from.parent) ||
      ts.isVariableDeclaration(from.parent) ||
      ts.isVariableDeclarationList(from.parent) ||
      ts.isVariableStatement(from.parent)
    ) {
      from = from.parent;
    }
    found.set(lineOf(source, from.getStart(source)), {
      to: lineOf(source, node.getEnd()),
      fn: node,
    });
  };
  visit(source);
  return found;
}

/** The identifier a declaration names: a variable, parameter, destructured binding, function or type. */
function declaredName(node: ts.Node) {
  const name =
    ts.isVariableDeclaration(node) ||
    ts.isParameter(node) ||
    ts.isBindingElement(node) ||
    ts.isFunctionDeclaration(node) ||
    ts.isClassDeclaration(node) ||
    ts.isInterfaceDeclaration(node) ||
    ts.isTypeAliasDeclaration(node) ||
    ts.isEnumDeclaration(node)
      ? node.name
      : undefined;
  return name && ts.isIdentifier(name) ? name : undefined;
}

/** The last line of a declaration worth showing: a function's signature, or all of the rest. */
function endOf(node: ts.Node, source: ts.SourceFile, fn: ts.FunctionLikeDeclaration | undefined) {
  if (ts.isParameter(node) || ts.isBindingElement(node)) return undefined;
  const value = ts.isVariableDeclaration(node) ? node.initializer : undefined;
  const body = (
    fn ??
    (ts.isFunctionDeclaration(node)
      ? node
      : value && (ts.isArrowFunction(value) || ts.isFunctionExpression(value))
        ? value
        : undefined)
  )?.body;
  if (body) return lineOf(source, body.getStart(source));
  const whole =
    ts.isVariableDeclaration(node) && ts.isVariableStatement(node.parent.parent)
      ? node.parent.parent
      : node;
  return lineOf(source, whole.getEnd());
}

/** A declaration's doc comment, or a parameter's `@param` text, with paragraphs kept. */
function docOf(node: ts.Node) {
  for (const doc of ts.getJSDocCommentsAndTags(node)) {
    const comment = ts.isJSDoc(doc) || ts.isJSDocParameterTag(doc) ? doc.comment : undefined;
    const text = ts.getTextOfJSDocComment(comment)?.replace(/^-\s*/, '').trim();
    if (text) return text;
  }
  return undefined;
}

/** The package a declaration file belongs to, or undefined for the language's own types. */
function packageOf(file: string) {
  if (file.startsWith(KIT)) return '@typegpu/kit';
  if (/\/typescript\/lib\/|\/tsover\/lib\/|\/@webgpu\/types\/|\/@types\//.test(file)) {
    return undefined;
  }
  return /.*\/node_modules\/((?:@[^/]+\/)?[^/]+)\//.exec(file)?.[1];
}

/** `name(a, b?, ...rest)` from a call signature's parameters. */
function callOf(name: string, signature: ts.Signature) {
  const parameters = signature.parameters.map((parameter) => {
    const declaration = parameter.valueDeclaration;
    const own = declaration && ts.isParameter(declaration) ? declaration : undefined;
    const written = parameter.name.startsWith('__') ? '{…}' : parameter.name;
    return `${own?.dotDotDotToken ? '...' : ''}${written}${own?.questionToken || own?.initializer ? '?' : ''}`;
  });
  return `${name}(${parameters.join(', ')})`;
}

// TypeGPU's types for what a shader reads, by how it reaches the shader
const BOUND =
  /^Tgpu(Uniform|Mutable|Readonly|Buffer\w*|Texture\w*|Sampler|FixedSampler|ComparisonSampler|FixedComparisonSampler|BindGroupLayout)$/;
const FILLED = /^Tgpu(Accessor|MutableAccessor|Slot)$/;
const BAKED = /^(v[234][fiuhb]|m[234]x[234][fh])$/;

export function read(key: string): Source {
  const dir = join(SRC, key);
  const all = readdirSync(dir).filter((name) => name.endsWith('.ts'));
  const built = program();
  const checker = built.getTypeChecker();
  const parsed = new Map(all.map((name) => [name, built.getSourceFile(join(dir, name))!]));

  const order: string[] = [];
  const queue = ['scene.ts'];
  const packages = new Set<string>();
  const kit = new Set<string>();
  while (queue.length > 0) {
    const name = queue.shift()!;
    if (order.includes(name) || !parsed.has(name)) continue;
    order.push(name);
    for (const statement of parsed.get(name)!.statements) {
      if (!ts.isImportDeclaration(statement) || !ts.isStringLiteral(statement.moduleSpecifier)) {
        continue;
      }
      const from = statement.moduleSpecifier.text;
      if (from.startsWith('./')) queue.push(from.slice(2));
      else if (from.startsWith('.')) continue;
      else if (from === '@typegpu/kit') {
        const bindings = statement.importClause?.namedBindings;
        if (bindings && ts.isNamedImports(bindings)) {
          for (const element of bindings.elements) {
            const typeOnly = statement.importClause?.phaseModifier === ts.SyntaxKind.TypeKeyword;
            if (!typeOnly && !element.isTypeOnly) kit.add(element.name.text);
          }
        }
      } else if (!from.startsWith('@typegpu/kit/')) packages.add(from);
    }
  }
  const files = [...order, ...all.filter((name) => !order.includes(name)).toSorted()];
  const gpu = new Map(files.map((name) => [name, gpuOf(parsed.get(name)!)]));
  const inExample = (node: ts.Node) => node.getSourceFile().fileName.startsWith(`${dir}/`);

  // Ids are names, numbered when two symbols share one
  const traced: Record<string, Traced> = {};
  // By declaration, as each instance of a generic type has its own symbol for a method
  const ids = new Map<ts.Node | ts.Symbol, string | null>();
  const add = (symbol: ts.Symbol, entry: () => Traced | undefined) => {
    const identity = symbol.declarations?.[0] ?? symbol;
    if (ids.has(identity)) return ids.get(identity) ?? undefined;
    ids.set(identity, null);
    const made = entry();
    let id = made?.name ?? null;
    for (let n = 2; id !== null && Object.hasOwn(traced, id); n++) id = `${made!.name}~${n}`;
    if (made && id !== null) traced[id] = made;
    ids.set(identity, id);
    return id ?? undefined;
  };

  const symbolOf = (node: ts.Identifier) => {
    const symbol = ts.isShorthandPropertyAssignment(node.parent)
      ? checker.getShorthandAssignmentValueSymbol(node.parent)
      : checker.getSymbolAtLocation(node);
    return symbol && symbol.flags & ts.SymbolFlags.Alias
      ? checker.getAliasedSymbol(symbol)
      : symbol;
  };

  const local = (symbol: ts.Symbol): Local | undefined => {
    const declaration = symbol.declarations?.find(inExample);
    const named = declaration && declaredName(declaration);
    if (!declaration || !named || symbol.declarations!.some((other) => !inExample(other))) {
      return undefined;
    }
    const source = declaration.getSourceFile();
    const file = source.fileName.slice(dir.length + 1);
    const line = lineOf(source, named.getStart(source));
    const shader = ts.isParameter(declaration) ? undefined : gpu.get(file)?.get(line);
    const doc = docOf(declaration);
    const captures = shader && capturesOf(shader.fn);
    const to = endOf(declaration, source, shader?.fn);
    return {
      kind: 'local',
      name: named.text,
      file,
      line,
      ...(to !== undefined && to > line ? { to } : {}),
      ...(doc ? { doc } : {}),
      ...(shader ? { gpu: true as const } : {}),
      ...(captures ? { captures } : {}),
    };
  };

  const api = (symbol: ts.Symbol, at: ts.Identifier): Api | undefined => {
    const declaration = symbol.declarations?.[0];
    const path = declaration?.getSourceFile().fileName ?? '';
    const from = packageOf(path);
    if (!declaration || !from || symbol.flags & ts.SymbolFlags.Module) return undefined;
    const signature = checker.getTypeOfSymbolAtLocation(symbol, at).getCallSignatures()[0];
    if (!signature) return undefined;
    // As written, since a package may export it under another name, or bundle it renamed
    const name = at.text;
    const owner = declaration.parent;
    const access = ts.isPropertyAccessExpression(at.parent) && at.parent.name === at;
    const through =
      access && ts.isIdentifier(at.parent.expression)
        ? checker.getSymbolAtLocation(at.parent.expression)
        : undefined;
    const title =
      (ts.isInterfaceDeclaration(owner) || ts.isClassDeclaration(owner)) && owner.name
        ? `${owner.name.text}.${name}`
        : through && through.flags & ts.SymbolFlags.Alias
          ? `${through.name}.${name}`
          : name;
    const doc = ts
      .displayPartsToString(
        symbol.getDocumentationComment(checker).length > 0
          ? symbol.getDocumentationComment(checker)
          : signature.getDocumentationComment(checker),
      )
      .trim();
    return {
      kind: 'api',
      name,
      title,
      call: callOf(name, signature),
      ...(doc ? { doc } : {}),
      from,
      ...(path.startsWith(KIT) ? { file: path.slice(KIT.length) } : {}),
    };
  };

  /** What a GPU function reads from the example around it, by how it reaches the shader. */
  function capturesOf(fn: ts.Node): Captures | undefined {
    const groups = {
      bound: new Set<string>(),
      filled: new Set<string>(),
      baked: new Set<string>(),
    };
    const visit = (node: ts.Node) => {
      if (
        ts.isIdentifier(node) &&
        !(ts.isPropertyAccessExpression(node.parent) && node.parent.name === node)
      ) {
        const symbol = symbolOf(node);
        const declaration = symbol?.declarations?.[0];
        if (
          symbol &&
          declaration &&
          inExample(declaration) &&
          (ts.isVariableDeclaration(declaration) ||
            ts.isParameter(declaration) ||
            ts.isBindingElement(declaration)) &&
          (declaration.getSourceFile() !== fn.getSourceFile() ||
            declaration.getStart() < fn.getStart() ||
            declaration.getStart() > fn.getEnd())
        ) {
          const type = checker.getTypeAtLocation(node);
          const named = type.aliasSymbol?.name ?? type.getSymbol()?.name ?? '';
          const group = BOUND.test(named)
            ? groups.bound
            : FILLED.test(named)
              ? groups.filled
              : type.flags & (ts.TypeFlags.NumberLike | ts.TypeFlags.BooleanLike) ||
                  BAKED.test(named)
                ? groups.baked
                : undefined;
          const id = group && add(symbol, () => local(symbol));
          if (id) group.add(id);
        }
      }
      ts.forEachChild(node, visit);
    };
    visit(fn);
    const found = Object.entries(groups).filter(([, names]) => names.size > 0);
    return found.length > 0
      ? Object.fromEntries(found.map(([group, names]) => [group, [...names]]))
      : undefined;
  }

  const marked = files.map((name) => {
    const source = parsed.get(name)!;
    const marks: [number, string][] = [];
    const visit = (node: ts.Node) => {
      if (ts.isIdentifier(node)) {
        const symbol = symbolOf(node);
        const id =
          symbol &&
          add(symbol, () => {
            if (symbol.flags & ts.SymbolFlags.Module) {
              const isKit = symbol.declarations?.[0]?.getSourceFile().fileName.startsWith(KIT);
              return isKit && kit.has(node.text) ? { kind: 'module', name: node.text } : undefined;
            }
            return local(symbol) ?? api(symbol, node);
          });
        if (id) marks.push([node.getStart(source), id]);
      }
      ts.forEachChild(node, visit);
    };
    visit(source);
    return marks;
  });

  // A local written once, without docs, has nothing to show
  const counts = new Map<string, number>();
  for (const [, id] of marked.flat()) counts.set(id, (counts.get(id) ?? 0) + 1);
  for (const [id, entry] of Object.entries(traced)) {
    if (entry.kind === 'local' && counts.get(id) === 1 && !entry.doc && !entry.gpu) {
      delete traced[id];
    }
  }

  return {
    files: files.map((name, i) => {
      const source = parsed.get(name)!;
      return {
        name,
        code: source.text,
        summary: name === 'scene.ts' ? SCENE : summaryOf(source),
        gpu: [...gpu.get(name)!].map(([from, { to }]) => [from, to] as const),
        marks: marked[i].filter(([, id]) => Object.hasOwn(traced, id)),
      };
    }),
    traced,
    kit: [...kit].toSorted(),
    packages: [...packages].toSorted(),
  };
}
