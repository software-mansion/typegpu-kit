import { readFileSync } from 'node:fs';
import { basename, dirname, join, relative } from 'node:path';
import { fileURLToPath } from 'node:url';
import ts from 'typescript';
import {
  objects,
  type ExportGroup,
  type Field,
  type GpuObject,
  type Handle,
  type Kind,
  type Member,
  type ModuleContract,
  type OptionField,
  type Options,
  type Parameter,
  type Recording,
  type Struct,
} from './contract.ts';
import { overrides as repoOverrides, type Overrides } from './overrides.ts';
import {
  firstSentence,
  parseDoc,
  readTags,
  splitDefault,
  tags as contractTags,
  unhyphened,
  type Doc,
} from './tags.ts';
import { readOwns } from './owns.ts';
import { readUsage, type Usage } from './usage.ts';

/** The calls that create each GPU object, in TypeGPU's and WebGPU's types. */
const CREATES: Record<string, GpuObject> = {
  createBuffer: 'buffer',
  createUniform: 'buffer',
  createMutable: 'buffer',
  createReadonly: 'buffer',
  createTexture: 'texture',
  createSampler: 'sampler',
  createComparisonSampler: 'sampler',
  createComputePipeline: 'pipeline',
  createComputePipelineAsync: 'pipeline',
  createRenderPipeline: 'pipeline',
  createRenderPipelineAsync: 'pipeline',
  createQuerySet: 'query set',
  createBindGroup: 'bind group',
};

/** TypeGPU's calls that submit commands of their own, besides an encoder's `submit`. */
const SUBMITS = new Set(['submit', 'generateMipmaps']);

/** Whether `text` names `noun` as a word, singular or plural: `3 bind groups` names `bind group`. */
const mentions = (text: string, noun: string) =>
  new RegExp(`\\b${noun.replace(' ', '\\s+')}s?\\b`, 'i').test(text);

export const repoRoot = fileURLToPath(new URL('../../../../', import.meta.url));

/** What the contracts are read from. */
export interface Workspace {
  /** The package's directory, whose `package.json` exports name `.ts` files. */
  readonly package: string;
  /** A directory of `<name>` examples, each with a `meta.json`. */
  readonly examples: string;
  /** A directory of the examples' pages by the same `<name>`, whose code counts as theirs. */
  readonly pages?: string;
  /** Hosts that run every example, by key such as `hosts/web`: a directory whose own files count. */
  readonly hosts?: Readonly<Record<string, string>>;
  readonly tsconfig: string;
  /** Development-only subpaths, with no module cards. */
  readonly skipped?: readonly string[];
  /**
   * Subpaths vendored from TypeGPU, each read as an entry of its own with its facts in overrides.
   * Their modules keep the package's ids, such as `kit/meshes` for `./geometry`.
   */
  readonly vendored?: readonly string[];
  readonly overrides: Overrides;
}

export const repo: Workspace = {
  package: join(repoRoot, 'packages/typegpu-kit'),
  examples: join(repoRoot, 'packages/typegpu-kit-examples/src'),
  pages: join(repoRoot, 'apps/typegpu-kit-docs/src/examples'),
  hosts: { 'hosts/web': join(repoRoot, 'apps/typegpu-kit-docs/src/examples') },
  tsconfig: join(repoRoot, 'tsconfig.base.json'),
  skipped: ['./testing/dawn', './testing/warnings', './testing/failOnWarnings'],
  vendored: ['./geometry'],
  overrides: repoOverrides,
};

export interface Extraction {
  readonly contracts: ModuleContract[];
  /** Per example, the ids of the modules its code uses. */
  readonly usage: Usage;
  /** Where the sources disagree with the code or with each other. */
  readonly issues: string[];
  /** The program it was read with, which the next extraction reuses for files that haven't changed. */
  readonly program: ts.Program;
}

interface Package {
  /** What code imports its root from: the package, or a vendored subpath. */
  readonly name: string;
  /** The npm package that ships it. */
  readonly package: string;
  readonly dir: string;
  readonly vendored: boolean;
  readonly entries: readonly { readonly subpath: string; readonly file: string }[];
}

/** A module's id: the package's short name, also for a vendored subpath, then `name`. */
const idOf = (pkg: Package, name: string) => `${pkg.package.split('/').at(-1)}/${name}`;

interface Module {
  readonly id: string;
  readonly pkg: Package;
  readonly namespace: string;
  /** What code imports it from: the package, or the subpath that holds it. */
  readonly specifier: string;
  /** Its file, or the package's entry when the whole package is one module. */
  readonly file: string;
  /** Exported names and the symbols they stand for. */
  readonly exports: ReadonlyMap<string, ts.Symbol>;
  /** Exported with `export * as`, so imported as a namespace whatever its tags say. */
  readonly namespaced: boolean;
  /** Imported by name from the package's root, as `memo` is. */
  readonly flat: boolean;
}

/** A parameter of a factory, or of a method that returns recorders, such as `renderer(camera)`. */
interface Taken {
  readonly type: ts.Type;
  readonly name: string;
  readonly method?: string;
}

interface Analysis {
  readonly factories: { name: string; signature: string; declared: ts.Signature }[];
  readonly returns: ts.Type[];
  /** Whether some of its factories take the root, and whether all of them do. */
  readonly takesRoot: boolean;
  readonly allTakeRoot: boolean;
  readonly records: { method: string; into: Recording }[];
  readonly injectable: boolean;
  readonly lifecycle: ModuleContract['lifecycle'];
  /** Parameters of the factories, and of methods that return recorders. */
  readonly parameters: Taken[];
  /** The module's block, then each factory's own JSDoc. */
  readonly docs: Doc[];
  /** The tags of each factory's own JSDoc, by its name. */
  readonly factoryTags: Map<string, Doc['tags']>;
  /** Where its factories' JSDoc comments start, whose tags it reads. */
  readonly read: { file: string; pos: number }[];
  readonly blockTags: Doc['tags'];
  readonly summary: string;
  readonly groups: Record<ExportGroup, string[]>;
  readonly structs: Struct[];
  readonly types: string[];
  readonly options: Options[];
  readonly handles: Handle[];
  readonly creates: boolean;
  /** GPU objects its code creates, and the files of other modules' factories it calls with the root. */
  readonly made: Set<GpuObject>;
  readonly nested: Set<string>;
  readonly submits: boolean;
  readonly features: Set<string>;
}

/** The package's entries, then each vendored subpath as an entry of its own. */
function readPackage(workspace: Workspace): Package[] {
  const dir = workspace.package;
  const json = JSON.parse(readFileSync(join(dir, 'package.json'), 'utf8')) as {
    name: string;
    exports?: Record<string, string>;
  };
  const entries = Object.entries(json.exports ?? {})
    .filter(([subpath, file]) => file.endsWith('.ts') && !workspace.skipped?.includes(subpath))
    .map(([subpath, file]) => ({ subpath, file: join(dir, file) }));
  const vendored = workspace.vendored ?? [];
  const pkg: Package = {
    name: json.name,
    package: json.name,
    dir,
    vendored: false,
    entries: entries.filter((entry) => !vendored.includes(entry.subpath)),
  };

  return [
    pkg,
    ...entries
      .filter((entry) => vendored.includes(entry.subpath))
      .map((entry) => ({
        ...pkg,
        name: `${pkg.name}${entry.subpath.slice(1)}`,
        vendored: true,
        entries: [{ subpath: '.', file: entry.file }],
      })),
  ];
}

const inNodeModules = (node: ts.Node) => node.getSourceFile().fileName.includes('/node_modules/');

const parts = (type: ts.Type) => (type.isUnionOrIntersection() ? type.types : [type]);
const symbolOf = (type: ts.Type) => type.aliasSymbol ?? type.getSymbol();
const isDispose = (property: ts.Symbol) => /^__@dispose@\d+$/.test(String(property.escapedName));

const foreignSymbol = (symbol: ts.Symbol | undefined) =>
  !!symbol?.declarations?.length && symbol.declarations.every(inNodeModules);

/** Declared outside the packages, as TypeGPU's and WebGPU's types are, in whole or in part. */
const foreign = (type: ts.Type): boolean =>
  type.isUnionOrIntersection() ? type.types.some(foreign) : foreignSymbol(symbolOf(type));

const fromTypegpu = (type: ts.Type) =>
  parts(type).some((part) =>
    symbolOf(part)?.declarations?.some((each) =>
      each.getSourceFile().fileName.includes('/node_modules/typegpu/'),
    ),
  );

/** An object of the packages' own, whose members are worth reading. */
const own = (type: ts.Type) =>
  !foreign(type) &&
  type.getCallSignatures().length === 0 &&
  parts(type).some((part) => (part.flags & ts.TypeFlags.Object) !== 0);

const isGpu = (body: ts.Node | undefined) => {
  const first = body && ts.isBlock(body) ? body.statements[0] : undefined;
  return (
    !!first &&
    ts.isExpressionStatement(first) &&
    ts.isStringLiteral(first.expression) &&
    first.expression.text === 'use gpu'
  );
};

/** The function a symbol declares, and whether it is a `'use gpu'` one. */
function functionOf(symbol: ts.Symbol) {
  const declaration = symbol.valueDeclaration;
  if (!declaration) return undefined;

  if (ts.isFunctionDeclaration(declaration)) {
    const implementation = symbol.declarations?.find(
      (each) => ts.isFunctionDeclaration(each) && !!each.body,
    ) as ts.FunctionDeclaration | undefined;
    return { declaration, gpu: isGpu(implementation?.body) };
  }

  const initializer = ts.isVariableDeclaration(declaration) ? declaration.initializer : undefined;
  if (initializer && (ts.isArrowFunction(initializer) || ts.isFunctionExpression(initializer))) {
    return { declaration: initializer, gpu: isGpu(initializer.body) };
  }

  return undefined;
}

/** Where a field's function runs, from its `@gpu` or `@cpu` tag where it is typed. */
const runsOf = (property: ts.Symbol) =>
  (property.declarations ?? []).flatMap((declaration) =>
    ts
      .getJSDocTags(declaration)
      .flatMap(({ tagName: { text } }) => (text === 'gpu' || text === 'cpu' ? [text] : [])),
  );

/** A parameter's name as its declaration spells it. */
function parameterName(parameter: ts.Symbol) {
  const declaration = parameter.valueDeclaration;
  return declaration && ts.isParameter(declaration) && ts.isIdentifier(declaration.name)
    ? declaration.name.text
    : parameter.name;
}

/** Whether a declaration's JSDoc keeps it off the docs with `@hidden`. */
const hidden = (symbol: ts.Symbol) =>
  !!symbol.declarations?.some((declaration) =>
    ts.getJSDocTags(declaration).some((tag) => tag.tagName.text === 'hidden'),
  );

/** Every JSDoc comment in a file, once each, by where it starts. */
function jsdocsOf(source: ts.SourceFile) {
  const text = source.getFullText();
  const found = new Map<number, Doc>();

  const visit = (node: ts.Node) => {
    for (const range of ts.getLeadingCommentRanges(text, node.pos) ?? []) {
      const comment = text.slice(range.pos, range.end);
      if (!found.has(range.pos) && comment.startsWith('/**')) {
        found.set(range.pos, parseDoc(comment));
      }
    }
    ts.forEachChild(node, visit);
  };

  visit(source);
  return found;
}

const hasTag = (doc: Doc, name: string) => doc.tags.some((tag) => tag.name === name);

/**
 * Reads every public module's contract from the packages' sources, with the repo's TypeScript, and
 * the examples that use each.
 */
export function extractContracts(workspace: Workspace = repo, previous?: Extraction): Extraction {
  const issues: string[] = [];
  const packages = readPackage(workspace);
  const { overrides } = workspace;

  const config = ts.readConfigFile(workspace.tsconfig, ts.sys.readFile);
  const configured = ts.convertCompilerOptionsFromJson(
    config.config.compilerOptions,
    dirname(workspace.tsconfig),
  ).options;
  const compilerOptions = { ...configured, noEmit: true, skipLibCheck: true };

  // TypeGPU and the kit as the examples import them: the types the contracts read by identity
  const resolved = (specifier: string) => {
    const from = join(workspace.examples, 'index.ts');
    const file = ts.resolveModuleName(specifier, from, compilerOptions, ts.sys).resolvedModule
      ?.resolvedFileName;
    if (!file) throw new Error(`The contracts cannot resolve ${specifier} from the examples.`);
    return file;
  };
  const typegpuFile = resolved('typegpu');
  const kitFile = resolved('@typegpu/kit');

  // A file whose text hasn't changed keeps the source the previous program parsed
  const host = ts.createCompilerHost(compilerOptions);
  const parse = host.getSourceFile.bind(host);
  host.getSourceFile = (fileName, language, onError, create) => {
    const old = previous?.program.getSourceFile(fileName);
    return old && old.text === host.readFile(fileName)
      ? old
      : parse(fileName, language, onError, create);
  };

  const program = ts.createProgram({
    rootNames: [
      ...packages.flatMap((pkg) => pkg.entries.map((entry) => entry.file)),
      typegpuFile,
      kitFile,
    ],
    options: compilerOptions,
    host,
    ...(previous ? { oldProgram: previous.program } : {}),
  });
  const checker = program.getTypeChecker();

  const sourceOf = (file: string) => {
    const source = program.getSourceFile(file);
    if (!source) throw new Error(`The contracts program has no ${file}.`);
    return source;
  };

  const resolve = (symbol: ts.Symbol) =>
    symbol.flags & ts.SymbolFlags.Alias ? checker.getAliasedSymbol(symbol) : symbol;

  const moduleSymbolOf = (file: string) => {
    const symbol = checker.getSymbolAtLocation(sourceOf(file));
    if (!symbol) throw new Error(`${file} is not a module.`);
    return symbol;
  };

  const exportsOf = (moduleSymbol: ts.Symbol) =>
    new Map(checker.getExportsOfModule(moduleSymbol).map((s) => [s.name, resolve(s)] as const));

  const fileOfSpecifier = (specifier: ts.Expression) =>
    checker.getSymbolAtLocation(specifier)?.valueDeclaration?.getSourceFile().fileName;

  const exported = (from: ts.Symbol, name: string, where: string) => {
    const symbol = exportsOf(from).get(name);
    if (!symbol) throw new Error(`${where} exports no ${name}, which the contracts read.`);
    return symbol;
  };

  // The types the contracts recognise, by symbol identity

  const typegpu = moduleSymbolOf(typegpuFile);
  const tgpu = (name: string) => exported(typegpu, name, 'typegpu');
  const data = tgpu('d');
  const dataType = (name: string) =>
    checker.getDeclaredTypeOfSymbol(exported(data, name, "typegpu's d"));
  const kit = moduleSymbolOf(kitFile);
  const kitPart = (namespace: string, name: string) =>
    exported(exported(kit, namespace, '@typegpu/kit'), name, `@typegpu/kit's ${namespace}`);

  const rootSymbol = tgpu('TgpuRoot');
  const frame = kitPart('frames', 'Frame');
  const wait = kitPart('coroutines', 'Wait');
  const layoutSymbol = kitPart('bindings', 'Layout');
  const outputSymbol = kitPart('passes', 'Output');
  const renderPass = checker.getDeclaredTypeOfSymbol(tgpu('TgpuRenderPass'));
  const computePass = checker.getDeclaredTypeOfSymbol(tgpu('TgpuComputePass'));
  const anyWgslData = dataType('AnyWgslData');
  const numeric = ['AnyVecInstance', 'm2x2f', 'm3x3f', 'm4x4f'].map(dataType);

  const recordings = new Map<ts.Symbol, Recording>([
    [tgpu('TgpuComputePass'), 'compute'],
    [tgpu('TgpuRenderPass'), 'render'],
    [tgpu('TgpuRenderCommands'), 'render'],
    [tgpu('TgpuCommandEncoder'), 'encoder'],
    [frame, 'frame'],
  ]);
  const gpuFunctions = new Set([tgpu('TgpuFn'), tgpu('TgpuGenericFn'), tgpu('TgpuComptime')]);
  const accessorSymbols = new Set([
    tgpu('TgpuAccessor'),
    tgpu('TgpuMutableAccessor'),
    tgpu('TgpuSlot'),
  ]);
  const filledSymbols = new Set([...accessorSymbols, tgpu('TgpuBindGroupLayout'), layoutSymbol]);

  /** Whether a type, or a part of it, is one of `symbols`. */
  const isOneOf = (type: ts.Type, symbols: ReadonlySet<ts.Symbol>) =>
    [type, ...parts(type)].some((part) => {
      const symbol = symbolOf(part);
      return !!symbol && symbols.has(resolve(symbol));
    });

  const isRoot = (type: ts.Type) => isOneOf(type, new Set([rootSymbol]));

  // The modules, and what each import specifier's names are

  const modules: Module[] = [];
  const imports = new Map<string, Map<string, string>>();
  const importsOf = (specifier: string) => {
    const names = imports.get(specifier) ?? new Map<string, string>();
    imports.set(specifier, names);
    return names;
  };

  for (const pkg of packages) {
    const entry = pkg.entries.find((each) => each.subpath === '.');
    if (!entry) throw new Error(`${pkg.name} has no root export.`);

    const say = (issue: string) => issues.push(`${pkg.name}: ${issue}`);
    const entrySource = sourceOf(entry.file);
    const rootNames = importsOf(pkg.name);
    const added: Module[] = [];
    const add = (module: Module) => {
      const twin = [...modules, ...added].find((other) => other.id === module.id);
      if (twin) {
        throw new Error(
          `${module.id} is both ${relative(pkg.dir, twin.file)} and ${relative(pkg.dir, module.file)}; module ids take a subpath's last segment, so rename one.`,
        );
      }
      added.push(module);
    };

    // Namespaces, and the names the root exports flat, by the file they come from
    const groups = new Map<string, Map<string, ts.Symbol>>();
    const groupOf = (file: string) => {
      const group = groups.get(file) ?? new Map<string, ts.Symbol>();
      groups.set(file, group);
      return group;
    };

    for (const statement of entrySource.statements) {
      if (!ts.isExportDeclaration(statement) || !statement.moduleSpecifier) continue;
      const target = checker.getSymbolAtLocation(statement.moduleSpecifier);
      const file = fileOfSpecifier(statement.moduleSpecifier);
      if (!target || !file) continue;

      const clause = statement.exportClause;
      if (clause && ts.isNamespaceExport(clause)) {
        const namespace = clause.name.text;
        add({
          id: idOf(pkg, namespace),
          pkg,
          namespace,
          specifier: pkg.name,
          file,
          exports: exportsOf(target),
          namespaced: true,
          flat: false,
        });
        rootNames.set(namespace, idOf(pkg, namespace));
        continue;
      }

      const group = groupOf(file);
      if (clause && ts.isNamedExports(clause)) {
        for (const element of clause.elements) {
          const symbol = checker.getSymbolAtLocation(element.name);
          if (symbol) group.set(element.name.text, resolve(symbol));
        }
      } else {
        for (const [name, symbol] of exportsOf(target)) group.set(name, symbol);
      }
    }

    for (const [name, symbol] of exportsOf(moduleSymbolOf(entry.file))) {
      if (symbol.declarations?.some((each) => each.getSourceFile() === entrySource)) {
        groupOf(entry.file).set(name, symbol);
      }
    }

    const subpaths = pkg.entries
      .filter((each) => each.subpath !== '.')
      .map((each) => ({
        ...each,
        specifier: `${pkg.name}${each.subpath.slice(1)}`,
        name: each.subpath.split('/').at(-1)!,
      }));
    const typesOnly: [string, Map<string, ts.Symbol>][] = [];

    // Values the root exports flat beside namespaces: a module per file they come from, named
    // by its subpath, or by the export if there is one. A vendored subpath's flat names are
    // those its overrides gather into modules, by the files they come from
    const gathered = new Map<string, string>();
    for (const [namespace, files] of Object.entries(overrides.flat?.[pkg.name] ?? {})) {
      for (const file of files) gathered.set(join(dirname(entry.file), file), namespace);
    }
    const flatExports = new Map<string, Map<string, ts.Symbol>>();

    for (const [file, group] of groups) {
      if (added.some((module) => module.file === file)) continue;

      const gatheredAs = pkg.vendored ? gathered.get(file) : undefined;
      if (gatheredAs) {
        const id = idOf(pkg, gatheredAs);
        const known = flatExports.get(id);
        if (known) {
          for (const [name, symbol] of group) known.set(name, symbol);
          continue;
        }
        const exports = new Map(group);
        flatExports.set(id, exports);
        add({
          id,
          pkg,
          namespace: gatheredAs,
          specifier: pkg.name,
          file,
          exports,
          namespaced: false,
          flat: true,
        });
        continue;
      }

      if (pkg.vendored) {
        say(
          `exports ${[...group.keys()].join(', ')} at its root, which no module covers; gather them into one in its overrides' flat.`,
        );
        continue;
      }

      const values = [...group].filter(([, symbol]) => symbol.flags & ts.SymbolFlags.Value);
      if (values.length === 0) {
        typesOnly.push([file, group]);
        continue;
      }

      const stem = basename(file, '.ts');
      const namespace =
        subpaths.find((each) => each.file === file)?.name ??
        (values.length === 1 ? values[0][0] : stem === 'index' ? basename(dirname(file)) : stem);
      add({
        id: idOf(pkg, namespace),
        pkg,
        namespace,
        specifier: pkg.name,
        file,
        exports: group,
        namespaced: false,
        flat: true,
      });
    }

    const wrappers = new Map<string, Map<string, ts.Symbol>>();
    for (const subpath of subpaths) {
      const source = sourceOf(subpath.file);
      const namespaces = source.statements.filter(
        (statement) =>
          ts.isExportDeclaration(statement) &&
          statement.exportClause &&
          ts.isNamespaceExport(statement.exportClause),
      );
      if (namespaces.length === 0) continue;

      const names = importsOf(subpath.specifier);
      for (const statement of namespaces) {
        if (
          !ts.isExportDeclaration(statement) ||
          !statement.exportClause ||
          !ts.isNamespaceExport(statement.exportClause) ||
          !statement.moduleSpecifier
        )
          continue;
        const target = checker.getSymbolAtLocation(statement.moduleSpecifier);
        const file = fileOfSpecifier(statement.moduleSpecifier);
        if (!target || !file) continue;
        const namespace = statement.exportClause.name.text;
        const id = idOf(pkg, namespace);
        add({
          id,
          pkg,
          namespace,
          specifier: subpath.specifier,
          file,
          exports: exportsOf(target),
          namespaced: true,
          flat: false,
        });
        names.set(namespace, id);
      }
      wrappers.set(subpath.specifier, exportsOf(moduleSymbolOf(subpath.file)));
    }

    for (const subpath of subpaths) {
      if (wrappers.has(subpath.specifier)) continue;
      const owner = added.find((module) => module.file === subpath.file);
      const id = owner?.id ?? idOf(pkg, subpath.name);
      const exports = owner?.exports ?? exportsOf(moduleSymbolOf(subpath.file));

      if (!owner) {
        add({
          id,
          pkg,
          namespace: subpath.name,
          specifier: subpath.specifier,
          file: subpath.file,
          exports,
          namespaced: false,
          flat: false,
        });
      }

      const names = importsOf(subpath.specifier);
      names.set('*', id);
      for (const name of exports.keys()) names.set(name, id);
    }

    for (const [specifier, exports] of wrappers) {
      const names = importsOf(specifier);
      for (const [name, symbol] of exports) {
        const owner = added.find((module) => [...module.exports.values()].includes(symbol));
        if (owner) names.set(name, owner.id);
      }
    }

    // Names the root exports flat belong to the module whose file they come from, or which
    // exports them too, as `@typegpu/kit/debug` exports its recorder's types
    const ownerOf = (file: string, group: ReadonlyMap<string, ts.Symbol>) =>
      added.find((module) => module.file === file) ??
      added.find((module) => {
        const symbols = new Set(module.exports.values());
        return [...group.values()].every((symbol) => symbols.has(symbol));
      });
    for (const [file, group] of typesOnly) {
      if (!ownerOf(file, group)) {
        say(
          `exports the types ${[...group.keys()].join(', ')} at its root from ${relative(pkg.dir, file)}, which no module exports.`,
        );
      }
    }
    for (const [file, group] of groups) {
      const owner = ownerOf(file, group);
      if (owner) for (const name of group.keys()) rootNames.set(name, owner.id);
    }

    modules.push(...added);
  }

  const namespaces = new Map<string, string>();
  for (const module of modules) {
    const other = namespaces.get(module.namespace);
    if (other) {
      issues.push(
        `${other} and ${module.id} are both called ${module.namespace}; their cards would share an anchor.`,
      );
    }
    namespaces.set(module.namespace, module.id);
  }

  const recordingOf = (type: ts.Type): Recording | undefined => {
    for (const part of parts(type)) {
      const symbol = symbolOf(part);
      const into = symbol && recordings.get(resolve(symbol));
      if (into) return into;
    }

    // Something shaped like a pass, such as the live lists' `IndirectDrawer`
    const methods = checker
      .getPropertiesOfType(type)
      .filter((property) => checker.getTypeOfSymbol(property).getCallSignatures().length > 0);
    if (!own(type) || methods.length === 0) return undefined;
    if (checker.isTypeAssignableTo(renderPass, type)) return 'render';
    if (checker.isTypeAssignableTo(computePass, type)) return 'compute';
    return undefined;
  };

  /** What a method records into, from its first required parameter that says. */
  const methodRecording = (signatures: readonly ts.Signature[]) => {
    for (const signature of signatures) {
      // A method that returns something, such as a surface's `view(frame)`, reads the frame
      // rather than recording into it
      const returned = checker.getReturnTypeOfSignature(signature);
      if (!(returned.flags & (ts.TypeFlags.Void | ts.TypeFlags.Undefined))) continue;

      for (const parameter of signature.parameters) {
        const declaration = parameter.valueDeclaration;
        if (!declaration || !ts.isParameter(declaration)) continue;
        if (declaration.dotDotDotToken || checker.isOptionalParameter(declaration)) continue;

        const into = recordingOf(checker.getTypeOfSymbol(parameter));
        if (into) return into;
      }
    }
    return undefined;
  };

  const waitMembers = new Set(
    wait.declarations?.flatMap((each) =>
      ts.isInterfaceDeclaration(each) ? each.members.map((member) => member) : [],
    ),
  );
  const skipped = (property: ts.Symbol) =>
    foreignSymbol(property) ||
    !!property.declarations?.some((each) => waitMembers.has(each as ts.TypeElement));

  const parameterText = (parameter: ts.Symbol) => {
    const declaration = parameter.valueDeclaration;
    if (!declaration || !ts.isParameter(declaration)) return parameter.name;

    const name = ts.isIdentifier(declaration.name) ? declaration.name.text : '{…}';
    if (declaration.dotDotDotToken) return `...${name}`;
    return checker.isOptionalParameter(declaration) ? `${name}?` : name;
  };

  const stringOf = (expression: ts.Expression | undefined): string | undefined => {
    let inner = expression;
    while (
      inner &&
      (ts.isParenthesizedExpression(inner) ||
        ts.isAsExpression(inner) ||
        ts.isSatisfiesExpression(inner) ||
        ts.isNonNullExpression(inner))
    ) {
      inner = inner.expression;
    }

    if (!inner) return undefined;
    if (ts.isStringLiteralLike(inner)) return inner.text;
    if (!ts.isIdentifier(inner)) return undefined;

    const declaration = checker.getSymbolAtLocation(inner)?.valueDeclaration;
    return declaration && ts.isVariableDeclaration(declaration)
      ? stringOf(declaration.initializer)
      : undefined;
  };

  // A module's own files: its file and the files it imports that no other module is
  const moduleFiles = new Set(modules.map((module) => module.file));
  const filesOf = (module: Module) => {
    const files = new Set([module.file]);
    const queue = [module.file];

    for (let file = queue.pop(); file; file = queue.pop()) {
      for (const statement of sourceOf(file).statements) {
        const specifier =
          ts.isImportDeclaration(statement) || ts.isExportDeclaration(statement)
            ? statement.moduleSpecifier
            : undefined;
        if (!specifier || !ts.isStringLiteral(specifier) || !specifier.text.startsWith('.')) {
          continue;
        }

        const next = fileOfSpecifier(specifier);
        if (!next || files.has(next) || moduleFiles.has(next)) continue;
        files.add(next);
        queue.push(next);
      }
    }

    return files;
  };

  // Every JSDoc in the packages' sources, and each module's block among them

  const jsdocs = new Map<string, Map<number, Doc>>();
  const docsIn = (file: string) => {
    const known = jsdocs.get(file) ?? jsdocsOf(sourceOf(file));
    jsdocs.set(file, known);
    return known;
  };

  /** The JSDoc comment in a module's file that carries `@module`. */
  const blockOf = (module: Module) => {
    const blocks = [...docsIn(module.file)].filter(([, doc]) => hasTag(doc, 'module'));
    if (blocks.length > 1) {
      issues.push(`${module.id}: ${blocks.length} blocks carry @module; give it one.`);
    }
    return blocks[0] && { pos: blocks[0][0], doc: blocks[0][1] };
  };

  /** Whether a type is plain data: numbers, strings, symbols, vectors, and arrays and objects of them. */
  const isData = (type: ts.Type, depth = 0): boolean => {
    if (depth > 4 || type.flags & (ts.TypeFlags.Any | ts.TypeFlags.Unknown)) return false;
    const plain =
      ts.TypeFlags.NumberLike |
      ts.TypeFlags.StringLike |
      ts.TypeFlags.BooleanLike |
      ts.TypeFlags.BigIntLike |
      ts.TypeFlags.ESSymbolLike |
      ts.TypeFlags.EnumLike |
      ts.TypeFlags.Null |
      ts.TypeFlags.Undefined;
    if (type.flags & plain) return true;
    if (type.isUnion()) return type.types.every((part) => isData(part, depth));
    if (!(type.flags & ts.TypeFlags.Object)) return false;
    if (numeric.some((each) => checker.isTypeAssignableTo(type, each))) return true;
    if (type.getCallSignatures().length > 0 || type.getConstructSignatures().length > 0) {
      return false;
    }
    if (checker.isArrayType(type) || checker.isTupleType(type)) {
      return checker
        .getTypeArguments(type as ts.TypeReference)
        .every((each) => isData(each, depth + 1));
    }

    const properties = checker.getPropertiesOfType(type);
    return (
      properties.length > 0 &&
      properties.every((property) => isData(checker.getTypeOfSymbol(property), depth + 1))
    );
  };

  /** Where a value export goes on the card, beside the factories. */
  const groupOf = (symbol: ts.Symbol): ExportGroup | undefined => {
    if (symbol.flags & ts.SymbolFlags.Module) return 'helpers';

    const type = checker.getTypeOfSymbol(symbol);
    if (isOneOf(type, gpuFunctions)) return 'functions';
    if (isOneOf(type, accessorSymbols)) return 'accessors';
    if (isOneOf(type, new Set([outputSymbol]))) return 'outputs';
    if (checker.isTypeAssignableTo(type, anyWgslData)) return 'schemas';
    if (isData(type)) return 'constants';
    if (isOneOf(type, filledSymbols) || !foreign(type)) return 'helpers';
    return undefined;
  };

  /** The type an options parameter names, `Options` or `…Options`, when the module declares it. */
  const optionsTypeOf = (parameter: ts.Symbol, files: ReadonlySet<string>) => {
    const declaration = parameter.valueDeclaration;
    if (!declaration || !ts.isParameter(declaration) || !declaration.type) return undefined;
    return optionsTypeIn(declaration.type, files);
  };

  /** The `Options` or `…Options` type a type node names, when the module declares it. */
  const optionsTypeIn = (written: ts.TypeNode, files: ReadonlySet<string>) => {
    let found: ts.Symbol | undefined;
    const visit = (node: ts.Node) => {
      if (found) return;
      if (ts.isTypeReferenceNode(node)) {
        const name = ts.isIdentifier(node.typeName) ? node.typeName : node.typeName.right;
        const symbol = name.text.endsWith('Options') && checker.getSymbolAtLocation(name);
        const target = symbol && resolve(symbol);
        const file = target && target.declarations?.[0]?.getSourceFile().fileName;
        const typeParameter = target && target.flags & ts.SymbolFlags.TypeParameter;
        if (target && file && files.has(file) && !typeParameter) {
          found = target;
          return;
        }
      }
      ts.forEachChild(node, visit);
    };

    visit(written);
    return found;
  };

  /** The fields of an options type whose type is another options type the module declares. */
  const nestedOptionsOf = (symbol: ts.Symbol, files: ReadonlySet<string>) =>
    checker.getPropertiesOfType(checker.getDeclaredTypeOfSymbol(symbol)).flatMap((property) => {
      const declaration = property.valueDeclaration ?? property.declarations?.[0];
      const written =
        declaration &&
        (ts.isPropertySignature(declaration) || ts.isPropertyDeclaration(declaration)) &&
        declaration.type;
      const nested = written && optionsTypeIn(written, files);
      return nested ? [{ field: property.name, symbol: nested }] : [];
    });

  /**
   * Requires a field of an options type that takes a function, other than TypeGPU's such as a
   * schema, to say where it runs. The vendored geometry keeps its upstream docs.
   */
  const checkRuns = (owner: ts.Symbol, property: ts.Symbol, given: ts.Type) => {
    const declared = property.valueDeclaration ?? property.declarations?.[0];
    const file = declared?.getSourceFile();
    if (!declared || !file || file.fileName.includes('/src/geometry/')) return;

    const runs = runsOf(property);
    // A signature declared in the packages, not one of TypeGPU's, such as a schema's
    const callable = parts(given).some((part) =>
      (part.isTypeParameter() ? (checker.getBaseConstraintOfType(part) ?? part) : part)
        .getCallSignatures()
        .some((signature) => signature.declaration && !inNodeModules(signature.declaration)),
    );
    const line = file.getLineAndCharacterOfPosition(declared.getStart()).line + 1;
    const field = `${owner.name}.${property.name} (${relative(repoRoot, file.fileName)}:${line})`;
    const issue =
      runs.length > 1
        ? `${field} has more than one of @gpu and @cpu.`
        : callable && runs.length === 0
          ? `${field} takes a function: tag where it runs, @gpu in a shader or @cpu on the CPU.`
          : !callable && runs.length > 0
            ? `${field} has @${runs[0]} but takes no function.`
            : undefined;
    if (issue && !issues.includes(issue)) issues.push(issue);
  };

  /**
   * A field of an object type, with its type as written, its default and its doc. A field of an
   * options type `owner` that takes a function says where it runs.
   */
  const fieldOf = (property: ts.Symbol, owner?: ts.Symbol): OptionField[] => {
    const type = checker.getTypeOfSymbol(property);
    const given = checker.getNonNullableType(type);
    if (given.flags & ts.TypeFlags.Never) return [];

    // A field of a union of options types, such as a shadow's `eye`, is `never` in the members
    // that refuse it, so its type is written where a member takes it
    const declarations = property.valueDeclaration
      ? [property.valueDeclaration]
      : (property.declarations ?? []);
    const typed = declarations.flatMap((declaration) =>
      (ts.isPropertySignature(declaration) || ts.isPropertyDeclaration(declaration)) &&
      declaration.type &&
      declaration.type.kind !== ts.SyntaxKind.NeverKeyword
        ? [declaration.type.getText()]
        : [],
    );
    const written = typed[0] ?? checker.typeToString(type);
    if (owner) checkRuns(owner, property, given);
    const { doc, default: fallback } = splitDefault(
      ts.displayPartsToString(property.getDocumentationComment(checker)),
    );

    return [
      {
        name: property.name,
        optional: (property.flags & ts.SymbolFlags.Optional) !== 0,
        // Without the TSDoc of an inline object type's fields
        type: written
          .replace(/\/\*[\s\S]*?\*\//g, '')
          .replace(/\s+/g, ' ')
          .replace(/;\s*}/g, ' }'),
        ...(fallback ? { default: fallback } : {}),
        ...(doc ? { doc } : {}),
        ...(runsOf(property).includes('gpu') ? { gpu: true as const } : {}),
      },
    ];
  };

  /**
   * A struct schema's fields, from the object `d.struct` took: each with its schema as written,
   * such as `d.vec3f` or `cameras.Ray`, and its doc. Nothing for a schema of another kind.
   */
  const structOf = (name: string, symbol: ts.Symbol): Struct[] => {
    const props = checker.getTypeOfSymbol(symbol).getProperty('propTypes');
    if (!props) return [];
    const fields = checker.getPropertiesOfType(checker.getTypeOfSymbol(props)).map((field) => {
      const declared = field.valueDeclaration;
      const type =
        declared && ts.isPropertyAssignment(declared)
          ? declared.initializer
              .getText()
              .replace(/\s+/g, ' ')
              .replace(/,\s*([)}])/g, ' $1')
          : checker.typeToString(checker.getTypeOfSymbol(field));
      const doc = firstSentence(ts.displayPartsToString(field.getDocumentationComment(checker)));
      return { name: field.name, type, ...(doc ? { doc } : {}) };
    });
    return [{ name, fields }];
  };

  /** Every field of an options type. */
  const fieldsOf = (symbol: ts.Symbol): OptionField[] =>
    checker
      .getPropertiesOfType(checker.getDeclaredTypeOfSymbol(symbol))
      .flatMap((property) => fieldOf(property, symbol));

  /**
   * A handle's members in declaration order, but those `@hidden`, internal ones marked `~`, and
   * those of TypeGPU's and WebGPU's types: a method by its short signature, once each, and a value
   * by its type as written. One level into a value of the packages' own, its recording methods, as
   * a lighting's `shadow.record`. `method` sees each method's signatures, by its path.
   */
  const membersOf = (
    type: ts.Type,
    method: (path: string, signatures: readonly ts.Signature[]) => void,
    prefix = '',
  ): Member[] =>
    checker.getPropertiesOfType(type).flatMap((property): Member[] => {
      if (skipped(property) || hidden(property) || isDispose(property)) return [];
      if (property.name.startsWith('~')) return [];

      const path = `${prefix}${property.name}`;
      const held = checker.getNonNullableType(checker.getTypeOfSymbol(property));
      const signatures = held.getCallSignatures();
      if (signatures.length === 0) {
        if (prefix) return [];
        const recorders = own(held)
          ? membersOf(held, method, `${path}.`).filter((member) => member.signature)
          : [];
        const values = fieldOf(property).map((field) => ({
          name: field.name,
          ...(field.optional ? { optional: true } : {}),
          type: field.type,
          ...(field.doc ? { doc: field.doc } : {}),
        }));
        return [...values, ...recorders];
      }
      if (prefix && !methodRecording(signatures)) return [];

      method(path, signatures);
      const doc = firstSentence(ts.displayPartsToString(property.getDocumentationComment(checker)));
      const texts = new Set(
        signatures.map(
          (signature) => `${path}(${signature.parameters.map(parameterText).join(', ')})`,
        ),
      );
      return [...texts].map((signature) => ({
        name: path,
        signature,
        ...(doc ? { doc } : {}),
      }));
    });

  /**
   * A name for what a factory returns: the type's own, or that of a type the module exports for
   * it, such as `type Particles = ReturnType<typeof create>`. A generic alias that adds members
   * to an exported type by its options, such as `LightingOf<O> = Lighting & …`, is named after
   * that type.
   */
  const returnName = (module: Module, signature: ts.Signature) => {
    const returned = checker.getReturnTypeOfSignature(signature);
    const awaited = checker.getAwaitedType(returned) ?? returned;
    if (foreign(awaited)) return undefined;

    const exportedAs = (type: ts.Type) =>
      [...module.exports.values()].find(
        (symbol) =>
          symbol.flags & (ts.SymbolFlags.TypeAlias | ts.SymbolFlags.Interface) &&
          checker.getDeclaredTypeOfSymbol(symbol) === type,
      );
    const refined =
      awaited.aliasTypeArguments && awaited.isIntersection()
        ? awaited.types.map(exportedAs).find((each) => each !== undefined)
        : undefined;
    const named = awaited.aliasSymbol ?? awaited.getSymbol();
    const exportedType = exportedAs(awaited);
    const name = [refined?.name, named?.name, exportedType?.name].find(
      (each) => each && /^[A-Z]\w*$/.test(each),
    );
    if (!name) return undefined;
    return awaited === returned ? name : `Promise<${name}>`;
  };

  const analyze = (module: Module): Analysis => {
    const override = overrides.modules[module.id];
    const helperNames = new Set(override?.helpers ?? []);
    const groups = Object.fromEntries(
      (['functions', 'accessors', 'outputs', 'schemas', 'constants', 'helpers'] as const).map(
        (group) => [group, [] as string[]],
      ),
    ) as Record<ExportGroup, string[]>;
    const types: string[] = [];
    const declared: { name: string; symbol: ts.Symbol; declaration: ts.SignatureDeclaration }[] =
      [];

    for (const [name, symbol] of module.exports) {
      if (hidden(symbol)) continue;
      if (!(symbol.flags & ts.SymbolFlags.Value)) {
        types.push(name);
        continue;
      }

      const found = functionOf(symbol);
      if (found?.gpu) groups.functions.push(name);
      else if (found && helperNames.has(name)) groups.helpers.push(name);
      else if (found) declared.push({ name, symbol, declaration: found.declaration });
      else {
        const group = groupOf(symbol);
        if (group) groups[group].push(name);
        else {
          issues.push(
            `${module.id}: exports ${name}, which is no factory, GPU function, accessor, output, schema, constant or helper; mark it @hidden or teach the contracts its kind.`,
          );
        }
      }
    }

    for (const name of helperNames) {
      if (!groups.helpers.includes(name)) {
        issues.push(
          `${module.id}: its overrides list ${name} as a helper, a function it doesn't export.`,
        );
      }
    }

    // `create` first, then the module's own file, each in export order
    const rank = (each: (typeof declared)[number]) =>
      each.name === 'create'
        ? 0
        : each.declaration.getSourceFile().fileName === module.file
          ? 1
          : 2;
    declared.sort((a, b) => rank(a) - rank(b));

    const factories: Analysis['factories'] = [];
    const returns: ts.Type[] = [];
    const parameters: Taken[] = [];
    let takesRoot = false;
    let allTakeRoot = true;

    for (const { name, symbol } of declared) {
      for (const signature of checker.getTypeOfSymbol(symbol).getCallSignatures()) {
        const text = `${name}(${signature.parameters.map(parameterText).join(', ')})`;
        // Overloads with the same parameters, such as an options object or another, are each listed
        const typed = `${name}${checker.signatureToString(signature)}`;
        const listed = (factory: (typeof factories)[number]) =>
          `${factory.name}${checker.signatureToString(factory.declared)}` === typed;
        if (!factories.some(listed)) {
          factories.push({ name, signature: text, declared: signature });
        }

        let rooted = false;
        for (const parameter of signature.parameters) {
          const type = checker.getTypeOfSymbol(parameter);
          parameters.push({ type, name: parameterName(parameter) });
          rooted ||= isRoot(type);
        }
        takesRoot ||= rooted;
        allTakeRoot &&= rooted;

        const returned = checker.getReturnTypeOfSignature(signature);
        returns.push(checker.getAwaitedType(returned) ?? returned);
      }
    }

    // The options types its factories take, each once, with every field, and the options types
    // their fields take, such as a lighting's `shadow`
    const files = filesOf(module);
    const tables = new Map<ts.Symbol, Options>();
    const tabulate = (symbol: ts.Symbol, name: string, parameter: string) => {
      const known = tables.get(symbol);
      if (known && !known.takenBy.includes(name)) known.takenBy.push(name);
      if (known) return;

      tables.set(symbol, {
        name: symbol.name,
        takenBy: [name],
        parameter,
        fields: fieldsOf(symbol),
      });
      for (const nested of nestedOptionsOf(symbol, files)) {
        tabulate(nested.symbol, name, `${parameter}.${nested.field}`);
      }
    };
    for (const { name, declared: signature } of factories) {
      for (const parameter of signature.parameters) {
        const symbol = optionsTypeOf(parameter, files);
        if (symbol) tabulate(symbol, name, parameterName(parameter));
      }
    }

    // What the factories return, each named type once with its members, and the options types
    // its methods take, as `Finish.record`
    const handles = new Map<string, Handle>();
    for (const { name, declared: signature } of factories) {
      const returned = checker.getReturnTypeOfSignature(signature);
      const type = checker.getAwaitedType(returned) ?? returned;
      const handle = returnName(module, signature)?.replace(/^Promise<(.*)>$/, '$1');
      // Callable ones too, such as an owner, by their members
      const object = parts(type).some((part) => (part.flags & ts.TypeFlags.Object) !== 0);
      if (!handle || !object) continue;

      const known = handles.get(handle);
      if (known) {
        if (!known.from.includes(name)) known.from.push(name);
        continue;
      }
      const members = membersOf(type, (path, signatures) => {
        for (const parameter of signatures.flatMap((each) => each.parameters)) {
          const symbol = optionsTypeOf(parameter, files);
          if (symbol) tabulate(symbol, `${handle}.${path}`, parameterName(parameter));
        }
      });
      if (members.length > 0) handles.set(handle, { name: handle, from: [name], members });
    }
    const options = [...tables.values()];

    // Recording methods of what the factories return, one level into members and into methods
    // that return recorders, such as `renderer()`, but never into TypeGPU's own types
    const records = new Map<string, Recording>();
    let injectable = false;
    const lifecycle = { initAsync: false, destroy: false, dispose: false };

    const visit = (type: ts.Type, prefix: string, depth: number) => {
      for (const property of checker.getPropertiesOfType(type)) {
        if (skipped(property)) continue;

        const propertyType = checker.getNonNullableType(checker.getTypeOfSymbol(property));
        const signatures = propertyType.getCallSignatures();
        if (signatures.length === 0) {
          if (depth === 0 && own(propertyType)) visit(propertyType, `${property.name}.`, 1);
          continue;
        }

        const into = methodRecording(signatures);
        if (into) {
          records.set(`${prefix}${property.name}`, into);
          continue;
        }

        if (depth > 0) continue;
        for (const signature of signatures) {
          const returned = checker.getReturnTypeOfSignature(signature);
          if (!own(returned)) continue;

          const before = records.size;
          visit(returned, `${property.name}().`, 1);
          if (records.size === before) continue;

          lifecycle.initAsync ||= !!checker.getPropertyOfType(returned, 'initAsync');
          for (const parameter of signature.parameters) {
            parameters.push({
              type: checker.getTypeOfSymbol(parameter),
              name: parameterName(parameter),
              method: property.name,
            });
          }
        }
      }
    };

    for (const returned of returns) {
      if (foreign(returned)) continue;
      visit(returned, '', 0);

      const properties = checker.getPropertiesOfType(returned);
      injectable ||= properties.some((property) => property.name === 'inject');
      lifecycle.initAsync ||= properties.some((property) => property.name === 'initAsync');
      lifecycle.destroy ||= properties.some((property) => property.name === 'destroy');
      lifecycle.dispose ||= properties.some(isDispose);
    }

    // What the code does: creates GPU objects, submits, checks device features
    let creates = false;
    let submits = false;
    const features = new Set<string>();
    const made = new Set<GpuObject>();
    const nested = new Set<string>();
    const publicFiles = new Set([
      module.file,
      ...[...module.exports.values()].flatMap((symbol) =>
        (symbol.declarations ?? []).map((declaration) => declaration.getSourceFile().fileName),
      ),
    ]);

    const read = (node: ts.Node) => {
      if (ts.isCallExpression(node)) {
        const callee = node.expression;
        if (
          ts.isPropertyAccessExpression(callee) &&
          callee.name.text === 'has' &&
          ts.isPropertyAccessExpression(callee.expression) &&
          callee.expression.name.text === 'features' &&
          foreignSymbol(checker.getSymbolAtLocation(callee.expression.name))
        ) {
          const argument = node.arguments[0];
          const type = argument && checker.getTypeAtLocation(argument);
          const feature = stringOf(argument) ?? (type?.isStringLiteral() ? type.value : undefined);

          // One typed as any feature is the caller's, such as one of `optionalFeatures`
          if (feature) features.add(feature);
          else if (!type?.isUnion())
            issues.push(`${module.id}: can't read the feature in '${node.getText()}'.`);
        }

        const signature = checker.getResolvedSignature(node);
        const declaration = signature?.getDeclaration();
        const name = declaration && ts.getNameOfDeclaration(declaration as ts.Declaration);
        const text = name && ts.isIdentifier(name) ? name.text : '';

        if (declaration && inNodeModules(declaration)) {
          // Dependencies can submit in modes this module never exposes
          submits ||= SUBMITS.has(text) && publicFiles.has(node.getSourceFile().fileName);
          creates ||= text.startsWith('create');
          if (CREATES[text]) made.add(CREATES[text]);
        }

        const rooted = !!signature?.parameters.some((each) =>
          isRoot(checker.getTypeOfSymbol(each)),
        );
        creates ||= rooted;
        if (rooted && declaration) nested.add(declaration.getSourceFile().fileName);
      }

      ts.forEachChild(node, read);
    };

    for (const file of files) read(sourceOf(file));

    // Docs: the module's block, then each factory's own JSDoc
    const block = blockOf(module);
    const factoryNodes = declared.map(({ name, symbol }) => ({
      name,
      nodes: (symbol.declarations ?? [])
        .flatMap((declaration) => ts.getJSDocCommentsAndTags(declaration))
        .filter(ts.isJSDoc),
    }));
    const factoryDocs = factoryNodes.map(({ name, nodes }) => ({
      name,
      docs: nodes.map((node) => parseDoc(node.getText())).filter((doc) => !hasTag(doc, 'module')),
    }));

    const factoryTags = new Map<string, Doc['tags']>();
    for (const { name, docs } of factoryDocs) {
      factoryTags.set(name, [...(factoryTags.get(name) ?? []), ...docs.flatMap((doc) => doc.tags)]);
    }

    const allDocs = factoryDocs.flatMap((each) => each.docs);
    const summary =
      override?.summary ??
      (block?.doc.description || allDocs.find((doc) => doc.description)?.description || '');

    const tabled = new Set([...options, ...handles.values()].map((table) => table.name));
    const structs = groups.schemas.flatMap((name) => structOf(name, module.exports.get(name)!));

    return {
      factories,
      returns,
      takesRoot,
      allTakeRoot,
      records: [...records].map(([method, into]) => ({ method, into })),
      injectable,
      lifecycle,
      parameters,
      docs: [...(block ? [block.doc] : []), ...allDocs],
      factoryTags,
      read: factoryNodes.flatMap(({ nodes }) =>
        nodes.map((node) => ({ file: node.getSourceFile().fileName, pos: node.pos })),
      ),
      blockTags: block?.doc.tags ?? [],
      summary: firstSentence(summary),
      groups,
      structs,
      types: types.filter((name) => !tabled.has(name)),
      options,
      handles: [...handles.values()],
      creates,
      made,
      nested,
      submits,
      features,
    };
  };

  const analyses = new Map(modules.map((module) => [module, analyze(module)]));

  // Contract tags are read only in a module's block or on a factory; anywhere else they say nothing
  const readAt = new Map<string, Set<number>>();
  const readHere = (file: string, pos: number) => {
    const known = readAt.get(file) ?? new Set<number>();
    known.add(pos);
    readAt.set(file, known);
  };
  for (const [module, analysis] of analyses) {
    const block = blockOf(module);
    if (block) readHere(module.file, block.pos);
    for (const { file, pos } of analysis.read) readHere(file, pos);
  }

  const checked = packages.filter((pkg) => !pkg.vendored).map((pkg) => `${pkg.dir}/`);
  for (const source of program.getSourceFiles()) {
    const file = source.fileName;
    if (!checked.some((dir) => file.startsWith(dir)) || file.includes('/node_modules/')) continue;

    const where = relative(workspace.package, file);
    const line = (pos: number) => source.getLineAndCharacterOfPosition(pos).line + 1;

    for (const [pos, doc] of docsIn(file)) {
      if (hasTag(doc, 'module') && !moduleFiles.has(file)) {
        issues.push(`${where}:${line(pos)}: @module in a file that is no module's.`);
        continue;
      }

      // A block that carries `@module` beside another is reported as such
      const stray = doc.tags.filter(
        (tag) => tag.name !== 'module' && (contractTags as readonly string[]).includes(tag.name),
      );
      if (stray.length > 0 && !readAt.get(file)?.has(pos)) {
        const names = [...new Set(stray.map((tag) => `@${tag.name}`))].join(', ');
        issues.push(
          `${where}:${line(pos)}: ${names} outside a module's block or a factory's JSDoc, where the docs never read it.`,
        );
      }
    }
  }

  // Each module's tags, its overrides checked against them

  const tagsOf = new Map(
    modules.map((module) => {
      const analysis = analyses.get(module)!;
      return [
        module,
        readTags(
          analysis.docs.flatMap((doc) => doc.tags),
          module.id,
          issues,
        ),
      ];
    }),
  );

  for (const [id, override] of Object.entries(overrides.modules)) {
    const module = modules.find((each) => each.id === id);
    if (!module) {
      issues.push(`The overrides name ${id}, which is no module.`);
      continue;
    }
    if (!module.pkg.vendored) {
      issues.push(`The overrides name ${id}, which isn't vendored; tag its source instead.`);
    }

    const tagged = tagsOf.get(module)!;
    for (const key of Object.keys(override) as (keyof typeof override)[]) {
      const value = (tagged as Record<string, unknown>)[key];
      const said = Array.isArray(value) ? value.length > 0 : value !== undefined;
      if (said && key !== 'helpers' && key !== 'summary') {
        issues.push(`${id}: its overrides and its tags both say what it ${key}.`);
      }
    }
  }

  const kindOf = (module: Module, analysis: Analysis): Kind => {
    const tagged = tagsOf.get(module)!.kind ?? overrides.modules[module.id]?.kind;
    if (tagged) return tagged;

    // Factories that all borrow the root and create nothing on it can only watch the device. One
    // that works without the root does the module's work
    if (analysis.takesRoot) return analysis.creates || !analysis.allTakeRoot ? 'module' : 'tool';
    if (analysis.factories.length === 0 || analysis.returns.every(fromTypegpu)) return 'shader';
    return 'cpu';
  };

  const kinds = new Map(modules.map((module) => [module, kindOf(module, analyses.get(module)!)]));
  const byFile = new Map(modules.map((module) => [module.file, module]));

  /**
   * Modules whose instances a type holds: the type itself, or one level into an options object of
   * the module's own, with the field that holds each.
   */
  const instancesIn = (module: Module, held: ts.Type) => {
    const files = filesOf(module);
    const seen = new Set<ts.Type>();
    const found: { module: string; field?: string }[] = [];

    const add = (type: ts.Type, field?: string) => {
      if (seen.has(type)) return;
      seen.add(type);

      // Flag tests rather than the type guards, which narrow what follows to `never`
      if (type.flags & ts.TypeFlags.UnionOrIntersection) {
        for (const part of (type as ts.UnionOrIntersectionType).types) add(part, field);
        return;
      }

      if (type.flags & ts.TypeFlags.TypeParameter) {
        const constraint = checker.getBaseConstraintOfType(type);
        if (constraint && constraint !== type) add(constraint, field);
        return;
      }

      const file = symbolOf(type)?.declarations?.[0]?.getSourceFile().fileName;
      const declaredHere = !!file && files.has(file);

      // A wrapper's type arguments, such as `Pick<Frames, 'run'>`'s; an options object of the
      // module's own is read by its fields instead
      if (!declaredHere) {
        for (const argument of type.aliasTypeArguments ?? []) add(argument, field);
        const reference =
          type.flags & ts.TypeFlags.Object &&
          (type as ts.ObjectType).objectFlags & ts.ObjectFlags.Reference;
        if (reference)
          for (const argument of checker.getTypeArguments(type as ts.TypeReference))
            add(argument, field);
      }
      const target = file ? byFile.get(file) : undefined;

      if (target && target !== module && kinds.get(target) === 'module') {
        const instance = ['inject', 'destroy'].some((name) =>
          checker.getPropertyOfType(type, name),
        );
        const made = analyses
          .get(target)!
          .returns.some((returned) => checker.isTypeAssignableTo(returned, type));
        const known = found.some((each) => each.module === target.id && each.field === field);
        if (instance && made && !known)
          found.push(field ? { module: target.id, field } : { module: target.id });
      }

      // An options object of the module's own: one level into its fields
      if (field === undefined && declaredHere && type.getCallSignatures().length === 0) {
        for (const property of checker.getPropertiesOfType(type)) {
          add(checker.getNonNullableType(checker.getTypeOfSymbol(property)), property.name);
        }
      }
    };

    add(held);
    return found;
  };

  /**
   * Modules whose instances a module's factories take, directly or in their options, and where:
   * `camera`, or `renderer(options.eye)` for a method that returns recorders.
   */
  const needsOf = (module: Module, analysis: Analysis) => {
    const via = new Map<string, string[]>();

    for (const { type, name, method } of analysis.parameters) {
      for (const each of instancesIn(module, type)) {
        const path = each.field ? `${name}.${each.field}` : name;
        const at = method ? `${method}(${path})` : path;
        const known = via.get(each.module) ?? [];
        if (!known.includes(at)) via.set(each.module, [...known, at]);
      }
    }

    const needs = [...via.keys()].toSorted();
    return {
      needs,
      via: Object.fromEntries(needs.map((id) => [id, via.get(id)!.join(' or ')])),
    };
  };

  /**
   * For a shader module, the modules whose accessors, slots or layouts its functions read: declared
   * in a file of another module, which its `inject()` fills in the calling pipeline.
   */
  const readsOf = (module: Module) => {
    const owners = (file: string) =>
      modules.filter(
        (other) => other !== module && kinds.get(other) === 'module' && filesOf(other).has(file),
      );
    const found = new Set<string>();

    const visit = (node: ts.Node) => {
      if (ts.isIdentifier(node)) {
        const symbol = checker.getSymbolAtLocation(node);
        const declaration = symbol && resolve(symbol).valueDeclaration;

        if (
          declaration &&
          ts.isVariableDeclaration(declaration) &&
          isOneOf(checker.getTypeAtLocation(declaration), filledSymbols)
        ) {
          for (const owner of owners(declaration.getSourceFile().fileName)) found.add(owner.id);
        }
      }

      ts.forEachChild(node, visit);
    };

    visit(sourceOf(module.file));
    return [...found].toSorted();
  };

  /**
   * The fields of a factory's `options` object, when the module declares its type: the header
   * shows the required ones, and the ones that hold a module or have a note.
   */
  const optionsOf = (module: Module, parameter: ts.Symbol) => {
    if (parameterName(parameter) !== 'options') return undefined;

    const files = filesOf(module);
    const type = checker.getNonNullableType(checker.getTypeOfSymbol(parameter));
    // Options that take a field only beside another are a union of intersections
    const declaredHere = (part: ts.Type): boolean => {
      const file = symbolOf(part)?.declarations?.[0]?.getSourceFile().fileName;
      return (
        (!!file && files.has(file)) ||
        parts(part).some((each) => each !== part && declaredHere(each))
      );
    };
    return declaredHere(type) && type.getCallSignatures().length === 0
      ? checker.getPropertiesOfType(type)
      : undefined;
  };

  /** Each factory's parameters, with the root, the modules they take and their notes marked. */
  const factoriesOf = (
    module: Module,
    analysis: Analysis,
    say: (issue: string) => void,
  ): ModuleContract['factories'] => {
    // What `@param` and `@returns` say, by factory name; checked against every signature of it
    const docs = new Map<string, { notes: Map<string, string>; result?: string }>();
    for (const [name, found] of analysis.factoryTags) {
      const notes = new Map<string, string>();
      let result: string | undefined;

      for (const tag of found) {
        if (tag.name === 'returns') result = unhyphened(tag.text) || undefined;
        if (tag.name !== 'param') continue;

        const [path = '', ...rest] = tag.text.split(/\s+/);
        const note = unhyphened(rest.join(' '));
        if (note) notes.set(path, note);
        else say(`@param ${path} on ${name} needs a few words.`);
      }
      docs.set(name, result ? { notes, result } : { notes });
    }

    const named = new Map<string, Set<string>>();
    for (const { name, declared } of analysis.factories) {
      const paths = named.get(name) ?? new Set<string>();
      named.set(name, paths);
      for (const parameter of declared.parameters) {
        paths.add(parameterName(parameter));
        for (const property of optionsOf(module, parameter) ?? []) {
          paths.add(`${parameterName(parameter)}.${property.name}`);
        }
      }
    }
    for (const [name, { notes }] of docs) {
      for (const path of notes.keys()) {
        if (!named.get(name)?.has(path)) say(`@param ${path} names no parameter of ${name}.`);
      }
    }

    return analysis.factories.map(({ name, signature, declared }) => {
      const notes = docs.get(name)?.notes ?? new Map<string, string>();

      const parameters = declared.parameters.map((parameter) => {
        const type = checker.getTypeOfSymbol(parameter);
        const text = parameterText(parameter);
        const takes = instancesIn(module, type);
        const direct = takes.find((take) => take.field === undefined);

        const entry: Parameter = { name: text.replace(/\?$/, '') };
        if (text.endsWith('?')) entry.optional = true;
        if (isRoot(type)) entry.root = true;
        if (direct) entry.module = direct.module;

        const note = notes.get(entry.name);
        if (note) entry.note = note;

        const properties = optionsOf(module, parameter);
        if (properties) {
          const fields: Field[] = [];
          for (const property of properties) {
            // A field typed `never` is one this signature refuses, such as `source` beside a glow
            const held = checker.getNonNullableType(checker.getTypeOfSymbol(property));
            if (held.flags & ts.TypeFlags.Never) continue;

            const field: Field = { name: property.name };
            const optional = (property.flags & ts.SymbolFlags.Optional) !== 0;
            const holds = takes.find((take) => take.field === property.name)?.module;
            const fieldNote = notes.get(`${entry.name}.${property.name}`);

            if (optional && !holds && !fieldNote) continue;
            if (optional) field.optional = true;
            if (holds) field.module = holds;
            if (fieldNote) field.note = fieldNote;
            fields.push(field);
          }

          if (fields.length > 0) entry.fields = fields;
          if (fields.length > 0 && fields.length < properties.length) entry.more = true;
        }

        return entry;
      });

      const returns = returnName(module, declared);
      const result = docs.get(name)?.result;
      return {
        name,
        signature,
        parameters,
        ...(returns ? { returns } : {}),
        ...(result ? { result } : {}),
      };
    });
  };

  /**
   * The objects and modules `@owns` names, to check against what the code creates. A module named
   * like an object, `pipelines`, is never owned, so `3 pipelines` names objects.
   */
  const ownedIn = (items: readonly string[]) => {
    const read = readOwns(items);
    const words = read.flatMap((part) => part.words).join(' ');
    // A namespace that reads as an object names a module only in backticks
    const owned = (module: Module) =>
      kinds.get(module) === 'module' &&
      (read.some((part) => part.module === module.namespace) ||
        (!objects.some((object) => mentions(module.namespace, object)) &&
          mentions(words, module.namespace)));

    return {
      objects: new Set(read.flatMap((part) => part.mentions)),
      modules: new Set(modules.filter(owned).map((module) => module.id)),
    };
  };

  // The contracts, checked against what the code shows

  // A host creates modules on a program's behalf, such as the frames `web.host` runs a scene with,
  // so a program that uses a host uses what it creates, and what the hosts among those create
  const created = (module: Module) =>
    [...analyses.get(module)!.nested]
      .map((file) => byFile.get(file))
      .filter((each): each is Module => each !== undefined && each !== module);
  const withHosted = (ids: readonly string[]) => {
    const used = new Set(ids);
    const visit = (module: Module | undefined) => {
      if (!module || kinds.get(module) !== 'platform') return;
      for (const each of created(module)) {
        if (used.has(each.id)) continue;
        used.add(each.id);
        visit(each);
      }
    };
    for (const id of ids) visit(modules.find((module) => module.id === id));
    return [...used].toSorted();
  };
  const found = readUsage(workspace.examples, imports, workspace.pages, workspace.hosts);
  for (const id of new Set(Object.values(found).flat())) {
    if (!modules.some((module) => module.id === id)) {
      issues.push(`A shared file's @uses names ${id}, which is no module.`);
    }
  }
  const usage: Usage = Object.fromEntries(
    Object.entries(found).map(([key, ids]) => [
      key,
      withHosted(ids.filter((id) => modules.some((module) => module.id === id))),
    ]),
  );

  const contracts = modules.map((module): ModuleContract => {
    const analysis = analyses.get(module)!;
    const override = overrides.modules[module.id] ?? {};
    const say = (issue: string) => issues.push(`${module.id}: ${issue}`);
    const tagged = tagsOf.get(module)!;

    for (const tag of analysis.blockTags) {
      if (tag.name === 'param' || tag.name === 'returns') {
        say(`@${tag.name} belongs on a factory, not in the module's block.`);
      }
    }
    if (tagged.flat && module.namespaced) {
      say('@flat, but its package exports it as a namespace.');
    }

    const injects = override.injects ?? tagged.injects;
    const binds = override.binds ?? tagged.binds;
    const requires = override.requires ?? tagged.requires;
    const submits = override.submits ?? tagged.submits;
    const allocates = override.allocates ?? tagged.allocates;
    const owns = override.owns ?? tagged.owns;
    const order = override.order ?? tagged.order;
    const platform = override.platform ?? tagged.platform;

    if (injects.length > 0 && !analysis.injectable)
      say('@injects, but nothing it makes has inject().');
    if (injects.length === 0 && analysis.injectable)
      say('inject() without @injects saying what it fills.');
    if (injects.length > 0 && binds.length === 0)
      say('@injects without @binds saying what its accessors bind.');
    for (const { feature } of requires) {
      if (!analysis.features.has(feature))
        say(`@requires ${feature}, which its code never checks.`);
    }
    for (const feature of analysis.features) {
      if (!requires.some((each) => each.feature === feature)) {
        say(`checks the feature ${feature} without @requires.`);
      }
    }
    if (submits && !analysis.submits) say(`@submits ${submits.when}, but its code never submits.`);
    if (!submits && analysis.submits) say('submits without @submits.');
    if (allocates && !analysis.creates) say('@allocates, but its code creates nothing.');
    if (!analysis.summary) say('no summary: give it a block with @module, or its factory a JSDoc.');

    // What `@owns` names, against the GPU objects the code creates and the modules it creates
    // with the root
    const inner = [...analysis.nested]
      .map((file) => byFile.get(file))
      .filter((each): each is Module => !!each && each !== module && kinds.get(each) === 'module');
    const nested = inner.map((each) => each.id);
    // What it creates through a module it creates counts as its own, such as the mesh pool's
    // textures through `images`
    const through = new Set(inner.flatMap((each) => [...analyses.get(each)!.made]));
    if (owns) {
      const named = ownedIn(owns.items);
      for (const object of analysis.made) {
        if (!named.objects.has(object)) say(`creates a ${object}, which @owns doesn't name.`);
      }
      for (const object of named.objects) {
        if (!analysis.made.has(object) && !through.has(object))
          say(`@owns names a ${object}, which its code never creates.`);
      }
      for (const id of nested) {
        if (!named.modules.has(id)) say(`creates a ${id}, which @owns doesn't name.`);
      }
      for (const id of named.modules) {
        if (!nested.includes(id)) say(`@owns names ${id}, which its code never creates.`);
      }
    } else if (
      // A shader or CPU module creates nothing; a file it shares with a module, such as the mesh
      // pool's textures, is the module's
      !['shader', 'cpu'].includes(kinds.get(module)!) &&
      (analysis.made.size > 0 || nested.length > 0)
    ) {
      say(`creates ${[...analysis.made, ...nested].join(', ')} without @owns saying what.`);
    }

    const kind = kinds.get(module)!;
    const { needs, via } = needsOf(module, analysis);
    const contract: ModuleContract = {
      id: module.id,
      package: module.pkg.package,
      namespace: module.namespace,
      specifier: module.specifier,
      file: relative(module.pkg.dir, module.file).split('\\').join('/'),
      summary: analysis.summary,
      factories: factoriesOf(module, analysis, say),
      ...analysis.groups,
      structs: analysis.structs,
      types: analysis.types,
      options: analysis.options,
      handles: analysis.handles,
      kind,
      records: analysis.records,
      injects,
      binds,
      needs,
      via,
      reads: kind === 'shader' ? readsOf(module) : [],
      requires,
      lifecycle: analysis.lifecycle,
      usedBy: Object.keys(usage).filter((key) => usage[key].includes(module.id)),
    };

    if (order) contract.order = order;
    if (submits) contract.submits = submits;
    if (owns) contract.owns = owns;
    if (allocates) contract.allocates = allocates;
    if (platform) contract.platform = platform;
    if (module.pkg.vendored) contract.vendored = true;
    if (module.pkg.vendored || tagged.unstable) contract.unstable = true;
    if (module.flat || tagged.flat) contract.flat = true;
    if (module.namespaced && module.specifier !== module.pkg.package)
      contract.namespaceExport = true;
    return contract;
  });

  return { contracts, usage, issues: [...new Set(issues)], program };
}
