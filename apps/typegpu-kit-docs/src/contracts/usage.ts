import { existsSync, readdirSync, readFileSync, statSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import ts from 'typescript';

/** Map import specifiers and exported names to module ids. `*` represents a specifier that exports one module. */
export type Imports = ReadonlyMap<string, Names>;
type Names = ReadonlyMap<string, string>;

/** Per example key, or host key such as `hosts/web`, the ids of the modules its code uses, sorted. */
export type Usage = Record<string, string[]>;

const directories = (dir: string) =>
  readdirSync(dir, { withFileTypes: true })
    .filter((entry) => entry.isDirectory())
    .map((entry) => entry.name);

const sources = (dir: string): string[] =>
  readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
    const path = join(dir, entry.name);
    if (entry.isDirectory()) return sources(path);
    return /\.tsx?$/.test(entry.name) ? [path] : [];
  });

/** The examples' keys: each `<name>` directory that holds a `meta.json`. */
export const examplesIn = (dir: string) =>
  directories(dir).filter((name) => existsSync(join(dir, name, 'meta.json')));

/**
 * Reads every example's `.ts` files without a type checker, and the files they import by relative
 * path, such as a helper the examples share. A value import counts as a use of its module: a
 * namespace (`import * as web`), a member (`import { memo }`), a namespace of a package read
 * through it (`kit.frames`), a re-export and a dynamic import. Type-only imports don't count,
 * since they run nothing. A shared file whose exports carry `@uses <id>...` tags counts by the
 * names imported from it, each for the modules its tag lists, so that importing a descriptor
 * such as `font` from the examples' `assets.ts` doesn't count its glTF loader.
 */
export function readUsage(
  examplesDir: string,
  imports: Imports,
  pagesDir?: string,
  hosts: Readonly<Record<string, string>> = {},
): Usage {
  // Each file is parsed once, however many examples reach it
  const read = new Map<string, Read>();
  const usedFrom = (files: readonly string[]) => {
    const used = new Set<string>();
    const seen = new Set<string>();
    const visit = (file: string) => {
      if (seen.has(file)) return;
      seen.add(file);
      const each = read.get(file) ?? readFile(file, imports);
      read.set(file, each);
      for (const id of each.used) used.add(id);
      for (const next of each.files) visit(next);
      for (const { file: next, names } of each.named) {
        const tagged = read.get(next) ?? readFile(next, imports);
        read.set(next, tagged);
        if (tagged.uses.size === 0) visit(next);
        else for (const name of names) for (const id of tagged.uses.get(name) ?? []) used.add(id);
      }
    };
    for (const file of files) visit(file);
    return [...used].toSorted();
  };

  const examples = examplesIn(examplesDir).map((key) => {
    const page =
      pagesDir === undefined ? [] : [join(pagesDir, key)].filter((dir) => existsSync(dir));
    return [key, usedFrom([join(examplesDir, key), ...page].flatMap(sources))] as const;
  });

  // A host is the files directly in its directory, not the examples' pages below them
  const hosted = Object.entries(hosts).map(([key, dir]) => {
    const files = readdirSync(dir, { withFileTypes: true })
      .filter((entry) => entry.isFile() && /\.tsx?$/.test(entry.name))
      .map((entry) => join(dir, entry.name));
    return [key, usedFrom(files)] as const;
  });

  return Object.fromEntries([...examples, ...hosted]);
}

/** What a file uses, and the files it imports by relative path for their values. */
interface Read {
  readonly used: ReadonlySet<string>;
  /** Files it imports whole: a namespace, a side effect, `export *` or a dynamic import. */
  readonly files: readonly string[];
  /** Files it imports names from, and the names. */
  readonly named: readonly { readonly file: string; readonly names: readonly string[] }[];
  /** Its exports' `@uses` tags: the module ids each export uses. */
  readonly uses: ReadonlyMap<string, readonly string[]>;
}

/** The `.ts` file a relative specifier names, if any. */
function resolveRelative(from: string, specifier: string) {
  if (!specifier.startsWith('.')) return undefined;
  const path = resolve(dirname(from), specifier);
  return [path, `${path}.ts`, `${path}.tsx`, join(path, 'index.ts')].find(
    (each) => /\.tsx?$/.test(each) && existsSync(each) && statSync(each).isFile(),
  );
}

function readFile(file: string, imports: Imports): Read {
  const used = new Set<string>();
  const files = new Set<string>();
  const named: { file: string; names: string[] }[] = [];
  const uses = new Map<string, string[]>();
  const follow = (
    specifier: ts.Expression | undefined,
    elements?: ts.NodeArray<ts.ImportSpecifier | ts.ExportSpecifier>,
  ) => {
    const next =
      specifier && ts.isStringLiteral(specifier)
        ? resolveRelative(file, specifier.text)
        : undefined;
    if (!next) return;
    if (!elements) files.add(next);
    else {
      const names = elements
        .filter((element) => !element.isTypeOnly)
        .map((element) => (element.propertyName ?? element.name).text);
      if (names.length > 0) named.push({ file: next, names });
    }
  };
  const source = ts.createSourceFile(
    file,
    readFileSync(file, 'utf8'),
    ts.ScriptTarget.Latest,
    true,
  );

  /** Local names bound to a package of several modules, whose members are its namespaces. */
  const packages = new Map<string, Names>();
  const useName = (names: Names, name: string) => {
    const module = names.get(name);
    if (module) used.add(module);
  };
  const useAll = (names: Names) => {
    for (const module of names.values()) used.add(module);
  };
  const bindWhole = (local: string, names: Names) => {
    const whole = names.get('*');
    if (whole) used.add(whole);
    else packages.set(local, names);
  };

  const specified = (specifier: ts.Expression | undefined) =>
    specifier && ts.isStringLiteral(specifier) ? imports.get(specifier.text) : undefined;

  for (const statement of source.statements) {
    const exported = ts.canHaveModifiers(statement)
      ? ts.getModifiers(statement)?.some((each) => each.kind === ts.SyntaxKind.ExportKeyword)
      : false;
    const tagged = exported
      ? ts.getJSDocTags(statement).filter((tag) => tag.tagName.text === 'uses')
      : [];
    for (const tag of tagged) {
      const ids = (ts.getTextOfJSDocComment(tag.comment) ?? '').split(/[\s,]+/).filter(Boolean);
      const declared = ts.isVariableStatement(statement)
        ? statement.declarationList.declarations.map((each) => each.name)
        : [(statement as ts.FunctionDeclaration).name];
      for (const name of declared) {
        if (name && ts.isIdentifier(name)) uses.set(name.text, ids);
      }
    }

    if (ts.isImportDeclaration(statement)) {
      const clause = statement.importClause;
      if (clause?.phaseModifier !== ts.SyntaxKind.TypeKeyword) {
        const bound = clause?.namedBindings;
        follow(
          statement.moduleSpecifier,
          bound && ts.isNamedImports(bound) ? bound.elements : undefined,
        );
      }
      const names = specified(statement.moduleSpecifier);
      const bound = clause?.namedBindings;
      if (!names || !bound || clause.phaseModifier === ts.SyntaxKind.TypeKeyword) continue;

      if (ts.isNamespaceImport(bound)) bindWhole(bound.name.text, names);
      else
        for (const element of bound.elements) {
          if (!element.isTypeOnly) useName(names, (element.propertyName ?? element.name).text);
        }
    }

    if (ts.isExportDeclaration(statement) && !statement.isTypeOnly) {
      const clause = statement.exportClause;
      follow(
        statement.moduleSpecifier,
        clause && ts.isNamedExports(clause) ? clause.elements : undefined,
      );
      const names = specified(statement.moduleSpecifier);
      if (!names) continue;

      if (clause && ts.isNamedExports(clause)) {
        for (const element of clause.elements) {
          if (!element.isTypeOnly) useName(names, (element.propertyName ?? element.name).text);
        }
      } else useAll(names);
    }
  }

  const visit = (node: ts.Node) => {
    // `await import('@typegpu/kit')`, bound to a name or taken apart
    if (ts.isCallExpression(node) && node.expression.kind === ts.SyntaxKind.ImportKeyword) {
      follow(node.arguments[0]);
    }
    if (
      ts.isCallExpression(node) &&
      node.expression.kind === ts.SyntaxKind.ImportKeyword &&
      specified(node.arguments[0])
    ) {
      const names = specified(node.arguments[0])!;
      const awaited = ts.isAwaitExpression(node.parent) ? node.parent : undefined;
      const declaration =
        awaited && ts.isVariableDeclaration(awaited.parent) ? awaited.parent : undefined;

      if (declaration && ts.isIdentifier(declaration.name)) {
        bindWhole(declaration.name.text, names);
      } else if (declaration && ts.isObjectBindingPattern(declaration.name)) {
        for (const element of declaration.name.elements) {
          const name = element.propertyName ?? element.name;
          if (ts.isIdentifier(name)) useName(names, name.text);
        }
      } else if (names.has('*')) {
        used.add(names.get('*')!);
      } else {
        useAll(names);
      }
    }

    if (
      ts.isPropertyAccessExpression(node) &&
      ts.isIdentifier(node.expression) &&
      packages.has(node.expression.text)
    ) {
      useName(packages.get(node.expression.text)!, node.name.text);
    }

    ts.forEachChild(node, visit);
  };

  // Bindings first: a dynamic import binds a package before the code that reads it is visited
  visit(source);
  visit(source);
  return { used, files: [...files], named, uses };
}
