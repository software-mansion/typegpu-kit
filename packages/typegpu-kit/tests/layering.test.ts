import { readdirSync, readFileSync } from 'node:fs';
import { dirname, join, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import ts from 'typescript';
import { expect, it } from 'vitest';

const root = fileURLToPath(new URL('../', import.meta.url));
const src = join(root, 'src');
const areas = new Set(['post', 'particles', 'render', 'text', 'gltf', 'geometry']);
/** The radix sort's own files and the only core file that imports it. */
const sorting = new Set(['sort', 'sorter.ts']);
const manifest = JSON.parse(readFileSync(join(root, 'package.json'), 'utf8')) as {
  name: string;
  exports: Record<string, string>;
};
const sourceOf = (file: string) =>
  ts.createSourceFile(file, readFileSync(file, 'utf8'), ts.ScriptTarget.Latest, true);
const index = sourceOf(join(src, 'index.ts'));
const entries = new Set(
  Object.entries(manifest.exports)
    .filter(([subpath, target]) => subpath !== '.' && target.endsWith('.ts'))
    .map(([, target]) => resolve(root, target)),
);
/** Lines read the miter join limit, which the geometry's entry does not export. */
const reached = new Set([join(src, 'geometry/lines/joins/miter.ts')]);
/** Private core files that areas reach, which the package entry keeps out of the API. */
const shared = [join(src, 'encoderSubmit.ts')];
const publicCore = new Set([...entries].filter((file) => dirname(file) === src).concat(shared));
for (const statement of index.statements) {
  if (
    ts.isExportDeclaration(statement) &&
    statement.exportClause &&
    ts.isNamedExports(statement.exportClause) &&
    statement.moduleSpecifier &&
    ts.isStringLiteral(statement.moduleSpecifier)
  ) {
    publicCore.add(resolve(src, statement.moduleSpecifier.text));
  }
}

function filesIn(dir: string): string[] {
  return readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
    const file = join(dir, entry.name);
    return entry.isDirectory() ? filesIn(file) : entry.name.endsWith('.ts') ? [file] : [];
  });
}

it('keeps higher-level modules on public files, core independent of them, and sort inside', () => {
  const offending: string[] = [];
  for (const file of filesIn(src)) {
    const area = relative(src, file).split('/')[0];
    const core = dirname(file) === src || area === 'sort';
    if (!core && !areas.has(area)) continue;

    const check = (specifier: string) => {
      if (!specifier.startsWith('.')) {
        if (specifier === manifest.name || specifier.startsWith(`${manifest.name}/`)) {
          offending.push(`${relative(root, file)} imports ${specifier}; use a core file directly`);
        }
        return;
      }
      const target = resolve(dirname(file), specifier);
      if (core && areas.has(relative(src, target).split('/')[0])) {
        offending.push(`${relative(root, file)} imports ${specifier}; core cannot import an area`);
      }
      if (relative(src, target).split('/')[0] === 'sort' && !sorting.has(area)) {
        offending.push(`${relative(root, file)} imports ${specifier}; sort through sorter.ts`);
      }
      if (!core && dirname(target) === src && !publicCore.has(target)) {
        offending.push(`${relative(root, file)} imports ${specifier}; the core file is not public`);
      }
      const into = relative(src, target).split('/')[0];
      if (
        !core &&
        areas.has(into) &&
        into !== area &&
        !entries.has(target) &&
        !reached.has(target)
      ) {
        offending.push(`${relative(root, file)} imports ${specifier}; it is internal to ${into}`);
      }
    };
    const visit = (node: ts.Node) => {
      if (ts.isImportDeclaration(node) && ts.isStringLiteral(node.moduleSpecifier)) {
        check(node.moduleSpecifier.text);
      } else if (
        !core &&
        ts.isExportDeclaration(node) &&
        node.moduleSpecifier &&
        ts.isStringLiteral(node.moduleSpecifier)
      ) {
        check(node.moduleSpecifier.text);
      } else if (
        ts.isCallExpression(node) &&
        node.expression.kind === ts.SyntaxKind.ImportKeyword &&
        node.arguments[0] &&
        ts.isStringLiteral(node.arguments[0])
      ) {
        check(node.arguments[0].text);
      } else if (
        ts.isImportTypeNode(node) &&
        ts.isLiteralTypeNode(node.argument) &&
        ts.isStringLiteral(node.argument.literal)
      ) {
        check(node.argument.literal.text);
      }
      ts.forEachChild(node, visit);
    };
    visit(sourceOf(file));
  }
  expect(offending).toEqual([]);
});
