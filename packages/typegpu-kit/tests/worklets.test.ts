import { readdirSync, readFileSync } from 'node:fs';
import { join, relative } from 'node:path';
import { fileURLToPath } from 'node:url';
import ts from 'typescript';
import { expect, it } from 'vitest';

const repo = fileURLToPath(new URL('../../../', import.meta.url));
const sources = [
  'packages/typegpu-kit/src',
  'packages/typegpu-kit-examples/src',
  'apps/typegpu-kit-native/src',
];

function filesIn(dir: string): string[] {
  return readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
    const file = join(dir, entry.name);
    return entry.isDirectory() ? filesIn(file) : /\.tsx?$/.test(entry.name) ? [file] : [];
  });
}

function isWorklet(node: ts.Node): node is ts.FunctionDeclaration & { name: ts.Identifier } {
  if (!ts.isFunctionDeclaration(node) || !node.name) return false;
  const first = node.body?.statements[0];
  return (
    !!first &&
    ts.isExpressionStatement(first) &&
    ts.isStringLiteral(first.expression) &&
    first.expression.text === 'worklet'
  );
}

function isMemberName(node: ts.Identifier) {
  const { parent } = node;
  return (
    (ts.isPropertyAccessExpression(parent) || ts.isPropertyAssignment(parent)) &&
    parent.name === node
  );
}

// The worklets plugin turns each worklet declaration into a variable set where it stands, and a
// worklet copies what it closes over as it is made. One that names a worklet declared below it
// copies `undefined`, which only a device shows.
it('declares a worklet above the worklets that call it', () => {
  const offending: string[] = [];
  for (const file of sources.flatMap((dir) => filesIn(join(repo, dir)))) {
    const source = ts.createSourceFile(
      file,
      readFileSync(file, 'utf8'),
      ts.ScriptTarget.Latest,
      true,
    );
    const visit = (node: ts.Node) => {
      if (ts.isSourceFile(node) || ts.isBlock(node)) {
        const worklets = node.statements.filter(isWorklet);
        for (const [index, worklet] of worklets.entries()) {
          const below = new Set(worklets.slice(index + 1).map((later) => later.name.text));
          const read = (inner: ts.Node) => {
            if (ts.isIdentifier(inner) && below.has(inner.text) && !isMemberName(inner)) {
              const { line } = source.getLineAndCharacterOfPosition(inner.getStart());
              offending.push(
                `${relative(repo, file)}:${line + 1} ${worklet.name.text} reads ${inner.text}, declared below it`,
              );
            }
            ts.forEachChild(inner, read);
          };
          ts.forEachChild(worklet.body!, read);
        }
      }
      ts.forEachChild(node, visit);
    };
    visit(source);
  }
  expect(offending).toEqual([]);
});
