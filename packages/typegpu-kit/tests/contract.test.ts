import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import ts from 'typescript';
import { expect, it } from 'vitest';

const kit = join(dirname(fileURLToPath(import.meta.url)), '..');

const CONTRACT: Record<string, readonly [file: string, ...names: string[]]> = {
  frames: ['frames.ts', 'Frames', 'create', 'encoderOf', 'dtOf'],
  Frame: ['frames.ts', 'Frame'],
  owner: ['owner.ts', 'Owner', 'owner'],
  Inject: ['inject.ts', 'Inject'],
  Injectable: ['inject.ts', 'Injectable'],
  byStage: ['inject.ts', 'byStage'],
  unfilled: ['inject.ts', 'unfilled'],
  Checked: ['inject.ts', 'Checked'],
  Drawing: ['passes.ts', 'Drawing'],
  Compound: ['passes.ts', 'Compound'],
  Into: ['passes.ts', 'Into'],
  Built: ['passes.ts', 'Built'],
  Drawn: ['passes.ts', 'Drawn'],
  Output: ['passes.ts', 'Output'],
  'passes.output': ['passes.ts', 'output'],
  'cameras.current': ['cameras.ts', 'current'],
  Camera: ['cameras.ts', 'Camera'],
  FloatTexture: ['targets.ts', 'FloatTexture'],
  post: ['post/post.ts', 'Rendered', 'Source'],
  Scene: ['hosting.ts', 'Scene'],
  CreateScene: ['hosting.ts', 'CreateScene'],
  Surface: ['targets.ts', 'Surface'],
  '@typegpu/kit/testing': [
    'testing.ts',
    'RecordingLog',
    'RecordingCounts',
    'RecordingRoot',
    'RecordingOptions',
    'recordingRoot',
  ],
};

const INTRO = `# The contract

These declarations change only by addition. A release can add an optional member, an overload or
a name, and removes or changes nothing here. \`tests/contract.test.ts\` prints them from the source
and fails on any difference. Accept an addition with
\`pnpm vitest run --project='!*(gpu)' -u packages/typegpu-kit/tests/contract.test.ts\`.
`;

/** The names a top-level statement of a declaration file declares. */
function namesOf(statement: ts.Statement): string[] {
  if (ts.isVariableStatement(statement)) {
    return statement.declarationList.declarations.map((each) => each.name.getText());
  }
  const named =
    ts.isFunctionDeclaration(statement) ||
    ts.isInterfaceDeclaration(statement) ||
    ts.isTypeAliasDeclaration(statement);
  return named && statement.name ? [statement.name.text] : [];
}

/** Each contract file's declarations as the compiler emits them, without comments. */
function report(): string {
  const { config } = ts.readConfigFile(join(kit, 'tsconfig.json'), ts.sys.readFile);
  const { options } = ts.parseJsonConfigFileContent(config, ts.sys, kit);
  const declarations: Record<string, string[]> = {};
  for (const [file, ...names] of Object.values(CONTRACT))
    (declarations[file] ??= []).push(...names);
  const files = Object.keys(declarations);
  const program = ts.createProgram(
    files.map((file) => join(kit, 'src', file)),
    {
      ...options,
      noEmit: false,
      declaration: true,
      emitDeclarationOnly: true,
      removeComments: true,
      declarationMap: false,
    },
  );

  const sections = files.map((file) => {
    let text = '';
    program.emit(
      program.getSourceFile(join(kit, 'src', file)),
      (_, emitted) => (text = emitted),
      undefined,
      true,
    );
    const declared = ts.createSourceFile(file, text, ts.ScriptTarget.Latest, true).statements;
    const printed = declarations[file].map((name) => {
      const found = declared.filter((statement) => namesOf(statement).includes(name));
      if (found.length === 0)
        throw new Error(`The contract names ${name}, which src/${file} lacks.`);
      return found.map((statement) => statement.getText()).join('\n');
    });
    return `## ${file}\n\n\`\`\`ts\n${printed.join('\n\n')}\n\`\`\`\n`;
  });

  return [INTRO, ...sections].join('\n');
}

it('prints the contract declarations for review of changes', async () => {
  await expect(report()).toMatchFileSnapshot('../api/contract.api.md');
}, 60_000);

it('names the same contract as the Versions list', () => {
  const page = readFileSync(
    join(kit, '../../apps/typegpu-kit-docs/src/content/docs/getting-started.mdx'),
    'utf8',
  );
  const list = page.split('## Versions')[1].split('  - how every module behaves')[0];
  const names = [...list.matchAll(/`([^`]+)`/g)]
    .map(([, name]) => name)
    .filter((name) => name !== 'passes');
  expect(names.toSorted()).toEqual(Object.keys(CONTRACT).toSorted());
});
