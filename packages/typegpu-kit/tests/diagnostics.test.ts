import { readdirSync } from 'node:fs';
import { basename, dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import ts from 'typescript';
import { expect, it } from 'vitest';

const kit = join(dirname(fileURLToPath(import.meta.url)), '..');
const fixtures = join(kit, 'tests/diagnostics');

/** The innermost messages of a diagnostic, which say what is wrong: one per overload tried. */
function reasons(message: string | ts.DiagnosticMessageChain): string[] {
  if (typeof message === 'string') return [message];
  return message.next?.flatMap(reasons) ?? [message.messageText];
}

/** The diagnostics for each fixture, with the innermost reasons. */
function diagnose() {
  const { config } = ts.readConfigFile(join(kit, 'tsconfig.json'), ts.sys.readFile);
  const { options } = ts.parseJsonConfigFileContent(config, ts.sys, kit);
  const files = readdirSync(fixtures)
    .toSorted()
    .map((file) => join(fixtures, file));
  const program = ts.createProgram(files, { ...options, noEmit: true });

  return Object.fromEntries(
    files.map((file) => [
      basename(file),
      ts.getPreEmitDiagnostics(program, program.getSourceFile(file)).map((found) => ({
        line: ts.getLineAndCharacterOfPosition(found.file!, found.start!).line + 1,
        code: found.code,
        message: reasons(found.messageText).join('\n'),
      })),
    ]),
  );
}

it('names the kit constraint at the mistake and contextually types inline hooks', () => {
  const found = diagnose();
  expect(found['bare.ts']).toEqual([]);
  expect(found['hooks.ts'].map(({ line, code }) => [line, code])).toEqual([
    [15, 2339],
    [24, 2339],
    [32, 2339],
  ]);
  expect(found['descriptions.ts'].map(({ line }) => line)).toEqual([22, 54, 66]);
  expect(found['descriptions.ts'][0].message).toContain('is not a target of this set, which has');
  expect(found['descriptions.ts'][1].message).toContain("target 'id' needs");
  expect(found['descriptions.ts'][2].message).toContain("'uv'");
  expect(found['rows.ts']).toHaveLength(1);
  expect(found['rows.ts'][0].message).toContain('"Shot"');
  expect(found['rows.ts'][0].message).toContain('"Enemy"');
  for (const [file, key] of [
    ['targets.ts', 'smapled'],
    ['lights.ts', 'extend'],
  ]) {
    expect(found[file]).toHaveLength(2);
    expect(found[file][0].message).toContain(`"${key} is not an option here"`);
  }
}, 60_000);
