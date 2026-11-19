import { readdirSync } from 'node:fs';
import { expect, it } from 'vitest';
import { exampleKeys, read } from '../src/source/read.ts';

const examplesDir = new URL('../../../packages/typegpu-kit-examples/src/', import.meta.url);

it.each(exampleKeys())('%s reads into a source page', (key) => {
  const source = read(key);
  const linesOf = (file: string) =>
    source.files.find((each) => each.name === file)!.code.split('\n');

  expect(source.files[0]?.name).toBe('scene.ts');
  expect(source.files.map((file) => file.name).toSorted()).toEqual(
    readdirSync(new URL(`${key}/`, examplesDir))
      .filter((name) => name.endsWith('.ts'))
      .toSorted(),
  );

  for (const [id, entry] of Object.entries(source.traced)) {
    if (entry.kind !== 'local') continue;
    expect(linesOf(entry.file)[entry.line - 1], id).toMatch(
      new RegExp(`\\b${entry.name.replace(/\$/g, '\\$')}\\b`),
    );
    for (const captured of Object.values(entry.captures ?? {}).flat()) {
      expect(Object.hasOwn(source.traced, captured), `${id} captures ${captured}`).toBe(true);
    }
  }

  for (const file of source.files) {
    for (const [offset, id] of file.marks) {
      const written = /^[\w$]+/.exec(file.code.slice(offset))?.[0] ?? '';
      const name = source.traced[id]?.name ?? '';
      // Or renamed where it's imported or destructured
      const renamed = new RegExp(`\\b${name}\\s*(as|:)\\s*${written}\\b`).test(file.code);
      expect(written === name || renamed, `${file.name} at ${offset}: ${written} as ${id}`).toBe(
        true,
      );
    }
    let last = 0;
    for (const [from, to] of file.gpu) {
      expect(from).toBeGreaterThan(last);
      expect(to).toBeGreaterThanOrEqual(from);
      expect(
        file.code
          .split('\n')
          .slice(from - 1, to)
          .join('\n'),
      ).toContain("'use gpu'");
      last = to;
    }
  }
});
