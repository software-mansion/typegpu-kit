import { existsSync, readFileSync, realpathSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

/** Type-only imports and exports, which leave nothing behind at run time. */
const typeOnly =
  /(?:import|export)\s+type\s*(?:\{[^}]*\}|[\w$*]+(?:\s+as\s+[\w$]+)?)\s*from\s*['"][^'"]+['"]/g;

/** What a module loads: its `import … from '…'`, `export … from '…'`, `import '…'` and `import('…')`. */
const loads =
  /^\s*(?:import|export)\s*[\w$\s{},*]*?\s*from\s*['"]([^'"]+)['"]|^\s*import\s*\(?\s*['"]([^'"]+)['"]|\bimport\(\s*['"]([^'"]+)['"]\s*\)/gm;

/** The export conditions a bundler for a phone or a browser picks from. */
const conditions = new Set(['react-native', 'browser', 'import', 'module', 'default']);

type Exports = string | null | readonly Exports[] | { readonly [key: string]: Exports };

function pick(target: Exports): string | undefined {
  if (target === null) return undefined;
  if (typeof target === 'string') return target;
  if (Array.isArray(target)) return target.map(pick).find(Boolean);

  for (const [key, value] of Object.entries(target)) {
    if (conditions.has(key)) return pick(value);
  }
  return undefined;
}

/** Where a bare specifier leads from `file`, as a bundler resolves the package's exports. */
function resolveBare(specifier: string, file: string) {
  const name = specifier
    .split('/')
    .slice(0, specifier.startsWith('@') ? 2 : 1)
    .join('/');
  const subpath = `.${specifier.slice(name.length)}`;

  let directory = path.dirname(file);
  while (!existsSync(path.join(directory, 'node_modules', name, 'package.json'))) {
    const parent = path.dirname(directory);
    if (parent === directory)
      throw new Error(`${file} imports ${specifier}, which isn't installed.`);
    directory = parent;
  }

  const root = path.join(directory, 'node_modules', name);
  const { exports, module, main } = JSON.parse(
    readFileSync(path.join(root, 'package.json'), 'utf8'),
  );
  const map: Record<string, Exports> =
    typeof exports === 'object' && exports !== null && Object.keys(exports)[0]?.startsWith('.')
      ? exports
      : { '.': exports ?? module ?? main ?? './index.js' };

  const target = pick(map[subpath] ?? null);
  if (target === undefined) throw new Error(`${name} exports no ${subpath}.`);
  return realpathSync(path.join(root, target));
}

/** Every file the module at `entry` loads at run time, with what it loads. */
function graph(entry: string) {
  const files = new Map<string, { source: string; specifiers: string[] }>();
  const pending = [entry];

  while (pending.length > 0) {
    const file = pending.pop()!;
    if (files.has(file)) continue;

    const source = readFileSync(file, 'utf8');
    const specifiers = [...source.replaceAll(typeOnly, '').matchAll(loads)].map(
      ([, from, imported, dynamic]) => (from ?? imported ?? dynamic)!,
    );
    files.set(file, { source, specifiers });

    for (const specifier of specifiers) {
      if (specifier.startsWith('node:')) continue;
      pending.push(
        specifier.startsWith('.')
          ? realpathSync(path.resolve(path.dirname(file), specifier))
          : resolveBare(specifier, file),
      );
    }
  }

  return files;
}

describe("the runtime's imports", () => {
  const entry = fileURLToPath(new URL('../../src/text/text.ts', import.meta.url));
  const files = graph(entry);

  it('reach no WebAssembly, no glyph and no Node built-in, through every package they load', () => {
    expect(files.size).toBeGreaterThan(20);

    for (const [file, { source, specifiers }] of files) {
      expect(file).not.toMatch(/\.wasm$|[\\/]@pmndrs[\\/]glyph[\\/]|[\\/]src[\\/]bake\.ts$/);
      expect(source, file).not.toMatch(/WebAssembly|\.wasm\b/);
      expect(
        specifiers.filter((specifier) => specifier.startsWith('node:')),
        file,
      ).toEqual([]);
    }
  });

  it('keep the text entry in typeset and text, with core and TypeGPU', () => {
    const own = [...files].filter(([file]) =>
      file.startsWith(fileURLToPath(new URL('../../src/text/', import.meta.url))),
    );
    const bare = own.flatMap(([, { specifiers }]) =>
      specifiers.filter((specifier) => !specifier.startsWith('.')),
    );

    expect(own.map(([file]) => path.basename(file)).toSorted()).toEqual(['text.ts', 'typeset.ts']);
    expect([...new Set(bare)].toSorted()).toEqual(['typegpu']);
  });
});
