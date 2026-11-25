// Builds the WebGL fallback for its tarball, as its prepack: dist/ with transformed GPU functions,
// the vendored @typegpu/gl bundled in, since it is not published with the fallback's patches, and
// the declarations the package entry reaches; and patches/ with the TypeGPU patch it needs

import { execFileSync } from 'node:child_process';
import { copyFile, mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import typegpu from 'unplugin-typegpu/vite';
import { build } from 'vite';

const repo = path.resolve(import.meta.dirname, '..');
const dir = path.join(repo, 'packages/typegpu-kit-gl');
const PATCH = 'typegpu@0.12.11.patch';

/** Packages the tarball depends on; everything else it imports ships inside dist. */
const external = (id: string) =>
  ['typegpu', 'tinyest'].some((name) => id === name || id.startsWith(`${name}/`));

await rm(path.join(dir, 'dist'), { recursive: true, force: true });

console.log('Building @typegpu/kit-gl');
await build({
  configFile: false,
  root: dir,
  logLevel: 'warn',
  plugins: [typegpu()],
  build: {
    outDir: 'dist',
    emptyOutDir: false,
    sourcemap: true,
    minify: false,
    target: 'es2024',
    lib: { entry: { index: 'src/index.ts' }, formats: ['es'] },
    rolldownOptions: {
      external,
      optimization: { inlineConst: false },
      output: {
        entryFileNames: '[name].js',
        chunkFileNames: '[name].js',
        // The vendored generator's source does not ship, so the maps carry what they point to
        sourcemapExcludeSources: false,
      },
    },
  },
});

// Declarations of what the entry reaches: emitted for the whole program, then only the files that
// index.d.ts imports, which must not name the bundled @typegpu/gl
const emitted = await mkdtemp(path.join(tmpdir(), 'kit-gl-types-'));
try {
  const config = path.join(dir, 'tsconfig.pack.json');
  await writeFile(
    config,
    JSON.stringify({
      extends: './tsconfig.json',
      compilerOptions: {
        noEmit: false,
        declaration: true,
        emitDeclarationOnly: true,
        declarationMap: true,
        rootDir: 'src',
        outDir: emitted,
      },
      include: ['src'],
    }),
  );
  try {
    execFileSync(path.join(repo, 'node_modules/.bin/tsc'), ['-p', config], { stdio: 'inherit' });
  } finally {
    await rm(config);
  }
  const out = emitted;
  const queue = ['index.d.ts'];
  const seen = new Set<string>();
  while (queue.length > 0) {
    const file = queue.pop()!;
    if (seen.has(file)) continue;
    seen.add(file);
    let text = await readFile(path.join(out, file), 'utf8');
    text = text.replace(/(['"]\.{1,2}\/[^'"]+)\.ts(['"])/g, '$1.js$2');
    for (const [, specifier] of text.matchAll(/(?:from|import\()\s*['"]([^'"]+)['"]/g)) {
      if (specifier!.startsWith('.')) {
        queue.push(path.posix.join(path.posix.dirname(file), specifier!.replace(/\.js$/, '.d.ts')));
      } else if (!external(specifier!)) {
        throw new Error(`dist/${file} names '${specifier}', which the tarball does not provide`);
      }
    }
    const target = path.join(dir, 'dist', file);
    await mkdir(path.dirname(target), { recursive: true });
    await writeFile(target, text);
    // The map points at src/, which the tarball ships
    const map = JSON.parse(await readFile(path.join(out, `${file}.map`), 'utf8')) as {
      sources: string[];
    };
    map.sources = map.sources.map((source) =>
      path.relative(path.dirname(target), path.resolve(path.dirname(path.join(out, file)), source)),
    );
    await writeFile(`${target}.map`, JSON.stringify(map));
  }
} finally {
  await rm(emitted, { recursive: true, force: true });
}

// One source of truth: the workspace registers the repository's patch, and the tarball carries a copy
await mkdir(path.join(dir, 'patches'), { recursive: true });
await copyFile(path.join(repo, 'patches', PATCH), path.join(dir, 'patches', PATCH));
