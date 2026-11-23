// Builds the kit for its tarball, as its prepack: docs/ after checking the skill's modules, then dist/
// with transformed GPU functions and declarations, so consumers need no kit-specific bundler setup

import { execFileSync } from 'node:child_process';
import { readdir, readFile, rm, writeFile } from 'node:fs/promises';
import path from 'node:path';
import ts from 'typescript';
import typegpu from 'unplugin-typegpu/vite';
import { build } from 'vite';
import {
  annotatedIndex,
  checkedContracts,
  declarationHeaders,
  docsDir,
  renderDocs,
  skillFile,
  writeDocs,
} from '../apps/typegpu-kit-docs/src/contracts/agents.ts';

const repo = path.resolve(import.meta.dirname, '..');
const DISPOSABLE = '/// <reference lib="esnext.disposable" />';
const dir = path.join(repo, 'packages/typegpu-kit');

interface Manifest {
  name: string;
  exports: Record<string, string>;
  publishConfig?: { exports?: Record<string, string> };
}

function entries(manifest: Manifest) {
  const found: Record<string, string> = {};

  for (const [subpath, target] of Object.entries(manifest.exports)) {
    // Ship non-TypeScript assets without compilation
    if (!target.endsWith('.ts')) continue;
    const name = target.replace(/^\.\/src\//, '').replace(/\.ts$/, '');
    found[name] = target;

    const published = manifest.publishConfig?.exports?.[subpath];
    if (published !== `./dist/${name}.js`) {
      throw new Error(
        `${manifest.name} must set publishConfig.exports['${subpath}'] to './dist/${name}.js' to match '${target}'`,
      );
    }
  }

  return found;
}

/** The modules a module re-exports at run time, or `undefined` if it does more than re-export. */
function reexported(source: ts.SourceFile) {
  const targets: string[] = [];
  for (const statement of source.statements) {
    if (!ts.isExportDeclaration(statement) || !statement.moduleSpecifier) return undefined;
    if (!statement.isTypeOnly) targets.push((statement.moduleSpecifier as ts.StringLiteral).text);
  }
  return targets;
}

/**
 * Splits the entries into modules for Vite and modules that only re-export, which ship as written:
 * Rolldown lowers `export * as` to getter objects, whose unused members bundlers cannot drop.
 */
async function inputs(names: Iterable<string>) {
  const built = new Map<string, string>();
  const reexports = new Map<string, string>();
  const visit = async (name: string) => {
    if (built.has(name) || reexports.has(name)) return;
    const file = `src/${name}.ts`;
    const text = await readFile(path.join(dir, file), 'utf8');
    const targets = reexported(ts.createSourceFile(file, text, ts.ScriptTarget.Latest));
    if (!targets) return void built.set(name, file);
    reexports.set(name, text);
    for (const target of targets)
      await visit(path.posix.join(path.posix.dirname(name), target).replace(/\.ts$/, ''));
  };
  for (const name of names) await visit(name);
  return { built: Object.fromEntries(built), reexports };
}

async function compile(
  manifest: Manifest,
  headers: ReadonlyMap<string, string>,
  index: (text: string) => string,
) {
  await rm(path.join(dir, 'dist'), { recursive: true, force: true });
  const { built, reexports } = await inputs(Object.keys(entries(manifest)));

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
      lib: { entry: built, formats: ['es'] },
      rolldownOptions: {
        external: (id) => !id.startsWith('.') && !path.isAbsolute(id),
        // Inlined constants leave a bare import of the module that held them, which esbuild warns
        // about in an app that links the kit outside node_modules, as a workspace does
        optimization: { inlineConst: false },
        output: {
          preserveModules: true,
          preserveModulesRoot: 'src',
          entryFileNames: '[name].js',
          // The maps point into the shipped src/, so they need no copy of it
          sourcemapExcludeSources: true,
        },
      },
    },
  });

  for (const [name, text] of reexports) {
    const out = path.join(dir, 'dist', `${name}.js`);
    const { outputText, sourceMapText } = ts.transpileModule(text, {
      fileName: `${path.basename(name)}.ts`,
      compilerOptions: {
        module: ts.ModuleKind.ESNext,
        rewriteRelativeImportExtensions: true,
        sourceMap: true,
        sourceRoot: path.relative(path.dirname(out), path.join(dir, 'src', path.dirname(name))),
      },
    });
    await writeFile(out, outputText);
    await writeFile(`${out}.map`, sourceMapText!);
  }

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
        outDir: 'dist',
      },
      include: ['src'],
    }),
  );

  try {
    execFileSync(path.join(repo, 'node_modules/.bin/tsc'), ['-p', config], { stdio: 'inherit' });
  } finally {
    await rm(config);
  }

  const unheaded = new Set(headers.keys());
  // tsc preserves `.ts` import extensions in declarations
  for (const file of await readdir(path.join(dir, 'dist'), { recursive: true })) {
    if (!file.endsWith('.d.ts')) continue;
    const declaration = path.join(dir, 'dist', file);
    const text = (await readFile(declaration, 'utf8')).replace(
      /(['"]\.{1,2}\/[^'"]+)\.ts(['"])/g,
      '$1.js$2',
    );
    // A project whose `lib` lacks this, as Vite's default does, reads `Disposable` and
    // `Symbol.dispose` as undefined, and `skipLibCheck` hides it: kit types then lose their option
    // checks and narrowed return types without an error
    const disposes = /\bDisposable\b|Symbol\.dispose/.test(text);
    const body = file === 'index.d.ts' ? index(text) : text;
    // Agents read declarations before docs, so each module's names where to read next
    const header = headers.get(file.replaceAll('\\', '/'));
    unheaded.delete(file.replaceAll('\\', '/'));
    const above = [header, disposes ? DISPOSABLE : undefined].filter(Boolean).join('\n');
    await writeFile(declaration, above ? `${above}\n${body}` : body);
    // Its map counts lines from the top, so it skips the lines added above, or go-to-definition
    // lands that many lines off in the source
    if (above) {
      const map = `${declaration}.map`;
      const json = JSON.parse(await readFile(map, 'utf8')) as { mappings: string };
      json.mappings = ';'.repeat(above.split('\n').length) + json.mappings;
      await writeFile(map, JSON.stringify(json));
    }
  }
  // A module moved or renamed keeps its header only if its path still matches
  if (unheaded.size > 0) throw new Error(`No declaration for headers: ${[...unheaded].join(', ')}`);
}

console.log("Writing the kit's docs");
const contracts = checkedContracts();
const docs = renderDocs(contracts);
const guide = await readFile(skillFile, 'utf8');
writeDocs(docsDir, docs);

console.log('Building @typegpu/kit');
await compile(
  JSON.parse(await readFile(path.join(dir, 'package.json'), 'utf8')) as Manifest,
  declarationHeaders(contracts, docs, guide),
  (text) => annotatedIndex(text, contracts, docs, guide),
);
