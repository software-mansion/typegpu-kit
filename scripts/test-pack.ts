import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { mkdtemp, readFile, readdir, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import ts from 'typescript';
import { build } from 'vite';

const repo = path.resolve(import.meta.dirname, '..');
const temporary = await mkdtemp(path.join(tmpdir(), 'kit-consumer-'));
const run = (command: string, args: string[], cwd = temporary) =>
  execFileSync(command, args, { cwd, stdio: 'inherit' });
const json = async (file: string) => JSON.parse(await readFile(file, 'utf8'));

/** The source line, from 1, that a source map's `mappings` give each generated line first. */
function sourceLines(mappings: string) {
  const digits = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';
  let line = 0;
  return mappings.split(';').map((segments) => {
    let first: number | undefined;
    for (const segment of segments.split(',').filter(Boolean)) {
      // Base64 VLQ: five bits a digit, continued while bit 6 is set, the sign in the lowest bit
      const values: number[] = [];
      let value = 0;
      let shift = 0;
      for (const char of segment) {
        const digit = digits.indexOf(char);
        value += (digit & 31) << shift;
        if (digit & 32) shift += 5;
        else {
          values.push(value & 1 ? -(value >>> 1) : value >>> 1);
          value = 0;
          shift = 0;
        }
      }
      if (values.length >= 4) {
        line += values[2]!;
        first ??= line + 1;
      }
    }
    return first;
  });
}

try {
  run('pnpm', ['--filter', '@typegpu/kit', 'pack', '--pack-destination', temporary], repo);
  const tarball = (await readdir(temporary)).find((file) => file.endsWith('.tgz'));
  assert(tarball, 'pack produced a tarball');
  // The consumer installs what the starter names, with the packed kit
  const starter = path.join(repo, 'packages/typegpu-create-kit');
  const { dependencies, devDependencies } = await json(path.join(starter, 'template/package.json'));
  await writeFile(
    path.join(temporary, 'package.json'),
    JSON.stringify({
      private: true,
      type: 'module',
      dependencies: { ...dependencies, ...devDependencies, '@typegpu/kit': `file:./${tarball}` },
    }),
  );
  run('pnpm', ['install', '--config.auto-install-peers=false']);
  const kit = path.join(temporary, 'node_modules/@typegpu/kit');
  const manifest = await json(path.join(kit, 'package.json'));
  const optional = new Map([
    ['./native/runtimes', 'react-native-webgpu'],
    ['./debug/gpuDebugPanel', 'react-native-webgpu'],
    ['./native/host', 'react-native-webgpu'],
    ['./native/react', 'react'],
    ['./fontBaker', '@pmndrs/glyph'],
    ['./testing/dawn', 'vitest'],
    ['./testing/warnings', 'vitest'],
    ['./testing/failOnWarnings', 'vitest'],
  ]);
  const exports = Object.entries(manifest.exports) as [string, string][];
  for (const key of optional.keys())
    assert(
      exports.some(([name]) => name === key),
      `stale skip: ${key}`,
    );
  for (const [subpath, target] of exports) {
    if (subpath.includes('*') || !target.endsWith('.js')) continue;
    const specifier = `@typegpu/kit${subpath === '.' ? '' : subpath.slice(1)}`;
    const peer = optional.get(subpath);
    const probe = `await import(${JSON.stringify(specifier)})`;
    if (peer) {
      let failed = false;
      try {
        execFileSync('node', ['--input-type=module', '-e', probe], {
          cwd: temporary,
          stdio: 'pipe',
        });
      } catch (error) {
        assert(
          String((error as { stderr?: Buffer }).stderr).includes(peer),
          `${subpath} failed for a different reason: ${error}`,
        );
        failed = true;
      }
      assert(failed, `stale optional-peer skip: ${subpath}`);
    } else run('node', ['--input-type=module', '-e', probe]);
  }
  // What agents read keeps its paths, which the exports name
  for (const file of ['skills/typegpu-kit/SKILL.md', 'docs/index.md', 'docs/modules.md']) {
    const found = `new URL(import.meta.resolve('@typegpu/kit/${file}'))`;
    run('node', ['--input-type=module', '-e', `(await import('node:fs')).accessSync(${found})`]);
  }
  const declared = new Set([
    ...Object.keys(manifest.dependencies),
    ...Object.keys(manifest.peerDependencies),
  ]);
  for (const file of await readdir(path.join(kit, 'dist'), { recursive: true })) {
    if (!file.endsWith('.js') && !file.endsWith('.d.ts')) continue;
    const source = ts.createSourceFile(
      file,
      await readFile(path.join(kit, 'dist', file), 'utf8'),
      ts.ScriptTarget.Latest,
    );
    const visit = (node: ts.Node) => {
      let specifier: string | undefined;
      if (
        (ts.isImportDeclaration(node) || ts.isExportDeclaration(node)) &&
        node.moduleSpecifier &&
        ts.isStringLiteral(node.moduleSpecifier)
      )
        specifier = node.moduleSpecifier.text;
      if (
        ts.isImportTypeNode(node) &&
        ts.isLiteralTypeNode(node.argument) &&
        ts.isStringLiteral(node.argument.literal)
      )
        specifier = node.argument.literal.text;
      if (
        ts.isCallExpression(node) &&
        (node.expression.kind === ts.SyntaxKind.ImportKeyword ||
          (ts.isIdentifier(node.expression) && node.expression.text === 'require')) &&
        node.arguments[0] &&
        ts.isStringLiteral(node.arguments[0])
      )
        specifier = node.arguments[0].text;
      if (specifier && !specifier.startsWith('.') && !specifier.startsWith('node:')) {
        const name = specifier
          .split('/')
          .slice(0, specifier.startsWith('@') ? 2 : 1)
          .join('/');
        assert(declared.has(name), `${file}: undeclared import ${specifier}`);
      }
      ts.forEachChild(node, visit);
    };
    visit(source);
  }
  // Go-to-definition follows each declaration map to the kit's source
  let followed = 0;
  for (const file of await readdir(path.join(kit, 'dist'), { recursive: true })) {
    if (!file.endsWith('.d.ts')) continue;
    const declaration = path.join(kit, 'dist', file);
    const map = await json(`${declaration}.map`);
    const source = (
      await readFile(path.resolve(path.dirname(declaration), map.sources[0]), 'utf8')
    ).split('\n');
    const mapped = sourceLines(map.mappings);
    (await readFile(declaration, 'utf8')).split('\n').forEach((line, at) => {
      const name = line.match(/^export (?:declare )?(?:function|const|interface|type) (\w+)/)?.[1];
      const to = mapped[at];
      if (!name || !to) return;
      assert(source[to - 1]?.includes(name), `${file}:${at + 1} maps ${name} to line ${to}`);
      followed++;
    });
  }
  assert(followed > 100, `only ${followed} declarations followed to their source`);
  await writeFile(
    path.join(temporary, 'probe.ts'),
    `import { tone, frames, type Frame } from '@typegpu/kit';\nimport { aces } from '@typegpu/kit/tone';\nimport type { TgpuRoot } from 'typegpu';\ndeclare const root: TgpuRoot;\nconst loop = frames.create(root, (frame: Frame) => { void frame.encoder; });\nloop.destroy();\nconst curve: typeof aces = tone.aces;\nvoid curve;\n`,
  );
  for (const resolution of ['bundler', 'nodenext']) {
    run(path.join(temporary, 'node_modules/.bin/tsc'), [
      '--noEmit',
      '--skipLibCheck',
      '--strict',
      '--target',
      'esnext',
      '--module',
      resolution === 'bundler' ? 'esnext' : 'nodenext',
      '--moduleResolution',
      resolution,
      '--types',
      '@webgpu/types',
      'probe.ts',
    ]);
  }
  // Each scene the starter offers type-checks and builds against the packed kit
  for (const name of await readdir(path.join(starter, 'scenes'))) {
    run('node', [path.join(starter, 'index.js'), name, '--scene', name]);
    const scene = path.join(temporary, name);
    run(path.join(temporary, 'node_modules/.bin/tsc'), ['-p', scene]);
    run(path.join(temporary, 'node_modules/.bin/vite'), ['build', '--logLevel', 'warn'], scene);
  }
  const sizes: number[] = [];
  for (const [name, code] of [
    ['root', "import { tone } from '@typegpu/kit'; export const curve = tone.aces;"],
    ['subpath', "export { aces as curve } from '@typegpu/kit/tone';"],
  ]) {
    const entry = path.join(temporary, `${name}.js`);
    await writeFile(entry, code!);
    const result = await build({
      configFile: false,
      root: temporary,
      logLevel: 'error',
      build: {
        write: false,
        minify: true,
        lib: { entry, formats: ['es'] },
        rolldownOptions: {
          external: (id) =>
            id === 'typegpu' ||
            id.startsWith('typegpu/') ||
            id.startsWith('@typegpu/color') ||
            id.startsWith('@typegpu/noise'),
        },
      },
    });
    const output = Array.isArray(result) ? result[0] : result;
    assert(output && 'output' in output);
    sizes.push(
      output.output.reduce(
        (sum, item) => sum + (item.type === 'chunk' ? Buffer.byteLength(item.code) : 0),
        0,
      ),
    );
  }
  assert(
    sizes[0]! <= sizes[1]! * 1.15 + 100,
    `root tone bundle ${sizes[0]} exceeds subpath ${sizes[1]}`,
  );
  console.log(
    `test:pack passed: exports, declarations, dependencies, starter, tree shaking (${sizes.join('/')} B)`,
  );
} finally {
  await rm(temporary, { recursive: true, force: true });
}
