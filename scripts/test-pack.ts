import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { mkdtemp, readFile, readdir, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import ts from 'typescript';
import { build } from 'vite';

const repo = path.resolve(import.meta.dirname, '..');
const temporary = await mkdtemp(path.join(tmpdir(), 'kit-consumer-'));
const run = (command: string, args: string[], cwd = temporary) =>
  execFileSync(command, args, { cwd, stdio: 'inherit' });
const json = async (file: string) => JSON.parse(await readFile(file, 'utf8'));
const bytes = (chunks: { code: string }[]) =>
  chunks.reduce((sum, chunk) => sum + Buffer.byteLength(chunk.code), 0);

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
      dependencies: {
        ...dependencies,
        ...devDependencies,
        '@typegpu/kit': `file:./${tarball}`,
        esbuild: '0.28.2',
      },
    }),
  );
  // Esbuild's install script only checks the binary its optional dependency brings
  await writeFile(path.join(temporary, 'pnpm-workspace.yaml'), 'allowBuilds:\n  esbuild: false\n');
  run('pnpm', ['install', '--config.auto-install-peers=false']);
  const kit = path.join(temporary, 'node_modules/@typegpu/kit');
  const manifest = await json(path.join(kit, 'package.json'));
  const optional = new Map([
    ['./native/runtimes', 'react-native-worklets'],
    ['./debug/gpuDebugPanel', 'react-native-webgpu'],
    ['./native/host', 'react-native-worklets'],
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
  // Namespaces ship as native ESM, the module itself rather than an object of getters, so
  // bundlers drop the members a program never reads
  const packed = (file: string) => import(pathToFileURL(path.join(kit, 'dist', file)).href);
  const geometry = await packed('geometry/index.js');
  assert.equal(geometry.meshes, await packed('geometry/mesh/index.js'));
  assert.equal(geometry.meshes.patches, await packed('geometry/mesh/patches/index.js'));
  const sourceKeys = execFileSync(
    'node',
    [
      '--input-type=module',
      '-e',
      "console.log(JSON.stringify(Object.keys((await import('./src/geometry/index.ts')).meshes)))",
    ],
    { cwd: path.join(repo, 'packages/typegpu-kit'), encoding: 'utf8' },
  );
  assert.deepEqual(Object.keys(geometry.meshes), JSON.parse(sourceKeys));
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
      // Esbuild warns about a bare import of a module "sideEffects" leaves out, in any app that
      // links the kit outside node_modules
      if (ts.isImportDeclaration(node) && !node.importClause && specifier?.startsWith('.')) {
        const target = `./${path.posix.join('dist', path.posix.dirname(file), specifier)}`;
        assert(manifest.sideEffects.includes(target), `${file}: bare import ${specifier}`);
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
  const typeProbe = `import { tone, frames, type Frame } from '@typegpu/kit';\nimport { aces } from '@typegpu/kit/tone';\nimport type { TgpuRoot } from 'typegpu';\ndeclare const root: TgpuRoot;\nconst loop = frames.create(root, (frame: Frame) => { void frame.encoder; });\nloop.destroy();\nconst curve: typeof aces = tone.aces;\nvoid curve;\n`;
  // No starter resolves as NodeNext, so the probe checks it alone
  await writeFile(path.join(temporary, 'probe.ts'), typeProbe);
  run(path.join(temporary, 'node_modules/.bin/tsc'), [
    '--noEmit',
    '--skipLibCheck',
    '--strict',
    '--target',
    'esnext',
    '--module',
    'nodenext',
    '--moduleResolution',
    'nodenext',
    '--types',
    '@webgpu/types',
    'probe.ts',
  ]);
  // Each scene the starter offers type-checks and builds against the packed kit, with the probe
  // beside its code, outside what it imports, under the Bundler settings it ships
  for (const name of await readdir(path.join(starter, 'scenes'))) {
    run('node', [path.join(starter, 'index.js'), name, '--scene', name]);
    const scene = path.join(temporary, name);
    await writeFile(path.join(scene, 'src/kit-types.ts'), typeProbe);
    run(path.join(temporary, 'node_modules/.bin/tsc'), ['-p', scene]);
    run(path.join(temporary, 'node_modules/.bin/vite'), ['build', '--logLevel', 'warn'], scene);
  }
  const external = ['typegpu', '@typegpu/color', '@typegpu/noise'];
  /** Vite's minified chunks for an entry module, with TypeGPU's packages external. */
  const vite = async (name: string, code: string) => {
    const entry = path.join(temporary, `${name}.js`);
    await writeFile(entry, code);
    const result = await build({
      configFile: false,
      root: temporary,
      logLevel: 'error',
      build: {
        write: false,
        minify: true,
        lib: { entry, formats: ['es'] },
        rolldownOptions: {
          external: (id) => external.some((peer) => id === peer || id.startsWith(`${peer}/`)),
        },
      },
    });
    const output = Array.isArray(result) ? result[0] : result;
    assert(output && 'output' in output);
    return output.output.flatMap((item) => (item.type === 'chunk' ? [item] : []));
  };
  const root = bytes(
    await vite('root', "import { tone } from '@typegpu/kit'; export const curve = tone.aces;"),
  );
  const subpath = bytes(
    await vite('subpath', "export { aces as curve } from '@typegpu/kit/tone';"),
  );
  assert(root <= subpath * 1.15 + 100, `root tone bundle ${root} exceeds subpath ${subpath}`);
  /** Esbuild's minified bundle of an entry module, with TypeGPU's packages external. */
  const esbuild = (name: string, code: string) =>
    execFileSync(
      path.join(temporary, 'node_modules/.bin/esbuild'),
      [
        `--sourcefile=${name}.js`,
        '--bundle',
        '--minify',
        '--format=esm',
        '--log-level=warning',
        ...external.flatMap((peer) => [`--external:${peer}`, `--external:${peer}/*`]),
      ],
      { cwd: temporary, input: code, encoding: 'utf8', stdio: ['pipe', 'pipe', 'inherit'] },
    );
  // Esbuild keeps every member of a root namespace a program reads, so the docs teach member
  // imports from subpaths, which must bundle as small under esbuild as under Vite
  const esbuilt = Buffer.byteLength(
    esbuild('member', "export { aces as curve } from '@typegpu/kit/tone';"),
  );
  assert(
    esbuilt <= subpath * 1.15 + 100,
    `esbuild tone bundle ${esbuilt} exceeds Vite's ${subpath}`,
  );
  const box = await vite(
    'box',
    "import { meshes } from '@typegpu/kit/geometry'; export const shape = meshes.box;",
  );
  const kept = box.flatMap((chunk) => Object.keys(chunk.modules));
  assert(
    !kept.some((id) => id.endsWith('/geometry/mesh/patches/rounded.js')),
    'meshes.box keeps the unused rounded patches',
  );
  // TypeGPU constructors look effectful to bundlers, so module-scope definitions marked pure
  // drop with the code they hold when a program never reads them. Each import's minified bytes
  // stay under about 1.3 times what it bundles to, which losing any one mark exceeds
  const ceilings: [name: string, code: string, max: { esbuild: number; vite: number }][] = [
    ['shading', "export { Material } from '@typegpu/kit/shading';", { esbuild: 710, vite: 940 }],
    [
      'environment',
      "export { sunAt } from '@typegpu/kit/environment';",
      { esbuild: 3_800, vite: 4_800 },
    ],
    [
      'instances',
      "export { Placement, rotate } from '@typegpu/kit/instances';",
      { esbuild: 12_900, vite: 17_100 },
    ],
    ['lighting', "export { WHITE } from '@typegpu/kit/lighting';", { esbuild: 5_950, vite: 6_950 }],
    [
      'passes',
      "export { color, depth } from '@typegpu/kit/passes';",
      { esbuild: 4_500, vite: 2_000 },
    ],
    ['dof', "export { defaults } from '@typegpu/kit/dof';", { esbuild: 5_500, vite: 2_250 }],
    ['bloom', "export { defaults } from '@typegpu/kit/bloom';", { esbuild: 1_050, vite: 1_250 }],
    ['taa', "export { defaults } from '@typegpu/kit/taa';", { esbuild: 7_350, vite: 2_850 }],
    ['ao', "export { defaults } from '@typegpu/kit/ao';", { esbuild: 10_900, vite: 2_650 }],
  ];
  for (const [name, code, max] of ceilings) {
    const measured = {
      esbuild: Buffer.byteLength(esbuild(name, code)),
      vite: bytes(await vite(name, code)),
    };
    for (const bundler of ['esbuild', 'vite'] as const)
      assert(
        measured[bundler] <= max[bundler],
        `${name} bundles to ${measured[bundler]} B with ${bundler}, over its ${max[bundler]} B ceiling`,
      );
  }
  // A ceiling on poses would sit within a few percent of what it bundles to, so a string only
  // its unread functions hold guards it
  const poses = "export { sample } from '@typegpu/kit/poses';";
  const vited = (await vite('poses', poses)).map((chunk) => chunk.code).join('');
  for (const output of [vited, esbuild('poses', poses)])
    assert(!output.includes('poses_'), 'poses keeps its unread functions');
  console.log(
    `test:pack passed: exports, declarations, dependencies, starter, tree shaking (${root}/${subpath}/${esbuilt} B)`,
  );
} finally {
  await rm(temporary, { recursive: true, force: true });
}
