#!/usr/bin/env node
// Writes a starter into a new directory: `npm create @typegpu/kit my-scene`.

import { cpSync, existsSync, readdirSync, readFileSync, renameSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { parseArgs } from 'node:util';
import { ask, banner, choose, finish } from './ui.js';

/** The scenes to start from, each a directory of `scenes/` and a section of the Starter scenes page. */
const SCENES = /** @type {const} */ ([
  { name: 'shader', title: 'Shader', hint: 'one fragment function fills the screen' },
  { name: 'lit', title: 'Lit scene', hint: 'a mesh, a sun and its shadow, an orbit camera' },
  {
    name: 'finished',
    title: 'Finished frame',
    hint: 'the lit scene with HDR, bloom and a tone curve',
  },
  { name: 'simulation', title: 'Simulation', hint: 'particles a compute kernel moves each step' },
]);

/** What each package manager runs to install and to start Vite, by its user agent's name. */
const COMMANDS = {
  npm: ['npm install', 'npm run dev'],
  pnpm: ['pnpm install', 'pnpm dev'],
  yarn: ['yarn', 'yarn dev'],
  bun: ['bun install', 'bun run dev'],
};

const USAGE = `Usage: npm create @typegpu/kit [directory] [-- --scene <name>]

Writes a web project that draws with TypeGPU Kit. In a terminal it asks for what you leave out.

Scenes:
${SCENES.map(({ name, hint }) => `  ${name.padEnd(12)}${hint}`).join('\n')}

Options:
  --scene <name>  the scene to start from
  -y, --yes       ask nothing: the shader scene unless --scene names another
  -h, --help      print this
`;

const here = import.meta.dirname;

/** Why `dir` cannot take a project, if it cannot. */
const taken = (/** @type {string} */ dir) =>
  existsSync(dir) && readdirSync(dir).length > 0
    ? `${dir} exists and is not empty. Name a new directory.`
    : undefined;

/**
 * @param {string} dir
 * @param {string} scene
 */
function create(dir, scene) {
  cpSync(path.join(here, 'template'), dir, { recursive: true });
  cpSync(path.join(here, 'scenes', scene), path.join(dir, 'src'), { recursive: true });
  // npm leaves a file named .gitignore out of the tarball
  renameSync(path.join(dir, 'gitignore'), path.join(dir, '.gitignore'));

  const manifest = path.join(dir, 'package.json');
  const name = path
    .basename(path.resolve(dir))
    .toLowerCase()
    .replace(/[^a-z0-9._~-]+/g, '-');
  const json = JSON.parse(readFileSync(manifest, 'utf8'));
  writeFileSync(manifest, `${JSON.stringify({ ...json, name }, null, 2)}\n`);
}

try {
  const { values, positionals } = parseArgs({
    allowPositionals: true,
    options: {
      scene: { type: 'string' },
      yes: { type: 'boolean', short: 'y' },
      help: { type: 'boolean', short: 'h' },
    },
  });
  if (values.help) process.stdout.write(USAGE);
  else {
    const asks = process.stdin.isTTY && process.stdout.isTTY && !values.yes;
    if (process.stdout.isTTY) {
      banner(JSON.parse(readFileSync(path.join(here, 'package.json'), 'utf8')).version);
    }

    const dir =
      positionals[0] ??
      (asks ? await ask('Where should the project go?', 'my-scene', taken) : undefined);
    if (!dir) throw new Error('Name the directory to write: npm create @typegpu/kit my-scene');
    const refusal = taken(dir);
    if (refusal) throw new Error(refusal);

    const scene = values.scene ?? (asks ? await choose('Start from', SCENES) : 'shader');
    if (!SCENES.some(({ name }) => name === scene)) {
      throw new Error(
        `There is no ${scene} scene. Choose from ${SCENES.map(({ name }) => name).join(', ')}.`,
      );
    }
    create(dir, scene);

    const agent = /** @type {keyof typeof COMMANDS} */ (
      process.env.npm_config_user_agent?.split('/')[0] ?? 'npm'
    );
    const inPlace = path.resolve(dir) === process.cwd();
    // Quoted when a shell would split or expand it
    const entered = /^[\w./-]+$/.test(dir) ? dir : `'${dir.replaceAll("'", "'\\''")}'`;
    finish(`Wrote the project${inPlace ? '' : ` into ${dir}`}. Next:`, [
      ...(inPlace ? [] : [`cd ${entered}`]),
      ...(COMMANDS[agent] ?? COMMANDS.npm),
    ]);
  }
} catch (error) {
  console.error(error instanceof Error ? error.message : error);
  process.exitCode = 1;
}
