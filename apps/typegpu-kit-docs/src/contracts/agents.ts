import { existsSync, mkdirSync, readdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { dirname, join, posix } from 'node:path';
import { moved } from '../build/moved.ts';
import { buildIndex, rowsOf } from '../components/site/nav.ts';
import { moduleById, type ModuleContract } from './contract.ts';
import { extractContracts, repoRoot } from './extract.ts';
import {
  cardOf,
  factMarkdown,
  importOf,
  kinds,
  optionsMarkdown,
  UNSTABLE,
  type FactKey,
} from './facets.ts';
import { convertPage, frontmatterOf, modulesOf } from './markdown.ts';
import { textOf } from './signature.ts';

/** The kit's guide for coding agents, an Agent Skill that TanStack Intent finds by the package's keyword. */
export const skillFile = join(repoRoot, 'packages/typegpu-kit/skills/typegpu-kit/SKILL.md');
/** Where the kit's docs/ is written, two folders above the skill. Not checked in. */
export const docsDir = join(repoRoot, 'packages/typegpu-kit/docs');
export const pagesDir = join(repoRoot, 'apps/typegpu-kit-docs/src/content/docs');
export const examplesDir = join(repoRoot, 'packages/typegpu-kit-examples/src');
/** The site's figures. A folder named for a recipe holds the files that recipe shows. */
export const vizDir = join(repoRoot, 'apps/typegpu-kit-docs/src/components/viz');
/** Getting started, whose setup between its BEGIN:setup and END:setup comments names the patches. */
export const setupPage = join(pagesDir, 'getting-started.mdx');

/** The kit's version, from its package.json, which the site's Markdown names. */
export const kitVersion = (): string =>
  JSON.parse(readFileSync(join(repoRoot, 'packages/typegpu-kit/package.json'), 'utf8')).version;

/** The index of every module in docs/. */
export const modulesFile = 'modules.md';

/** The examples' files that belong to the docs site's host, which runs them, rather than to the kit. */
export const harnessFiles = ['contract.ts', 'hosts.ts', 'settings.ts', 'ui.ts', 'assets.ts'];

const GLOSSARY = [
  'Each entry links the import to its page and lists factories, purpose and non-default facts.',
  `**kind** describes the module. ${(['cpu', 'shader', 'platform', 'tool'] as const).map((kind) => `${kind} means ${kinds[kind]}`).join('; ')}. By default it is a module that ${kinds.module}.`,
  '**platform** names the supported platform.',
  `**stability** marks an unstable module, which ${UNSTABLE}.`,
  '**records** lists recording methods and their compute pass, render pass, encoder or frame arguments. **owns** lists resources, **allocates** says when recording allocates, and **lifecycle** lists compilation and cleanup methods.',
  "**injects** describes placeholders filled by `inject()`: hooks and a definition's members, one instance per pipeline, piped as `root.pipe(x.inject())`. An instance's members carry their fill, so one pipeline reads several instances.",
  '**needs** names module arguments. **reads** names modules whose fills the calling pipeline needs. **requires** names device features. **submits** declares independent submissions.',
  "Exports follow the factories. Each options type lists its fields, with `?` marking optional fields. A field that takes a `'use gpu'` function, which runs in a shader, says so; other functions run on the CPU.",
  'The entry ends with its source file and examples. Example source is in `examples/<key>/`, and `examples/README.md` says how to run one.',
].join(' ');

/** Facts the line leaves to other places: the table of bindings, and the examples at its end. */
const elsewhere = new Set<FactKey>(['binds', 'usedBy']);

const sentence = (text: string) => (/[.!?]$/.test(text) ? text : `${text}.`);

/**
 * A module's line in modules.md, and the head of its card on a page: everything its contract says.
 * `listed` leaves out the options' fields, which the card lists after it.
 */
export function lineOf(
  contract: ModuleContract,
  contracts: readonly ModuleContract[],
  page?: string,
  listed = false,
) {
  const card = cardOf(contract, contracts);
  const { lines, rest } = card.signature;
  const signatures = [...lines.map(textOf), ...rest].map((text) => `\`${text}\``).join(', ');
  // The docs site's host doesn't ship in docs/
  const shown = contract.usedBy.filter((key) => !key.startsWith('hosts/'));
  const examples = shown.length > 0 ? `; ${shown.join(', ')}` : '';
  const imported = `\`${importOf(contract)}\``;

  return [
    `${page ? `[${imported}](${page})` : imported}:`,
    signatures ? `${signatures}.` : undefined,
    sentence(contract.summary),
    ...card.facts.filter((fact) => !elsewhere.has(fact.key)).map(factMarkdown),
    listed ? undefined : optionsMarkdown(card.options),
    `\`${contract.file}\`${examples}.`,
  ]
    .filter(Boolean)
    .join(' ');
}

/** Code in backticks, double ones around code that holds one. */
const ticked = (text: string) => (text.includes('`') ? `\`\` ${text} \`\`` : `\`${text}\``);

/** A field or member as a list item: its type or signature, its doc and its default. */
const item = (head: string, doc?: string, fallback?: string) => {
  const words = [doc && sentence(doc), fallback && `Default ${sentence(fallback)}`];
  return `- ${[ticked(head), ...words].filter(Boolean).join(' ')}`;
};

/** A field or value with its type. */
const typed = (name: string, optional: boolean | undefined, type: string) =>
  `${name}${optional ? '?' : ''}: ${type}`;

/**
 * Each options type and each handle of a module as its card on a page lists them: a line per field
 * or member, with its type or signature, its doc and its default.
 */
export function listingsOf(contract: ModuleContract) {
  return [
    ...contract.options.map((table) => {
      const field = table.parameter.includes('.') ? ` as ${ticked(table.parameter)}` : '';
      return [
        `${ticked(table.name)}, taken by ${table.takenBy.map(ticked).join(', ')}${field}:`,
        ...table.fields.map((each) =>
          item(
            typed(each.name, each.optional, each.type),
            each.gpu ? `A \`'use gpu'\` function. ${each.doc ?? ''}`.trim() : each.doc,
            each.default,
          ),
        ),
      ].join('\n');
    }),
    ...contract.structs.map((struct) =>
      [
        `${ticked(struct.name)}, a schema:`,
        ...struct.fields.map((each) => item(typed(each.name, false, each.type), each.doc)),
      ].join('\n'),
    ),
    ...contract.handles.map((handle) =>
      [
        `${ticked(handle.name)}, from ${handle.from.map(ticked).join(', ')}:`,
        ...handle.members.map((each) =>
          item(each.signature ?? typed(each.name, each.optional, each.type!), each.doc),
        ),
      ].join('\n'),
    ),
  ];
}

/** A module's card on a page: its line, then its options types and handles. */
export const cardMarkdown = (contract: ModuleContract, contracts: readonly ModuleContract[]) =>
  [lineOf(contract, contracts, undefined, true), ...listingsOf(contract)].join('\n\n');

/**
 * The index: a glossary, then one line per module, under its package. `pages` gives a module's
 * section in docs/, `kit/frames.md#x`, which the line links.
 */
export function renderIndex(
  contracts: readonly ModuleContract[],
  pages: ReadonlyMap<string, string> = new Map(),
) {
  const lines = [
    `- \`${contracts[0]?.package}\``,
    ...contracts.map((contract) => `  - ${lineOf(contract, contracts, pages.get(contract.id))}`),
  ];

  return [GLOSSARY, '', ...lines].join('\n');
}

/**
 * What each module's accessors bind in a stage, a row per `@binds`, for budgeting a stage's 8. Its
 * columns are padded as the formatter pads them.
 */
export function renderBinds(contracts: readonly ModuleContract[]) {
  const rows = contracts.flatMap((contract) =>
    contract.binds.map((each) => [
      `\`${contract.namespace}\``,
      String(each.storage),
      String(each.uniform),
      sentence(each.note),
    ]),
  );

  const head = ['Module', 'Storage', 'Uniform', 'When a shader reads'];
  const widths = head.map((title, i) =>
    Math.max(title.length, ...rows.map((row) => row[i].length)),
  );
  const line = (cells: readonly string[]) =>
    `| ${cells.map((cell, i) => cell.padEnd(widths[i])).join(' | ')} |`;

  return [line(head), line(widths.map((width) => '-'.repeat(width))), ...rows.map(line)].join('\n');
}

/** docs/modules.md: the index, then the table of bindings. */
export function renderModules(
  contracts: readonly ModuleContract[],
  pages: ReadonlyMap<string, string>,
) {
  return [
    '# Modules',
    '',
    renderIndex(contracts, pages),
    '',
    '## What modules bind',
    '',
    "What a stage binds when its shaders read a module's accessors, from the contracts, for budgeting a stage's 8 storage and 12 uniform buffers, the minimum WebGPU guarantees. A buffer counts once however many accessors read it.",
    '',
    renderBinds(contracts),
    '',
  ].join('\n');
}

/**
 * The docs site's pages that ship as Markdown, all but 404 and the map, which `modules.md` stands
 * for: each page's collection id, `kit/frames`, and its file.
 */
export function pageFiles(dir = pagesDir) {
  return readdirSync(dir, { recursive: true, encoding: 'utf8' })
    .map((file) => file.replaceAll('\\', '/'))
    .filter((file) => /\.mdx?$/.test(file) && !/(^|\/)_/.test(file))
    .map((file) => ({ id: file.replace(/\.mdx?$/, ''), file: join(dir, file) }))
    .filter(({ id }) => id !== '404' && id !== 'map')
    .toSorted((a, b) => a.id.localeCompare(b.id));
}

/**
 * Every example's source as docs/ ships it, by path from docs/: its `.ts` files and `meta.json`
 * under `examples/<name>/`, and the files they share in `examples/`. Not tests, markup or assets.
 */
export function exampleFiles(dir = examplesDir) {
  const files = new Map<string, string>();
  const paths = readdirSync(dir, { recursive: true, encoding: 'utf8' })
    .map((file) => file.replaceAll('\\', '/'))
    .filter(
      (path) =>
        (/\.tsx?$/.test(path) && !/\.test(\.gpu)?\.tsx?$/.test(path)) ||
        (path.includes('/') && path.endsWith('/meta.json')),
    )
    .toSorted();
  for (const path of paths) files.set(`examples/${path}`, readFileSync(join(dir, path), 'utf8'));
  return files;
}

/**
 * Each recipe's files, which its page shows in parts, whole beside its Markdown by path from docs/:
 * `recipes/lit-models/look.ts`, from the figure that runs them.
 */
export function recipeFiles(pages = pageFiles(), dir = vizDir) {
  const files = new Map<string, string>();
  for (const { id } of pages) {
    const folder = join(dir, id.replace(/^recipes\//, ''));
    if (!id.startsWith('recipes/') || !existsSync(folder)) continue;
    for (const file of readdirSync(folder)
      .filter((each) => each.endsWith('.ts'))
      .toSorted()) {
      files.set(`${id}/${file}`, readFileSync(join(folder, file), 'utf8'));
    }
  }
  return files;
}

/** docs/examples/README.md: what an example is, which files are the host's, and how to run one. */
export function examplesReadme() {
  const harness = harnessFiles.map((file) => `\`${file}\``);
  return `# Examples

Each directory holds one example. Its \`scene.ts\` exports \`createScene(root, surface, options?)\`, which builds the example's modules and returns a scene: \`frame(frame)\` records a frame and its optional \`destroy()\` frees its textures, non-uniform buffers and modules. The modules beside it are the example's own.

${harness.slice(0, -1).join(', ')} and ${harness.at(-1)} here, and an example's own \`ui.ts\` and \`assets.ts\`, belong to the docs site's host, which runs every example with its settings panel, buttons and files. They are not part of \`@typegpu/kit\`, and a program of your own needs none of them. The other files here are helpers several examples share.

Run an example on a canvas with \`web.host\` from \`@typegpu/kit/web\`:

\`\`\`ts
import * as web from '@typegpu/kit/web';
import { createScene } from './slime-mould/scene.ts';

web.host(createScene, { canvas: document.querySelector('canvas')! });
\`\`\`

An example with an \`assets.ts\` takes its loaded files as the scene's options, \`web.host\`'s second argument.
`;
}

export interface Docs {
  /** Every file of docs/, by its path there. */
  readonly files: ReadonlyMap<string, string>;
  /** Each module's section, from docs/: `kit/frames.md`, `kit/finish.md#bloom`. */
  readonly sections: ReadonlyMap<string, string>;
}

/**
 * The pages as Markdown and modules.md, in memory, as docs/ and the site serve them. `example`
 * gives an example's file from its slug where a page links it, by default its folder.
 */
export function renderPages(
  contracts: readonly ModuleContract[],
  example?: (slug: string) => string,
): Docs {
  const files = new Map<string, string>();
  const sections = new Map<string, string>();

  for (const { id, file } of pageFiles()) {
    const page = convertPage(id, readFileSync(file, 'utf8'), {
      module: (each) => cardMarkdown(moduleById(contracts, each), contracts),
      index: modulesFile,
      example,
    });
    files.set(page.path, page.text);
    for (const [each, anchor] of page.modules) {
      sections.set(each, `${page.path}${anchor ? `#${anchor}` : ''}`);
    }
  }

  // A page that moved keeps its Markdown address, as a pointer to where its content is
  for (const [from, to] of Object.entries(moved)) {
    const [page = '', anchor] = to.split('#');
    const target = `${page.replace(/\/$/, '')}.md${anchor ? `#${anchor}` : ''}`;
    const href = posix.relative(posix.dirname(from), target);
    files.set(`${from}.md`, `# Moved\n\nThis page is at [${target}](${href}).\n`);
  }

  files.set(modulesFile, renderModules(contracts, sections));
  return { files, sections };
}

/** docs/, in memory: the pages, modules.md, the examples and the recipes' files. */
export function renderDocs(contracts: readonly ModuleContract[]): Docs {
  const { files, sections } = renderPages(contracts);
  const all = new Map(files);
  for (const [path, text] of exampleFiles()) all.set(path, text);
  for (const [path, text] of recipeFiles()) all.set(path, text);
  all.set('examples/README.md', examplesReadme());
  return { files: all, sections };
}

/** How a module's entry names it: its namespace, or the names it exports flat. */
function namesOf(contract: ModuleContract) {
  if (!contract.flat) return contract.namespace;
  const factories = [...new Set(contract.factories.map((factory) => factory.name))];
  const names = factories.length > 0 ? factories : contract.functions;
  return names.length > 0 ? names.join(', ') : contract.namespace;
}

/**
 * Each link to an example in the skill's own words, with the names its sentence puts in code
 * before it, since the link before: `ao` for robot-arm in "Creases: `ao`; [robot-arm](…)".
 */
export function exampleLinks(skill: string) {
  const links: { href: string; names: string[] }[] = [];
  const parts = skill.split('\n').flatMap((line) => line.split(/(?<=\.) /));
  for (const part of parts) {
    let from = 0;
    for (const match of part.matchAll(/\]\((\.\.\/\.\.\/docs\/examples\/[^)]+)\)/g)) {
      const names = [...part.slice(from, match.index).matchAll(/`([^`]+)`/g)].flatMap(
        ([, code]) => code!.match(/\b[a-zA-Z]\w*/g) ?? [],
      );
      links.push({ href: match[1], names });
      from = match.index + match[0].length;
    }
  }
  return links;
}

/** Each namespace's examples, in order, that the skill links after naming it. */
function skillPicks(skill: string, contracts: readonly ModuleContract[]) {
  const known = new Set(contracts.filter((each) => !each.flat).map((each) => each.namespace));
  const picks = new Map<string, string[]>();
  for (const { href, names } of exampleLinks(skill)) {
    const key = href.match(/docs\/examples\/([\w-]+)\//)?.[1];
    if (!key) continue;
    for (const name of names.filter((each) => known.has(each))) {
      picks.set(name, [...new Set([...(picks.get(name) ?? []), key])]);
    }
  }
  return picks;
}

/**
 * Up to three examples for a module: those the skill links beside it, then those whose files call
 * it most.
 */
function examplesOf(contract: ModuleContract, docs: Docs, picks: ReadonlyMap<string, string[]>) {
  const calls = new RegExp(`\\b${contract.namespace}\\.`, 'g');
  const count = (key: string) =>
    [...docs.files]
      .filter(([path]) => path.startsWith(`examples/${key}/`))
      .reduce((sum, [, text]) => sum + (text.match(calls)?.length ?? 0), 0);
  const users = contract.usedBy
    .filter((key) => !key.startsWith('hosts/'))
    .map((key) => [key, count(key)] as const)
    .toSorted((a, b) => b[1] - a[1])
    .map(([key]) => key);
  // A pick whose files never call the module came from a sentence naming several
  const picked = (picks.get(contract.namespace) ?? []).filter((key) => count(key) > 0);
  return [...new Set([...picked, ...users])].slice(0, 3);
}

/** A module's guide and examples, as paths from a file `root` names the package's folder from. */
function pointersOf(
  contract: ModuleContract,
  docs: Docs,
  picks: ReadonlyMap<string, string[]>,
  root: string,
) {
  const examples = examplesOf(contract, docs, picks).map((key) => `${root}docs/examples/${key}/`);
  return [
    `Guide: ${root}docs/${docs.sections.get(contract.id) ?? modulesFile}.`,
    examples.length > 0 ? `Examples: ${examples.join(', ')}.` : undefined,
  ]
    .filter(Boolean)
    .join(' ');
}

/**
 * The comments that open each module's declaration file in dist/, by its path there
 * (`render/ao.d.ts`): what the module is, its guide and examples, and where every module is listed,
 * as paths from that file. Agents read declarations before docs, so each one names where to read
 * next.
 */
export function declarationHeaders(
  contracts: readonly ModuleContract[],
  docs: Docs,
  skill: string,
) {
  const picks = skillPicks(skill, contracts);
  const headers = new Map<string, string[]>();
  for (const contract of contracts) {
    if (contract.vendored) continue;
    const path = contract.file.replace(/^src\//, '').replace(/\.tsx?$/, '.d.ts');
    // From dist/ and the folders below it, up to the package's folder
    const root = '../'.repeat(path.split('/').length);
    const lines = headers.get(path) ?? [
      `Every module: ${root}docs/modules.md. Rules and tasks: ${root}skills/typegpu-kit/SKILL.md.`,
    ];
    // Before the line naming every module
    lines.splice(
      -1,
      0,
      `\`${namesOf(contract)}\`: ${sentence(contract.summary)} ${pointersOf(contract, docs, picks, root)}`,
    );
    headers.set(path, lines);
  }
  return new Map(
    [...headers].map(([path, lines]) => [path, lines.map((line) => `// ${line}`).join('\n')]),
  );
}

/**
 * `declaration`, the kit's dist/index.d.ts, with each `export * as name` preceded by its module's
 * summary and guide, so the barrel reads as a catalogue. Each comment shares its export's line, so
 * the declaration map's lines still match.
 */
export function annotatedIndex(
  declaration: string,
  contracts: readonly ModuleContract[],
  docs: Docs,
  skill: string,
) {
  const picks = skillPicks(skill, contracts);
  const main = contracts.filter((each) => each.specifier === each.package && !each.flat);
  return declaration.replace(/^export \* as (\w+) from .*$/gm, (line, name: string) => {
    const contract = main.find((each) => each.namespace === name);
    if (!contract) return line;
    return `/** ${sentence(contract.summary)} ${pointersOf(contract, docs, picks, '../')} */ ${line}`;
  });
}

/** A page of the site as /llms.txt lists it. */
export interface ListedPage {
  /** Its collection id: `kit/frames`, `index`. */
  readonly id: string;
  readonly title: string;
  readonly description?: string | undefined;
}

/** The pages /llms.txt lists, under the sidebar's groups and in its order, each page once. */
export function llmsGroups(contracts: readonly ModuleContract[]) {
  const listed = new Map(
    pageFiles().map(({ id, file }) => {
      const source = readFileSync(file, 'utf8');
      const { fields } = frontmatterOf(source);
      const page: ListedPage = {
        id,
        title: fields.get('title') ?? id,
        description: fields.get('description'),
      };
      return [id === 'index' ? '' : id, { page, modules: modulesOf(source) }] as const;
    }),
  );
  const pages = [...listed].map(([id, { page, modules }]) => ({
    id,
    title: page.title,
    modules: modules.map((each) => ({ id: each })),
  }));

  return buildIndex(contracts, pages).map((group) => ({
    label: group.label,
    pages: [
      ...new Set(
        rowsOf(group)
          .filter((row) => !row.block)
          .map((row) => row.page),
      ),
    ].flatMap((id) => listed.get(id)?.page ?? []),
  }));
}

/** Where an agent with the kit installed reads instead of the site, which may describe another version. */
const installed =
  'If `@typegpu/kit` is installed, read `node_modules/@typegpu/kit/docs/` and its `skills/typegpu-kit/SKILL.md` instead of this site. They match the installed code.';

/**
 * /llms.txt: the site's pages as Markdown, in `groups` of the sidebar's order, each with its
 * description, under the kit's `version`. `href` gives a Markdown file's address, `kit/frames.md`.
 */
export function renderLlms(
  groups: readonly { readonly label: string; readonly pages: readonly ListedPage[] }[],
  href: (path: string) => string,
  version: string,
) {
  const line = (page: ListedPage) =>
    `- [${page.title}](${href(`${page.id}.md`)})${page.description ? `: ${page.description}` : ''}`;

  return [
    '# TypeGPU Kit',
    '',
    "> GPU modules on TypeGPU for WebGPU and React Native. A module borrows the caller's TypeGPU root and records into the encoder or pass it is given; the caller owns the device, the loop and the submit.",
    '',
    `These pages describe \`@typegpu/kit\` ${version}. ${installed}`,
    '',
    `Every page is served as Markdown at its address with \`.md\` in place of the trailing slash. [${modulesFile}](${href(modulesFile)}) lists every module with its import, factories, options and bindings.`,
    ...groups.flatMap((group) => ['', `## ${group.label}`, '', ...group.pages.map(line)]),
    '',
  ].join('\n');
}

/** A page of the site as Markdown, its title followed by the kit's `version` it describes. */
export function stamped(text: string, version: string) {
  const stamp = `> \`@typegpu/kit\` ${version}. If it is installed, read \`node_modules/@typegpu/kit/docs/\` instead.`;
  const title = /^# .*\n/.exec(text)?.[0] ?? '';
  return `${title}${title ? '\n' : ''}${stamp}\n\n${text.slice(title.length).replace(/^\n/, '')}`;
}

/** Writes docs/ into `dir`, emptying it first. */
export function writeDocs(dir: string, docs: Docs) {
  rmSync(dir, { recursive: true, force: true });
  for (const [path, text] of docs.files) {
    const file = join(dir, path);
    mkdirSync(dirname(file), { recursive: true });
    writeFileSync(file, text);
  }
}

/** The contracts; throws on the issues that keep them from matching the code. */
export function checkedContracts() {
  const { contracts, issues } = extractContracts();
  if (issues.length > 0) {
    throw new Error(
      [
        'The contracts disagree with the code; fix them first:',
        ...issues.map((each) => `- ${each}`),
      ].join('\n'),
    );
  }
  return contracts;
}

if (import.meta.main) {
  const contracts = checkedContracts();
  const docs = renderDocs(contracts);
  writeDocs(docsDir, docs);
  console.log(`Wrote ${docs.files.size} files into ${docsDir}.`);
}
