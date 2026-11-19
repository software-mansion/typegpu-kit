import { existsSync, readFileSync } from 'node:fs';
import { join, posix } from 'node:path';
import { describe, expect, it } from 'vitest';
import {
  annotatedIndex,
  declarationHeaders,
  exampleFiles,
  exampleLinks,
  harnessFiles,
  llmsGroups,
  pageFiles,
  renderDocs,
  renderLlms,
  renderPages,
  skillFile,
  stamped,
} from '../src/contracts/agents.ts';
import {
  exampleListing,
  exampleMarkdown,
  renderExample,
  renderGallery,
} from '../src/build/markdown.ts';
import { moved } from '../src/build/moved.ts';
import { exportGroups, type ModuleContract } from '../src/contracts/contract.ts';
import { extractContracts, repoRoot } from '../src/contracts/extract.ts';
import { anchorsOf, convertPage, linksOf, proseOf } from '../src/contracts/markdown.ts';
import { read } from '../src/source/read.ts';
import { examples as everyExample } from 'typegpu-kit-examples';

const { contracts } = extractContracts();
const docs = renderDocs(contracts);
const skill = readFileSync(skillFile, 'utf8');
const skillPath = 'skills/typegpu-kit/SKILL.md';

/** The kit's guide and docs/, by path from the kit's root. */
const kit = new Map([
  [skillPath, skill],
  ...[...docs.files].map(([path, text]) => [`docs/${path}`, text] as const),
]);
const pages = [...kit.keys()].filter((path) => path.endsWith('.md'));

/** Whether `path`, from the kit's root, is in docs/ as generated or in the package. */
const shipped = (path: string) =>
  kit.has(path) ||
  [...kit.keys()].some((each) => each.startsWith(path.endsWith('/') ? path : `${path}/`)) ||
  existsSync(join(repoRoot, 'packages/typegpu-kit', path));

/** The skill's own words, without its code blocks. */
const written = skill.replace(/^```[\s\S]*?^```$/gm, '');
/** What `text` puts in backticks. */
const spans = (text: string) => [...text.matchAll(/`([^`]+)`/g)].map(([, code]) => code!);
const namespaces = new Map(
  contracts.filter((each) => !each.flat).map((each) => [each.namespace, each] as const),
);
/** How code names a module: its namespace, or the names it exports flat. */
const namesOf = (contract: ModuleContract) =>
  contract.flat
    ? [...contract.factories.map((each) => each.name), ...contract.functions]
    : [contract.namespace];
const membersOf = (contract: ModuleContract) =>
  new Set([
    ...contract.factories.map((each) => each.name),
    ...exportGroups.flatMap((group) => contract[group]),
    ...contract.types,
  ]);

const options = { module: (id: string) => `The card of ${id}.`, index: 'modules.md' };

describe("the kit's guide", () => {
  it('keeps its rules under 8 KB and each line of its index under 256 bytes', () => {
    const body = skill.replace(/^---\n[\s\S]*?\n---\n/, '');
    const index = body.match(/^## Find it before you build it\n[\s\S]*?(?=^## )/m)![0];
    const long = index.split('\n').filter((line) => Buffer.byteLength(line) > 256);

    expect(
      Buffer.byteLength(body.replace(index, '')),
      'Rules are obeyed less as they multiply: move one to its page, or to the index if it routes',
    ).toBeLessThanOrEqual(8 * 1024);
    expect(long, 'An index line names one need: split it').toEqual([]);
  });

  it('and its docs link only to files and headings that are there', () => {
    const broken = pages.flatMap((from) =>
      linksOf(kit.get(from)!)
        .filter((href) => !/^[a-z]+:/.test(href))
        .filter((href) => {
          const [path = '', anchor] = href.split('#');
          const target = path ? posix.normalize(posix.join(posix.dirname(from), path)) : from;
          // A directory: an example, or all of them.
          if (path.endsWith('/')) return ![...kit.keys()].some((each) => each.startsWith(target));
          const text = kit.get(target);
          if (text === undefined) return true;
          return anchor !== undefined && !anchorsOf(text).has(anchor);
        })
        .map((href) => `${from}: ${href}`),
    );

    expect(broken).toEqual([]);
  });

  it('keeps a Markdown file at every published page', () => {
    const published: string[] = JSON.parse(
      readFileSync(join(repoRoot, 'apps/typegpu-kit-docs/src/build/addresses.json'), 'utf8'),
    );
    // Examples, their sources and the map are pages of the site alone
    const gone = published
      .filter((address) => !/#|^\/(examples|source|map)\//.test(address))
      .map((address) => (address === '/' ? 'docs/index.md' : `docs${address.slice(0, -1)}.md`))
      .filter((path) => !kit.has(path));

    expect(gone, 'Name where a moved page is in src/build/moved.ts').toEqual([]);
  });

  it('routes every module', () => {
    const named = new Set(spans(written).flatMap((code) => code.match(/\w+/g) ?? []));
    const unrouted = contracts
      .filter((each) => !namesOf(each).some((name) => named.has(name)))
      .map((each) => each.id);

    expect(unrouted, 'Say what each is for in "Find it before you build it"').toEqual([]);
  });

  it('names only members its modules export', () => {
    const unknown = spans(written).flatMap((code) =>
      [...code.matchAll(/\b([a-z]\w*)\.([a-zA-Z]\w*)/g)].flatMap(([ref, name, member]) => {
        const contract = namespaces.get(name!);
        return contract && !membersOf(contract).has(member!) ? [ref] : [];
      }),
    );

    expect(unknown).toEqual([]);
  });

  it('links examples that use every module named before them', () => {
    const files = [...exampleFiles()];
    const sourceOf = (href: string) => {
      const path = href.replace(/^\.\.\/\.\.\/docs\//, '');
      return files
        .filter(([each]) => (path.endsWith('/') ? each.startsWith(path) : each === path))
        .map(([, text]) => text)
        .join('\n');
    };
    const stale = exampleLinks(skill).flatMap(({ href, names }) => {
      const modules = names.filter((name) => namespaces.has(name));
      // A module's calls when the sentence names one, otherwise any name it puts in code
      const uses =
        modules.length > 0
          ? modules.map((each) => new RegExp(`\\b${each}\\.`))
          : names.map((each) => new RegExp(`\\b${each}\\b`));
      const source = sourceOf(href);
      return modules.length > 0
        ? uses.filter((use) => !use.test(source)).map((use) => `${href} never calls ${use.source}`)
        : uses.length > 0 && !uses.some((use) => use.test(source))
          ? [`${href} uses none of ${names.join(', ')}`]
          : [];
    });

    expect(stale).toEqual([]);
  });

  it('names in backticks only paths that ship, from its own folder', () => {
    expect(backtickPaths('`../../docs/a.md`, [`b/c.ts`](x), `@typegpu/kit/web`, `a / b`')).toEqual([
      '../../docs/a.md',
    ]);
    const missing = backtickPaths(skill).filter(
      (path) => !shipped(posix.normalize(posix.join(posix.dirname(skillPath), path))),
    );

    expect(missing).toEqual([]);
  });
});

/** The paths `markdown` names in backticks outside code blocks and link texts; not package names. */
function backtickPaths(markdown: string) {
  const prose = markdown
    .split(/^```[\s\S]*?^```$/m)
    .join('\n')
    .replace(/\[(?:[^\]`]|`[^`]*`)*\]\([^)]*\)/g, '');
  return [...prose.matchAll(/`([^`]+)`/g)]
    .map(([, code]) => code!)
    .filter((code) => /^[\w.-]+(\/[\w.-]+)+\/?$/.test(code));
}

describe('the declarations', () => {
  it('name their module, guide and examples, at paths from themselves that ship', () => {
    const headers = declarationHeaders(contracts, docs, skill);
    const barrel = annotatedIndex("export * as ao from './render/ao.js';", contracts, docs, skill);
    const named = [...headers, ['index.d.ts', barrel] as const].flatMap(([file, text]) =>
      [...text.matchAll(/((?:\.\.\/)+[^\s,]+)/g)].map(([, path]) => {
        const [target = '', anchor] = path!.replace(/\.$/, '').split('#');
        return { path: posix.normalize(posix.join('dist', posix.dirname(file), target)), anchor };
      }),
    );
    const missing = named.filter(({ path, anchor }) =>
      anchor === undefined ? !shipped(path) : !anchorsOf(kit.get(path) ?? '').has(anchor),
    );

    expect(headers.get('render/ao.d.ts')).toMatch(
      /^\/\/ `ao`: .* Guide: \.\.\/\.\.\/docs\/kit\/ao\.md\. Examples: \.\.\/\.\.\/docs\/examples\/robot-arm\//,
    );
    expect(barrel).toMatch(/^\/\*\* .*\.\.\/docs\/kit\/ao\.md.* \*\/ export \* as ao/);
    expect(missing).toEqual([]);
  });
});

describe('the pages as Markdown', () => {
  it('are every page but 404, the module index and a pointer for each page that moved', () => {
    const converted = pages.filter(
      (path) => path.startsWith('docs/') && !path.startsWith('docs/examples/'),
    );
    expect(converted.length).toBe(pageFiles().length + 1 + Object.keys(moved).length);
    expect(kit.get('docs/kit/shadows.md')).toContain('(lighting.md#shadows)');
    expect(converted).toContain('docs/kit/frames.md');
    expect(converted).toContain('docs/modules.md');
  });

  it('keep no component, import, MDX comment or twoslash marker', () => {
    const left = pages
      .filter((path) => path.startsWith('docs/'))
      .flatMap((path) => {
        const text = kit.get(path)!;
        const prose = proseOf(text).filter((line) =>
          /<\/?[A-Z]|^import\s|\{\/\*|^\s*:::/.test(line),
        );
        const markers = text
          .split('\n')
          .filter((line) =>
            /\/\/ ---cut|^\s*\/\/ @\w+|^\s*\/\/\s*\^\?|```\w*\s+twoslash/.test(line),
          );
        return [...prose, ...markers].map((line) => `${path}: ${line.trim()}`);
      });

    expect(left).toEqual([]);
  });

  it('ship every example with its source and meta.json, and a README naming the harness', () => {
    const examples = [...docs.files.keys()].filter((path) => path.startsWith('examples/'));
    expect(examples).toContain('examples/dominoes/scene.ts');
    expect(examples).toContain('examples/dominoes/meta.json');
    for (const file of harnessFiles) {
      expect(examples).toContain(`examples/${file}`);
      expect(docs.files.get('examples/README.md')).toContain(`\`${file}\``);
    }
    expect(
      examples.filter((path) => /\.test\.|\.html$|\.glb$|^examples\/hosts\//.test(path)),
    ).toEqual([]);
  });
});

describe('the examples as Markdown on the site', () => {
  it('list every example once, in the gallery and in /llms.txt', () => {
    const keys = everyExample.map((example) => example.key).toSorted();
    const linked = [...renderGallery().matchAll(/\]\(examples\/([\w-]+)\.md\)/g)].map(
      ([, key]) => key,
    );
    const listed = exampleListing().pages.map((page) => page.id.replace('examples/', ''));

    expect(linked.toSorted()).toEqual(keys);
    expect(listed.toSorted()).toEqual(keys);
  });

  it('hold each file whole, in reading order, and link the modules it imports', () => {
    const { sections } = renderPages(contracts, exampleMarkdown);
    const source = read('lattice');
    const example = everyExample.find((each) => each.key === 'lattice')!;
    const text = renderExample(example, source, contracts, sections);

    const files = [...text.matchAll(/^```ts title="([^"]+)"\n([\s\S]*?)\n```$/gm)];
    expect(files.map(([, name]) => name)).toEqual(source.files.map((file) => file.name));
    for (const [, name, code] of files) {
      expect(`${code}\n`).toBe(source.files.find((file) => file.name === name)!.code);
    }
    expect(text).toContain('[`taa`](../kit/taa.md)');
    expect(text).not.toContain('`Particle`');
  });
});

describe('/llms.txt', () => {
  it('lists every page as Markdown once, under the sidebar groups', () => {
    const llms = renderLlms(llmsGroups(contracts), (path) => `/base/${path}`, '1.2.3');
    const listed = [...llms.matchAll(/^- \[[^\]]*\]\(\/base\/([^)]*)\.md\)/gm)].map(([, id]) => id);

    expect(listed.toSorted()).toEqual(pageFiles().map(({ id }) => id));
    expect(llms).toContain('## Concepts');
    expect(llms).toContain('- [Frames](/base/kit/frames.md): ');
  });

  it('names the version it describes, and sends agents to the installed docs first', () => {
    const llms = renderLlms(llmsGroups(contracts), (path) => `/base/${path}`, '1.2.3');
    const head = llms.slice(0, llms.indexOf('\n## '));

    expect(head).toContain('`@typegpu/kit` 1.2.3');
    expect(head).toContain('read `node_modules/@typegpu/kit/docs/`');
    expect(stamped('# Frames\n\nOne encoder a frame.\n', '1.2.3')).toBe(
      '# Frames\n\n> `@typegpu/kit` 1.2.3. If it is installed, read `node_modules/@typegpu/kit/docs/` instead.\n\nOne encoder a frame.\n',
    );
  });
});

describe('converting a page', () => {
  it('heads it with its title and description, and drops imports, comments and figures with their captions', () => {
    const page = [
      '---',
      'title: Frames',
      'description: One encoder a frame.',
      '---',
      '',
      "import Figure from '../Figure.astro';",
      "import { Aside } from '@astrojs/starlight/components';",
      '',
      '{/* a note',
      '   on two lines */}',
      '<Figure />',
      '',
      '<Figure stage="one">',
      '  The <code>frame</code>, drawn.',
      '</Figure>',
      '',
      'See [the surface](/kit/surface/#sizing), [an example](/examples/blockfall/?debug) and [TypeGPU](https://typegpu.com).',
    ].join('\n');

    expect(convertPage('kit/frames', page, options).text).toBe(
      [
        '# Frames',
        '',
        'One encoder a frame.',
        '',
        'See [the surface](surface.md#sizing), [an example](../examples/blockfall/) and [TypeGPU](https://typegpu.com).',
        '',
      ].join('\n'),
    );
  });

  it('links examples where the site serves them, and rejects a heading a dropped component empties', () => {
    const page = [
      '---',
      'title: Clock',
      '---',
      '',
      '## Examples',
      '',
      '<Highlights>',
      '',
      '[Every example](/examples/) and [the jump flood](/examples/jump-flood/).',
      '',
      '</Highlights>',
    ].join('\n');
    const site = { ...options, example: exampleMarkdown };

    expect(convertPage('kit/clock', page, site).text).toContain(
      '[Every example](../examples.md) and [the jump flood](../examples/jump-flood.md).',
    );
    expect(() => convertPage('kit/clock', page.replaceAll('Highlights', 'Gallery'), site)).toThrow(
      '"## Examples" has nothing under it',
    );
  });

  it("keeps a twoslash block's shown lines, and of its meta only the title", () => {
    const page = [
      '```ts twoslash title="scene.ts" ins={2}',
      "import { tgpu } from 'typegpu';",
      '// ---cut---',
      'const root = await tgpu.init();',
      '// ---cut-start---',
      'declare const hidden: number;',
      '// ---cut-end---',
      '// @errors: 2345',
      'const x = root;',
      '//    ^?',
      '```',
      '',
      '```ts del={1} ins={2}',
      'a();',
      'b();',
      '```',
    ].join('\n');

    expect(convertPage('x', page, options).text).toBe(
      [
        '# x',
        '',
        '```ts title="scene.ts"',
        'const root = await tgpu.init();',
        'const x = root;',
        '```',
        '',
        '```diff',
        '-a();',
        '+b();',
        '```',
        '',
      ].join('\n'),
    );
  });

  it('turns asides into quotes, tabs into labelled blocks, cards into links and module cards into lines', () => {
    const page = [
      '---',
      'title: Finish',
      'modules: [kit/finish, kit/bloom]',
      '---',
      '',
      '## Bloom',
      '',
      '<Module id="kit/bloom" />',
      '',
      ':::caution[An open pass]',
      'Close it.',
      ':::',
      '',
      '<Aside type="tip">',
      '  Warm up.',
      '</Aside>',
      '',
      '<Tabs syncKey="platform">',
      '<TabItem label="Web">',
      'On the web.',
      '</TabItem>',
      '</Tabs>',
      '',
      '<LinkCard title="Resizing" description="The shadow map." href="/kit/lighting/#resizing" />',
      '',
      '<Install add="@typegpu/kit" dev="vitest" />',
    ].join('\n');

    const converted = convertPage('kit/finish', page, options);
    expect(converted.text).toBe(
      [
        '# Finish',
        '',
        '## Bloom',
        '',
        '> The card of kit/bloom.',
        '',
        '> **Caution: An open pass**',
        '>',
        '> Close it.',
        '',
        '> **Tip**',
        '>',
        '> Warm up.',
        '',
        '**Web**',
        '',
        'On the web.',
        '',
        '[Resizing](lighting.md#resizing): The shadow map.',
        '',
        '```sh',
        'npm i @typegpu/kit',
        'npm i -D vitest',
        '```',
        '',
      ].join('\n'),
    );
    expect([...converted.modules]).toEqual([['kit/bloom', 'bloom']]);
  });
});
