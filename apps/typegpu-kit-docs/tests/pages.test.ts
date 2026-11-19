import { readdirSync, readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import {
  checkPlacement,
  pageLedBy,
  pageless,
  placedIn,
  withPages,
} from '../src/components/site/modules.ts';
import {
  areas,
  buildIndex,
  checkAreas,
  isCurrent,
  linksOf,
  neighbours,
} from '../src/components/site/nav.ts';
import { extractContracts } from '../src/contracts/extract.ts';

const { contracts } = extractContracts();

/** A docs page as the collection gives it, with its modules as references. */
const page = (id: string, ...modules: string[]) => ({
  id,
  title: id,
  modules: modules.map((each) => ({ id: each, collection: 'modules' })),
});

/** Every module on a page of its own, but those on `shared` and the ones that have no page. */
const pagesWith = (...shared: ReturnType<typeof page>[]) => [
  ...contracts
    .filter((contract) => !pageless.includes(contract.id))
    .filter((contract) => !shared.some((each) => each.modules.some((m) => m.id === contract.id)))
    .map((contract) => page(pageLedBy(contract), contract.id)),
  ...shared,
];

describe('pages and modules', () => {
  it('give each module the page that lists it', () => {
    const paged = withPages(contracts, pagesWith(page('kit/web', 'kit/web', 'kit/native')));
    expect(paged.find((module) => module.id === 'kit/native')?.page).toBe('kit/web');
    expect(paged.filter((module) => !module.page).map((module) => module.id)).toEqual(pageless);
  });

  it('throw on an unknown module, a module on two pages, and a module on none', () => {
    expect(() => withPages(contracts, [...pagesWith(), page('kit/odd', 'kit/nothing')])).toThrow(
      'The page kit/odd lists kit/nothing, which is no module.',
    );
    expect(() =>
      withPages(contracts, [
        ...pagesWith(page('kit/clock', 'kit/clock', 'kit/frames')),
        page('kit/frames', 'kit/frames'),
      ]),
    ).toThrow('kit/frames is on two pages, kit/clock and kit/frames.');
    expect(() =>
      withPages(
        contracts,
        pagesWith().filter((each) => each.id !== 'kit/frames'),
      ),
    ).toThrow('No page lists kit/frames');
    expect(() => withPages(contracts, [...pagesWith(), page('kit/inject', 'kit/inject')])).toThrow(
      'kit/inject is on kit/inject; take it off pageless.',
    );
  });

  it("live in kit/, each at its lead's kebab-case name", () => {
    expect(pageLedBy(contracts.find((contract) => contract.id === 'kit/meshPool')!)).toBe(
      'kit/mesh-pool',
    );
    expect(() =>
      withPages(contracts, pagesWith(page('packages/finish', 'kit/finish', 'kit/bloom'))),
    ).toThrow('The page packages/finish leads with kit/finish, so it belongs at kit/finish.mdx');
    expect(() =>
      withPages(contracts, pagesWith(page('kit/bloom', 'kit/finish', 'kit/bloom'))),
    ).toThrow('belongs at kit/finish.mdx');
    expect(() => withPages(contracts, [...pagesWith(), page('kit/guide')])).toThrow(
      'The page kit/guide lists no modules, but kit/ holds only module pages.',
    );
    expect(() => withPages(contracts, [...pagesWith(), page('concepts/guide')])).not.toThrow();
  });

  it("give each module on a page a row in the index, the lead's opening the page", () => {
    const links = linksOf(
      buildIndex(contracts, pagesWith(page('kit/finish', 'kit/finish', 'kit/bloom'))),
    );
    const rows = links.filter((link) => link.module);

    expect(rows.length).toBe(contracts.length - pageless.length);
    expect(rows.find((link) => link.module === 'finish')).toMatchObject({ href: '/kit/finish/' });
    expect(rows.find((link) => link.module === 'bloom')).toMatchObject({
      href: '/kit/finish/#module-bloom',
      block: true,
    });
    expect(
      links.filter((link) => isCurrent(link, 'kit/finish')).map((link) => link.module),
    ).toEqual(['finish']);
  });
  it("run each area's rows under labels, and throw on a module none places", () => {
    const groups = buildIndex(contracts, pagesWith());
    expect(groups.map((group) => group.label)).toEqual([
      'Start',
      'Tutorials',
      'Concepts',
      'Recipes',
      'Core',
      'Rendering',
      'Post-processing',
      'Particles',
      'Text',
      'Assets',
    ]);
    for (const group of groups.slice(4)) {
      expect(group.links, group.label).toEqual([]);
      expect(group.sections?.length, group.label).toBeGreaterThan(0);
    }
    expect(
      groups.find((group) => group.label === 'Rendering')?.sections?.[0]?.links[0],
    ).toMatchObject({ module: 'meshes', href: '/kit/meshes/' });
    const kit = groups.find((group) => group.label === 'Core');
    expect(kit?.sections?.map((section) => section.label)).toEqual([
      'Frame',
      'Passes and targets',
      'Tables and lists',
      'Kernels',
      'Reading back',
      'Hosts',
      'Testing and debugging',
    ]);
    expect(kit?.sections?.[0]?.links.map((link) => link.module)).toEqual([
      'frames',
      'clock',
      'coroutines',
      'pipelines',
    ]);

    const frames = contracts.find((contract) => contract.id === 'kit/frames')!;
    const extra = { ...frames, id: 'kit/extra', namespace: 'extra' };
    expect(() =>
      buildIndex([...contracts, extra], [...pagesWith(), page('kit/extra', 'kit/extra')]),
    ).toThrow('missing kit/extra');
  });

  it('throw on an area without sections, an empty section and a package for a label', () => {
    const [core, ...rest] = areas;
    expect(() => checkAreas(contracts)).not.toThrow();
    expect(() => checkAreas(contracts, [...areas, { label: 'Empty', sections: [] }])).toThrow(
      'the area Empty has no sections',
    );
    expect(() =>
      checkAreas(contracts, [
        ...areas,
        {
          label: 'More',
          sections: [{ label: 'Nothing', concept: 'concepts/modules', modules: [] }],
        },
      ]),
    ).toThrow('the section Nothing is empty');
    expect(() =>
      checkAreas(contracts, [{ ...core!, label: '@typegpu/kit/geometry' }, ...rest]),
    ).toThrow('the area @typegpu/kit/geometry is named like a package');
    expect(() =>
      checkAreas(contracts, [
        {
          ...core!,
          sections: [
            ...core!.sections,
            { label: 'Again', concept: 'concepts/modules', modules: ['kit/frames'] },
          ],
        },
        ...rest,
      ]),
    ).toThrow('twice kit/frames');
  });

  it('end with the pages beside them in their own group, a guide leading on to the next', () => {
    const pages = pagesWith();
    const groups = buildIndex(contracts, pages);
    expect(neighbours(groups, pages, 'concepts/modules')).toMatchObject({
      prev: { page: 'concepts/how-the-kit-works', kicker: 'Previous concept' },
      next: { page: 'concepts/connecting-modules', kicker: 'Next concept' },
    });
    expect(neighbours(groups, pages, 'concepts/data-on-the-gpu').next).toMatchObject({
      page: 'recipes/light-from-textures',
      kicker: 'Recipes',
    });
    expect(neighbours(groups, pages, 'recipes/other-renderers').next).toBeUndefined();
    expect(neighbours(groups, pages, 'kit/frames')).toMatchObject({
      prev: undefined,
      next: { page: 'kit/clock', kicker: 'Next in Core' },
    });
  });
});

describe('module cards on a page', () => {
  it('are placed with <Module id="…" />, outside code', () => {
    const body = [
      "import Module from '../../../components/site/Module.astro';",
      '',
      '<Module id="kit/finish" />',
      '',
      '## Bloom',
      '',
      '<Module id="kit/bloom"/>',
      '',
      'Write `<Module id="kit/fxaa" />` to place one.',
      '',
      '```mdx',
      '<Module id="kit/taa" />',
      '```',
    ].join('\n');

    expect(placedIn(body)).toEqual(['kit/finish', 'kit/bloom']);
  });

  it('throw on a card spelled any other way', () => {
    expect(() => placedIn('<Module id={"kit/taa"} />')).toThrow(/Write a module's card/);
    expect(() => placedIn('<Module id="kit/taa" compact />')).toThrow(/Write a module's card/);
    expect(() => placedIn('<Module id="kit/taa"></Module>')).toThrow(/Write a module's card/);
  });

  it('show one module under the title, and several each once where the page places it', () => {
    expect(() => checkPlacement('kit/frames', ['kit/frames'], [])).not.toThrow();
    expect(() => checkPlacement('kit/frames', ['kit/frames'], ['kit/frames'])).toThrow(
      /places none/,
    );

    const listed = ['kit/finish', 'kit/bloom'];
    expect(() => checkPlacement('kit/finish', listed, ['kit/finish', 'kit/bloom'])).not.toThrow();
    expect(() => checkPlacement('kit/finish', listed, ['kit/finish'])).toThrow(
      /places kit\/bloom nowhere/,
    );
    expect(() =>
      checkPlacement('kit/finish', listed, ['kit/finish', 'kit/bloom', 'kit/bloom']),
    ).toThrow(/places kit\/bloom 2 times/);
    expect(() =>
      checkPlacement('kit/finish', listed, ['kit/finish', 'kit/bloom', 'kit/taa']),
    ).toThrow(/places kit\/taa, which its modules don't list/);
  });
});

describe('the tutorials', () => {
  const folder = new URL('../src/content/docs/tutorials/', import.meta.url);

  it.each(readdirSync(folder))("%s imports nothing from the docs app's example host", (file) => {
    const text = readFileSync(new URL(file, folder), 'utf8');
    const imports =
      /from '(?:[^']*\/(?:contract|settings|ui|hosts)\.ts|typegpu-kit-examples[^']*)'/g;
    expect(text.match(imports)).toBeNull();
  });
});
