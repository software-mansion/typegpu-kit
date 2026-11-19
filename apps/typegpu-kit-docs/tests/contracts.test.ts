import { readFileSync } from 'node:fs';
import { basename, dirname, join } from 'node:path';
import ts from 'typescript';
import { describe, expect, it } from 'vitest';
import { renderIndex } from '../src/contracts/agents.ts';
import { moduleContract, type Factory } from '../src/contracts/contract.ts';
import { extractContracts, repo } from '../src/contracts/extract.ts';
import { cardOf, exportKeys, factMarkdown, factsOf } from '../src/contracts/facets.ts';
import { readTags } from '../src/contracts/tags.ts';

const { contracts, issues } = extractContracts();
const byId = (id: string) => contracts.find((contract) => contract.id === id)!;
const facts = (id: string) => factsOf(byId(id), contracts);
const factOf = (id: string, key: string) => facts(id).find((each) => each.key === key);

/** Every public namespace, read from the package's entries without a type checker. */
function publicNamespaces() {
  const ids: string[] = [];
  const json = JSON.parse(readFileSync(join(repo.package, 'package.json'), 'utf8'));
  const short = json.name.split('/')[1];
  const entries = Object.entries(json.exports as Record<string, string>).filter(
    ([subpath, file]) => file.endsWith('.ts') && !repo.skipped?.includes(subpath),
  );
  const files = new Set<string>();
  const wrappers = new Set<string>();
  for (const [subpath, target] of entries) {
    const entry = join(repo.package, target);
    const source = ts.createSourceFile(entry, readFileSync(entry, 'utf8'), ts.ScriptTarget.Latest);
    for (const statement of source.statements) {
      if (
        ts.isExportDeclaration(statement) &&
        statement.exportClause &&
        ts.isNamespaceExport(statement.exportClause) &&
        statement.moduleSpecifier &&
        ts.isStringLiteral(statement.moduleSpecifier)
      ) {
        ids.push(`${short}/${statement.exportClause.name.text}`);
        files.add(join(dirname(entry), statement.moduleSpecifier.text));
        wrappers.add(subpath);
      }
    }
  }
  for (const [subpath, file] of entries) {
    if (subpath !== '.' && !wrappers.has(subpath) && !files.has(join(repo.package, file))) {
      ids.push(`${short}/${basename(subpath)}`);
    }
  }

  return ids.toSorted();
}

describe('module contracts', () => {
  it('agree with the code, and each passes the strict schema', () => {
    expect(issues).toEqual([]);
    for (const contract of contracts) moduleContract.parse(contract);
  });

  it('cover every public namespace', () => {
    const ids = new Set(contracts.map((contract) => contract.id));
    expect(publicNamespaces().filter((id) => !ids.has(id))).toEqual([]);
  });

  it("name a factory's result by its type, not by a generic alias that adds to it", () => {
    const create = byId('kit/lighting').factories.find((factory) => factory.name === 'create');
    expect(create?.returns).toBe('Lighting');
  });

  it('take a method that returns something, such as a surface view, for a read, not a recording', () => {
    expect(byId('kit/testing').records).toEqual([]);
  });

  it('call a module a tool only when every factory borrows the root and creates nothing', () => {
    expect(byId('kit/census').kind).toBe('tool');
    expect(byId('kit/readback').kind).toBe('module');
  });

  it('take kind, platform and flatness from tags, one platform or none', () => {
    expect(byId('kit/web')).toMatchObject({ kind: 'platform', platform: 'web' });
    expect(byId('kit/hosting').kind).toBe('platform');
    expect(byId('kit/hosting').platform).toBeUndefined();
    expect(byId('kit/react')).toMatchObject({ kind: 'platform', platform: 'native', flat: true });
    expect(byId('kit/gpuDebug')).toMatchObject({
      kind: 'tool',
      platform: 'web',
      namespaceExport: true,
    });
    expect(byId('kit/fontBaker')).toMatchObject({ kind: 'tool', flat: true });
  });

  it('say where a factory takes each module it needs', () => {
    expect(byId('kit/instances').via).toEqual({
      'kit/cameras': 'renderer(options.eye)',
      'kit/meshPool': 'pool',
    });
    expect(byId('kit/finish').via).toEqual({ 'kit/exposure': 'options.exposure' });
    expect(byId('kit/particles').via).toEqual({ 'kit/cameras': 'camera' });
  });

  it("show an options object's required fields, its noted ones and the modules they hold", () => {
    const [final] = byId('kit/finish').factories;
    expect(final!.parameters[1]).toEqual({
      name: 'options',
      fields: [
        { name: 'format' },
        { name: 'source', note: 'the picture, rendered or a texture, read at every `draw`' },
        {
          name: 'region',
          note: 'the part of the source that holds the picture, read at every `draw`',
          optional: true,
        },
        {
          name: 'glow',
          note: 'a bloom, or a glow of your own, added before tone mapping',
          optional: true,
        },
        {
          name: 'exposure',
          module: 'kit/exposure',
          note: 'an adapted exposure, read on the GPU',
          optional: true,
        },
      ],
      more: true,
    });
  });

  it('table every field of the options types the factories take, with its default', () => {
    const orbit = byId('kit/cameras').options.find(({ name }) => name === 'OrbitOptions');
    expect(orbit).toMatchObject({ name: 'OrbitOptions', takenBy: ['orbit'], parameter: 'options' });
    expect(orbit!.fields.find((field) => field.name === 'pitch')).toEqual({
      name: 'pitch',
      optional: true,
      type: 'number',
      default: '0.6',
      doc: 'Sets the initial pitch in radians above the horizon.',
    });

    const [smooth] = byId('kit/fxaa').options;
    expect(smooth!.fields.map((field) => field.name)).toEqual([
      'format',
      'source',
      'region',
      'settings',
    ]);
    expect(byId('kit/cameras').types).not.toContain('OrbitOptions');
  });

  it('table the options types an options type holds, under the field that holds them', () => {
    const tables = byId('kit/lighting').options;
    expect(tables.map(({ name, takenBy, parameter }) => ({ name, takenBy, parameter }))).toEqual([
      { name: 'Options', takenBy: ['create'], parameter: 'options' },
      { name: 'PointsOptions', takenBy: ['create'], parameter: 'options.points' },
      { name: 'ShadowOptions', takenBy: ['create'], parameter: 'options.shadow' },
      { name: 'PointShadowOptions', takenBy: ['create'], parameter: 'options.pointShadow' },
      { name: 'ShadowRecordOptions', takenBy: ['Lighting.shadow.record'], parameter: 'options' },
      {
        name: 'PointShadowRecordOptions',
        takenBy: ['Lighting.pointShadow.record'],
        parameter: 'options',
      },
    ]);
    expect(byId('kit/lighting').types).not.toContain('ShadowOptions');

    // A union's field takes its type from the member that accepts it, not the one that refuses it
    const shadow = tables.find(({ name }) => name === 'ShadowOptions')!;
    const typeOf = (name: string) => shadow.fields.find((field) => field.name === name)?.type;
    expect([typeOf('eye'), typeOf('extent'), typeOf('cascades')]).toEqual([
      'CpuView',
      'number',
      '1 | 2 | 3 | 4',
    ]);
  });

  it('take the root first, then at most three inputs by position before the options', () => {
    const shapes = contracts
      .filter((contract) => !contract.vendored && contract.kind !== 'platform')
      .flatMap((contract) =>
        contract.factories.flatMap((factory) => {
          const [first, ...rest] = factory.parameters;
          const after = first?.root ? rest : factory.parameters;
          const positional = after.filter(
            (parameter, i) => !(i === after.length - 1 && /options$/i.test(parameter.name)),
          );
          const name = `${contract.id} ${factory.signature}`;
          return [
            ...(after.some((parameter) => parameter.root)
              ? [`${name}: a root after the first`]
              : []),
            ...(positional.length > 3 ? [`${name}: ${positional.length} by position`] : []),
          ];
        }),
      );
    expect(shapes).toEqual([]);
  });

  it('give every optional field of an options type its default', () => {
    const undefaulted = contracts
      .filter((contract) => !contract.vendored)
      .flatMap((contract) =>
        contract.options.flatMap((table) =>
          table.fields
            .filter((field) => field.optional && field.default === undefined)
            .map((field) => `${contract.id} ${table.name}.${field.name}`),
        ),
      );
    expect(undefaulted).toEqual([]);
  });

  it('say what every factory returns, and what each input it takes by position is', () => {
    const unsaid = contracts
      .filter((contract) => !contract.vendored)
      .flatMap((contract) =>
        contract.factories.flatMap((factory) => {
          const name = `${contract.id} ${factory.signature}`;
          const last = factory.parameters.length - 1;
          const inputs = factory.parameters.filter(
            (parameter, i) => !parameter.root && !(i === last && /options$/i.test(parameter.name)),
          );
          return [
            ...(factory.result === undefined ? [`${name}: @returns`] : []),
            ...inputs
              .filter((parameter) => parameter.note === undefined)
              .map((parameter) => `${name}: @param ${parameter.name}`),
          ];
        }),
      );
    expect(unsaid).toEqual([]);
  });

  it('describe every required field of an options type', () => {
    const undescribed = contracts
      .filter((contract) => !contract.vendored)
      .flatMap((contract) =>
        contract.options.flatMap((table) =>
          table.fields
            .filter((field) => !field.optional && field.doc === undefined)
            .map((field) => `${contract.id} ${table.name}.${field.name}`),
        ),
      );
    expect(undescribed).toEqual([]);
  });

  it('sort every value export beside the factories', () => {
    expect(byId('kit/cameras')).toMatchObject({
      functions: [
        'toClip',
        'farDepth',
        'nearerDepth',
        'forwardOf',
        'lensOf',
        'distanceOf',
        'rayOf',
        'towards',
        'sees',
      ],
      accessors: ['current', 'reversed'],
      schemas: ['Camera', 'Ray'],
    });
    expect(byId('kit/shading').accessors).toEqual([
      'reflect',
      'ambient',
      'fog',
      'sunShadow',
      'attenuation',
    ]);
    expect(byId('kit/meshPool')).toMatchObject({
      functions: ['positionAt', 'vertexAt', 'uv1At', 'influencesAt', 'boundsOf'],
      helpers: ['textureLayout'],
    });
    expect(byId('kit/text').functions).toEqual(['facing', 'ground', 'screen', 'pinned']);
    expect(byId('kit/passes')).toMatchObject({
      outputs: ['shaded', 'accumulation', 'revealage'],
      constants: ['color', 'depth'],
      helpers: ['fullScreen'],
    });
    expect(byId('kit/taa').outputs).toEqual(['motion']);
    expect(byId('kit/picker').constants).toEqual(['none', 'defaults']);
    expect(byId('kit/meshes').functions).toEqual(['triangleBarycentrics', 'triangleIndexAt']);
    expect(byId('kit/meshes')).toMatchObject({
      package: '@typegpu/kit',
      specifier: '@typegpu/kit/geometry',
      namespaceExport: true,
      file: 'src/geometry/mesh/index.ts',
      vendored: true,
    });
    expect(contracts.some((contract) => contract.id === 'sort/sort')).toBe(false);
    expect(byId('kit/grid').submits).toBeUndefined();
  });

  it("list a shader module's functions, and the modules whose accessors they read", () => {
    expect(byId('kit/tone').functions).toContain('aces');
    expect(byId('kit/materials').reads).toEqual(['kit/meshPool']);
    expect(byId('kit/finish').reads).toEqual([]);
  });
});

/** Whether a name is PascalCase, not a constant in capitals. */
const pascal = (name: string) => /^[A-Z]/.test(name) && !/^[A-Z][A-Z0-9_]*$/.test(name);
/** A React component, which takes `props`. */
const component = (factory: Factory) => factory.parameters[0]?.name === 'props';

/** Pairs of names that differ only in case. */
const clashesIn = (where: string, names: readonly string[]) => {
  const seen = new Map<string, string>();
  return [...new Set(names)].flatMap((name) => {
    const other = seen.get(name.toLowerCase());
    seen.set(name.toLowerCase(), name);
    return other === undefined ? [] : [`${where}: ${other} / ${name}`];
  });
};

describe('names', () => {
  const kit = contracts.filter((contract) => !contract.vendored);
  const valueGroups = ['functions', 'accessors', 'outputs', 'constants', 'helpers'] as const;

  it('write PascalCase values only for schemas, table definitions and React components', async () => {
    const others: string[] = [];
    for (const contract of kit) {
      for (const factory of contract.factories) {
        if (pascal(factory.name) && !component(factory))
          others.push(`${contract.id}.${factory.name}`);
      }
      const named = valueGroups.flatMap((group) => contract[group]).filter(pascal);
      if (named.length === 0) continue;
      const entry = await import(contract.specifier);
      const values = entry[contract.namespace] ?? entry;
      for (const name of named) {
        if (!('~parts' in values[name])) others.push(`${contract.id}.${name}`);
      }
    }
    expect(others).toEqual([]);
  });

  it('never name a factory result like a schema', () => {
    const schemas = new Set(kit.flatMap((contract) => contract.schemas));
    const clashes = kit.flatMap((contract) =>
      contract.factories
        .map((factory) => factory.returns?.match(/^(?:Promise<)?(\w+)/)?.[1])
        .filter((type) => type !== undefined && schemas.has(type))
        .map((type) => `${contract.id}: ${type}`),
    );
    expect(clashes).toEqual([]);
  });

  it('never give two values names that differ only in case', async () => {
    const clashes = kit.flatMap((contract) =>
      clashesIn(contract.id, [
        contract.namespace,
        ...contract.factories.filter((factory) => !component(factory)).map(({ name }) => name),
        ...valueGroups.flatMap((group) => contract[group]),
        ...contract.schemas,
      ]),
    );
    clashes.push(...clashesIn('@typegpu/kit', Object.keys(await import('@typegpu/kit'))));
    expect(clashes).toEqual([]);
  });
});

describe('module cards', () => {
  it('show each fact that is set once, the kind in the title, and leave the defaults out', () => {
    for (const contract of contracts) {
      const card = cardOf(contract, contracts);
      const shown = [...card.exports, ...card.groups.flatMap((group) => group.facts)].map(
        (fact) => fact.key,
      );
      const set = card.facts
        .map((fact) => fact.key)
        .filter((key) => !['kind', 'stability', 'usedBy'].includes(key));
      expect(shown.toSorted(), contract.id).toEqual(set.toSorted());
    }
    expect(factOf('kit/liveLists', 'submits')).toBeUndefined();
    expect(factOf('kit/frames', 'submits')).toMatchObject({ exception: true });
    expect(factOf('kit/frames', 'binds')).toBeUndefined();
  });

  it('list each set fact in docs/modules.md, but bindings and examples, which it has elsewhere', () => {
    const index = renderIndex(contracts);
    for (const contract of contracts) {
      for (const fact of facts(contract.id)) {
        if (fact.key === 'binds' || fact.key === 'usedBy') continue;
        expect(index).toContain(factMarkdown(fact));
      }
    }
  });

  it('list what a module exports beside its factories, a row per kind of export', () => {
    const card = cardOf(byId('kit/cameras'), contracts);
    expect(card.exports.map((fact) => fact.key)).toEqual([
      'functions',
      'accessors',
      'schemas',
      'types',
    ]);
    expect(exportKeys).toContain('types');
  });

  it('say what a stage binds for what a shader reads, a line per cost', () => {
    expect(factOf('kit/trails', 'binds')).toMatchObject({
      entries: [
        { value: [{ type: 'text', text: '2 storage' }] },
        { value: [{ type: 'text', text: '1 storage' }] },
        { value: [{ type: 'text', text: '2 storage · 2 uniform' }] },
      ],
    });
  });

  it('list what a module owns item by item, with its note', () => {
    const owns = factOf('kit/liveLists', 'owns');
    expect(owns).toMatchObject({
      value: [
        { type: 'text', text: '4 buffers ·' },
        { type: 'text', text: '2 pipelines' },
      ],
      note: [
        {
          type: 'text',
          text: 'the live slots per group, their counts, the indirect draws and the indirect dispatches',
        },
      ],
    });
    expect(factMarkdown(owns!)).toBe(
      '**owns** 4 buffers, 2 pipelines: the live slots per group, their counts, the indirect draws and the indirect dispatches.',
    );
  });
});

describe('module tags', () => {
  it('take a value, then a note on what the value does not say', () => {
    const found: string[] = [];
    const read = readTags(
      [
        { name: 'module', text: '' },
        { name: 'kind', text: 'platform' },
        { name: 'platform', text: 'native' },
        { name: 'flat', text: '' },
        { name: 'unstable', text: '' },
        { name: 'injects', text: 'member emit and the pool, for kernels piped with it' },
        { name: 'binds', text: '2 storage: `emit`, the pool and its counter' },
        { name: 'binds', text: '1 storage, 1 uniform: `alive(i)`, the live bits and the span' },
        { name: 'binds', text: '1 uniform: `cameras.current.$`' },
        { name: 'requires', text: 'indirect-first-instance option when firstInstance is set' },
        { name: 'submits', text: 'setup' },
        { name: 'allocates', text: 'on resize: the levels' },
        { name: 'owns', text: '5 buffers, 3 pipelines, live lists: the pool and its counters' },
        { name: 'records', text: 'In this order, every frame.' },
      ],
      'test',
      found,
    );

    expect(found).toEqual([]);
    expect(read).toEqual({
      kind: 'platform',
      platform: 'native',
      flat: true,
      unstable: true,
      injects: [{ kind: 'member', note: 'emit and the pool, for kernels piped with it' }],
      binds: [
        { storage: 2, uniform: 0, note: '`emit`, the pool and its counter' },
        { storage: 1, uniform: 1, note: '`alive(i)`, the live bits and the span' },
        { storage: 0, uniform: 1, note: '`cameras.current.$`' },
      ],
      requires: [
        { feature: 'indirect-first-instance', mode: 'option', note: 'when firstInstance is set' },
      ],
      submits: { when: 'setup' },
      allocates: { when: 'on resize', note: 'the levels' },
      owns: {
        items: ['5 buffers', '3 pipelines', 'live lists'],
        note: 'the pool and its counters',
      },
      order: 'In this order, every frame.',
    });
  });

  it('say what is wrong with them', () => {
    const found: string[] = [];
    readTags(
      [
        { name: 'injects', text: 'everything' },
        { name: 'binds', text: '2 storage' },
        { name: 'binds', text: 'two buffers: the rings' },
        { name: 'requires', text: 'timestamp-query' },
        { name: 'owns', text: 'a buffer' },
        { name: 'owns', text: 'a texture' },
        { name: 'kind', text: 'definition' },
        { name: 'platform', text: 'web native' },
        { name: 'colour', text: 'blue' },
      ],
      'test',
      found,
    );

    expect(found).toEqual([
      "test: @injects takes member, definition, hook first, got 'everything'.",
      "test: @binds takes '<n> storage, <n> uniform: what', got '2 storage'.",
      "test: @binds takes '<n> storage, <n> uniform: what', got 'two buffers: the rings'.",
      "test: @requires takes a feature and required, optional, option, got 'timestamp-query'.",
      'test: @owns twice; give it one tag.',
      "test: @kind takes module, cpu, shader, platform, tool, got 'definition'.",
      "test: @platform takes web or native, got 'web native'; leave it out for both.",
      'test: unknown tag @colour.',
    ]);
  });
});
