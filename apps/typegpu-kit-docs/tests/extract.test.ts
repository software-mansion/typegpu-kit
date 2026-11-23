import { fileURLToPath } from 'node:url';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { extractContracts, repo, type Workspace } from '../src/contracts/extract.ts';

const fixture = fileURLToPath(new URL('fixtures/contracts/', import.meta.url));

/** A small package, each export an edge case of the extractor. */
const workspace: Workspace = {
  package: join(fixture, 'package'),
  examples: join(fixture, 'examples'),
  tsconfig: repo.tsconfig,
  skipped: ['./dev'],
  vendored: ['./vendored'],
  overrides: {
    modules: {
      'fixture/shapes': { summary: 'Shapes.', helpers: ['combine', 'missing'] },
    },
  },
};

const extraction = extractContracts(workspace);
const { contracts, usage, issues } = extraction;
const byId = (id: string) => contracts.find((contract) => contract.id === id)!;

describe('the contracts extractor', () => {
  it('finds a module per namespace, flat root export and subpath, but not in skipped ones', () => {
    expect(contracts.map((contract) => [contract.id, contract.specifier, !!contract.flat])).toEqual(
      [
        ['fixture/spinner', '@kit-fixtures/fixture', false],
        ['fixture/wait', '@kit-fixtures/fixture', true],
        ['fixture/shade', '@kit-fixtures/fixture/shade', false],
        ['fixture/tool', '@kit-fixtures/fixture/tool', true],
        ['fixture/shapes', '@kit-fixtures/fixture/vendored', false],
      ],
    );
  });

  it('infers a kind from the code, and takes one from @kind', () => {
    expect(contracts.map((contract) => contract.kind)).toEqual([
      'module',
      'cpu',
      'shader',
      'tool',
      'cpu',
    ]);
    expect(byId('fixture/tool').platform).toBe('web');
  });

  it("reads a summary and tags from the block marked @module, and a factory's JSDoc", () => {
    expect(byId('fixture/spinner')).toMatchObject({
      summary: "Turns a number on the GPU, a module for the contracts' tests.",
      owns: { items: ['a buffer'], note: 'the speed' },
      order: '`spin` in a compute pass.',
      records: [{ method: 'spin', into: 'compute' }],
      lifecycle: { initAsync: false, destroy: true, dispose: false },
    });
    expect(byId('fixture/wait').summary).toBe('A wait of a few frames, on the CPU.');
  });

  it('lists each overload of a factory', () => {
    expect(byId('fixture/spinner').factories.map((factory) => factory.signature)).toEqual([
      'create(root, options)',
      'create(root, options, label)',
    ]);
  });

  it('sorts every value export, leaves out @hidden ones and lists the types but tabled ones', () => {
    expect(byId('fixture/spinner')).toMatchObject({
      functions: ['turn'],
      accessors: ['speed'],
      outputs: [],
      schemas: ['Settings'],
      constants: ['defaults'],
      helpers: ['presets'],
      types: [],
    });
    expect(byId('fixture/shapes')).toMatchObject({
      factories: [{ name: 'cube' }],
      helpers: ['combine'],
      vendored: true,
    });
  });

  it('tables every field of an options type, with its default', () => {
    expect(byId('fixture/spinner').options.slice(0, 1)).toEqual([
      {
        name: 'Options',
        takenBy: ['create'],
        parameter: 'options',
        fields: [
          { name: 'label', optional: false, type: 'string', doc: 'What the spinner is called.' },
          {
            name: 'speed',
            optional: true,
            type: 'number',
            default: '`1`',
            doc: 'Turns a second, e.g. one.',
          },
          { name: 'backwards', optional: true, type: 'boolean', doc: 'Turns it backwards.' },
          {
            name: 'ease',
            optional: true,
            type: '(t: number) => number',
            doc: 'Eases each turn.',
            gpu: true,
          },
          { name: 'turned', optional: true, type: '() => void', doc: 'Runs after each turn.' },
        ],
      },
    ]);
  });

  it("lists a returned type's members but @hidden ones, and tables what its methods take", () => {
    const spinner = byId('fixture/spinner');
    expect(spinner.handles).toEqual([
      {
        name: 'Spinner',
        from: ['create'],
        members: [
          { name: 'turns', type: 'number', doc: 'Counts the turns so far.' },
          { name: 'spin', signature: 'spin(pass, options?)', doc: 'Spins it in a compute pass.' },
          { name: 'destroy', signature: 'destroy()' },
        ],
      },
    ]);
    expect(spinner.options[1]).toEqual({
      name: 'SpinOptions',
      takenBy: ['Spinner.spin'],
      parameter: 'options',
      fields: [
        {
          name: 'turns',
          optional: true,
          type: 'number',
          default: '`1`',
          doc: 'Turns this many times.',
        },
      ],
    });
  });

  it('counts the modules an example uses: by value, through a namespace, re-exported, imported late, through a shared file or by the @uses of the names it imports', () => {
    expect(usage).toEqual({
      five: ['fixture/shapes', 'fixture/wait'],
      four: ['fixture/spinner'],
      one: ['fixture/spinner', 'fixture/tool'],
      two: ['fixture/shade', 'fixture/wait'],
    });
    expect(byId('fixture/spinner').usedBy).toEqual(['four', 'one']);
  });

  it('reports what it cannot place, and tags where the docs never read them', () => {
    expect(issues.toSorted()).toEqual(
      [
        "A shared file's @uses names fixture/ghost, which is no module.",
        "@kit-fixtures/fixture/vendored: exports loose at its root, which no module covers; gather them into one in its overrides' flat.",
        'fixture/spinner: exports registry, which is no factory, GPU function, accessor, output, schema, constant or helper; mark it @hidden or teach the contracts its kind.',
        "fixture/shapes: its overrides list missing as a helper, a function it doesn't export.",
        'fixture/shade: 2 blocks carry @module; give it one.',
        "src/stray.ts:1: @module in a file that is no module's.",
        'Options.turned (apps/typegpu-kit-docs/tests/fixtures/contracts/package/src/spinner.ts:22) takes a function: tag where it runs, @gpu in a shader or @cpu on the CPU.',
        "src/stray.ts:7: @owns outside a module's block or a factory's JSDoc, where the docs never read it.",
      ].toSorted(),
    );
  });

  it('reports overrides of a module that is not vendored, or of none', () => {
    const { issues: found } = extractContracts(
      {
        ...workspace,
        overrides: {
          modules: { 'fixture/spinner': { kind: 'cpu' }, 'fixture/ghost': {} },
        },
      },
      extraction,
    );

    expect(found).toContain(
      "The overrides name fixture/spinner, which isn't vendored; tag its source instead.",
    );
    expect(found).toContain('The overrides name fixture/ghost, which is no module.');
  });
});
