import { randf } from '@typegpu/noise';
import { d, tgpu, type TgpuComputePass, type TgpuRoot } from 'typegpu';
import { describe, expect, it } from 'vitest';
import * as grid from '../src/grid.ts';
import { recordingRoot } from '../src/testing.ts';

const Item = d.struct({ position: d.vec3f });

function setup(options: Partial<grid.Options<typeof Item>> = {}) {
  const recording = recordingRoot();
  const { root } = recording;
  const places = root.createReadonly(d.arrayOf(d.vec3f, 256));

  const cells = grid.create(root, {
    capacity: 256,
    low: d.vec3f(-4, 0, -4),
    size: d.vec3f(8, 2, 8),
    sight: 1,
    item: Item,
    alive: (i) => {
      'use gpu';
      return i % 2 === 0;
    },
    position: (i) => {
      'use gpu';
      return d.vec3f(places.$[i]);
    },
    itemOf: (i) => {
      'use gpu';
      return Item({ position: places.$[i] });
    },
    ...options,
  });

  return { recording, root, cells };
}

/** One step that sorts the grid in a compute pass of its own encoder, and submits. */
function step(root: TgpuRoot, cells: grid.Grid<typeof Item>) {
  const encoder = root.createCommandEncoder();
  const pass = encoder.beginComputePass();
  cells.sort(pass);
  pass.end();
  encoder.submit();
}

// Emptying the last sort's ranges first, then finding the new ones and copying the items
const around = (...binning: string[]) => [
  'begin-compute',
  'dispatch-indirect',
  ...binning,
  'dispatch',
  'dispatch-indirect',
  'end-compute',
  'submit',
];

describe('grid', () => {
  it("sorts in the caller's pass and submits nothing of its own", () => {
    // Counting, sizing, reserving per line and scattering
    const counted = setup();
    const dispatcher = { dispatchWorkgroupsIndirect: () => {} };
    expect(() => counted.cells.dispatch(dispatcher)).toThrow('Record sort(pass)');
    step(counted.root, counted.cells);
    counted.cells.dispatch(dispatcher);
    expect(counted.recording.log.events).toEqual(
      around('dispatch', 'dispatch', 'dispatch-indirect', 'dispatch'),
    );

    // Keys, then one 8-bit pass for 128 keys: count, scan, scatter
    const sorted = setup({ binning: grid.radix });
    step(sorted.root, sorted.cells);
    expect(sorted.recording.log.events).toEqual(around(...Array<string>(4).fill('dispatch')));
  });

  it('creates nothing in a steady step', () => {
    const { recording, root, cells } = setup();
    step(root, cells);
    step(root, cells);
    const before = recording.log.counts();

    step(root, cells);
    step(root, cells);
    expect(recording.log.counts()).toEqual(before);
  });

  it('lets a kernel walk it with nothing piped', () => {
    const { root, cells } = setup();
    const probe = tgpu.computeFn({ workgroupSize: [1] })(() => {
      'use gpu';
      randf.seed(0.5);
      const range = cells.range(cells.walk(d.vec3f()), 0);
      if (range.count > 0) cells.neighbour(range, 0);
    });

    expect(() => root.createComputePipeline({ compute: probe }).initSync()).not.toThrow();
  });

  it('binds two storage buffers for a walk, and a third for slot', async () => {
    const { recording, root, cells } = setup();
    const storage = () => recording.log.shaders.at(-1)!.match(/var<storage/g)!.length;
    const walked = root.createMutable(d.u32);

    const walk = tgpu.computeFn({ workgroupSize: [1] })(() => {
      'use gpu';
      randf.seed(0.5);
      const range = cells.range(cells.walk(d.vec3f()), 0);
      if (range.count > 0) walked.$ = d.u32(cells.neighbour(range, 0).position.x);
    });
    const slotted = tgpu.computeFn({ workgroupSize: [1] })(() => {
      'use gpu';
      randf.seed(0.5);
      const range = cells.range(cells.walk(d.vec3f()), 0);
      if (range.count > 0)
        walked.$ = cells.slot(range, 0) + d.u32(cells.neighbour(range, 0).position.x);
    });

    const injected = root.pipe(cells.inject());
    await injected.createComputePipeline({ compute: walk }).initAsync();
    expect(storage()).toBe(3); // and `walked`
    await injected.createComputePipeline({ compute: slotted }).initAsync();
    expect(storage()).toBe(4);
  });

  it('counts, scatters and bounds only up to the span, once a shrink is emptied', () => {
    let span = 64;
    const { cells } = setup({ span: () => span });
    const sizes: number[] = [];
    const pass = {
      setPipeline: () => {},
      dispatchWorkgroups: (x: number) => sizes.push(x),
      dispatchWorkgroupsIndirect: () => {},
    } as unknown as TgpuComputePass;
    const sorted = () => {
      sizes.length = 0;
      cells.sort(pass);
      return [...sizes];
    };

    // Count, size, scatter, then bound one key past the span. The first sort empties every slot
    expect([sorted(), sorted()]).toEqual([
      [4, 1, 4, 2],
      [1, 1, 1, 2],
    ]);
    span = 200;
    expect(sorted()).toEqual([4, 1, 4, 4]);
    span = 0;
    expect([sorted(), sorted()]).toEqual([
      [4, 1, 4, 1],
      [0, 1, 0, 1],
    ]);
  });

  it('refuses a sight of 0, naming grid.create', () => {
    expect(() => setup({ sight: 0 })).toThrow(
      new RangeError('grid.create: `sight` must be above 0, got 0.'),
    );
  });
});
