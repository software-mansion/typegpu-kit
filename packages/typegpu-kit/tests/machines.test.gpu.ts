import { gpuRoot } from '@typegpu/kit/testing/dawn';
import { randf } from '@typegpu/noise';
import { d, std, tgpu, type TgpuComputePass, type TgpuRoot } from 'typegpu';
import { afterEach, expect, it } from 'vitest';
import { machines } from '../src/index.ts';

const STEP = 1 / 60;
const ROWS = 256;
const HALF = ROWS / 2;

let cleanup: (() => void) | undefined;

afterEach(() => {
  cleanup?.();
  cleanup = undefined;
});

async function setup() {
  const root = await gpuRoot();
  cleanup = () => root.destroy();
  return root;
}

/** Records `count` steps, each in its own pass, and submits them. */
function steps(root: TgpuRoot, rows: { step(pass: TgpuComputePass): void }, count: number) {
  const encoder = root.createCommandEncoder();
  for (let k = 0; k < count; k++) {
    const pass = encoder.beginComputePass();
    rows.step(pass);
    pass.end();
  }
  encoder.submit();
}

it('steps every row on the GPU: guards, delays, hooks and events from the CPU', async () => {
  const root = await setup();
  const scared = root.createMutable(d.arrayOf(d.u32, ROWS));
  const entered = root.createMutable(d.arrayOf(d.atomic(d.u32), ROWS));
  const left = root.createMutable(d.arrayOf(d.atomic(d.u32), ROWS));

  const Mood = machines.define('Mood', {
    events: ['startle'],
    states: {
      calm: {
        enter: (i) => {
          'use gpu';
          std.atomicAdd(entered.$[i], 1);
        },
        next: [
          {
            to: 'flee',
            when: (i) => {
              'use gpu';
              return scared.$[i] === 1;
            },
          },
          { to: 'flee', on: 'startle' },
        ],
      },
      flee: {
        exit: (i) => {
          'use gpu';
          std.atomicAdd(left.$[i], 1);
        },
        next: [{ to: 'calm', after: 0.5 }],
      },
    },
  });
  const moods = machines.create(root, Mood, { capacity: ROWS, step: STEP });

  for (let slot = 0; slot < ROWS; slot++) moods.start(slot);
  scared.write(Array.from({ length: ROWS }, (_, i) => (i % 4 === 0 ? 1 : 0)));
  moods.flush();
  steps(root, moods, 1); // enters calm

  moods.send.startle(1);
  moods.send.startle(2);
  moods.flush();
  steps(root, moods, 1);

  let rows = await moods.rows.read();
  expect(rows.slice(0, 5).map((row) => Mood.nameOf(row.state as typeof Mood.calm))).toEqual([
    'flee',
    'flee',
    'flee',
    'calm',
    'flee',
  ]);
  expect(rows[3].age).toBe(1);
  expect(rows[1].age).toBe(0);

  // The event was taken once: the next step doesn't see it again.
  scared.write(Array.from({ length: ROWS }, () => 0));
  steps(root, moods, 30);
  rows = await moods.rows.read();
  expect(rows.slice(0, 5).map((row) => Mood.nameOf(row.state as typeof Mood.calm))).toEqual([
    'calm',
    'calm',
    'calm',
    'calm',
    'calm',
  ]);
  const enters = await entered.read();
  const exits = await left.read();
  expect([enters[0], enters[1], enters[3]]).toEqual([2, 2, 1]);
  expect([exits[0], exits[1], exits[3]]).toEqual([1, 1, 0]);
});

it('takes events kernels send, and skips rows that aren’t live', async () => {
  const root = await setup();
  const live = root.createReadonly(
    d.arrayOf(d.u32, ROWS),
    Array.from({ length: ROWS }, (_, i) => (i < ROWS / 2 ? 1 : 0)),
  );
  const Switch = machines.define('Switch', {
    events: ['flip'],
    states: { off: { next: [{ to: 'on', on: 'flip' }] }, on: {} },
  });
  const switches = machines.create(root, Switch, {
    capacity: ROWS,
    step: STEP,
    alive: (i) => {
      'use gpu';
      return live.$[i] === 1;
    },
  });

  const flip = tgpu.computeFn({ workgroupSize: [64], in: { gid: d.builtin.globalInvocationId } })(({
    gid,
  }) => {
    'use gpu';
    if (gid.x < ROWS && gid.x % 2 === 0) Switch.send.flip(gid.x);
  });
  const flipping = root.pipe(switches.inject()).createComputePipeline({ compute: flip });

  const encoder = root.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(flipping);
  pass.dispatchWorkgroups(ROWS / 64);
  switches.step(pass);
  pass.end();
  encoder.submit();

  const rows = await switches.rows.read();
  const on = (i: number) => rows[i].state === Switch.on;
  expect([on(0), on(1), on(2), on(ROWS / 2), on(ROWS / 2 + 1)]).toEqual([
    true,
    false,
    true,
    false,
    false,
  ]);
  expect([rows[1].age, rows[ROWS / 2 + 1].age]).toEqual([1, 0]);
});

it('shares a roll among chances, from a roll function of the row and its tick', async () => {
  const root = await setup();
  const COUNT = 4096;
  const Coin = machines.define('Coin', {
    states: {
      toss: {
        next: [
          { to: 'heads', chance: 0.25 },
          { to: 'tails', chance: 0.75 },
        ],
      },
      heads: {},
      tails: {},
    },
  });
  const coins = machines.create(root, Coin, {
    capacity: COUNT,
    step: STEP,
    roll: (i, tick) => {
      'use gpu';
      randf.seed2(d.vec2f(d.f32(i), d.f32(tick)));
      return randf.sample();
    },
  });
  for (let slot = 0; slot < COUNT; slot++) coins.start(slot);
  coins.flush();
  steps(root, coins, 2);

  const rows = await coins.rows.read();
  const heads = rows.filter((row) => row.state === Coin.heads).length;
  const tails = rows.filter((row) => row.state === Coin.tails).length;
  expect(heads + tails).toBe(COUNT);
  expect(heads / COUNT).toBeGreaterThan(0.22);
  expect(heads / COUNT).toBeLessThan(0.28);
});

it('rolls anew at every step of a row', async () => {
  const root = await setup();
  const COUNT = 4096;
  const STEPS = 20;
  const Fuse = machines.define('Fuse', {
    states: { lit: { next: [{ to: 'out', chance: 0.1 }] }, out: {} },
  });
  const fuses = machines.create(root, Fuse, {
    capacity: COUNT,
    step: STEP,
    roll: (i, tick) => {
      'use gpu';
      randf.seed2(d.vec2f(d.f32(i), d.f32(tick)));
      return randf.sample();
    },
  });
  for (let slot = 0; slot < COUNT; slot++) fuses.start(slot);
  fuses.flush();
  steps(root, fuses, STEPS + 1);

  const rows = await fuses.rows.read();
  const out = rows.filter((row) => row.state === Fuse.out).length / COUNT;
  expect(out).toBeGreaterThan(1 - 0.9 ** STEPS - 0.03);
  expect(out).toBeLessThan(1 - 0.9 ** STEPS + 0.03);
});

it('takes a rate as a share of each step, from the step’s length', async () => {
  const root = await setup();
  const COUNT = 4096;
  const Ember = machines.define('Ember', {
    states: { glowing: { next: [{ to: 'dark', rate: 1 }] }, dark: {} },
  });
  const embers = machines.create(root, Ember, {
    capacity: COUNT,
    step: STEP,
    roll: (i, tick) => {
      'use gpu';
      randf.seed2(d.vec2f(d.f32(i), d.f32(tick)));
      return randf.sample();
    },
  });
  for (let slot = 0; slot < COUNT; slot++) embers.start(slot);
  embers.flush();
  steps(root, embers, 1 + 60); // the enter, then a second

  const rows = await embers.rows.read();
  const dark = rows.filter((row) => row.state === Ember.dark).length / COUNT;
  expect(dark).toBeGreaterThan(1 - Math.exp(-1) - 0.03);
  expect(dark).toBeLessThan(1 - Math.exp(-1) + 0.03);
});

it('enters the states kernels start rows in, over the CPU’s start and the row’s events', async () => {
  const root = await setup();
  const entered = root.createMutable(d.arrayOf(d.atomic(d.u32), ROWS));
  const Ring = machines.define('Ring', {
    events: ['fade'],
    states: {
      idle: { next: [{ to: 'idle', on: 'fade' }] },
      burning: {
        enter: (i) => {
          'use gpu';
          std.atomicAdd(entered.$[i], 1);
        },
        next: [{ to: 'idle', on: 'fade' }],
      },
    },
  });
  const rings = machines.create(root, Ring, { capacity: ROWS, step: STEP, kernelStarts: true });
  const spawn = tgpu.computeFn({ workgroupSize: [64], in: { gid: d.builtin.globalInvocationId } })(
    ({ gid }) => {
      'use gpu';
      if (gid.x < HALF) {
        Ring.send.fade(gid.x);
        Ring.start(gid.x, Ring.burning);
      }
    },
  );
  const spawning = root.pipe(rings.inject()).createComputePipeline({ compute: spawn });

  rings.start(0, Ring.idle);
  rings.start(ROWS - 1, Ring.idle);
  const encoder = root.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(spawning);
  pass.dispatchWorkgroups(ROWS / 64);
  rings.step(pass);
  pass.end();
  encoder.submit();

  const rows = await rings.rows.read();
  const enters = await entered.read();
  expect([rows[0].state, rows[HALF - 1].state, rows[HALF].state]).toEqual([
    Ring.burning,
    Ring.burning,
    Ring.idle,
  ]);
  expect([enters[0], enters[1], enters[HALF]]).toEqual([1, 1, 0]);
  // The fade sent with the start was dropped: the next step keeps them burning.
  steps(root, rings, 1);
  const after = await rings.rows.read();
  expect([after[0].state, after[0].age]).toEqual([Ring.burning, 1]);
});

it('draws a fresh roll in a hook each time a row enters', async () => {
  const root = await setup();
  const drawn = root.createMutable(d.arrayOf(d.f32, ROWS * 2));
  const Blink = machines.define('Blink', {
    states: {
      shut: { next: [{ to: 'open', after: STEP }] },
      open: {
        enter: (i) => {
          'use gpu';
          // The first time at i, the second at ROWS + i.
          if (drawn.$[i] === 0) drawn.$[i] = Blink.roll(i);
          else drawn.$[ROWS + i] = Blink.roll(i);
        },
        next: [{ to: 'shut', after: STEP }],
      },
    },
  });
  const blinks = machines.create(root, Blink, {
    capacity: ROWS,
    step: STEP,
    roll: (i, tick) => {
      'use gpu';
      randf.seed2(d.vec2f(d.f32(i), d.f32(tick)));
      return randf.sample();
    },
  });
  for (let slot = 0; slot < ROWS; slot++) blinks.start(slot);
  steps(root, blinks, 5); // enter shut, open, shut, open

  const values = await drawn.read();
  const first = values.slice(0, ROWS);
  const second = values.slice(ROWS);
  expect(first.every((value) => value > 0 && value < 1)).toBe(true);
  expect(second.every((value) => value > 0 && value < 1)).toBe(true);
  expect(new Set(first).size).toBeGreaterThan(ROWS * 0.95);
  expect(first.filter((value, i) => value === second[i]).length).toBe(0);
  const mean = first.reduce((sum, value) => sum + value, 0) / ROWS;
  expect(mean).toBeGreaterThan(0.4);
  expect(mean).toBeLessThan(0.6);
});

it('lists the live rows in each listed state after every step, for a kernel over each', async () => {
  const root = await setup();
  const COUNT = 1000; // not a whole number of workgroups
  const visits = root.createMutable(d.arrayOf(d.atomic(d.u32), COUNT));
  const live = root.createReadonly(
    d.arrayOf(d.u32, COUNT),
    Array.from({ length: COUNT }, (_, i) => (i % 10 === 9 ? 0 : 1)),
  );
  const Tide = machines.define('Tide', {
    events: ['turn'],
    states: { low: { next: [{ to: 'high', on: 'turn' }] }, high: {}, slack: {} },
  });
  const tides = machines.create(root, Tide, {
    capacity: COUNT,
    step: STEP,
    alive: (i) => {
      'use gpu';
      return live.$[i] === 1;
    },
    lists: [Tide.high, Tide.slack],
  });
  const listed = tides.listed!;
  const visitor = (group: number, amount: number) =>
    root.createComputePipeline({
      compute: tgpu.computeFn({
        workgroupSize: [64],
        in: { gid: d.builtin.globalInvocationId },
      })(({ gid }) => {
        'use gpu';
        if (gid.x >= listed.count(group)) return;
        std.atomicAdd(visits.$[listed.slot(group, gid.x)], amount);
      }),
    });
  const visiting = [visitor(0, 1), visitor(1, 100)];

  for (let slot = 0; slot < COUNT; slot++) {
    tides.start(slot, slot % 3 === 0 ? Tide.slack : Tide.low);
  }
  steps(root, tides, 1); // enters
  for (let slot = 1; slot < COUNT; slot += 3) tides.send.turn(slot);
  // Two steps, each followed by the work: a list holds what its own step listed.
  const encoder = root.createCommandEncoder();
  const pass = encoder.beginComputePass();
  for (let k = 0; k < 2; k++) {
    tides.step(pass);
    for (const [group, pipeline] of visiting.entries()) {
      pass.setPipeline(pipeline);
      listed.dispatch(pass, group);
    }
  }
  pass.end();
  encoder.submit();

  const counts = await visits.read();
  const expected = Array.from({ length: COUNT }, (_, i) => {
    if (i % 10 === 9) return 0;
    return [200, 2, 0][i % 3];
  });
  expect(counts).toEqual(expected);
});
