import { d, tgpu } from 'typegpu';
import { describe, expect, expectTypeOf, it } from 'vitest';
import { machines, type State } from '../src/index.ts';
import { recordingRoot } from '../src/testing.ts';

const STEP = 1 / 60;

/** A definition as plain JavaScript may pass it, past the types. */
const spec = (states: object, events?: string[]) => () =>
  // oxlint-disable-next-line typescript/no-explicit-any -- what plain JavaScript passes
  machines.define('Bad', { events, states } as any);

describe('machines.define', () => {
  it('numbers the states in order and brands them with the name', () => {
    const Door = machines.define('Door', {
      events: ['push'],
      states: { shut: { next: [{ to: 'open', on: 'push' }] }, open: {} },
    });

    expect([Door.shut, Door.open]).toEqual([0, 1]);
    expect(Door.names).toEqual(['shut', 'open']);
    expect(Door.events).toEqual(['push']);
    expect(Door.nameOf(Door.open)).toBe('open');
    expectTypeOf(Door.open).toEqualTypeOf<State<'Door'>>();
    expectTypeOf(Door.nameOf(Door.shut)).toEqualTypeOf<'shut' | 'open'>();
    expectTypeOf<machines.StateOf<typeof Door>>().toEqualTypeOf<State<'Door'>>();
    expectTypeOf<machines.EventOf<typeof Door>>().toEqualTypeOf<'push'>();
  });

  it('checks targets, events and guards against the definition', () => {
    expect(() =>
      machines.define('Typed', {
        events: ['startle'],
        states: {
          calm: {
            next: [
              // @ts-expect-error -- no such state
              { to: 'fleee' },
              // @ts-expect-error -- no such event
              { to: 'flee', on: 'startel' },
              // @ts-expect-error -- a guard returns a boolean
              { to: 'flee', when: (i: number) => i },
            ],
          },
          flee: {},
        },
      }),
    ).toThrow("sets `to` to 'fleee'");

    // @ts-expect-error -- a member's name
    expect(() => machines.define('Dice', { states: { roll: {} } })).toThrow(
      'as a definition member',
    );
    // @ts-expect-error -- a number as a state's name
    expect(() => machines.define('Count', { states: { idle: {}, 2: {} } })).toThrow(
      'rejects numeric state name',
    );

    const Gait = machines.define('Gait', { states: { walk: {} } });
    const Mood = machines.define('Mood', { states: { calm: {} } });
    // @ts-expect-error -- states of two machines don't compare
    expect(Gait.walk === Mood.calm).toBe(true);
  });

  it('throws for what the types catch, for callers without them', () => {
    expect(spec({})).toThrow('a machine needs a state');
    expect(spec({ a: { next: [{ to: 'b' }] } })).toThrow(
      "sets `to` to 'b'. Use the name of a defined state.",
    );
    expect(spec({ a: { next: [{ to: 'a', on: 'x' }] } })).toThrow(
      "sets `on` to 'x'. Use the name of a defined event.",
    );
    expect(spec({ a: { next: [{ to: 'a', after: -1 }] } })).toThrow('has `after` set to -1 s');
    expect(
      spec({
        a: {
          next: [
            { to: 'a', chance: 0.6 },
            { to: 'a', chance: 0.6 },
          ],
        },
      }),
    ).toThrow('add up to 1.2');
    expect(spec({ a: { next: [{ to: 'a', chance: 0.5, rate: 1 }] } })).toThrow(
      'has a chance and a rate',
    );
    expect(spec({ a: { next: [{ to: 'a', rate: 0 }] } })).toThrow('has `rate` set to 0');
    expect(spec({ state: {} })).toThrow("reserves 'state' as a definition member");
    expect(spec({ start: {} })).toThrow("reserves 'start' as a definition member");
    expect(spec({ b: {}, 1: {} })).toThrow("rejects numeric state name '1'");
    expect(spec({ a: {}, __proto__: {} })).toThrow("'__proto__' can't name a state");
    expect(spec({ a: {}, ['__proto__']: {} })).toThrow("'__proto__' can't name a state");
    expect(spec({ a: {} }, ['x', 'x'])).toThrow('unique events');
    expect(
      spec(
        { a: {} },
        Array.from({ length: 32 }, (_, i) => `e${i}`),
      ),
    ).toThrow('at most 31');
  });
});

describe('machines.run', () => {
  it('enters the first state at the first step, then counts the time in it', () => {
    const log: string[] = [];
    const Light = machines.define('Light', {
      states: {
        off: { enter: () => log.push('enter off'), next: [{ to: 'on', after: 0.5 }] },
        on: { enter: () => log.push('enter on') },
      },
    });
    const light = machines.run(Light);

    expect(light.state()).toBe(Light.off);
    light.step(STEP);
    expect(log).toEqual(['enter off']);
    expect(light.since()).toBe(0);

    for (let k = 0; k < 29; k++) light.step(STEP);
    expect(light.state()).toBe(Light.off);
    expect(light.since()).toBeCloseTo(29 * STEP);

    light.step(STEP);
    expect(light.state()).toBe(Light.on);
    expect(light.since()).toBe(0);
    expect(log).toEqual(['enter off', 'enter on']);
  });

  it('takes the first transition that holds, and at most one a step', () => {
    let hungry = false;
    let scared = false;
    const Mood = machines.define('Mood', {
      states: {
        calm: {
          next: [
            { to: 'flee', when: () => scared },
            { to: 'graze', when: () => hungry },
          ],
        },
        graze: { next: [{ to: 'flee', when: () => scared }] },
        flee: {},
      },
    });
    const mood = machines.run(Mood);
    mood.step(STEP);

    hungry = true;
    scared = true;
    mood.step(STEP);
    expect(mood.state()).toBe(Mood.flee);

    mood.start(0, Mood.calm);
    mood.step(STEP); // enters calm
    scared = false;
    mood.step(STEP);
    expect(mood.state()).toBe(Mood.graze);
    scared = true;
    mood.step(STEP);
    expect(mood.state()).toBe(Mood.flee);
  });

  it('runs the exit, then the enter, with the state already changed for the enter', () => {
    const log: string[] = [];
    const Flow = machines.define('Flow', {
      events: ['go'],
      states: {
        title: {
          exit: () => log.push(`exit title in ${flow.state()}`),
          next: [{ to: 'play', on: 'go' }],
        },
        play: { enter: () => log.push(`enter play in ${flow.state()}`) },
      },
    });
    const flow = machines.run(Flow);
    flow.step(STEP);
    flow.send.go();
    flow.step(STEP);

    expect(log).toEqual(['exit title in 0', 'enter play in 1']);
  });

  it('gives an event to the next step only, and an event a hook sends to the step after', () => {
    const Pair = machines.define('Pair', {
      events: ['poke'],
      states: {
        idle: {
          next: [
            { to: 'poked', on: 'poke', when: (i) => i === 0 || ready },
            { to: 'idle', after: 100 },
          ],
        },
        poked: { enter: (i) => i === 0 && pair.send.poke(1) },
      },
    });
    let ready = false;
    const pair = machines.run(Pair, { count: 2 });
    pair.step(STEP);

    pair.send.poke(1); // ignored: its guard doesn't hold, and the event is dropped
    pair.step(STEP);
    ready = true;
    pair.step(STEP);
    expect(pair.state(1)).toBe(Pair.idle);

    pair.send.poke(0);
    pair.step(STEP); // 0 is poked, and pokes 1
    expect([pair.state(0), pair.state(1)]).toEqual([Pair.poked, Pair.idle]);
    pair.step(STEP);
    expect(pair.state(1)).toBe(Pair.poked);
  });

  it('shares one roll a step among the chances of a state, in order', () => {
    const Walk = machines.define('Walk', {
      events: ['done'],
      states: {
        walk: {
          next: [
            { to: 'die', on: 'done', chance: 0.05 },
            { to: 'taunt', on: 'done', chance: 0.15 },
            { to: 'attack', on: 'done', chance: 0.3 },
            { to: 'idle', on: 'done' },
          ],
        },
        die: {},
        taunt: {},
        attack: {},
        idle: {},
      },
    });

    const after = (roll: number) => {
      const walker = machines.run(Walk, { random: () => roll });
      walker.step(STEP);
      walker.send.done();
      walker.step(STEP);
      return Walk.nameOf(walker.state());
    };

    expect([0, 0.049, 0.05, 0.19, 0.2, 0.49, 0.5, 0.99].map(after)).toEqual([
      'die',
      'die',
      'taunt',
      'taunt',
      'attack',
      'attack',
      'idle',
      'idle',
    ]);
  });

  it('gives a rate the share of the roll it holds in a step of each length', () => {
    const Lamp = machines.define('Lamp', {
      states: {
        lit: {
          next: [
            { to: 'flicker', chance: 0.25 },
            { to: 'out', rate: Math.LN2 }, // half the time in a second
          ],
        },
        flicker: {},
        out: {},
      },
    });

    const after = (roll: number, dt: number) => {
      const lamp = machines.run(Lamp, { random: () => roll });
      lamp.step(dt);
      lamp.step(dt);
      return Lamp.nameOf(lamp.state());
    };

    // A second's step: the rate takes [0.25, 0.75).
    expect([0.2, 0.3, 0.74, 0.76].map((roll) => after(roll, 1))).toEqual([
      'flicker',
      'out',
      'out',
      'lit',
    ]);
    // Half a second's: 1 - 2^-0.5, about 0.29, after the chance.
    expect([0.53, 0.55].map((roll) => after(roll, 0.5))).toEqual(['out', 'lit']);
    // Ten seconds': cut at 1.
    expect(after(0.999, 10)).toBe('out');
  });

  it('reads a delay from a function of the machine at each step', () => {
    const lengths = [0.1, 0.3];
    const Clip = machines.define('Clip', {
      states: { play: { next: [{ to: 'rest', after: (i) => lengths[i] }] }, rest: {} },
    });
    const clips = machines.run(Clip, { count: 2 });
    clips.step(STEP);

    for (let k = 0; k < 6; k++) clips.step(STEP);
    expect([clips.state(0), clips.state(1)]).toEqual([Clip.rest, Clip.play]);
    for (let k = 0; k < 12; k++) clips.step(STEP);
    expect(clips.state(1)).toBe(Clip.rest);
  });

  it('tells subscribers at a start and once after a step that changed states, until they stop', () => {
    const Flow = machines.define('Flow', {
      events: ['play'],
      states: {
        title: { next: [{ to: 'playing', on: 'play' }] },
        playing: { enter: (i) => i === 0 && flow.start(1, Flow.playing) },
      },
    });
    const flow = machines.run(Flow, { count: 2 });
    const told: string[] = [];
    const stop = flow.subscribe(() =>
      told.push(`${Flow.nameOf(flow.state(0))} ${Flow.nameOf(flow.state(1))}`),
    );

    flow.step(STEP);
    flow.step(STEP);
    expect(told).toEqual([]);
    flow.send.play(0);
    flow.step(STEP); // 0 plays, and starts 1
    expect(told).toEqual(['playing playing']);

    flow.start(0, Flow.title);
    expect(told).toEqual(['playing playing', 'title playing']);

    stop();
    flow.step(STEP);
    flow.send.play(0);
    flow.step(STEP);
    expect(flow.state(0)).toBe(Flow.playing);
    expect(told).toHaveLength(2);
  });

  it('enters a state a hook starts at the next step, whichever machine it is', () => {
    const log: string[] = [];
    const Wave = machines.define('Wave', {
      states: {
        still: { next: [{ to: 'rising', when: (i) => i === 1 }] },
        rising: {
          enter: (i) => {
            log.push(`rise ${i}`);
            wave.start(0, Wave.fallen);
            wave.start(2, Wave.fallen);
          },
        },
        fallen: { enter: (i) => log.push(`fall ${i}`) },
      },
    });
    const wave = machines.run(Wave, { count: 4 });
    wave.step(STEP);
    wave.step(STEP);
    expect(log).toEqual(['rise 1']);
    expect([wave.state(0), wave.state(2)]).toEqual([Wave.fallen, Wave.fallen]);

    wave.step(STEP);
    expect(log).toEqual(['rise 1', 'fall 0', 'fall 2']);
  });

  it('puts a machine where a start from its exit says, over the transition', () => {
    const Door = machines.define('Door', {
      events: ['push'],
      states: {
        shut: { exit: () => door.start(0, Door.locked), next: [{ to: 'open', on: 'push' }] },
        open: {},
        locked: {},
      },
    });
    const door = machines.run(Door);
    door.step(STEP);
    door.send.push();
    door.step(STEP);
    expect(door.state()).toBe(Door.locked);
  });

  it("throws for a machine or a state it doesn't have", () => {
    const One = machines.define('One', { events: ['e'], states: { only: {} } });
    const one = machines.run(One, { count: 2 });

    expect(() => one.state(2)).toThrow('machines 0 to 1, not 2');
    expect(() => one.send.e(-1)).toThrow('not -1');
    expect(() => one.start(0, 5 as State<'One'>)).toThrow('no state 5');
    expect(() => machines.run(One, { count: 0 })).toThrow('count must be');
  });
});

/** A roll for resolving shaders: the row and its tick, in thousandths. */
const counted = (i: number, tick: number) => {
  'use gpu';
  return d.f32(i + tick) / 1000;
};

/** The words of a buffer the rows wrote, by its label. */
const wordsOf = (recording: ReturnType<typeof recordingRoot>, label: string) => {
  const buffer = recording.log.writes.find((write) => write.buffer.label === label)?.buffer;
  return buffer ? new Uint32Array(recording.bytes(buffer).buffer.slice(0)) : new Uint32Array();
};

describe('machines.create', () => {
  const Herd = machines.define('Herd', {
    events: ['startle', 'calm'],
    states: {
      graze: {
        next: [
          { to: 'flee', on: 'startle' },
          { to: 'walk', after: 2 },
        ],
      },
      walk: { next: [{ to: 'graze', after: 1 }] },
      flee: { next: [{ to: 'graze', on: 'calm' }] },
    },
  });

  it('refuses a capacity past the workgroups the device dispatches', () => {
    const { root } = recordingRoot({ limits: { maxComputeWorkgroupsPerDimension: 4 } });
    expect(() => machines.create(root, Herd, { capacity: 257, step: STEP })).toThrow(
      'The largest capacity it allows is 256.',
    );
  });

  it('uploads starts and sends at the flush or else the step, adding to what no step took', () => {
    const recording = recordingRoot();
    const { root } = recording;
    const herd = machines.create(root, Herd, { capacity: 8, step: STEP });
    const rows = () => wordsOf(recording, 'Herd_rows');
    const mail = () => wordsOf(recording, 'Herd_mail');
    const step = () => {
      const pass = root.createCommandEncoder().beginComputePass();
      herd.step(pass);
      pass.end();
    };

    herd.start(3, Herd.flee);
    herd.send.calm(3);
    herd.send.startle(5);
    expect(rows()[3 * 2]).toBe(undefined);
    herd.flush();
    expect([rows()[3 * 2], rows()[3 * 2 + 1]]).toEqual([Herd.flee, 0]);
    expect(mail()[3]).toBe((0x80000000 | 2) >>> 0);
    expect(mail()[5]).toBe(1);

    herd.send.calm(5);
    herd.flush();
    expect(mail()[5]).toBe(3);

    step();
    herd.start(4, Herd.flee);
    herd.send.calm(5);
    step();
    expect(rows()[4 * 2]).toBe(Herd.flee);
    expect(mail()[5]).toBe(2);
  });

  it('writes starts, and events alone, in consecutive slots at once', () => {
    const recording = recordingRoot();
    const { root, log } = recording;
    const herd = machines.create(root, Herd, { capacity: 1024, step: STEP });
    const before = log.writes.length;

    for (let slot = 0; slot < 1000; slot++) herd.start(slot, Herd.walk);
    herd.start(1010);
    for (let slot = 1012; slot < 1016; slot++) herd.send.calm(slot);
    herd.flush();
    // The rows, the CPU's events, the kernels' events cleared, and the roll ticks.
    expect(
      log.writes.slice(before).map(({ buffer, offset, size }) => [buffer.label, offset, size]),
    ).toEqual([
      ['Herd_rows', 0, 8000],
      ['Herd_mail', 0, 4000],
      ['Herd_inbox', 0, 4000],
      ['Herd_ticks', 0, 4000],
      ['Herd_rows', 1010 * 8, 8],
      ['Herd_mail', 1010 * 4, 4],
      ['Herd_inbox', 1010 * 4, 4],
      ['Herd_ticks', 1010 * 4, 4],
      ['Herd_mail', 1012 * 4, 16],
    ]);
    const rows = wordsOf(recording, 'Herd_rows');
    expect([rows[999 * 2], rows[1010 * 2]]).toEqual([Herd.walk, Herd.graze]);
  });

  it('resolves a step and shaders that read the rows', () => {
    const recording = recordingRoot();
    const { root } = recording;
    const scared = root.createReadonly(d.arrayOf(d.f32, 64));
    const Mood = machines.define('Mood', {
      events: ['startle'],
      states: {
        calm: {
          next: [
            {
              to: 'flee',
              when: (i) => {
                'use gpu';
                return scared.$[i] > 0.5;
              },
            },
            { to: 'flee', on: 'startle' },
          ],
        },
        flee: { next: [{ to: 'calm', after: 1.5 }] },
      },
    });
    const moods = machines.create(root, Mood, { capacity: 64, step: STEP });

    const encoder = root.createCommandEncoder();
    const pass = encoder.beginComputePass();
    moods.step(pass);
    pass.end();

    const step = recording.log.shaders.at(-1)!;
    expect(step).toContain('fn Mood_next');
    expect(step).toContain('90u'); // 1.5 s at 60 steps a second

    // A shader reads two sets of one definition through their bound members, with nothing piped
    const others = machines.create(root, Mood, { capacity: 64, step: STEP });
    const out = root.createMutable(d.arrayOf(d.f32, 64));
    const read = tgpu.computeFn({ workgroupSize: [64], in: { gid: d.builtin.globalInvocationId } })(
      ({ gid }) => {
        'use gpu';
        if (moods.gpu.state(gid.x) === Mood.flee) out.$[gid.x] = moods.gpu.since(gid.x);
        others.gpu.send.startle(gid.x);
      },
    );
    const wgsl = tgpu.resolve([root.createComputePipeline({ compute: read })]);
    expect(wgsl).toContain('Mood_state');
    expect(wgsl.match(/var<storage, read_write> Mood_inbox\w*/g)).toEqual([
      'var<storage, read_write> Mood_inbox',
    ]);
    expect(wgsl.match(/var<storage, read> Mood_rows\w*/g)).toEqual([
      'var<storage, read> Mood_rows',
    ]);
  });

  it('starts rows from kernels given kernelStarts, and rolls only in the step', () => {
    const { root } = recordingRoot();
    const Spark = machines.define('Spark', {
      states: {
        lit: {
          enter: (i) => {
            'use gpu';
            if (Spark.roll(i) < 0.5) Spark.start(i, Spark.out);
          },
        },
        out: {},
      },
    });
    const spawn = tgpu.computeFn({
      workgroupSize: [64],
      in: { gid: d.builtin.globalInvocationId },
    })(({ gid }) => {
      'use gpu';
      Spark.start(gid.x, Spark.lit);
    });

    const sparks = machines.create(root, Spark, {
      capacity: 64,
      step: STEP,
      roll: counted,
      kernelStarts: true,
    });
    const encoder = root.createCommandEncoder();
    const pass = encoder.beginComputePass();
    sparks.step(pass);
    pass.end();
    expect(
      tgpu.resolve([root.pipe(sparks.inject()).createComputePipeline({ compute: spawn })]),
    ).toContain('atomicStore');

    const unstarted = machines.create(root, Spark, { capacity: 64, step: STEP, roll: counted });
    expect(() =>
      tgpu.resolve([root.pipe(unstarted.inject()).createComputePipeline({ compute: spawn })]),
    ).toThrow('given `kernelStarts`');
    const rolling = tgpu.computeFn({ workgroupSize: [1] })(() => {
      'use gpu';
      if (Spark.roll(0) > 2) return;
    });
    expect(() =>
      tgpu.resolve([root.pipe(sparks.inject()).createComputePipeline({ compute: rolling })]),
    ).toThrow('only in step guards, hooks');
  });

  it('lists rows by state in the step, for a kernel dispatched over a list', () => {
    const { root, log } = recordingRoot();
    const out = root.createMutable(d.arrayOf(d.f32, 100));
    const herd = machines.create(root, Herd, {
      capacity: 100,
      step: STEP,
      lists: [Herd.flee, Herd.walk],
    });
    const listed = herd.listed!;
    const flee = tgpu.computeFn({
      workgroupSize: [64],
      in: { gid: d.builtin.globalInvocationId },
    })(({ gid }) => {
      'use gpu';
      if (gid.x >= listed.count(0)) return;
      const i = listed.slot(0, gid.x);
      out.$[i] = herd.gpu.since(i);
    });
    const fleeing = root.createComputePipeline({ compute: flee });

    const encoder = root.createCommandEncoder();
    const pass = encoder.beginComputePass();
    const before = log.events.length;
    herd.step(pass);
    pass.setPipeline(fleeing);
    listed.dispatch(pass, 0);
    pass.end();
    expect(log.events.slice(before, -1)).toEqual(['dispatch', 'dispatch', 'dispatch-indirect']);
    expect(log.shaders.join('\n')).toContain('workgroupBarrier');

    expect(() =>
      machines.create(root, Herd, { capacity: 4, step: STEP, lists: [Herd.flee, Herd.flee] }),
    ).toThrow("'flee' twice");
    expect(() =>
      machines.create(root, Herd, { capacity: 4, step: STEP, lists: [3 as State<'Herd'>] }),
    ).toThrow("isn't a state of 'Herd'");
    const Other = machines.define('Other', { states: { only: {} } });
    // @ts-expect-error -- a state of another definition
    machines.create(root, Herd, { capacity: 4, step: STEP, lists: [Other.only] }).destroy();

    herd.destroy();
    expect(log.alive().join()).not.toContain('Herd_');
  });

  it('names the inject a shader is missing', () => {
    const { root } = recordingRoot();
    const read = tgpu.computeFn({ workgroupSize: [1] })(() => {
      'use gpu';
      if (Herd.state(0) === Herd.flee) return;
    });
    expect(() => tgpu.resolve([root.createComputePipeline({ compute: read })])).toThrow(
      "The pipeline is missing a fill for `Herd`'s rows",
    );
  });

  it('starts the ticks of rows started from the CPU where `random` draws', () => {
    const recording = recordingRoot();
    const Coin = machines.define('Coin', {
      states: { heads: { next: [{ to: 'tails', chance: 0.5 }] }, tails: {} },
    });
    const coins = machines.create(recording.root, Coin, {
      capacity: 4,
      step: STEP,
      roll: counted,
      random: () => 0.25,
    });

    coins.start(2);
    coins.flush();
    expect(wordsOf(recording, 'Coin_ticks')[2]).toBe(2 ** 22);
  });

  it('rolls from randf by default, and checks its options and slots', () => {
    const recording = recordingRoot();
    const { root } = recording;
    const Coin = machines.define('Coin', {
      states: { heads: { next: [{ to: 'tails', chance: 0.5 }] }, tails: {} },
    });

    const coins = machines.create(root, Coin, { capacity: 4, step: STEP });
    const encoder = root.createCommandEncoder();
    const pass = encoder.beginComputePass();
    coins.step(pass);
    pass.end();
    expect(recording.log.shaders.at(-1)).toContain('randSeed2');

    expect(() => machines.create(root, Herd, { capacity: 0, step: STEP })).toThrow('capacity');
    expect(() => machines.create(root, Herd, { capacity: 4, step: 0 })).toThrow(
      'step must be a positive',
    );

    const herd = machines.create(root, Herd, { capacity: 4, step: STEP });
    expect(() => herd.send.startle(4)).toThrow('slots 0 to 3, not 4');
    expect(() => herd.start(0, 7 as State<'Herd'>)).toThrow('no state 7');
  });
});
