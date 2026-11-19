/**
 * Defines states and transitions for machines that run on the CPU or for each table row on the
 * GPU.
 *
 * @module
 * @unstable
 * @injects definition its definition's rows, events and starts, for guards, hooks and code written
 * over any rows of the definition; each set's `gpu` members carry its own
 * @binds 1 storage: `state(i)`, the rows
 * @binds 1 storage, 1 uniform: `since(i)`, the rows and the step's length
 * @binds 1 storage: `send`, the events kernels send
 * @binds 1 storage: `start`, the starts kernels make
 * @owns 3 to 5 buffers, a uniform, a pipeline, liveLists with `lists`: the rows, the CPU's starts
 * and events, the events kernels send when the definition has any, the starts kernels make with
 * `kernelStarts`, the ticks that seed rolls, the step's length, the step and, with `lists`, the
 * rows in those states
 * @records `step(pass)` at each fixed step; it uploads the starts and sends made since the last
 * with `queue.writeBuffer`, which `flush()` does sooner for a frame that reads them and records no
 * step. With `lists`, dispatch work over `listed` after the step whose lists it reads.
 */

import {
  d,
  std,
  tgpu,
  type StorageFlag,
  type TgpuAccessor,
  type TgpuBuffer,
  type TgpuComputePass,
  type TgpuGenericFn,
  type TgpuMutableAccessor,
  type TgpuRoot,
  type TgpuSlot,
} from 'typegpu';
import { randf } from '@typegpu/noise';
import * as pipelines from './pipelines.ts';
import { unfilled, type Inject } from './inject.ts';
import * as liveLists from './liveLists.ts';
import { memo } from './memo.ts';
import { named } from './named.ts';

declare const stateBrand: unique symbol;
/** Identifies a state by its index in machine `N`. Compiles to a literal in WGSL. */
export type State<N extends string = string> = number & { readonly [stateBrand]: N };

/**
 * Tests whether machine `i` can take a transition. On the GPU, it is a shader function and `i` is
 * a row slot. On the CPU, `i` is the machine index.
 */
export type Guard = (i: number) => boolean;

/** Runs when machine `i` enters or leaves a state. On the GPU, it is a shader function. */
export type Hook = (i: number) => void;

/** Defines a transition out of a state. Every specified condition must hold. */
export interface Transition<K extends string, E extends string> {
  /** Names the destination state. */
  readonly to: K;
  /**
   * Allows the transition when the guard returns true. Checked last, after the other conditions
   * hold.
   */
  readonly when?: Guard;
  /**
   * Sets the delay in seconds after entering the state. A function computes the delay for machine
   * `i`.
   */
  readonly after?: number | ((i: number) => number);
  /** Requires this event in the current step. */
  readonly on?: E;
  /**
   * Assigns the transition a share of one roll in [0, 1) per step. The state's chances and rates
   * take consecutive shares of that roll in order. Chances of `0.05`, `0.15` and `0.3` receive 5,
   * 15 and 30 % respectively. The chances must sum to at most 1.
   */
  readonly chance?: number;
  /**
   * Sets an average transition rate per second. For a step of `dt` seconds, reserves a share `1 -
   * e^(-rate * dt)` of the roll after earlier shares. Truncates shares at 1.
   */
  readonly rate?: number;
}

/** Defines hooks for entering and leaving a state, and its outgoing transitions. */
export interface StateSpec<K extends string, E extends string> {
  /** Runs when a step enters the state, after the prior state's `exit`. */
  readonly enter?: Hook;
  /** Runs when a step leaves the state, before the next state's `enter`. */
  readonly exit?: Hook;
  /**
   * Lists transitions in the order each step checks them. The first whose conditions hold is
   * taken.
   */
  readonly next?: readonly Transition<K, E>[];
}

const RESERVED = [
  'name',
  'names',
  'events',
  'nameOf',
  'state',
  'since',
  'send',
  'start',
  'roll',
  '~parts',
] as const;
type Reserved = (typeof RESERVED)[number];

const numeric = (name: string) => String(Number(name)) === name;

/** CPU starts from hooks enter one step later to avoid entering twice in the same step. */
const ENTERS = 1;
const ENTERS_LATER = 2;

/** Reserves one bit per event in a `u32`, leaving one bit for starts. */
const MAX_EVENTS = 31;
const START = 0x80000000;
const NONE = 0xffffffff;
const MAX_CAPACITY = 2 ** 24;
/** Wrapping below 2^24 keeps the tick and slot exactly representable as `f32` seeds. */
const TICKS = 0xffffff;
const WORKGROUP = 64;
/** Live lists hold at most this many groups. */
const LISTS = 256;
/**
 * Allows rounding error in sums of CPU step durations when comparing against `after`, in seconds.
 */
const EPSILON = 1e-6;

interface Edge {
  readonly to: number;
  readonly mask: number;
  readonly after: number | ((i: number) => number);
  readonly when: Guard | undefined;
  /** Records whether the transition uses a chance or rate. Both are 0 otherwise. */
  readonly rolled: boolean;
  readonly chance: number;
  readonly rate: number;
}

interface Compiled {
  readonly enter: Hook | undefined;
  readonly exit: Hook | undefined;
  readonly next: readonly Edge[];
  readonly rolls: boolean;
}

const rowSchema = () => d.struct({ state: d.u32, age: d.u32 });
type RowSchema = ReturnType<typeof rowSchema>;

interface Parts {
  readonly states: readonly Compiled[];
  readonly row: RowSchema;
  readonly rows: TgpuAccessor<d.WgslArray<RowSchema>>;
  readonly inbox: TgpuMutableAccessor<d.WgslArray<d.Atomic<d.U32>>>;
  readonly starts: TgpuMutableAccessor<d.WgslArray<d.Atomic<d.U32>>>;
  readonly ticks: TgpuMutableAccessor<d.WgslArray<d.U32>>;
  readonly roller: TgpuSlot<(i: number, tick: number) => number>;
  readonly dt: TgpuAccessor<d.F32>;
}

/** Provides state names and shader functions shared by every definition. */
export interface Members<N extends string, K extends string, E extends string> {
  /** Labels types and WGSL. It is not a lookup key. */
  readonly name: N;
  /** Lists states in order. Each state's index is its number. Machines start in the first state. */
  readonly names: readonly K[];
  /** Lists events in order. Each event's bit is its index. */
  readonly events: readonly E[];
  /** Returns a state name for labels and logs. */
  nameOf(state: State<N>): K;
  /**
   * Reads the state of row `i` in a shader written over any rows of the definition, such as a
   * guard. A `machines.create`'s `gpu.state` reads its own rows.
   */
  readonly state: TgpuGenericFn<(i: number) => State<N>>;
  /** Reads the time spent in the current state by row `i`, in seconds measured as whole steps. */
  readonly since: TgpuGenericFn<(i: number) => number>;
  /** Sends events to row `i` for its next step. Available only in compute shaders. */
  readonly send: { readonly [P in E]: TgpuGenericFn<(i: number) => void> };
  /**
   * Starts row `i` in `state` at its next step, which runs `enter` and drops the row's events.
   * Available only in compute shaders, for rows created with `kernelStarts`.
   */
  readonly start: TgpuGenericFn<(i: number, state: State<N>) => void>;
  /**
   * Draws a fresh number in [0, 1) for row `i` using the rows' `roll` function. Available in step
   * guards, hooks and `after` functions. The step calls `after` functions every time. To keep a
   * random duration, draw it in `enter` and store it.
   */
  readonly roll: TgpuGenericFn<(i: number) => number>;
  readonly '~parts': Parts;
}

/**
 * Gives a definition's shader members bound to one set of machine rows: `state`, `since`, `send`
 * and `start`, which read those rows in any pipeline without a fill.
 */
type GpuMembers<N extends string, K extends string, E extends string> = Pick<
  Members<N, K, E>,
  'state' | 'since' | 'send' | 'start'
>;

/** Defines states, transitions and shader functions that read machine rows. Needs no root. */
export type Definition<N extends string, K extends string, E extends string> = Members<N, K, E> & {
  readonly [P in K]: State<N>;
};

// oxlint-disable-next-line typescript/no-explicit-any -- matches a definition of any states and events
export type AnyDefinition = Members<any, any, any>;

/** Extracts the branded state type from a definition. */
export type StateOf<D extends AnyDefinition> = State<D['name']>;

/** Extracts the event names from a definition. */
export type EventOf<D extends AnyDefinition> = D['events'][number];

/** JavaScript lists integer keys first, so numeric state names would change the declared order. */
type Unnumbered = {
  readonly [
    index: number
  ]: 'a number, which JavaScript orders before the other states, names no state';
};

type Validated<K extends string, E extends string> = {
  readonly [P in K]: P extends Reserved
    ? `'${P}' is a member of every definition; name the state otherwise`
    : StateSpec<NoInfer<K>, NoInfer<E>>;
};

/**
 * Writes each transition's roll interval [from, until) at `2k` in `bounds`. Uses chances directly
 * and converts rates for steps of `dt` seconds. Assigns shares in order and truncates at 1.
 */
function fillShares(next: readonly Edge[], dt: number, bounds: Float64Array) {
  let share = 0;
  for (let k = 0; k < next.length; k++) {
    const edge = next[k];
    const from = share;
    if (edge.chance > 0) share += edge.chance;
    else if (edge.rate > 0) share -= Math.expm1(-edge.rate * dt);
    bounds[k * 2] = Math.min(from, 1);
    bounds[k * 2 + 1] = Math.min(share, 1);
  }
}

/**
 * Declares states, ordered transitions and events for a state machine. Each state can run hooks
 * when entered or left. Use `run` on the CPU with plain functions, or `create` on the GPU with
 * shader functions that take row slots. Needs no root, so define it where the resources its hooks
 * read live.
 *
 * `name` brands states as `State<'Mood'>` and labels WGSL functions and errors. It is not a lookup
 * key. Throws for undeclared transition targets or events, invalid delays, chances or rates,
 * chances that sum above 1, or a transition with both a chance and a rate. Also throws for missing
 * states, duplicate events, more than 31 events, numeric state names or reserved names such as
 * `state` and `__proto__`.
 *
 * @param name a label for types and WGSL
 * @param spec the events, then the states; machines start in the first
 * @returns a definition that `run` runs on the CPU and `create` runs per row on the GPU
 */
export function define<const N extends string, K extends string, const E extends string = never>(
  name: N,
  spec: { readonly events?: readonly E[]; readonly states: Validated<K, E> & Unnumbered },
): Definition<N, K, E> {
  const events = (spec.events ?? []) as readonly E[];
  const states = spec.states as unknown as Record<K, StateSpec<K, E>>;
  const names = Object.keys(states) as K[];
  const where = `machines.define('${name}')`;

  if (names.length === 0) throw new Error(`${where}: a machine needs a state.`);
  if (events.length > MAX_EVENTS || new Set(events).size !== events.length) {
    throw new RangeError(
      `${where} accepts at most ${MAX_EVENTS} unique events, got ${events.join(', ')}.`,
    );
  }
  // `{ __proto__: … }` sets the object prototype instead of declaring a state
  const prototype = Object.getPrototypeOf(states);
  if ((prototype !== Object.prototype && prototype !== null) || names.includes('__proto__' as K)) {
    throw new Error(`${where}: '__proto__' can't name a state.`);
  }
  for (const state of names) {
    if ((RESERVED as readonly string[]).includes(state)) {
      throw new Error(
        `${where} reserves '${state}' as a definition member. Choose another state name.`,
      );
    }
    if (numeric(state)) {
      throw new Error(
        `${where} rejects numeric state name '${state}' because JavaScript orders numeric keys first. Choose a nonnumeric name.`,
      );
    }
  }

  const compiled = names.map((state): Compiled => {
    const { enter, exit, next = [] } = states[state];
    let share = 0;
    const edges = next.map((transition, k): Edge => {
      const at = `${where}: '${state}', transition ${k}`;
      const to = names.indexOf(transition.to);
      if (to < 0) throw new Error(`${at} leads to '${transition.to}', which isn't a state.`);
      let mask = 0;
      if (transition.on !== undefined) {
        const bit = events.indexOf(transition.on);
        if (bit < 0) throw new Error(`${at} takes '${transition.on}', which isn't an event.`);
        mask = 2 ** bit;
      }
      const after = transition.after ?? 0;
      if (typeof after === 'number' && !(after >= 0 && Number.isFinite(after))) {
        throw new RangeError(`${at} waits ${after} s; give a finite number of seconds, 0 or more.`);
      }
      const { chance = 0, rate = 0 } = transition;
      if (transition.chance !== undefined && transition.rate !== undefined) {
        throw new Error(`${at} has a chance and a rate; give one.`);
      }
      if (transition.chance !== undefined) {
        if (!(chance > 0 && chance <= 1)) {
          throw new RangeError(`${at} has a chance of ${chance}; give one in (0, 1].`);
        }
        share += chance;
        if (share > 1 + 1e-9) {
          throw new RangeError(
            `${where}: the chances out of '${state}' add up to ${share}, past 1.`,
          );
        }
      }
      if (transition.rate !== undefined && !(rate > 0 && Number.isFinite(rate))) {
        throw new RangeError(
          `${at} has a rate of ${rate}; give a finite number a second, above 0.`,
        );
      }
      const rolled = chance > 0 || rate > 0;
      return { to, mask, after, when: transition.when, rolled, chance, rate };
    });
    return { enter, exit, next: edges, rolls: edges.some((edge) => edge.rolled) };
  });

  const row = rowSchema().$name(`${name}Row`);
  const rows = tgpu
    .accessor(
      d.arrayOf(row),
      unfilled(`\`${name}\`'s rows`, 'states', `a \`machines.create\` of \`${name}\``),
    )
    .$name(`${name}_rows`);
  const inbox = tgpu
    .mutableAccessor(
      d.arrayOf(d.atomic(d.u32)),
      unfilled(`\`${name}\`'s events`, 'states', `a \`machines.create\` of \`${name}\``),
    )
    .$name(`${name}_inbox`) as TgpuMutableAccessor<d.WgslArray<d.Atomic<d.U32>>>;
  const dt = tgpu
    .accessor(
      d.f32,
      unfilled(`\`${name}\`'s step`, 'states', `a \`machines.create\` of \`${name}\``),
    )
    .$name(`${name}_dt`);
  const starts = tgpu
    .mutableAccessor(
      d.arrayOf(d.atomic(d.u32)),
      unfilled(
        `\`${name}\`'s starts from kernels`,
        'states',
        `a \`machines.create\` of \`${name}\` given \`kernelStarts\``,
      ),
    )
    .$name(`${name}_starts`) as TgpuMutableAccessor<d.WgslArray<d.Atomic<d.U32>>>;
  const ticks = tgpu
    .mutableAccessor(d.arrayOf(d.u32), (): never => {
      throw new Error(`Call \`${name}.roll\` only in step guards, hooks and \`after\` functions.`);
    })
    .$name(`${name}_ticks`) as TgpuMutableAccessor<d.WgslArray<d.U32>>;
  const roller = tgpu.slot<(i: number, tick: number) => number>().$name(`${name}_roller`);

  const state = named(`${name}_state`, (i: number) => {
    'use gpu';
    return rows.$[i].state;
  }) as unknown as TgpuGenericFn<(i: number) => State<N>>;
  const since = named(`${name}_since`, (i: number) => {
    'use gpu';
    return d.f32(rows.$[i].age) * dt.$;
  });
  const send = Object.fromEntries(
    events.map((event, bit) => {
      const mask = 2 ** bit;
      const fn = named(`${name}_send_${event}`, (i: number) => {
        'use gpu';
        std.atomicOr(inbox.$[i], d.u32(mask));
      });
      return [event, fn];
    }),
  ) as Members<N, K, E>['send'];

  const startRow = named(`${name}_start`, (i: number, to: number) => {
    'use gpu';
    std.atomicStore(starts.$[i], d.u32(to) + 1);
  }) as unknown as TgpuGenericFn<(i: number, state: State<N>) => void>;
  const roll = named(`${name}_roll`, (i: number) => {
    'use gpu';
    const tick = (ticks.$[i] + 1) & TICKS;
    ticks.$[i] = tick;
    return roller.$(i, tick);
  });

  const definition: Record<string, unknown> = {
    name,
    names,
    events,
    nameOf: (value: State<N>) => {
      const found = names[value];
      if (found === undefined) throw new RangeError(`${where} has no state ${value}.`);
      return found;
    },
    state,
    since,
    send,
    start: startRow,
    roll,
    '~parts': {
      states: compiled,
      row,
      rows,
      inbox,
      starts,
      ticks,
      roller,
      dt,
    } satisfies Parts,
  };
  names.forEach((key, index) => {
    definition[key] = index;
  });

  return definition as unknown as Definition<N, K, E>;
}

export interface RunOptions {
  /** Sets the number of CPU machines, each with its own index. Default 1. */
  readonly count?: number;
  /**
   * Draws a number in [0, 1) for each step in a state with chances or rates. Default
   * `Math.random`.
   * @cpu
   */
  readonly random?: () => number;
}

/** Runs machines on the CPU. Methods that select a machine take its index, with a default of 0. */
export interface Running<N extends string, K extends string, E extends string> {
  /** Holds the definition the machines run, from `machines.define`. */
  readonly definition: Definition<N, K, E>;
  /** Reports the number of machines, from the `count` option. */
  readonly count: number;
  /** Returns the state of machine `i`. */
  state(i?: number): State<N>;
  /** Returns the seconds machine `i` has spent in its state. */
  since(i?: number): number;
  /** Sends an event to machine `i` for its next step. That step drops the event if unused. */
  readonly send: { readonly [P in E]: (i?: number) => void };
  /**
   * Starts machine `i` in `state` and notifies subscribers. Defaults to the first state. Runs
   * `enter` at the next step. When called from a hook, notifies after the current step and runs
   * `enter` one step later.
   */
  start(i?: number, state?: State<N>): void;
  /** Advances every machine by `dt` seconds. */
  step(dt: number): void;
  /**
   * Calls `listener` after steps that change a state and after each `start`. Returns a function
   * that unsubscribes. Use it with `useSyncExternalStore(flow.subscribe, () => flow.state())`.
   */
  subscribe(listener: () => void): () => void;
}

/**
 * Runs a definition on the CPU with every machine in the first state. Runs that state's `enter` at
 * the first step. Throws unless `count` is an integer above 0.
 *
 * @allocates never in a step
 * @param def the definition, from `machines.define`
 * @returns machines that step when you call `step`
 */
export function run<N extends string, K extends string, E extends string>(
  def: Definition<N, K, E>,
  options: RunOptions = {},
): Running<N, K, E> {
  const { count = 1, random = Math.random } = options;
  if (!Number.isInteger(count) || count < 1) {
    throw new RangeError(`machines.run: count must be a whole number, 1 or more, got ${count}.`);
  }
  const parts = def['~parts'];
  const states = parts.states;
  const current = new Uint32Array(count);
  const age = new Float64Array(count);
  const entering = new Uint8Array(count).fill(ENTERS);
  let inbox = new Uint32Array(count);
  let taking = new Uint32Array(count);
  const listeners = new Set<() => void>();
  const notify = () => {
    for (const listener of listeners) listener();
  };
  let stepping = false;
  let startedInStep = false;
  // Recalculate roll intervals when the step duration changes
  const bounds = states.map((state) => new Float64Array(state.next.length * 2));
  let boundsDt = Number.NaN;

  const at = (i: number) => {
    if (!(Number.isInteger(i) && i >= 0 && i < count)) {
      throw new RangeError(`'${def.name}' has machines 0 to ${count - 1}, not ${i}.`);
    }
    return i;
  };

  const send = Object.fromEntries(
    def.events.map((event, bit) => {
      const mask = 2 ** bit;
      return [
        event,
        (i = 0) => {
          inbox[at(i)] |= mask;
        },
      ];
    }),
  ) as Running<N, K, E>['send'];

  const step = (dt: number) => {
    // Events sent by hooks must wait until the next step
    const taken = inbox;
    inbox = taking;
    taking = taken;
    inbox.fill(0);
    if (dt !== boundsDt) {
      for (let s = 0; s < states.length; s++) {
        if (states[s].rolls) fillShares(states[s].next, dt, bounds[s]);
      }
      boundsDt = dt;
    }

    let changed = false;
    stepping = true;
    try {
      for (let i = 0; i < count; i++) {
        const from = current[i];
        if (entering[i] === ENTERS_LATER) continue;
        if (entering[i] === ENTERS) {
          entering[i] = 0;
          age[i] = 0;
          states[from].enter?.(i);
          continue;
        }

        const ev = taken[i];
        const seconds = age[i] + dt;
        const state = states[from];
        const shares = bounds[from];
        let roll = -1;
        let to = -1;
        for (let k = 0; k < state.next.length; k++) {
          const edge = state.next[k];
          if ((ev & edge.mask) !== edge.mask) continue;
          const after = typeof edge.after === 'number' ? edge.after : edge.after(i);
          if (seconds + EPSILON < after) continue;
          if (edge.rolled) {
            if (roll < 0) roll = random();
            if (roll < shares[k * 2] || roll >= shares[k * 2 + 1]) continue;
          }
          if (edge.when !== undefined && !edge.when(i)) continue;
          to = edge.to;
          break;
        }

        if (to < 0) {
          age[i] = seconds;
          continue;
        }
        state.exit?.(i);
        // A start from `exit` overrides the pending transition
        if (entering[i] === ENTERS_LATER) continue;
        current[i] = to;
        age[i] = 0;
        changed = true;
        states[to].enter?.(i);
      }
    } finally {
      stepping = false;
    }

    if (startedInStep) {
      startedInStep = false;
      changed = true;
      for (let i = 0; i < count; i++) if (entering[i] === ENTERS_LATER) entering[i] = ENTERS;
    }
    if (changed) notify();
  };

  return {
    definition: def,
    count,
    state: (i = 0) => current[at(i)] as State<N>,
    since: (i = 0) => age[at(i)],
    send,
    start(i = 0, state = 0 as State<N>) {
      at(i);
      if (!(Number.isInteger(state) && state >= 0 && state < states.length)) {
        throw new RangeError(`'${def.name}' has no state ${state}.`);
      }
      current[i] = state;
      age[i] = 0;
      if (stepping) {
        entering[i] = ENTERS_LATER;
        startedInStep = true;
      } else {
        entering[i] = ENTERS;
        notify();
      }
    },
    step,
    subscribe(listener) {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
  };
}

export interface Options<N extends string = string> {
  /** Sets the number of row slots, from `0` to `capacity - 1`. Must be an integer in [1, 2^24]. */
  readonly capacity: number;
  /**
   * Sets the length of a step in seconds, such as a fixed-step clock's `sim.step`. Transitions
   * compile it into their waits and rate shares.
   */
  readonly step: number;
  /**
   * Tests whether slot `i` holds a live row, such as a table's `live()` function. Skips dead rows.
   * Default every slot is live.
   * @gpu
   */
  readonly alive?: (i: number) => boolean;
  /**
   * Returns the highest live slot plus one, read when `step` records, such as a table's from
   * `live()`. Skips the slots above it. Default `capacity`.
   * @cpu
   */
  readonly span?: () => number;
  /**
   * Draws a number in [0, 1) for row `i` at each step, for transitions with a chance or rate.
   * `tick` counts steps from a random start and wraps below 2^24. Also supplies the definition's
   * `roll(i)`. Default `randf.seed2(d.vec2f(d.f32(i), d.f32(tick)))`, then `randf.sample()`.
   * @gpu
   */
  readonly roll?: (i: number, tick: number) => number;
  /**
   * Allows compute shaders to start rows with the definition's `start(i, state)`. Uses one word
   * per row and one load per step. Default false.
   */
  readonly kernelStarts?: boolean;
  /**
   * Lists rows in these states after each step into `listed`, for kernels that run over them.
   * Uses one word per row for each listed state. Accepts at most 256 states. Default none.
   */
  readonly lists?: readonly State<N>[];
  /**
   * Draws a number in [0, 1) to set the initial roll tick of a row started from the CPU. Default
   * `Math.random`.
   * @cpu
   */
  readonly random?: () => number;
}

/** Runs one state machine per row on the GPU. */
export interface Machines<N extends string, K extends string, E extends string> extends Disposable {
  /** Holds the definition the rows run, from `machines.define`. */
  readonly definition: Definition<N, K, E>;
  /** Reports the number of row slots, from the `capacity` option. */
  readonly capacity: number;
  /** Reports the step duration in seconds, from the `step` option. */
  readonly dt: number;
  /**
   * Holds the rows in each state of `lists` after the last step, group `k` for `lists[k]`, for
   * work of your own through `listed.slot` and `listed.dispatch`. Undefined without `lists`.
   */
  readonly listed: liveLists.LiveLists | undefined;
  /**
   * Provides the definition's `state`, `since`, `send` and `start` bound to these rows, such as
   * `gaits.gpu.state(i)`. `start` needs `kernelStarts`.
   */
  readonly gpu: GpuMembers<N, K, E>;
  /**
   * Fills the definition's `state`, `since`, `send` and, with `kernelStarts`, `start`, for code
   * written over any rows of the definition, such as guards.
   */
  inject(): Inject;
  /**
   * Adds `fill` to the step pipeline, such as a table's `inject()` for guards and hooks. Returns
   * the same rows with that fill.
   */
  pipe(fill: Inject): Machines<N, K, E>;
  /**
   * Starts row `slot` in `state` when the CPU writes are uploaded. Defaults to the first state.
   * Runs `enter` at the next step.
   */
  start(slot: number, state?: State<N>): void;
  /** Sends an event to a row for its next step. That step drops the event if unused. */
  readonly send: { readonly [P in E]: (slot: number) => void };
  /**
   * Uploads CPU starts and events since the last upload. `step` does this too. Call before the
   * frame's steps when the frame may read this state without recording a step.
   */
  flush(): void;
  /**
   * Uploads pending CPU starts and events, records one step for every live row, then lists the
   * rows in each state of `lists`.
   */
  step(pass: TgpuComputePass): void;
  /** Compiles the step pipeline and the pipelines of its lists. Calling it is optional. */
  initAsync(): Promise<void>;
  /** Stores one row per slot, with its state and the number of steps spent in it. */
  readonly rows: TgpuBuffer<d.WgslArray<RowSchema>> & StorageFlag;
  /** Destroys the buffers and lists. */
  destroy(): void;
}

/** Seeds `randf` by row and tick, then samples it. */
const randomRoll = (i: number, tick: number) => {
  'use gpu';
  randf.seed2(d.vec2f(d.f32(i), d.f32(tick)));
  return randf.sample();
};

/** Builds a function that runs state `current`'s `enter` or `exit` hook, if it has one. */
function hookFn(label: string, states: readonly Compiled[], hook: 'enter' | 'exit') {
  const hooks = states.flatMap((state, index) => {
    const fn = state[hook];
    return fn ? [{ index, fn }] : [];
  });
  return named(label, (i: number, current: number) => {
    'use gpu';
    for (const h of tgpu.unroll(hooks)) {
      if (current === h.index) h.fn(i);
    }
  });
}

/**
 * Creates a GPU state machine for every slot in a table or another set of rows. Rows start in the
 * first state without running `enter`. Call `start(slot)` when adding a row. Shaders read rows
 * with `gpu.state(i)` and `gpu.since(i)`, and send events with `gpu.send`. Throws for invalid capacity or step duration, invalid or duplicate
 * listed states, more than 256 lists, or lists exceeding `maxStorageBufferBindingSize`.
 *
 * @param def the definition, from `machines.define`
 * @param options.alive whether a slot holds a live row, such as a table's from `rows.live()`,
 * spread into the options with its capacity
 * @returns rows you step at each fixed step and destroy
 */
export function create<N extends string, K extends string, E extends string>(
  root: TgpuRoot,
  def: Definition<N, K, E>,
  options: Options<N>,
): Machines<N, K, E> {
  const {
    capacity,
    step: dt,
    alive,
    span,
    roll = randomRoll,
    kernelStarts = false,
    lists = [],
    random = Math.random,
  } = options;
  const parts = def['~parts'];
  const name = def.name;
  const where = `machines.create('${name}')`;
  if (!Number.isInteger(capacity) || capacity < 1 || capacity > MAX_CAPACITY) {
    throw new RangeError(
      `${where}: capacity must be a whole number in [1, 2^24], got ${capacity}.`,
    );
  }
  if (!(dt > 0 && Number.isFinite(dt))) {
    throw new RangeError(`${where}: step must be a positive number of seconds, got ${dt}.`);
  }
  if (lists.length > LISTS) {
    throw new RangeError(`${where}: lists must be at most ${LISTS} states, got ${lists.length}.`);
  }
  for (const [k, listed] of lists.entries()) {
    if (!(Number.isInteger(listed) && listed >= 0 && listed < parts.states.length)) {
      throw new RangeError(`${where}: lists has ${listed}, which isn't a state of '${name}'.`);
    }
    if (lists.indexOf(listed) !== k) {
      throw new Error(`${where}: lists has '${def.names[listed]}' twice.`);
    }
  }
  const rowsBuffer = root
    .createBuffer(d.arrayOf(parts.row, capacity))
    .$usage('storage')
    .$name(`${name}_rows`);
  const mailBuffer = root
    .createBuffer(d.arrayOf(d.u32, capacity))
    .$usage('storage')
    .$name(`${name}_mail`);
  const inboxBuffer =
    def.events.length > 0
      ? root
          .createBuffer(d.arrayOf(d.atomic(d.u32), capacity))
          .$usage('storage')
          .$name(`${name}_inbox`)
      : undefined;
  const startsBuffer = kernelStarts
    ? root
        .createBuffer(d.arrayOf(d.atomic(d.u32), capacity))
        .$usage('storage')
        .$name(`${name}_starts`)
    : undefined;
  const ticksBuffer = root
    .createBuffer(d.arrayOf(d.u32, capacity))
    .$usage('storage')
    .$name(`${name}_ticks`);
  const listed =
    lists.length > 0
      ? liveLists.create(root, { capacity, groups: lists.length, label: `${name}_lists` })
      : undefined;
  const dtUniform = root.createUniform(d.f32, dt);
  const destroy = () => {
    for (const buffer of [rowsBuffer, mailBuffer, ticksBuffer]) buffer.destroy();
    inboxBuffer?.destroy();
    startsBuffer?.destroy();
    listed?.destroy();
  };
  const rowsGpu = root.unwrap(rowsBuffer);
  const mailGpu = root.unwrap(mailBuffer);
  const inboxGpu = inboxBuffer && root.unwrap(inboxBuffer);
  const startsGpu = startsBuffer && root.unwrap(startsBuffer);
  const ticksGpu = root.unwrap(ticksBuffer);
  const queue = root.device.queue;

  const reading = rowsBuffer.as('readonly');
  const writing = rowsBuffer.as('mutable');
  const mailing = mailBuffer.as('mutable');
  const sending = inboxBuffer?.as('mutable');
  const starting = startsBuffer?.as('mutable');
  const ticking = ticksBuffer.as('mutable');

  const filling =
    (rows: typeof reading | typeof writing): Inject =>
    (cfg) => {
      let filled = cfg.with(parts.rows, rows).with(parts.dt, dtUniform);
      if (sending) filled = filled.with(parts.inbox, sending);
      if (starting) filled = filled.with(parts.starts, starting);
      return filled;
    };
  const inject = filling(reading);
  const gpu: GpuMembers<N, K, E> = {
    state: def.state.pipe(inject),
    since: def.since.pipe(inject),
    send: Object.fromEntries(
      def.events.map((event) => [event, def.send[event].pipe(inject)]),
    ) as GpuMembers<N, K, E>['send'],
    start: def.start.pipe(inject),
  };
  // Step hooks must read the same mutable rows and roll data as the step
  const stepping: Inject = (cfg) => {
    const filled = filling(writing)(cfg);
    return filled.with(parts.ticks, ticking).with(parts.roller, roll);
  };

  // Unrolls every state's transitions into one function, in the order `run` checks them
  const table = parts.states
    .map((state, s) => {
      const shares = new Float64Array(state.next.length * 2);
      fillShares(state.next, dt, shares);
      const edges = state.next.map((edge, k) => ({
        to: edge.to,
        mask: edge.mask,
        wait:
          typeof edge.after === 'number' ? Math.max(0, Math.ceil(edge.after / dt - EPSILON)) : 0,
        after: typeof edge.after === 'number' ? undefined : edge.after,
        rolled: edge.rolled,
        from: shares[k * 2],
        until: shares[k * 2 + 1],
        when: edge.when,
      }));
      return { state: s, rolls: state.rolls, edges };
    })
    .filter((s) => s.edges.length > 0);
  const pick = named(`${name}_next`, (i: number, current: number, age: number, ev: number) => {
    'use gpu';
    for (const s of tgpu.unroll(table)) {
      if (current === s.state) {
        const drawn = s.rolls ? def.roll(i) : d.f32(0);
        for (const e of tgpu.unroll(s.edges)) {
          // Tests of an edge's own fields resolve in JS, keeping always-true conditions out of WGSL
          if (e.mask === 0 || (ev & e.mask) === e.mask) {
            if (
              e.after === undefined ? e.wait === 0 || age >= e.wait : d.f32(age) * dt >= e.after(i)
            ) {
              if (!e.rolled || (drawn >= e.from && drawn < e.until)) {
                if (e.when === undefined || e.when(i)) return d.u32(e.to);
              }
            }
          }
        }
        return d.u32(NONE);
      }
    }
    return d.u32(NONE);
  });
  const enter = hookFn(`${name}_enter`, parts.states, 'enter');
  const exit = hookFn(`${name}_exit`, parts.states, 'exit');
  const RowOf = parts.row;

  const advance = named(`${name}_advance`, (i: number) => {
    'use gpu';
    if (alive !== undefined && !alive(i)) return d.u32(NONE);

    const row = RowOf(writing.$[i]);
    let ev = mailing.$[i];
    if (ev !== 0) mailing.$[i] = 0;
    // Load before exchanging so rows with no events do not write
    if (sending !== undefined && std.atomicLoad(sending.$[i]) !== 0) {
      ev = ev | std.atomicExchange(sending.$[i], 0);
    }
    // A start step only enters the state. Kernel starts override CPU starts, whose flush has
    // already written the whole row. They store state plus one so zero means no start
    if (starting !== undefined) {
      if (std.atomicLoad(starting.$[i]) !== 0) {
        const to = std.atomicExchange(starting.$[i], 0) - 1;
        writing.$[i] = RowOf({ state: to, age: 0 });
        enter(i, to);
        return to;
      }
    }
    if ((ev & START) !== 0) {
      enter(i, row.state);
      return row.state;
    }

    const age = row.age + 1;
    const to = pick(i, row.state, age, ev);
    if (to === NONE) {
      writing.$[i].age = age;
      return row.state;
    }
    exit(i, row.state);
    writing.$[i] = RowOf({ state: to, age: 0 });
    enter(i, to);
    return to;
  });

  const kernel = tgpu
    .computeFn({
      workgroupSize: [WORKGROUP],
      in: { gid: d.builtin.globalInvocationId },
    })(({ gid }) => {
      'use gpu';
      const i = gid.x;
      if (i >= capacity) return;
      advance(i);
    })
    .$name(`${name}_step`);

  // Listing synchronises the workgroup, so every thread reaches it
  const groups = lists.map((state, group) => ({ state, group }));
  const listing = listed
    ? tgpu
        .computeFn({
          workgroupSize: [WORKGROUP],
          in: { gid: d.builtin.globalInvocationId },
        })(({ gid }) => {
          'use gpu';
          const i = gid.x;
          let which = d.u32(NONE);
          if (i < capacity) {
            const state = advance(i);
            for (const g of tgpu.unroll(groups)) {
              if (state === g.state) which = d.u32(g.group);
            }
          }
          listed.list(which, i);
        })
        .$name(`${name}_step`)
    : undefined;

  // Keep flushed starts and events until a step consumes them, because another flush may occur
  // first
  const starts = new Int32Array(capacity).fill(-1);
  const mail = new Uint32Array(capacity);
  const flying = new Uint32Array(capacity);
  const dirty = new Uint32Array(capacity);
  const marked = new Uint8Array(capacity);
  const flown = new Uint32Array(capacity);
  const flownMarked = new Uint8Array(capacity);
  let dirtyCount = 0;
  let flownCount = 0;
  // CPU starts clear pending kernel events and starts as they upload the row and its roll tick
  const started = new Uint32Array(capacity * 2);
  const posted = new Uint32Array(capacity);
  const seeds = new Uint32Array(capacity);
  const empty = inboxGpu || startsGpu ? new Uint32Array(capacity) : undefined;

  const mark = (slot: number) => {
    if (!(Number.isInteger(slot) && slot >= 0 && slot < capacity)) {
      throw new RangeError(`${where} has slots 0 to ${capacity - 1}, not ${slot}.`);
    }
    if (marked[slot] === 0) {
      marked[slot] = 1;
      dirty[dirtyCount++] = slot;
    }
  };

  const send = Object.fromEntries(
    def.events.map((event, bit) => {
      const mask = 2 ** bit;
      return [
        event,
        (slot: number) => {
          mark(slot);
          mail[slot] |= mask;
        },
      ];
    }),
  ) as Machines<N, K, E>['send'];

  // Combine consecutive slots into one write per buffer
  let runFrom = 0;
  let runTo = 0;
  const upload = () => {
    if (runTo === runFrom) return;
    const rows = runTo - runFrom;
    queue.writeBuffer(rowsGpu, runFrom * 8, started.buffer, runFrom * 8, rows * 8);
    queue.writeBuffer(mailGpu, runFrom * 4, posted.buffer, runFrom * 4, rows * 4);
    if (inboxGpu && empty) {
      queue.writeBuffer(inboxGpu, runFrom * 4, empty.buffer, runFrom * 4, rows * 4);
    }
    if (startsGpu && empty) {
      queue.writeBuffer(startsGpu, runFrom * 4, empty.buffer, runFrom * 4, rows * 4);
    }
    queue.writeBuffer(ticksGpu, runFrom * 4, seeds.buffer, runFrom * 4, rows * 4);
    runFrom = runTo;
  };

  const flush = () => {
    for (let k = 0; k < dirtyCount; k++) {
      const slot = dirty[k];
      marked[slot] = 0;
      const bits = (flying[slot] | mail[slot]) >>> 0;
      const state = starts[slot];
      if (state >= 0) {
        if (slot !== runTo) {
          upload();
          runFrom = slot;
        }
        runTo = slot + 1;
        started[slot * 2] = state;
        started[slot * 2 + 1] = 0;
        posted[slot] = (bits | START) >>> 0;
        seeds[slot] = Math.floor(random() * (TICKS + 1));
        starts[slot] = -1;
      } else {
        posted[slot] = bits;
        queue.writeBuffer(mailGpu, slot * 4, posted.buffer, slot * 4, 4);
      }
      flying[slot] = posted[slot];
      mail[slot] = 0;
      if (flownMarked[slot] === 0) {
        flownMarked[slot] = 1;
        flown[flownCount++] = slot;
      }
    }
    upload();
    dirtyCount = 0;
  };

  const make = (fills: Inject): Machines<N, K, E> => {
    const built = memo(() =>
      root
        .pipe(fills)
        .pipe(stepping)
        .createComputePipeline({ compute: listing ?? kernel })
        .$name(`${name}_stepping`),
    );

    return {
      definition: def,
      capacity,
      dt,
      listed,
      gpu,
      inject: () => inject,
      pipe: (fill) => make((cfg) => fill(fills(cfg))),
      start(slot, state = 0 as State<N>) {
        if (!(Number.isInteger(state) && state >= 0 && state < parts.states.length)) {
          throw new RangeError(`'${name}' has no state ${state}.`);
        }
        mark(slot);
        starts[slot] = state;
      },
      send,
      flush,
      step(pass) {
        if (dirtyCount > 0) flush();
        pass.setPipeline(built());
        pass.dispatchWorkgroups(Math.ceil((span ? span() : capacity) / WORKGROUP));
        listed?.finish(pass);
        for (let k = 0; k < flownCount; k++) {
          const slot = flown[k];
          flownMarked[slot] = 0;
          flying[slot] = 0;
        }
        flownCount = 0;
      },
      initAsync: () => pipelines.compile(listed ? [built(), listed] : [built()]),
      rows: rowsBuffer,
      destroy,
      [Symbol.dispose]: destroy,
    };
  };

  return make((cfg) => cfg);
}
