// The trips the readback and counters figures share: a value copied on the GPU, carried in a
// staging buffer while its map takes, and landing on the CPU. Also their programs' delayed maps
import { action, animation, fail } from './figure.ts';
import { badge, CHAR, stipple, svg, text } from './svg.ts';

/** Frames the figure shows at most. */
const COLUMNS = 24;

/** Frames a program keeps, more than it shows, so trips copied before the first column still draw. */
export const KEPT = COLUMNS + 8;

/** The programs' frame interval, in seconds. */
const FRAME_TIME = 1 / 60;

/** Pixels left of the first column, for the lanes' names. */
export const LEFT = 52;

/** The widest a column is drawn, in pixels. */
const CELL = 30;

interface DelayedMaps {
  /** Per staging buffer, the frame its map lands in, or -1. */
  readonly busy: number[];
  /** Lands the maps due in the next frame, lets what awaited them run, and runs that frame of `loop`. */
  step(loop: { run(seconds: number): unknown }): Promise<void>;
}

/**
 * Patches `device` so each staging buffer's map lands `latency` frames after it starts, calling
 * `mapped` with the buffer's slot and that frame as it starts. `slots` buffers are free at first.
 */
export function delayedMaps(
  device: GPUDevice,
  latency: number,
  mapped: (slot: number, until: number) => void,
  slots = 0,
): DelayedMaps {
  const maps: { resolve: () => void; at: number }[] = [];
  const busy = Array.from({ length: slots }, () => -1);
  let created = 0;
  let now = 0;

  const createBuffer = device.createBuffer.bind(device);
  device.createBuffer = (descriptor) => {
    const buffer = createBuffer(descriptor);
    if (!(descriptor.usage & GPUBufferUsage.MAP_READ)) return buffer;

    const slot = created++;
    if (slot >= busy.length) busy.push(-1);
    const mapAsync = buffer.mapAsync.bind(buffer);
    buffer.mapAsync = (...args) => {
      const at = now + latency;
      busy[slot] = at;
      mapped(slot, at);
      void mapAsync(...args);
      return new Promise<undefined>((resolve) =>
        maps.push({ resolve: () => resolve(undefined), at }),
      );
    };
    return buffer;
  };

  /** Lands the maps due in frame `index` and lets what awaited them run. */
  const settle = async (index: number) => {
    now = index;
    for (const map of maps.filter((pending) => pending.at === index)) map.resolve();
    await Promise.resolve();
    await Promise.resolve();
  };
  let index = 0;

  return {
    busy,

    async step(loop) {
      await settle(index);
      loop.run(index * FRAME_TIME);
      index += 1;
    },
  };
}

interface Wants {
  /** Steps the cadence in a frame with a free staging buffer, as `capture` does, and says if a copy is due. */
  free(index: number): boolean;
  /** Says if the rate's next tick has come by a frame whose staging buffers are all mapping. */
  full(index: number): boolean;
}

/**
 * When a readback's rate wants a copy, in whole frames. Mirrors `capture` in the kit's
 * `readback.ts`, which asks its cadence only in a frame with a free staging buffer, passing the time
 * since the last such frame, and `due` in `clock.ts`, which runs in the frame closest to the next
 * tick and restarts the schedule after a stall. Without `hz`, nothing is due.
 */
export function cadenceOf(hz: number | undefined): Wants {
  // Frames are FRAME_TIME apart, so a tick of hz is a whole number of frames for 60, 30 and 10
  const interval = hz ? 1 / (hz * FRAME_TIME) : 0;
  let next = 0;
  let counted = 0;
  let first = true;

  return {
    free(index) {
      if (!hz) return false;
      const run = first || index > next - (index - counted) / 2;
      counted = index;
      if (!run) return false;

      const onSchedule = !first && Math.abs(index - next) < interval / 2;
      next = onSchedule ? next + interval : index + interval;
      first = false;
      return true;
    },

    full: (index) => !!hz && (first || index > next - 1 / 2),
  };
}

/** The last columns that fit in `width`, at least 10, and where each frame's column is. */
export function columnsIn<C extends { readonly index: number }>(all: readonly C[], width: number) {
  const count = Math.max(10, Math.min(COLUMNS, Math.floor((width - LEFT) / CELL)));
  const cell = (width - LEFT) / count;
  const columns = all.slice(-count);
  const first = columns[0]?.index ?? 0;
  return {
    columns,
    cell,
    /** The right edge of the last column. */
    end: LEFT + count * cell,
    /** The centre of frame `index`'s column, left of LEFT for a frame before the first. */
    x: (index: number) => LEFT + (index - first) * cell + cell / 2,
  };
}

type View = ReturnType<typeof columnsIn>;

/** Heights in the picture: the GPU lane, a bar per staging buffer, the CPU lane and a row below. */
interface Lanes {
  readonly gpu: number;
  /** The top of staging buffer `slot`'s bars. */
  readonly slot: (slot: number) => number;
  readonly cpu: number;
  readonly below: number;
  readonly height: number;
}

const BAR = 16;
/** Pixels a bar starts after the frame that copies and ends before the frame it lands in. */
const GAP = 1.5;

const slot = (s: number) => 60 + s * (BAR + 6);

export function lanes(slots: number): Lanes {
  const cpu = slot(slots) + 16;
  return { gpu: 42, slot, cpu, below: cpu + 48, height: cpu + 58 };
}

/** Draws the frame badges, a faint rule under each lane and the lanes' names. */
export function frameAxis(
  figure: SVGSVGElement,
  view: View,
  at: Lanes,
  slots: number,
  below: string,
) {
  const { columns, x, end } = view;
  const now = columns.at(-1);
  for (const column of columns) {
    // An odd newest frame hides the even one beside it
    const crowded = now && now.index % 2 === 1 && column.index === now.index - 1;
    if ((column.index % 2 === 0 && !crowded) || column === now) {
      badge(figure, x(column.index), 0, column.index, column === now);
    }
  }

  const rule = (y: number) =>
    svg('line', { x1: LEFT, x2: end, y1: y, y2: y, stroke: 'var(--viz-line)' }, figure);
  rule(at.gpu);
  rule(at.cpu);
  for (let s = 0; s < slots; s++) {
    svg(
      'rect',
      {
        x: LEFT,
        y: at.slot(s),
        width: end - LEFT,
        height: BAR,
        fill: stipple(figure, 'var(--viz-line)', 0.12),
      },
      figure,
    );
  }

  text(figure, 0, at.gpu + 4, 'GPU', { class: 'strong', style: 'fill: var(--viz-b)' });
  for (let s = 0; s < slots; s++) text(figure, 0, at.slot(s) + BAR / 2 + 4, `slot ${s + 1}`);
  text(figure, 0, at.cpu + 4, 'CPU', { class: 'strong', style: 'fill: var(--viz-c)' });
  text(figure, 0, at.below + 4, below, { class: 'strong' });
}

export interface Trip {
  /** The frame that copied. */
  readonly from: number;
  readonly slot: number;
  /** The frame its map lands in. */
  readonly until: number;
  /** What it carries, shown at the bar's start where it fits. */
  readonly label?: string | undefined;
  /** `answer` carries a waiting read's value; `dropped` lands after a reset and is thrown away. */
  readonly look?: 'answer' | 'dropped' | undefined;
}

const COLOUR = { copy: 'var(--viz-d)', answer: 'var(--viz-accent)', dropped: 'var(--viz-bad)' };

/**
 * Draws each trip: a dot where the GPU copies, a drop into its staging buffer, a bar for as long as
 * its map takes and a drop onto the CPU lane where it lands. Clipped to the columns shown.
 */
export function drawTrips(figure: SVGSVGElement, view: View, at: Lanes, trips: readonly Trip[]) {
  const { x, end } = view;
  const shown = trips.filter((trip) => x(trip.until) > LEFT && x(trip.from) < end);

  // Drops first, so a bar of another buffer hides the part of a drop behind it. A bar keeps clear
  // of the next one in its buffer, which may start in the frame it lands
  const edges = (trip: Trip) => [x(trip.from) + GAP, x(trip.until) - GAP] as const;
  for (const trip of shown) {
    const colour = COLOUR[trip.look ?? 'copy'];
    const top = at.slot(trip.slot);
    const [x0, x1] = edges(trip);
    if (x0 >= LEFT) {
      svg('line', { x1: x0, x2: x0, y1: at.gpu, y2: top, stroke: colour }, figure);
    }
    if (x1 <= end && trip.look !== 'dropped') {
      svg('line', { x1, x2: x1, y1: top + BAR, y2: at.cpu - 5, stroke: colour }, figure);
      svg('path', { d: `M${x1 - 3.5},${at.cpu - 6}h7l-3.5,6z`, fill: colour }, figure);
    }
  }

  for (const trip of shown) {
    const colour = COLOUR[trip.look ?? 'copy'];
    const top = at.slot(trip.slot);
    const [from, until] = edges(trip);
    const x0 = Math.max(from, LEFT);
    const x1 = Math.min(until, end);
    const box = { x: x0, y: top, width: x1 - x0, height: BAR };
    svg('rect', { ...box, fill: 'var(--ground)' }, figure);
    if (trip.look === 'dropped') {
      svg(
        'rect',
        { ...box, rx: 2, fill: 'none', stroke: colour, 'stroke-dasharray': '3 2' },
        figure,
      );
    } else {
      svg('rect', { ...box, rx: 2, fill: stipple(figure, colour, 0.25), stroke: colour }, figure);
    }
    if (from >= LEFT) {
      svg('circle', { cx: from, cy: at.gpu, r: 3.5, fill: colour }, figure);
      const { label } = trip;
      const room = label ? label.length * CHAR + 6 : 0;
      if (label && room + 4 <= x1 - x0) {
        svg(
          'rect',
          { x: x0 + 3, y: top + 2, width: room, height: BAR - 4, fill: 'var(--ground)' },
          figure,
        );
        text(figure, x0 + 6, top + BAR / 2 + 4, label, { class: 'strong' });
      }
    }
  }
}

/** A coral cross on the GPU lane: a capture with every staging buffer still mapping. */
export function skipped(figure: SVGSVGElement, x: number, at: Lanes) {
  const r = 3.5;
  svg(
    'path',
    {
      d: `M${x - r},${at.gpu - r}l${2 * r},${2 * r}m0,${-2 * r}l${-2 * r},${2 * r}`,
      stroke: 'var(--viz-bad)',
      'stroke-width': 1.5,
    },
    figure,
  );
}

/**
 * Draws a value held across the columns on the line at `y`: a line in `colour` where it is held,
 * and the value where it changes, or at the first column, above or below the line.
 */
export function heldValue(
  figure: SVGSVGElement,
  view: View,
  y: number,
  values: readonly (string | undefined)[],
  colour: string,
  side: 'above' | 'below',
) {
  const { columns, x, cell } = view;
  const ty = side === 'above' ? y - 7 : y + 16;
  values.forEach((value, k) => {
    if (value === undefined) return;
    const fx = x(columns[k].index);
    svg(
      'line',
      { x1: fx - cell / 2, x2: fx + cell / 2, y1: y, y2: y, stroke: colour, 'stroke-width': 2 },
      figure,
    );
    if (k === 0 || values[k - 1] !== value) {
      text(figure, fx, ty, value, { 'text-anchor': 'middle', class: 'strong' });
    }
  });
}

/** A count in at most four characters: 1200 is 1.2k. */
export const short = (n: number) =>
  n < 1000 ? String(n) : `${Number((n / 1000).toFixed(n < 10_000 ? 1 : 0))}k`;

interface Stepped {
  /** Runs one frame. */
  step(): Promise<void>;
  destroy(): void;
}

interface SteppedRun<P extends Stepped> {
  /** The program drawn, once it has filled the figure. */
  readonly program: P | undefined;
  /** Starts over with a new program, shown once it has run KEPT frames. */
  reset(): void;
  /** Applies `act` to the program, and runs a frame at once while paused. */
  act(act: (program: P) => void): void;
}

/**
 * Steps a timeline's program every 0.3 s with the figure's play button, never two steps at once,
 * and redraws after each. `before` runs before each frame, with the frames the program ran so far.
 */
export function steppedRun<P extends Stepped>(
  root: HTMLElement,
  options: {
    readonly create: () => P;
    readonly before?: (program: P, frame: number) => void;
    readonly draw: () => void;
  },
): SteppedRun<P> {
  let program: P | undefined;
  let generation = 0;
  let stepping = false;
  const ran = new WeakMap<P, number>();

  const step = (stepped: P) => {
    const frame = ran.get(stepped) ?? 0;
    ran.set(stepped, frame + 1);
    options.before?.(stepped, frame);
    return stepped.step();
  };

  const stop = (error: unknown) => {
    timeline.destroy();
    fail(root, error);
  };

  const timeline = animation(root, {
    every: 0.3,
    step() {
      const current = program;
      if (!current || stepping) return;

      stepping = true;
      step(current).then(() => {
        stepping = false;
        if (current === program) timeline.redraw();
      }, stop);
    },
    draw: () => options.draw(),
    play: action(root, 'play'),
    destroy: () => program?.destroy(),
  });

  const reset = async () => {
    const mine = ++generation;
    const next = options.create();

    try {
      for (let i = 0; i < KEPT; i++) await step(next);
    } catch (error) {
      next.destroy();
      stop(error);
      return;
    }

    if (mine !== generation) return next.destroy();
    program?.destroy();
    program = next;
    timeline.redraw();
  };

  return {
    get program() {
      return program;
    },

    reset: () => void reset(),

    act(act) {
      if (!program) return;
      act(program);
      if (!timeline.playing) void step(program).then(() => timeline.redraw(), stop);
    },
  };
}
