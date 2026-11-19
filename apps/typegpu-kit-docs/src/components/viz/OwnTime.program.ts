import { createRecorder, type GpuCommand } from '@typegpu/kit/debug/recording';

export interface Pass {
  readonly name: string;
  /** Its first timestamp, in ms from the frame's start. */
  readonly start: number;
  /** Its last. */
  readonly end: number;
}

const ends: [string, number][] = [
  ['shadow', 0.12],
  ['scene', 1.05],
  ['down 1', 1.14],
  ['down 2', 1.2],
  ['down 3', 1.24],
  ['down 4', 1.27],
  ['down 5', 1.29],
  ['down 6', 1.31],
  ['up 5', 1.33],
  ['up 4', 1.36],
  ['up 3', 1.41],
  ['up 2', 1.49],
  ['up 1', 1.61],
  ['output', 2.03],
];

export const passes: readonly Pass[] = ends.map(([name, end], i) => ({
  name,
  start: i === 0 ? 0 : i === 1 ? 0.01 : 0.02 + ((i - 2) * 0.6) / 66,
  end,
}));

export interface Program {
  readonly passes: readonly Pass[];
  /** What the recorder charges each pass, in ms, in the order of `passes`. */
  readonly own: readonly number[];
}

/** Feeds the recorder the timestamps the instrument would report for `passes`, and reads its bar. */
export function createProgram(): Program {
  const recorder = createRecorder();
  const commands: GpuCommand[] = passes.map(({ name }) => ({
    kind: 'render',
    name,
    label: name,
    group: undefined,
    groups: [],
    pass: undefined,
    split: false,
    pipelines: [],
    wgsl: [],
    calls: 1,
    uses: [],
    frame: 0,
    timestamps: 'written',
  }));

  recorder.frameStarted(0);
  recorder.submitted(commands);
  commands.forEach((command, i) => {
    command.start = BigInt(Math.round(passes[i].start * 1e6));
    command.end = BigInt(Math.round(passes[i].end * 1e6));
  });
  recorder.frameStarted(100);
  recorder.timed(commands);

  const segments = recorder.bars()[0].segments.gpu;
  const own = passes.map(({ name }) => {
    const segment = segments.find((each) => each.name === name);
    if (!segment) throw new Error(`The recorder charged nothing to ${name}.`);
    return segment.ms;
  });
  return { passes, own };
}
