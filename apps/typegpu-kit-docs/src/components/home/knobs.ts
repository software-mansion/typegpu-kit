/** The numbers the reader drags in the page's code: their range, step and first value. */

export interface Knobs {
  count: number;
  sight: number;
  elevation: number;
  range: number;
  intensity: number;
  length: number;
  threshold: number;
  strength: number;
  cell: number;
  levels: number;
  toward: number;
  radius: number;
  crowd: number;
}

export interface Knob {
  readonly min: number;
  readonly max: number;
  readonly step: number;
  readonly value: number;
}

export const KNOBS: { readonly [K in keyof Knobs]: Knob } = {
  count: { min: 64, max: 1024, step: 32, value: 448 },
  sight: { min: 0.2, max: 1.4, step: 0.1, value: 1.4 },
  elevation: { min: 8, max: 80, step: 2, value: 26 },
  range: { min: 0.3, max: 1.4, step: 0.1, value: 1 },
  intensity: { min: 0, max: 4, step: 0.1, value: 0.8 },
  length: { min: 0.2, max: 0.9, step: 0.05, value: 0.5 },
  threshold: { min: 0, max: 3, step: 0.1, value: 1.2 },
  strength: { min: 0, max: 2, step: 0.05, value: 0.35 },
  cell: { min: 1, max: 8, step: 1, value: 1 },
  levels: { min: 2, max: 32, step: 1, value: 32 },
  toward: { min: 0, max: 4, step: 0.1, value: 1 },
  radius: { min: 1, max: 6.5, step: 0.5, value: 4 },
  crowd: { min: 1, max: 12, step: 1, value: 6 },
};

/** Each knob's first value, by name, as the frame's code reads it. */
export const knob = Object.fromEntries(
  Object.entries(KNOBS).map(([key, each]) => [key, each.value]),
) as unknown as Readonly<Knobs>;

const decimals = (step: number) => (String(step).split('.')[1] ?? '').length;

/** The knob's value as the code shows it. */
export const shown = (key: keyof Knobs, value: number) => value.toFixed(decimals(KNOBS[key].step));
