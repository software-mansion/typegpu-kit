// What a scene lets hosts change: its values, a knob per value that hosts build controls from,
// and panel actions and readouts

/** Numeric, boolean or named-choice setting. */
export type Value = number | boolean | string;

export type Values = Record<string, Value>;

interface Shown {
  /** Visible label. Defaults to the value name. */
  readonly label?: string;
  /** Control folder, such as `look`. */
  readonly folder?: string;
  /** Follows scene-side value changes, such as the hex isle clock. */
  readonly live?: boolean;
  /** Reason the control is disabled, such as a missing device feature. */
  readonly disabled?: string;
}

export interface Range {
  readonly min: number;
  readonly max: number;
  readonly step: number;
}

/** A choice among `options`, such as shadow map sizes or named views. */
export interface Choice<V> {
  readonly options: readonly V[];
}

/** Control for a numeric range or choice, a boolean checkbox or a named choice. */
export type Knob<V extends Value = Value> = Shown &
  ([V] extends [number] ? Range | Choice<V> : [V] extends [string] ? Choice<V> : unknown);

export type Knobs<T extends Values> = { readonly [K in keyof T]: Knob<T[K]> };

export interface Settings<T extends Values = Values> {
  /** Scene values. Change a field, then call `write`. */
  readonly values: T;
  /** One knob per value, in display order. */
  readonly knobs: Knobs<T>;
  /**
   * Uploads the current values. `key` names the value that changed, when one did. Resolves after
   * any rebuild, such as changing jump-flood `immediates`. Hosts call it once without a key after
   * `createScene` and before the first frame, then with the key of each value they change. Default
   * none, for a scene that reads its values each frame.
   */
  write?(key?: keyof T): void | Promise<void>;
  /** Panel buttons, such as fish-tank `feed`. */
  readonly actions?: Readonly<Record<string, () => void>>;
  /** Readouts sampled each frame, such as flow-field `settings writes`. */
  readonly readouts?: Readonly<Record<string, () => Value>>;
}
