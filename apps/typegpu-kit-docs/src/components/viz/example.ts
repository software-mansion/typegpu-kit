import type { Surface } from '@typegpu/kit';
import type { TgpuRoot } from 'typegpu';
import { exampleOf } from 'typegpu-kit-examples';
import type { ExampleScene, Loading } from 'typegpu-kit-examples/contract';
import { create, WARMUP_FRAMES } from 'typegpu-kit-examples/hosts';
import { loading as pageLoading } from '../../examples/files.ts';
import { connect, fitRing } from '../../examples/host.ts';
import type {
  Choice,
  Knob,
  Range,
  Settings,
  Value,
  Values,
} from 'typegpu-kit-examples/settings.ts';
import type { Camera } from 'typegpu-kit-examples/ui.ts';
import { film } from '../../examples/camera.ts';
import { choice, fail, live, note, slider, toggle, uniqueName } from './figure.ts';
import { posterOf } from '../site/posters.ts';
import { veil } from '../site/veil.ts';
import { pageGpu } from './gpu.ts';

/** What ExampleFigure.astro hands its script, in the figure's `data-example`. */
export interface Embed {
  readonly key: string;
  /** Scene options overriding the example card options. */
  readonly options: Record<string, unknown>;
  readonly controls: readonly string[];
  /** Settings the scene starts from, by name. */
  readonly values: Values;
}

/** Merges figure options over the example card options. */
export const optionsOf = (
  card: Record<string, unknown> = {},
  options: Record<string, unknown> = {},
): Record<string, unknown> => ({ ...card, ...options });

interface Shown {
  readonly name: string;
  readonly label: string;
  /** Why it can't change, such as a feature the device lacks. */
  readonly disabled?: string;
  /** The scene changes it too, so the control follows it. */
  readonly live: boolean;
}

/** A control a figure shows for one of a scene's settings or actions. */
export type Control =
  | (Shown & {
      readonly kind: 'slider';
      readonly min: number;
      readonly max: number;
      readonly step: number;
    })
  | (Shown & { readonly kind: 'choice'; readonly options: readonly (string | number)[] })
  | (Shown & { readonly kind: 'toggle' })
  | { readonly kind: 'action'; readonly name: string; readonly label: string }
  | { readonly kind: 'readout'; readonly name: string; readonly label: string };

/** What `settings` has to show, for an error that names what a figure asked for. */
const offered = (settings: Settings | undefined) => {
  const names = [
    ...Object.keys(settings?.knobs ?? {}),
    ...Object.keys(settings?.actions ?? {}),
    ...Object.keys(settings?.readouts ?? {}),
  ];
  return names.length > 0 ? `it has ${names.join(', ')}` : 'it has none';
};

/** Builds controls in name order using sliders, choices, toggles, action buttons and readouts. Throws if a name is absent from the scene settings, actions and readouts. */
export function controlsOf(
  key: string,
  settings: Settings | undefined,
  names: readonly string[],
): Control[] {
  return names.map((name): Control => {
    const knob = settings?.knobs[name] as Knob | undefined;
    const value = settings?.values[name];

    if (!knob || value === undefined) {
      if (settings?.actions?.[name]) return { kind: 'action', name, label: name };
      if (settings?.readouts?.[name]) return { kind: 'readout', name, label: name };
      throw new Error(`${key} has no setting, action or readout ${name}; ${offered(settings)}.`);
    }

    const shown: Shown = {
      name,
      label: knob.label ?? name,
      live: knob.live ?? false,
      ...(knob.disabled ? { disabled: knob.disabled } : {}),
    };
    if ('options' in knob) {
      return { ...shown, kind: 'choice', options: (knob as Choice<string | number>).options };
    }
    if (typeof value === 'number') {
      const { min, max, step } = knob as Range;
      return { ...shown, kind: 'slider', min, max, step };
    }
    return { ...shown, kind: 'toggle' };
  });
}

/** Applies and writes initial settings before the first frame, leaving a disabled knob's as it is. Throws for missing names, type mismatches or values outside knob limits. */
export async function start(key: string, settings: Settings | undefined, values: Values) {
  for (const [name, value] of Object.entries(values)) {
    const current = settings?.values[name];
    if (!settings || current === undefined) {
      throw new Error(`${key} has no setting ${name}; ${offered(settings)}.`);
    }
    if (typeof value !== typeof current) {
      throw new Error(`${key}'s ${name} is a ${typeof current}, not ${JSON.stringify(value)}.`);
    }

    const knob = settings.knobs[name] as Knob | undefined;
    const options = knob && 'options' in knob ? (knob as Choice<Value>).options : undefined;
    const range = knob as Partial<Range> | undefined;
    if (options && !options.includes(value)) {
      throw new Error(`${key}'s ${name} is one of ${options.join(', ')}, not ${value}.`);
    }
    if (typeof value === 'number' && range?.min !== undefined && range.max !== undefined) {
      if (value < range.min || value > range.max) {
        throw new Error(`${key}'s ${name} goes from ${range.min} to ${range.max}, not ${value}.`);
      }
    }

    if (knob?.disabled) continue;
    settings.values[name] = value;
    await settings.write?.(name);
  }
}

export interface Built {
  readonly scene: ExampleScene;
  readonly controls: readonly Control[];
}

/** Creates a scene with loaded assets, figure options, initial values and controls. Destroys it if a control or value is invalid. */
export async function build(
  embed: Embed,
  root: TgpuRoot,
  surface: Surface,
  loading: Loading,
): Promise<Built> {
  const scene = await create(embed.key, root, surface, loading, embed.options);
  try {
    const controls = controlsOf(embed.key, scene.settings, embed.controls);
    await start(embed.key, scene.settings, embed.values);
    return { scene, controls };
  } catch (error) {
    scene.destroy?.();
    throw error;
  }
}

/** Decimals a value on a slider of `step` shows. */
const decimalsOf = (step: number) => (String(step).split('.')[1] ?? '').length;

const element = <K extends keyof HTMLElementTagNameMap>(
  tag: K,
  attributes: Record<string, string> = {},
  ...children: (Node | string)[]
) => {
  const made = document.createElement(tag);
  for (const [name, value] of Object.entries(attributes)) made.setAttribute(name, value);
  made.append(...children);
  return made;
};

const labelOf = (control: Control) =>
  'disabled' in control && control.disabled
    ? `${control.label} (${control.disabled})`
    : control.label;

/** Creates a Slider, Choice, button or readout element initialized to its value. */
function render(control: Control, value: Value | undefined): HTMLElement {
  const disabled = 'disabled' in control && control.disabled ? { disabled: '' } : {};

  switch (control.kind) {
    case 'slider': {
      const { name, min, max, step } = control;
      const input = element('input', {
        type: 'range',
        min: String(min),
        max: String(max),
        step: String(step),
        value: String(value),
        autocomplete: 'off',
        'data-control': name,
        ...disabled,
      });
      return element(
        'label',
        { class: 'viz-slider' },
        `${labelOf(control)} `,
        element('span', { class: 'viz-range' }, input, element('output')),
      );
    }

    case 'choice': {
      const group = uniqueName(control.name);
      const options = control.options.map((option) => {
        const input = element('input', {
          type: 'radio',
          name: group,
          value: String(option),
          autocomplete: 'off',
          ...disabled,
        });
        input.checked = option === value;
        return element('label', {}, input, element('span', {}, String(option)));
      });
      return element(
        'div',
        {
          class: 'viz-choice',
          role: 'radiogroup',
          'aria-label': labelOf(control),
          'data-choice': control.name,
        },
        element('span', { 'aria-hidden': 'true' }, labelOf(control)),
        ...options,
      );
    }

    case 'toggle':
      return element(
        'button',
        { type: 'button', 'aria-pressed': String(value === true), ...disabled },
        labelOf(control),
      );

    case 'action':
      return element('button', { type: 'button' }, control.label);

    case 'readout':
      return element('span', { class: 'viz-readout' }, `${control.label} `, element('output'));
  }
}

/**
 * Sets a value of the settings `settingsOf` gives, those of the scene the figure holds, and writes
 * it, then calls `changed`. A value set while its write runs goes once that write has finished. A
 * scene the figure let go gets no further write, and its failure is dropped; others call
 * `onError`.
 */
export function writerFor(
  settingsOf: () => Settings | undefined,
  changed: () => void,
  onError: (error: unknown) => void,
): (name: string, value: Value) => void {
  /** The settings each setting's running write goes to. */
  const writing = new Map<string, Settings>();

  const send = async (settings: Settings, name: string) => {
    const sent = settings.values[name];
    try {
      await settings.write?.(name);
    } finally {
      if (writing.get(name) === settings) writing.delete(name);
    }
    const latest = settings.values[name];
    if (settingsOf() === settings && latest !== sent) write(name, latest);
    else changed();
  };

  const write = (name: string, value: Value) => {
    const settings = settingsOf();
    if (!settings) return;
    settings.values[name] = value;
    if (writing.get(name) === settings) return;
    writing.set(name, settings);
    send(settings, name).catch((error: unknown) => {
      if (settingsOf() === settings) onError(error);
    });
  };
  return write;
}

/**
 * Inserts controls after `after`, showing the values of `first`, the first scene's settings. A
 * change writes the settings `settingsOf` gives, those of the scene the figure holds, and calls
 * `changed`; failures call `onError`. Returns updaters for readouts and settings that the scene
 * can change.
 */
function controlsFor(
  figure: HTMLElement,
  after: HTMLElement,
  controls: readonly Control[],
  first: Settings | undefined,
  settingsOf: () => Settings | undefined,
  changed: () => void,
  onError: (error: unknown) => void,
): (() => void)[] {
  const write = writerFor(settingsOf, changed, onError);
  const valueOf = (name: string) => settingsOf()?.values[name];

  const shown = controls.map((control) => render(control, first?.values[control.name]));
  after.after(...shown);

  return controls.flatMap((control, i) => {
    const made = shown[i];
    switch (control.kind) {
      case 'slider': {
        const decimals = decimalsOf(control.step);
        const format = (value: number) => value.toFixed(decimals);
        const input = slider(figure, control.name, (value) => write(control.name, value), format);
        if (!control.live) return [];
        const output = made.querySelector('output')!;
        return [
          () => {
            const value = valueOf(control.name) as number;
            if (input.valueAsNumber === value || document.activeElement === input) return;
            input.valueAsNumber = value;
            output.textContent = format(value);
          },
        ];
      }

      case 'choice': {
        // Radio inputs hold text, so a numeric choice writes the option the text names
        const optionOf = (text: string) =>
          control.options.find((option) => String(option) === text) ?? text;
        choice<string>(figure, control.name, (text) => write(control.name, optionOf(text)));
        if (!control.live) return [];
        const inputs = [...made.querySelectorAll('input')];
        return [
          () => {
            const current = String(valueOf(control.name));
            for (const input of inputs) input.checked = input.value === current;
          },
        ];
      }

      case 'toggle': {
        const state = toggle(made as HTMLButtonElement, (on) => write(control.name, on));
        return control.live ? [() => (state.on = valueOf(control.name) === true)] : [];
      }

      case 'action':
        made.addEventListener('click', () => {
          settingsOf()?.actions?.[control.name]?.();
          changed();
        });
        return [];

      case 'readout': {
        const output = made.querySelector('output')!;
        const show = () => {
          const read = settingsOf()?.readouts?.[control.name];
          if (!read) return;
          const text = String(read());
          if (output.textContent !== text) output.textContent = text;
        };
        show();
        return [show];
      }
    }
  });
}

/**
 * Inserts a button after `after` for each camera the example declares. Returns what asks for each
 * at once for a scene, until `signal` stops them.
 */
function camerasFor(key: string, after: HTMLElement, onError: (error: unknown) => void) {
  const parts = Object.entries(exampleOf(key).ui ?? {}).flatMap(([name, part]) =>
    part.kind === 'camera' ? [{ name, part }] : [],
  );
  const toggles = new Map<string, () => void>();
  const buttons = parts.map(({ name, part }) => {
    const button = element('button', { type: 'button' }, part.label);
    button.addEventListener('click', () => toggles.get(name)?.());
    return button;
  });
  after.after(...buttons);

  return (scene: ExampleScene, signal: AbortSignal) => {
    const bound = (scene.ui ?? {}) as Readonly<Record<string, unknown>>;
    signal.addEventListener('abort', () => toggles.clear());
    parts.forEach(({ name, part }, i) => {
      const label = (text: string) => (buttons[i].textContent = text);
      toggles.set(name, film(part, bound[name] as Camera, label, onError, signal).toggle);
    });
  };
}

/**
 * Runs an ExampleFigure scene behind a cover until ready and warmed for `WARMUP_FRAMES`. Then
 * connects controls, pointer input and playback. Pauses off screen and starts paused under reduced
 * motion. A scene built again, after the figure let it go, starts from the values the reader set.
 */
export async function mount(figure: HTMLElement) {
  const embed = JSON.parse(figure.dataset.example ?? '{}') as Embed;
  const canvas = figure.querySelector('canvas');
  const cover = figure.querySelector<HTMLElement>('[data-cover]');
  const play = figure.querySelector<HTMLButtonElement>('[data-action=play]');
  const ring = figure.querySelector<HTMLElement>('[data-reach]');
  if (!canvas || !cover || !play)
    throw new Error('The example figure has no canvas, cover or play.');

  const curtain = veil(canvas.parentElement ?? figure, {
    owner: figure,
    poster: posterOf(embed.key),
  });

  // Without WebGPU the poster stays, above the note saying why
  const gpu = await pageGpu();
  if (!gpu.root) {
    note(figure, gpu.why);
    cover.textContent = '';
    curtain.rest();
    return;
  }

  const loading = pageLoading((text) => (cover.textContent = text));
  let drawn = 0;
  let values = embed.values;
  let held: Built | undefined;
  let follow: (() => void)[] | undefined;
  let filmFor: ReturnType<typeof camerasFor> | undefined;

  const stop = (error: unknown) => {
    shown.destroy();
    canvas.hidden = true;
    void curtain.lift();
    fail(figure, error);
  };
  const again = () => shown.redraw();

  const shown = live(
    figure,
    async () => ({
      createScene: async (root: TgpuRoot, surface: Surface) => {
        const built = await build({ ...embed, values }, root, surface, loading);
        return {
          built,
          frame: built.scene.frame,
          ...(built.scene.ready ? { ready: built.scene.ready } : {}),
          destroy: () => built.scene.destroy?.(),
        };
      },
    }),
    undefined,
    {
      animate: true,
      play,
      surface: { alphaMode: 'opaque' },
      onFrame(_frame, { scene: { built }, ready }) {
        if (built.scene.settling) again();
        if (drawn < WARMUP_FRAMES) {
          // Frames under the cover, playing or not, until it comes off
          if (ready && ++drawn === WARMUP_FRAMES) {
            figure.classList.add('live');
          } else again();
          return;
        }
        for (const each of follow ?? []) each();
      },

      ready({ built }) {
        held = built;
        const { scene, controls } = built;
        if (!follow) {
          follow = controlsFor(
            figure,
            play,
            controls,
            scene.settings,
            () => held?.scene.settings,
            again,
            stop,
          );
          filmFor = camerasFor(embed.key, play, stop);
          const { pointing } = scene;
          if (ring && pointing?.reach !== undefined) {
            follow.push(() => fitRing(ring, held?.scene.pointing?.reach ?? 0));
          }
        }
        const filming = new AbortController();
        let listening: Disposable;
        try {
          filmFor?.(scene, filming.signal);
          listening = connect(canvas, scene, { page: true, ring, redraw: again });
        } catch (error) {
          filming.abort();
          throw error;
        }

        return {
          [Symbol.dispose]() {
            // What the reader set, for the next scene to start from
            const set = controls.filter(
              (control) => control.kind !== 'action' && control.kind !== 'readout',
            );
            const next: Record<string, Value> = { ...values };
            for (const { name } of set) {
              const value = scene.settings?.values[name];
              if (value !== undefined) next[name] = value;
            }
            values = next;
            held = undefined;
            drawn = 0;
            figure.classList.remove('live');
            filming.abort();
            listening[Symbol.dispose]();
          },
        };
      },
    },
  );
}
