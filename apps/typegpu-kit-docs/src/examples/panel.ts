import GUI, { type Controller } from 'lil-gui';
import type { Knob, Range, Settings, Values } from 'typegpu-kit-examples/settings.ts';

/**
 * Builds lil-gui controls that write settings when changed. Orders ungrouped knobs first, then
 * actions, readouts and folders. Without settings, the panel starts empty.
 *
 * @param onError Receives failed writes, including rebuild failures.
 */
export function panel<T extends Values>(
  settings: Settings<T> | undefined,
  title: string,
  onError: (error: unknown) => void,
): GUI {
  const gui = new GUI({ title });
  // Start closed in docs frames and on phones so the panel does not cover the figure
  if (window.self !== window.top || matchMedia('(max-width: 560px)').matches) gui.close();
  if (!settings) return gui;
  const { values, knobs } = settings;
  const folders = new Map<string, GUI>();
  const keys = Object.keys(knobs) as (keyof T & string)[];

  const add = (parent: GUI, key: keyof T & string, knob: Knob) => {
    const value = values[key];
    let control: Controller;
    if ('options' in knob) {
      control = parent.add(values, key, [...(knob.options as readonly (string | number)[])]);
    } else if (typeof value === 'number') {
      const { min, max, step } = knob as Range;
      control = parent.add(values, key, min, max, step);
    } else {
      control = parent.add(values, key);
    }

    const label = knob.label ?? key;
    control.name(knob.disabled ? `${label} (${knob.disabled})` : label);
    if (knob.disabled) control.disable();
    if (knob.live) control.listen();

    control.onChange(() => {
      try {
        void Promise.resolve(settings.write?.(key)).catch(onError);
      } catch (error) {
        onError(error);
      }
    });
  };

  for (const key of keys) {
    const knob = knobs[key];
    if (!knob.folder) add(gui, key, knob);
  }

  for (const [name, act] of Object.entries(settings.actions ?? {})) gui.add({ [name]: act }, name);

  for (const [name, read] of Object.entries(settings.readouts ?? {})) {
    const shown = {
      get [name]() {
        return read();
      },
    };
    gui.add(shown, name).disable().listen();
  }

  for (const key of keys) {
    const knob = knobs[key];
    if (!knob.folder) continue;

    let folder = folders.get(knob.folder);
    if (!folder) folders.set(knob.folder, (folder = gui.addFolder(knob.folder)));
    add(folder, key, knob);
  }

  return gui;
}
