import type { ExampleScene } from '../../../typegpu-kit-examples/src/contract.ts';
import type { Ui, Camera, Held, Tap } from '../../../typegpu-kit-examples/src/ui.ts';
import type { Choice, Range, Value } from '../../../typegpu-kit-examples/src/settings.ts';

export interface InteractionReport {
  completed: string[];
  skipped: { action: string; reason: string }[];
}

export async function interact(
  scene: ExampleScene,
  ui: Ui,
  run: (count: number) => Promise<void>,
): Promise<InteractionReport> {
  const report: InteractionReport = { completed: [], skipped: [] };
  const act = async (label: string, action: () => unknown) => {
    try {
      await action();
      await run(2);
      report.completed.push(label);
    } catch (error) {
      throw new Error(`Interaction ${label}: ${String(error)}`, { cause: error });
    }
  };
  const pointer = async (prefix: string) => {
    const pointing = scene.pointing;
    if (!pointing) return;
    for (const y of [0.2, 0.5, 0.8])
      for (const x of [0.2, 0.5, 0.8]) {
        if (pointing.move) await act(`${prefix}.move(${x},${y})`, () => pointing.move!(x, y));
        if (pointing.press) await act(`${prefix}.press(${x},${y})`, () => pointing.press!(x, y));
      }
    if (pointing.drag)
      for (let i = 0; i < 5; i++)
        await act(`${prefix}.drag[${i}]`, () => pointing.drag!(0.2 + i * 0.15, 0.5, i === 0));
    if (pointing.end) await act(`${prefix}.end`, () => pointing.end!());
    if (pointing.leave) await act(`${prefix}.leave`, () => pointing.leave!());
    if (pointing.zoom) {
      await act(`${prefix}.zoom.out`, () => pointing.zoom!(1.2));
      await act(`${prefix}.zoom.in`, () => pointing.zoom!(1 / 1.2));
    }
  };
  await pointer('pointer');
  for (const [name, element] of Object.entries(ui)) {
    const handler = scene.ui?.[name];
    if (!handler) throw new Error(`Missing UI handler ${name}`);
    if (element.kind === 'buttons') {
      const buttons = handler as Record<string, Tap | Held>;
      for (const key of Object.keys(element.buttons)) {
        const button = buttons[key]!;
        await act(`ui.${name}.${key}.press`, () => button.press());
        if ('release' in button) await act(`ui.${name}.${key}.release`, () => button.release());
      }
    } else if (element.kind === 'text') {
      report.completed.push(`ui.${name}: ${(handler as () => string)()}`);
    } else if (element.kind === 'image') {
      await act(`ui.${name}.image`, () =>
        (handler as (picture: object) => void)({
          width: 2,
          height: 2,
          pixels: new Uint8Array([
            255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255,
          ]),
        }),
      );
    } else {
      await act(`ui.${name}.detach`, () => (handler as Camera).attach(undefined));
      report.skipped.push({
        action: `ui.${name}.attach`,
        reason: 'requires a real external camera texture',
      });
    }
  }
  const settings = scene.settings;
  if (settings) {
    for (const [name, knob] of Object.entries(settings.knobs)) {
      if (knob.disabled) {
        report.skipped.push({ action: `setting.${name}`, reason: knob.disabled });
        continue;
      }
      const original = settings.values[name]!;
      const values: readonly Value[] =
        'options' in knob
          ? (knob as Choice<Value>).options
          : typeof original === 'boolean'
            ? [false, true]
            : typeof original === 'number'
              ? [
                  (knob as Range).min,
                  (knob as Range).max,
                  Math.min(
                    (knob as Range).max,
                    (knob as Range).min +
                      Math.round(
                        ((knob as Range).max - (knob as Range).min) / (2 * (knob as Range).step),
                      ) *
                        (knob as Range).step,
                  ),
                ]
              : [original];
      for (const value of new Set(values)) {
        await act(`setting.${name}=${value}`, async () => {
          settings.values[name] = value;
          await settings.write?.(name);
        });
        // Tool/mode settings may change which pointer callback reaches a kernel.
        if (typeof value !== 'number' || 'options' in knob)
          await pointer(`setting.${name}=${value}`);
      }
      await act(`setting.${name}.restore`, async () => {
        settings.values[name] = original;
        await settings.write?.(name);
      });
    }
    for (const [name, action] of Object.entries(settings.actions ?? {}))
      await act(`action.${name}`, action);
    for (const read of Object.values(settings.readouts ?? {})) read();
  }
  const goal = scene.view?.goal as Record<string, unknown> | undefined;
  if (goal) {
    for (const [field, delta] of [
      ['yaw', 0.25],
      ['pitch', 0.1],
      ['distance', 0.25],
    ] as const) {
      const original = goal[field];
      if (typeof original !== 'number') continue;
      await act(`view.${field}`, () => {
        goal[field] = original + delta;
      });
      await act(`view.${field}.restore`, () => {
        goal[field] = original;
      });
    }
  }
  await run(6);
  return report;
}
