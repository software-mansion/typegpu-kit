// Worklets that run on the scene's runtime. Imports from the examples package load again there, so
// the worker runs the package's list and only the scenes it opens.

import { cameras } from '@typegpu/kit';
import type { CreateScene, Stage } from '@typegpu/kit/native/host';
import type { Source } from '@typegpu/kit/video';
import { exampleOf } from 'typegpu-kit-examples';
import type { ExampleScene } from 'typegpu-kit-examples/contract';
import {
  begin,
  orbits,
  readingsOf,
  shapeOf,
  type Readings,
  type Shape,
} from 'typegpu-kit-examples/hosts';
import type { Value, Values } from 'typegpu-kit-examples/settings.ts';
import type { Camera, Held, Tap } from 'typegpu-kit-examples/ui.ts';

type ExampleStage = Stage<ExampleScene>;

export interface ExampleOptions {
  /** The example's directory in the package, such as `blockfall`. */
  readonly key: string;
}

/** Creates the example that `options.key` names, with its files. One worklet hosts every example. */
export const createExample: CreateScene<ExampleOptions, ExampleScene> = async (
  root,
  surface,
  options,
) => {
  'worklet';
  const { createScene } = await exampleOf(options.key).scene();
  return begin(await createScene(root, surface, options));
};

export function shape({ scene }: ExampleStage): Shape {
  'worklet';
  return shapeOf(scene);
}

export function readings(stage: ExampleStage, key: string): Readings {
  'worklet';
  return readingsOf(stage, exampleOf(key).ui);
}

/** Turns an orbiting view by a drag of `dx` and `dy` points. */
export function orbit({ scene }: ExampleStage, dx: number, dy: number) {
  'worklet';
  if (!orbits(scene)) return;
  cameras.turn(scene.view.goal, dx, dy);
}

/** Zooms, multiplying the view's distance by `by`, above 1 farther. */
export function zoom({ scene }: ExampleStage, by: number) {
  'worklet';
  if (orbits(scene)) scene.view.goal.distance *= by;
  else scene.pointing?.zoom?.(by);
}

/** A tap at `x` and `y`, from 0 to 1 across the canvas. */
export function tap({ scene }: ExampleStage, x: number, y: number) {
  'worklet';
  scene.pointing?.move?.(x, y);
  scene.pointing?.press?.(x, y);
  // A finger lifts after a tap, so nothing stays hovered under it as the view turns
  scene.pointing?.leave?.();
}

/** A stroke point at `x` and `y`; `start` marks its first. */
export function drag({ scene }: ExampleStage, x: number, y: number, start: boolean) {
  'worklet';
  scene.pointing?.drag?.(x, y, start);
}

export function end({ scene }: ExampleStage) {
  'worklet';
  scene.pointing?.end?.();
}

/** Writes a setting and resolves after any rebuild it causes. */
export async function write({ scene }: ExampleStage, key: string, value: Value) {
  'worklet';
  const settings = scene.settings;
  if (!settings) return;
  (settings.values as Values)[key] = value;
  await settings.write?.(key);
}

export function act({ scene }: ExampleStage, name: string) {
  'worklet';
  scene.settings?.actions?.[name]?.();
}

type Bindings = Readonly<Record<string, unknown>>;

const buttonOf = (scene: ExampleScene, element: string, name: string) => {
  'worklet';
  const buttons = (scene.ui as Bindings | undefined)?.[element] as Bindings | undefined;
  return buttons?.[name] as (Partial<Held> & Tap) | undefined;
};

export function press({ scene }: ExampleStage, element: string, name: string) {
  'worklet';
  buttonOf(scene, element, name)?.press();
}

export function release({ scene }: ExampleStage, element: string, name: string) {
  'worklet';
  buttonOf(scene, element, name)?.release?.();
}

/** Gives an image element the picture the reader picked, decoded on the JS thread. */
export function show({ scene }: ExampleStage, element: string, picture: ImageBitmap) {
  'worklet';
  const bound = (scene.ui as Bindings | undefined)?.[element];
  if (typeof bound === 'function') bound(picture);
}

/** Points camera element `element` at `source`, or stops it with `undefined`. */
export function watch({ scene }: ExampleStage, element: string, source: Source | undefined) {
  'worklet';
  const camera = (scene.ui as Bindings | undefined)?.[element] as Partial<Camera> | undefined;
  camera?.attach?.(source);
}
