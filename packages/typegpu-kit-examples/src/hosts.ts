// What every host of the examples shares: creating a scene with its files, reading its shape and
// readings as copyable data, and the warm-up under the loading cover

import { backends, type Surface } from '@typegpu/kit';
import type { Stage } from '@typegpu/kit/hosting';
import type { TgpuRoot } from 'typegpu';
import { resolve } from './assets.ts';
import type { ExampleScene, Loading } from './contract.ts';
import { exampleOf } from './index.ts';
import type { Knobs, Value, Values } from './settings.ts';
import type { Ui } from './ui.ts';

/**
 * Frames drawn under the loading cover. WebKit builds Metal pipeline state at first draw, which
 * can stall that frame.
 */
export const WARMUP_FRAMES = 10;

/** Compilation status text for progress `done` from 0 to 1. */
export const compiling = (done: number) => `compiling ${Math.round(done * 100)} %`;

/** Loads the files of example `key` as scene options. Most examples ship none. */
export async function filesOf(key: string, loading: Loading): Promise<object> {
  const module = await exampleOf(key).assets?.();
  if (!module) return {};
  return resolve(key, module.assets, loading);
}

/**
 * Loads the example's files and creates its scene, passing `onProgress` and saying `compiling`
 * through `loading.say`. Where compute runs on the CPU, the example's `fallback` options size the
 * scene for it. `options`, such as a card's, override all of these.
 */
export async function create(
  key: string,
  root: TgpuRoot,
  surface: Surface,
  loading: Loading,
  options: object = {},
): Promise<ExampleScene> {
  const { say } = loading;
  say?.('loading');
  const [{ createScene }, files] = await Promise.all([
    exampleOf(key).scene(),
    filesOf(key, loading),
  ]);
  say?.('compiling');
  const progress = say ? { onProgress: (done: number) => say(compiling(done)) } : {};

  const sized = backends.of(root).compute === 'cpu' ? exampleOf(key).fallback : undefined;

  return begin(await createScene(root, surface, { ...files, ...progress, ...sized, ...options }));
}

/**
 * Writes a new scene's settings once, before its first frame, so it starts from its values. Every
 * host calls it after `createScene`. Destroys the scene if the write throws.
 */
export async function begin<S extends ExampleScene>(scene: S): Promise<S> {
  try {
    await scene.settings?.write?.();
  } catch (error) {
    scene.destroy?.();
    throw error;
  }
  return scene;
}

/** The goal orbit controls write. */
export interface Orbit {
  yaw: number;
  pitch: number;
  distance: number;
}

/**
 * Whether hosts orbit the view: its goal has yaw, pitch and distance. While `pointing` takes
 * strokes, hosts orbit with the secondary button and two fingers.
 */
export function orbits(
  scene: ExampleScene,
): scene is ExampleScene & { readonly view: { readonly goal: Orbit } } {
  const goal = scene.view?.goal as Partial<Record<keyof Orbit, unknown>> | undefined;
  if (!goal) return false;
  return (
    typeof goal.yaw === 'number' &&
    typeof goal.pitch === 'number' &&
    typeof goal.distance === 'number'
  );
}

/** What a host builds controls from, as data that copies between runtimes. Read once per scene. */
export interface Shape {
  readonly orbit: boolean;
  readonly pointing?: { readonly press: boolean; readonly drag: boolean; readonly zoom: boolean };
  readonly settings?: {
    readonly values: Values;
    readonly knobs: Knobs<Values>;
    readonly actions: readonly string[];
    readonly readouts: readonly string[];
  };
}

/** Reads the scene's shape, as data a host on another runtime can receive. */
export function shapeOf(scene: ExampleScene): Shape {
  const { pointing, settings } = scene;
  return {
    orbit: orbits(scene),
    ...(pointing && {
      pointing: { press: !!pointing.press, drag: !!pointing.drag, zoom: !!pointing.zoom },
    }),
    ...(settings && {
      settings: {
        values: { ...settings.values },
        knobs: settings.knobs,
        actions: Object.keys(settings.actions ?? {}),
        readouts: Object.keys(settings.readouts ?? {}),
      },
    }),
  };
}

/** What changes while the scene runs: readiness, the UI's texts, readouts and live knobs. */
export interface Readings {
  readonly ready: boolean;
  /** Each text element's current value, by name. */
  readonly texts: Readonly<Record<string, string>>;
  readonly readouts: Readonly<Record<string, Value>>;
  readonly values: Values;
}

/** Reads the scene's current readings; `ui` is the example's declaration. */
export function readingsOf(stage: Stage<ExampleScene>, ui: Ui | undefined): Readings {
  const { scene, ready } = stage;
  const bound = scene.ui as Readonly<Record<string, unknown>> | undefined;
  const texts: Record<string, string> = {};
  for (const [name, element] of Object.entries(ui ?? {})) {
    const read = bound?.[name];
    if (element.kind === 'text' && typeof read === 'function') texts[name] = String(read());
  }
  const readouts: Record<string, Value> = {};
  for (const [name, read] of Object.entries(scene.settings?.readouts ?? {}))
    readouts[name] = read();
  return { ready, texts, readouts, values: { ...scene.settings?.values } };
}
