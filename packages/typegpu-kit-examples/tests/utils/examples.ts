import { readFile } from 'node:fs/promises';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import { frames, gltf, type Frames, type Surface, type images } from '@typegpu/kit';
import {
  offscreenSurface,
  recordingRoot,
  type OffscreenSurface,
  type RecordingRoot,
} from '@typegpu/kit/testing';
import { createDecoderModule } from 'draco3d';
import type { TgpuRoot } from 'typegpu';
import { describe, expect, it } from 'vitest';
import type { ExampleScene, Loading, Read } from '../../src/contract.ts';
import { begin, filesOf, readingsOf, shapeOf } from '../../src/hosts.ts';
import { exampleOf } from '../../src/index.ts';
import type { Choice, Knob, Range, Value } from '../../src/settings.ts';
import type { Camera, Speaker, Ui } from '../../src/ui.ts';

const packageRoot = fileURLToPath(new URL('../../', import.meta.url));
const require = createRequire(import.meta.url);

/** Reads a file the package ships from disk, by path from its `src/`. */
export const readFromDisk: Read = async (path) => {
  const bytes = await readFile(`${packageRoot}src/${path}`);
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer;
};

/** Reads a font the kit ships from its package. */
export const fontFromDisk: Loading['font'] = async (name) => {
  const [json, atlas] = await Promise.all([
    readFile(require.resolve(`@typegpu/kit/fonts/${name}.json`), 'utf8'),
    readFile(require.resolve(`@typegpu/kit/fonts/${name}.rgba`)),
  ]);
  return {
    json: JSON.parse(json) as unknown,
    atlas: atlas.buffer.slice(atlas.byteOffset, atlas.byteOffset + atlas.byteLength) as ArrayBuffer,
  };
};

/** Neither Node nor dawn.node can decode an image, so each is a white pixel instead. */
export const white: images.Decode = async () => ({
  width: 1,
  height: 1,
  pixels: new Uint8Array([255, 255, 255, 255]),
});

let decoder: Promise<gltf.Draco> | undefined;

/** Decodes Draco meshes with `draco3d`'s WebAssembly decoder, made on first use. */
export const dracoFromDisk = () =>
  (decoder ??= createDecoderModule().then((module) => gltf.draco(module)));

/** Loads an example's files from disk, with images as white pixels. */
export const fromDisk: Loading = {
  read: readFromDisk,
  font: fontFromDisk,
  decode: white,
  draco: dracoFromDisk,
};

/** A render pass recording `draws`. */
export const pass = (...draws: string[]) => ['begin-render', ...draws, 'end-render'];

/** A compute pass recording `count` dispatches. */
export const compute = (count = 1) => [
  'begin-compute',
  ...Array<string>(count).fill('dispatch'),
  'end-compute',
];

/** What `bloom.record` records: six levels down and five up, a draw each. */
export const BLOOM = Array.from({ length: 11 }, () => pass('draw')).flat();

/**
 * What a neighbour grid's `sort` records with its default binning: emptying the last sort's ranges,
 * counting, sizing, reserving, scattering, finding the ranges and copying the items.
 */
export const SORTING = [
  'dispatch-indirect',
  'dispatch',
  'dispatch',
  'dispatch-indirect',
  'dispatch',
  'dispatch',
  'dispatch-indirect',
];

export { recorded } from '../../../typegpu-kit/tests/recorded.ts';

/** Lets readbacks started in a frame map. */
export const settle = () => new Promise((resolve) => setTimeout(resolve, 0));

/** A loop's frames run one at a time, as a page's display runs them. */
export interface Driven {
  /** Runs a frame `gap` seconds after the last, the driver's by default; the first runs at 0. */
  next(gap?: number): void;
  /** Runs `count` frames `gap` apart, letting readbacks map after each. */
  run(count: number, gap?: number): Promise<void>;
  /** Runs frames until the scene is ready, as behind a page's cover, failing after 100. */
  warmUp(): Promise<void>;
}

/**
 * Drives `loop` a frame at a time, `gap` seconds apart from 0. It watches `ready` from now, so a
 * scene ready before its first frame runs none to warm up, and a scene that fails to get ready
 * throws its error from the next frame.
 */
export function driven(loop: Frames, ready: PromiseLike<unknown> | undefined, gap = 0.04): Driven {
  const state: { ready: boolean; failure?: { error: unknown } } = { ready: ready === undefined };
  ready?.then(
    () => (state.ready = true),
    (error: unknown) => (state.failure = { error }),
  );

  let seconds: number | undefined;
  const next = (step = gap) => {
    if (state.failure) throw state.failure.error;
    seconds = seconds === undefined ? 0 : seconds + step;
    loop.run(seconds);
  };

  const run = async (count: number, step?: number) => {
    for (let i = 0; i < count; i++) {
      next(step);
      await settle();
    }
  };

  const warmUp = async () => {
    for (let i = 0; i < 100 && !state.ready; i++) await run(1);
    expect(state.ready, 'ready after 100 frames').toBe(true);
  };

  return { next, run, warmUp };
}

/** An example's scene on `recordingRoot()`, and its frames. */
export interface Built<S> extends RecordingRoot, Driven {
  readonly surface: OffscreenSurface;
  readonly scene: S;
  readonly loop: Frames;
}

type CreateScene<O, S> = (root: TgpuRoot, surface: Surface, options: O) => S | Promise<S>;

export interface Contract<O = object> {
  /** Render and compute pipelines the scene has created by its steady frames. */
  readonly pipelines: number;
  /** Scene options every build starts from, over the page's, such as fewer trees. */
  readonly options?: Partial<O>;
  /** Device features the recording root reports, such as `indirect-first-instance`. Default none. */
  readonly features?: readonly GPUFeatureName[];
  /** Seconds between the frames `next`, `run` and `warmUp` run. Default 0.04. */
  readonly gap?: number;
}

/**
 * Describes an example as the page runs it, from its key: builds its scene with its files on
 * `recordingRoot()` and a 64 × 36 `offscreenSurface()`, and checks what every example keeps: a
 * steady frame creates nothing, submits once and holds the contract's pipelines, `destroy` frees
 * every texture and buffer the scene made other than uniforms, a resize replaces the textures
 * without new pipelines, and each value has a knob, in range, that writes.
 * `body` adds the example's own tests, building the scene with `build`, which takes options over
 * the page's.
 */
export function describeExample<O, S extends ExampleScene>(
  key: string,
  createScene: CreateScene<O, S>,
  body: (build: (options?: Partial<O>) => Promise<Built<S>>) => void,
  contract: Contract<O>,
) {
  const example = exampleOf(key);

  let files: Promise<object> | undefined;

  const build = async (options: Partial<O> = {}): Promise<Built<S>> => {
    files ??= filesOf(key, fromDisk);
    const recording = recordingRoot(contract.features ? { features: contract.features } : {});
    const { root } = recording;
    const surface = offscreenSurface(root, { width: 64, height: 36, format: 'bgra8unorm' });

    const scene = await begin(
      await createScene(root, surface, {
        ...(await files),
        ...contract.options,
        ...options,
      } as O),
    );
    const loop = frames.create(root, scene.frame);
    return { ...recording, surface, scene, loop, ...driven(loop, scene.ready, contract.gap) };
  };

  describe(example.title, () => {
    it('compiles while loading, creates nothing and submits once in a steady frame, and frees what it made', async () => {
      const { log, scene, loop, run, warmUp } = await build();
      const { blocking } = log;
      await warmUp();
      await run(6);
      expect(log.blocking - blocking, 'pipelines a frame waited for').toBe(0);

      const counts = log.counts();
      const { submissions } = log;
      await run(4);
      expect(log.counts()).toEqual(counts);
      expect(log.submissions - submissions, 'submits in four frames').toBe(4);
      expect(log.pipelines, 'pipelines').toBe(contract.pipelines);

      loop.destroy();
      scene.destroy?.();
      expect(log.alive({ uniforms: false }), 'textures and buffers left after destroy').toEqual([
        '[Texture "offscreen surface"]',
      ]);
    });

    it('replaces its textures once on each resize, with no new pipelines', async () => {
      const { log, surface, run, warmUp } = await build();
      await warmUp();
      await run(2);
      const { pipelines } = log;
      const alive = log.alive().length;

      /** Resizes the surface and counts the textures the next frames create. */
      const resize = async (width: number, height: number) => {
        const { textures } = log;
        surface.setSize({ width, height });
        await run(2);
        const counts = log.counts();
        await run(2);
        expect(log.counts(), `frames after the resize to ${width} × ${height}`).toEqual(counts);
        return log.textures - textures;
      };

      const first = await resize(96, 54);
      expect(await resize(128, 72), 'textures the second resize creates').toBe(first);
      expect(log.pipelines - pipelines, 'pipelines a resize creates').toBe(0);
      expect(log.alive(), 'textures and buffers alive').toHaveLength(alive);
    });

    it('gives each value a knob in its range, and writes it moved to the end', async () => {
      const { scene, run, warmUp } = await build();
      await warmUp();
      if (!scene.settings) return;
      const settings = scene.settings;
      const { values, knobs } = settings;
      const hidden = Object.keys(values).filter((name) => !(name in knobs));
      expect(hidden, 'values without a knob').toEqual([]);

      for (const [name, shown] of Object.entries(knobs)) {
        const value = values[name];
        const knob = shown as Knob;
        expect(value, name).toBeDefined();

        let moved: Value = !value;
        if ('options' in knob) {
          const { options } = knob as Choice<Value>;
          expect(options, name).toContain(value);
          moved = options.at(-1)!;
        } else if (typeof value === 'number') {
          const { min, max } = knob as Range;
          expect(value, name).toBeGreaterThanOrEqual(min);
          expect(value, name).toBeLessThanOrEqual(max);
          moved = max;
        }
        if (knob.disabled) continue;

        values[name] = moved;
        await settings.write?.(name);
        await run(1);
      }

      await settings.write?.();
      await run(1);
    });

    it('gives hosts copyable data and binds every element its UI declares', async () => {
      const { root, surface, scene, loop, next, run, warmUp } = await build();
      await warmUp();
      const shape = shapeOf(scene);
      expect(structuredClone(shape)).toEqual(shape);
      const readings = readingsOf({ scene, root, surface, loop, ready: true }, example.ui);
      expect(structuredClone(readings)).toEqual(readings);

      await checkUi(key, example.ui ?? {}, scene.ui ?? {}, () => next());
      await run(1);
    });

    body(build);
  });
}

/** The WGSL of a scene built with the page's defaults on a device with `features`. */
export async function shadersWith(
  createScene: (root: TgpuRoot, surface: Surface) => ExampleScene | Promise<ExampleScene>,
  features: GPUFeatureName[],
) {
  const { root, log } = recordingRoot({ features });
  const surface = offscreenSurface(root, { width: 64, height: 36, format: 'bgra8unorm' });
  (await createScene(root, surface)).destroy?.();
  return log.shaders.join('\n');
}

/**
 * Checks that `bound` has a handler of the right kind for every element `ui` declares, that each
 * button has keys or a pad and each sound's file of example `example` exists, and that pressing
 * each button, picking a 2 × 2 image, attaching a camera and attaching a player leave the next
 * frame running.
 */
async function checkUi(
  example: string,
  ui: Ui,
  bound: Readonly<Record<string, unknown>>,
  next: () => void,
) {
  const pixels = new Uint8Array(16).fill(255);
  for (const [name, element] of Object.entries(ui)) {
    const handler = bound[name];
    expect(handler, `the ${name} binding`).toBeDefined();

    if (element.kind === 'text') {
      expect(typeof (handler as () => unknown)(), name).toBe('string');
    } else if (element.kind === 'image') {
      (handler as (picture: object) => void)({ width: 2, height: 2, pixels });
      next();
      await settle();
    } else if (element.kind === 'camera') {
      const camera = handler as Camera;
      const frame = {
        texture: {} as GPUExternalTexture,
        width: 4,
        height: 3,
        rotation: 90 as const,
        serial: 1,
      };
      camera.attach({ latest: () => frame });
      next();
      next();
      expect(camera.frames, `${name} receives a camera frame`).toBe(1);
      camera.attach(undefined);
      next();
    } else if (element.kind === 'sounds') {
      for (const path of Object.values(element.files)) {
        expect((await readFromDisk(`${example}/${path}`)).byteLength, path).toBeGreaterThan(0);
      }
      const speaker = handler as Speaker;
      speaker.attach((sound) => expect(Object.keys(element.files), name).toContain(sound));
      next();
      speaker.attach(undefined);
      next();
    } else {
      const padded = new Set([...(element.pads.left ?? []), ...(element.pads.right ?? [])].flat());
      const handlers = handler as Readonly<Record<string, { press(): void; release?(): void }>>;
      for (const [key, button] of Object.entries(element.buttons)) {
        expect(
          button.keys?.length || padded.has(key),
          `${name}.${key} has keys or a pad`,
        ).toBeTruthy();
        handlers[key].press();
        next();
        if (button.held) handlers[key].release!();
        next();
        await settle();
      }
    }
  }
}
