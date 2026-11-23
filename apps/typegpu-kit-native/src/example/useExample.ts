import { useScene, type Hosted, type SceneHandle } from '@typegpu/kit/native/react';
import { File } from 'expo-file-system';
import { ImageManipulator, SaveFormat } from 'expo-image-manipulator';
import { launchImageLibraryAsync } from 'expo-image-picker';
import { useCallback, useEffect, useRef, useState } from 'react';
import type { Example } from 'typegpu-kit-examples';
import type { ExampleScene } from 'typegpu-kit-examples/contract';
import {
  compiling,
  filesOf,
  WARMUP_FRAMES,
  type Readings,
  type Shape,
} from 'typegpu-kit-examples/hosts';
import type { Value } from 'typegpu-kit-examples/settings.ts';
import { loading } from '../files.ts';
import { useGpu } from '../gpu.tsx';
import { useActive } from '../useActive.ts';
import { useDrawn } from '../useDrawn.ts';
import * as worklets from '../worklets.ts';
import type { ExampleOptions } from '../worklets.ts';

/** Readings are read this long after the last read while the screen shows. */
const POLL_MS = 250;

/** Picked photos are shrunk so their longer side is at most this, in pixels. */
const PHOTO_SIZE = 1024;

const toError = (reason: unknown) => (reason instanceof Error ? reason : Error(String(reason)));

/** Loads an example's files on the JS thread. Most examples ship none. */
export function useFiles(example: Example) {
  const [files, setFiles] = useState<object>();
  const [status, setStatus] = useState('loading');
  const [error, setError] = useState<Error>();

  useEffect(() => {
    let live = true;
    const say = (text: string) => live && setStatus(text);
    filesOf(example.key, loading(say)).then(
      (loaded) => live && setFiles(loaded),
      (reason: unknown) => live && setError(toError(reason)),
    );
    return () => {
      live = false;
    };
  }, [example.key]);

  return { files, status, error };
}

export interface ExampleHandle {
  readonly canvas: SceneHandle<ExampleScene>['canvas'];
  /** The hosted scene; pads and gestures send to it from the UI thread. */
  readonly scene: Hosted<ExampleScene> | undefined;
  /** The scene's controls, once it is up. */
  readonly shape: Shape | undefined;
  readonly readings: Readings | undefined;
  /** Compilation status, then undefined once the scene is ready and has drawn its warm-up frames. */
  readonly status: string | undefined;
  readonly error: Error | undefined;
  write(key: string, value: Value): void;
  act(name: string): void;
  /** Opens the photo library and gives the picked image to image element `element`. */
  pick(element: string): void;
}

/**
 * Hosts an example with its loaded files on the app's worker, and reads it while it runs. The
 * scene starts once its canvas is laid out, so mount the canvas only after `files` arrive.
 */
export function useExample(example: Example, files: object | undefined): ExampleHandle {
  const { device, worker } = useGpu();
  const [progress, setProgress] = useState<string>('compiling');
  const [shape, setShape] = useState<Shape>();
  const [readings, setReadings] = useState<Readings>();
  const [failure, setFailure] = useState<Error>();
  const fail = useCallback(
    (reason: unknown) => setFailure((first) => first ?? toError(reason)),
    [],
  );

  const options: ExampleOptions & { onProgress: (done: number) => void } = {
    ...files,
    key: example.key,
    onProgress: (done) => setProgress(compiling(done)),
  };
  const warm = useDrawn(WARMUP_FRAMES);
  const { canvas, scene, error } = useScene(worklets.createExample, options, {
    device,
    on: worker,
    onFrame: warm.onFrame,
  });

  useEffect(() => {
    if (!scene) return;
    let live = true;
    scene.run(worklets.shape).then((next) => live && setShape(next), fail);
    return () => {
      live = false;
    };
  }, [scene, fail]);

  // One read at a time, the next one `POLL_MS` after the last, while the app is in the foreground
  const active = useActive();
  const last = useRef('');
  useEffect(() => {
    if (!scene || !active) return;
    let live = true;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const poll = () => {
      scene.run(worklets.readings, example.key).then(
        (next) => {
          if (!live) return;
          // Skip renders while nothing changes
          const text = JSON.stringify(next);
          if (text !== last.current) {
            last.current = text;
            setReadings(next);
          }
          timer = setTimeout(poll, POLL_MS);
        },
        (reason: unknown) => live && fail(reason),
      );
    };
    poll();
    return () => {
      live = false;
      clearTimeout(timer);
    };
  }, [scene, active, example.key, fail]);

  // Each setting sends its latest value once the write before it has finished
  const writes = useRef(new Map<string, { value: Value; busy: boolean }>());
  const write = useCallback(
    (key: string, value: Value) => {
      if (!scene) return;
      const entry = writes.current.get(key);
      if (entry?.busy) {
        entry.value = value;
        return;
      }
      const next = { value, busy: true };
      writes.current.set(key, next);
      const flush = (sent: Value): Promise<void> =>
        scene.run(worklets.write, key, sent).then(() => {
          if (next.value !== sent) return flush(next.value);
          next.busy = false;
          return undefined;
        });
      flush(value).catch((reason: unknown) => {
        next.busy = false;
        fail(reason);
      });
    },
    [scene, fail],
  );

  const act = useCallback((name: string) => scene?.send(worklets.act, name), [scene]);

  // The scene on show, which a photo picked for an earlier one must not reach
  const current = useRef<typeof scene>(undefined);
  useEffect(() => {
    current.current = scene;
    return () => {
      current.current = undefined;
    };
  }, [scene]);

  const pick = useCallback(
    (element: string) => {
      if (!scene) return;
      pickPhoto()
        .then((bitmap) => {
          if (!bitmap) return undefined;
          if (current.current !== scene) return bitmap.close();
          return scene.run(worklets.show, element, bitmap);
        })
        .catch(fail);
    },
    [scene, fail],
  );

  return {
    canvas,
    scene,
    shape,
    readings,
    status: warm.drawn ? undefined : progress,
    error: error ?? failure,
    write,
    act,
    pick,
  };
}

/** Lets the reader pick a photo from the library, shrunk and decoded, or undefined on cancel. */
async function pickPhoto(): Promise<ImageBitmap | undefined> {
  const picked = await launchImageLibraryAsync({ mediaTypes: ['images'] });
  const asset = picked.canceled ? undefined : picked.assets[0];
  if (!asset) return undefined;
  const scale = Math.min(1, PHOTO_SIZE / Math.max(asset.width, asset.height));
  const context = ImageManipulator.manipulate(asset.uri);
  if (scale < 1) context.resize({ width: Math.round(asset.width * scale) });
  const image = await (await context.renderAsync()).saveAsync({ format: SaveFormat.PNG });
  const bytes = await new File(image.uri).arrayBuffer();
  // Worker runtimes have no createImageBitmap, so decode here and send the bitmap
  return createImageBitmap(bytes, { colorSpaceConversion: 'none', premultiplyAlpha: 'none' });
}
