import { useScene, type Hosted, type SceneHandle } from '@typegpu/kit/native/react';
import { File } from 'expo-file-system';
import { ImageManipulator, SaveFormat } from 'expo-image-manipulator';
import { launchImageLibraryAsync } from 'expo-image-picker';
import { useCallback, useEffect, useRef, useState } from 'react';
import type { Example } from 'typegpu-kit-examples';
import type { ExampleScene } from 'typegpu-kit-examples/contract';
import { compiling, filesOf, type Readings, type Shape } from 'typegpu-kit-examples/hosts';
import type { Value } from 'typegpu-kit-examples/settings.ts';
import { loading } from '../files.ts';
import { useGpu } from '../gpu.tsx';
import * as worklets from '../worklets.ts';
import type { ExampleOptions } from '../worklets.ts';

/** Readings are polled this often while the screen shows. */
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
  /** Compilation status, then undefined once the scene is ready. */
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
  const { canvas, scene, error } = useScene(worklets.createExample, options, {
    device,
    on: worker,
  });

  useEffect(() => {
    if (!scene) return;
    let live = true;
    let last = '';
    scene.run(worklets.shape).then((next) => live && setShape(next), fail);
    const poll = () => {
      scene.run(worklets.readings, example.key).then((next) => {
        // Skip renders while nothing changes
        const text = JSON.stringify(next);
        if (!live || text === last) return;
        last = text;
        setReadings(next);
      }, fail);
    };
    poll();
    const timer = setInterval(poll, POLL_MS);
    return () => {
      live = false;
      clearInterval(timer);
    };
  }, [scene, example.key, fail]);

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
      flush(value).catch(fail);
    },
    [scene, fail],
  );

  const act = useCallback((name: string) => scene?.send(worklets.act, name), [scene]);

  const pick = useCallback(
    (element: string) => {
      if (!scene) return;
      pickPhoto()
        .then((bitmap) => bitmap && scene.run(worklets.show, element, bitmap))
        .catch(fail);
    },
    [scene, fail],
  );

  return {
    canvas,
    scene,
    shape,
    readings,
    status: readings?.ready ? undefined : progress,
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
