// What an example's scene implements, and what a host gives it while it loads

import type { Scene } from '@typegpu/kit/hosting';
import type { gltf, images } from '@typegpu/kit';
import type { Settings } from './settings.ts';
import type { Bound, Ui } from './ui.ts';

/** What hosts read from a scene besides `frame`, `destroy` and `ready`. */
export interface ExampleScene<U extends Ui = Ui> extends Scene {
  /** Requests another draw without advancing time, such as settling light after an edit. */
  readonly settling?: boolean;
  /**
   * The camera. Hosts orbit it when its goal has yaw, pitch and distance: with the primary button
   * and one finger, or with the secondary button and two fingers while `pointing` takes strokes.
   */
  readonly view?: { readonly goal: object };
  readonly settings?: Settings;
  /** Scene pointer callbacks connected by pages and figures. */
  readonly pointing?: Pointing;
  /** Handlers for each element the example's `ui.ts` declares. Default none. */
  readonly ui?: Bound<U>;
}

/** Surface coordinates x and y from 0 to 1, measured from the top left. */
export interface Pointing {
  /** Hover move with no button down, and a tap's point before its `press`. Excludes orbit drags. */
  move?(x: number, y: number): void;
  /** The pointer left the surface. On touch, follows each tap at once. */
  leave?(): void;
  /** Click or tap at `x` and `y` without dragging. */
  press?(x: number, y: number): void;
  /** Primary-pointer stroke. `start` marks its first point. Hosts then orbit with the secondary button and two fingers. */
  drag?(x: number, y: number, start: boolean): void;
  /** Ends the stroke `drag` began: the pointer lifted, or a second finger started a pinch. */
  end?(): void;
  /**
   * Zooms a view the scene turns itself, multiplying its distance by `by`, above 1 farther. Hosts
   * send the wheel, as e^(deltaY × 0.001), and two-finger pinches here.
   */
  zoom?(by: number): void;
  /** Picker radius in points. Figures draw a ring at this radius. */
  readonly reach?: number;
}

/** Reads a file the package ships, by path from `src/`, such as `hex-isle/models/tree.glb`. */
export type Read = (path: string) => Promise<ArrayBuffer>;

/** Receives loading status such as `loading models` or `compiling 40 %`. */
export type Say = (text: string) => void;

/** A font the kit ships. */
export type FontName = 'inter';

/** What a host gives an example while it loads its files. */
export interface Loading {
  readonly read: Read;
  /** Reads a font the kit ships: its JSON parsed, and its atlas. */
  readonly font: (
    name: FontName,
  ) => Promise<{ readonly json: unknown; readonly atlas: ArrayBuffer }>;
  readonly say?: Say;
  /**
   * Decodes images on the thread that loads them. Tests decode to pixels; the native host decodes
   * to bitmaps with `images.decodeBitmap`, since its scenes run on a worker without
   * `createImageBitmap`. The web host leaves images to the scene.
   */
  readonly decode?: images.Decode;
  /**
   * Makes the decoder for models the manifest marks as Draco, once. The web host and tests use
   * `draco3d`'s WebAssembly decoder; the native host Google's JavaScript build, since Hermes has no
   * WebAssembly.
   */
  readonly draco?: () => Promise<gltf.Draco>;
}

/** A picked or loaded image: an `ImageBitmap` on the web and on native, unpremultiplied pixels in tests. */
export type Picture = ImageBitmap | images.DecodedImage;
