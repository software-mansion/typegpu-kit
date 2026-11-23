// The example page: its canvas, loading cover and error display, the generic page that hosts an
// example, and the UI its ui.ts declares

import type { Hosted } from '@typegpu/kit/web';
import type GUI from 'lil-gui';
import { exampleOf } from 'typegpu-kit-examples';
import type { ExampleScene, Picture, Say } from 'typegpu-kit-examples/contract';
import { WARMUP_FRAMES } from 'typegpu-kit-examples/hosts';
import type { Camera, CameraKind, Ui } from 'typegpu-kit-examples/ui.ts';
import { posterOf } from '../components/site/posters.ts';
import { reducedMotion } from '../components/viz/figure.ts';
import { veil } from '../components/site/veil.ts';
import { film, type Filming } from './camera.ts';
import { connectPad, listenKeys, padsOf, type Pad, type PadRows, type Pressed } from './buttons.ts';
import { host } from './host.ts';
import { panel } from './panel.ts';

/** Finds an element in the example markup. Throws if none matches `selector`. */
const element = <E extends HTMLElement>(selector: string) => {
  const found = document.querySelector<E>(selector);
  if (!found) throw new Error(`The example page has no ${selector}.`);
  return found;
};

const canvas = element<HTMLCanvasElement>('#example canvas');
const cover = element('#cover');
const words = element('#cover > span');
const curtain = veil(cover, { cell: 4, poster: posterOf(element('#example').dataset.key) });
const notice = element<HTMLOutputElement>('#notice');
const pause = element<HTMLButtonElement>('#pause');

function lift() {
  cover.classList.add('lifting');
  void curtain.lift().then(() => cover.remove());
}

/** Displays loading text on the cover. */
const say: Say = (text) => {
  words.textContent = text;
};

/** Shows a message over the poster, which stays. */
export function tell(text: string) {
  words.textContent = '';
  curtain.rest();
  notice.textContent = text;
}

/** Displays the first error over the canvas and logs every error. */
export function fail(error: unknown) {
  // oxlint-disable-next-line no-console -- retain each error and its stack in the console
  console.error(error);
  if (notice.textContent) return;
  tell(error instanceof Error ? error.message : String(error));
}

/** A pad's button, which `connectPad` connects. */
function padButton(pad: Pad): HTMLButtonElement {
  const button = document.createElement('button');
  button.type = 'button';
  button.textContent = pad.label;
  if (pad.name) button.setAttribute('aria-label', pad.name);
  connectPad(button, pad);
  return button;
}

/**
 * On a touch screen, buttons over the example's foot, on the `left` and on the `right`. Returns
 * the blocks it adds.
 */
function pads(sides: {
  readonly left?: PadRows;
  readonly right?: PadRows;
}): readonly HTMLElement[] {
  if (!matchMedia('(pointer: coarse)').matches) return [];
  const blocks: HTMLElement[] = [];
  for (const [side, rows] of Object.entries(sides)) {
    const block = document.createElement('div');
    block.className = `pads ${side}`;
    for (const row of rows) {
      const line = document.createElement('div');
      for (const pad of row) line.append(pad ? padButton(pad) : document.createElement('span'));
      block.append(line);
    }
    document.body.append(block);
    blocks.push(block);
  }
  return blocks;
}

/**
 * Opens a file picker from the panel's button, and takes an image dropped on the canvas. Shows only
 * the latest pick, unless `signal` aborted first, and closes the bitmaps it drops.
 */
function images(
  label: string,
  show: (picture: Picture) => void,
  gui: GUI | undefined,
  signal: AbortSignal,
) {
  if (!gui) throw new Error(`The image element ${label} needs a panel.`);
  let picks = 0;
  const take = (file: File) => {
    const pick = ++picks;
    const latest = () => pick === picks && !signal.aborted;
    void createImageBitmap(file)
      .then((bitmap) => {
        if (latest()) show(bitmap);
        else bitmap.close();
      })
      .catch((error: unknown) => {
        if (latest()) fail(error);
      });
  };

  const chooser = document.createElement('input');
  chooser.type = 'file';
  chooser.accept = 'image/*';
  chooser.addEventListener(
    'change',
    () => {
      const file = chooser.files?.[0];
      if (file) take(file);
    },
    { signal },
  );
  gui.add({ [label]: () => chooser.click() }, label);

  canvas.addEventListener('dragover', (event) => event.preventDefault(), { signal });
  canvas.addEventListener(
    'drop',
    (event) => {
      event.preventDefault();
      const file = event.dataTransfer?.files[0];
      if (file?.type.startsWith('image/')) take(file);
    },
    { signal },
  );
}

/** Shows the camera in a panel button that turns it off and on. */
function cameras(part: CameraKind, camera: Camera, gui: GUI | undefined, signal: AbortSignal) {
  if (!gui) throw new Error(`The camera element ${part.label} needs a panel.`);
  let filming: Filming | undefined;
  const button = gui.add({ [part.label]: () => filming?.toggle() }, part.label);
  filming = film(part, camera, (text) => button.name(text), fail, signal);
}

/** What `ui` connected. */
interface Shown extends Disposable {
  /** Writes texts that changed; call it each frame. */
  update(): void;
}

/**
 * Shows the example's declared UI and connects it to the scene's bindings: texts at the canvas's
 * corners, keys and touch pads pressing buttons, an image picker in `gui` with drops on the
 * canvas, and a camera button in `gui`. Throws if the scene binds no handler for a declared element.
 */
function ui(declared: Ui | undefined, scene: ExampleScene, gui?: GUI): Shown {
  const listening = new AbortController();
  const { signal } = listening;
  const bound = (scene.ui ?? {}) as Readonly<Record<string, unknown>>;
  const added: HTMLElement[] = [];
  const texts: { readonly box: HTMLElement; readonly read: () => string }[] = [];

  for (const [name, part] of Object.entries(declared ?? {})) {
    const binding = bound[name];
    if (binding === undefined) throw new Error(`The scene binds no ${name}, which ui.ts declares.`);

    switch (part.kind) {
      case 'text': {
        const box = document.createElement('div');
        box.className = `hud ${part.corner}`;
        document.body.append(box);
        added.push(box);
        texts.push({ box, read: binding as () => string });
        break;
      }

      case 'buttons': {
        const handlers = binding as Pressed;
        listenKeys(part, handlers, window, signal);
        added.push(...pads(padsOf(part, handlers)));
        break;
      }

      case 'image':
        images(part.label, binding as (picture: Picture) => void, gui, signal);
        break;

      case 'camera':
        cameras(part, binding as Camera, gui, signal);
        break;
    }
  }

  return {
    update() {
      for (const { box, read } of texts) {
        const text = read();
        if (box.textContent !== text) box.textContent = text;
      }
    },
    [Symbol.dispose]() {
      listening.abort();
      for (const each of added) each.remove();
    },
  };
}

/**
 * Hosts example `key` on the page: its input, a settings panel titled with its
 * title, its declared UI and the bar's Pause button. Removes the cover when the scene is ready and
 * has drawn `WARMUP_FRAMES` frames, then pauses under reduced motion.
 */
export function run(key: string): Hosted<ExampleScene> {
  const example = exampleOf(key);
  const picks = Object.values(example.ui ?? {}).some(
    (part) => part.kind === 'image' || part.kind === 'camera',
  );
  let shown: Shown | undefined;
  let drawn = 0;
  const hold = (on: boolean) => {
    if (on) hosted.pause();
    else hosted.resume();
    pause.setAttribute('aria-pressed', String(on));
  };

  const hosted = host(key, canvas, {
    say,
    onError: fail,
    attach: ({ scene }) => {
      const gui =
        scene.settings || picks
          ? panel(scene.settings, example.title.toLowerCase(), fail)
          : undefined;
      const made = ui(example.ui, scene, gui);
      shown = made;
      return {
        [Symbol.dispose]() {
          made[Symbol.dispose]();
          gui?.destroy();
        },
      };
    },
    onFrame: (stage) => {
      shown?.update();
      if (stage.ready && drawn < WARMUP_FRAMES && ++drawn === WARMUP_FRAMES) {
        lift();
        if (reducedMotion()) hold(true);
      }
    },
  });
  pause.hidden = false;
  pause.addEventListener('click', () => hold(pause.getAttribute('aria-pressed') !== 'true'));
  return hosted;
}
