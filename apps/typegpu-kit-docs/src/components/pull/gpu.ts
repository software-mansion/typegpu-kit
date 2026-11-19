// Everything a pull draws on the GPU, in one frame: the vessel, a box or a foil pack, and the foil
// on the card's face. Each canvas has a texel a dither cell. Compiled once a page, when it first pulls.

import * as frames from '@typegpu/kit/frames';
import * as web from '@typegpu/kit/web';
import { d } from 'typegpu';
import type { Colors, Rgb } from '../site/dither.ts';
import { lostOf, pageRoot } from '../viz/gpu.ts';
import type { BoxInput } from 'typegpu-kit-pull/box.ts';
import { createBox } from 'typegpu-kit-pull/box.scene.ts';
import type { Finish } from 'typegpu-kit-pull/card.ts';
import { createFoil } from 'typegpu-kit-pull/foil.scene.ts';
import { gpuFrame } from 'typegpu-kit-pull/vessel.scene.ts';
import { offsetIn } from './card.ts';
import type { PouchInput } from './pouch.ts';
import { createPouch } from './pouch.scene.ts';
import { CELL, type Cells } from './print.ts';

/** How the card's face looks this frame. */
export interface FaceState {
  /** Where its face points, -1 to 1 across and down. */
  readonly tilt: readonly [number, number];
  readonly sheen: number;
  readonly sweep: number;
  readonly flare: number;
}

export type Vessel = 'box' | 'pack';
export type VesselState =
  | { readonly kind: 'box'; readonly input: BoxInput }
  | { readonly kind: 'pack'; readonly input: PouchInput };

export interface Gpu {
  /** Moves the foil onto `face`, printing `cells` in `art`, its poster, in the card's `finish`. */
  attach(face: HTMLElement, art: HTMLElement, cells: Cells, finish: Finish): void;
  /** Takes the foil off the card. */
  detach(): void;
  /** Sizes the vessel's canvas, in cells. */
  fit(columns: number, rows: number): void;
  /** Draws one frame: the vessel, if shown, and the face, if any. */
  draw(vessel: VesselState | undefined, face: FaceState | undefined): void;
  /** Another canvas the vessel alone draws into, such as its place on the page. */
  elsewhere(canvas: HTMLCanvasElement): VesselTarget;
  /** Settles once the device is lost, as iOS loses it for a page in the background. */
  readonly lost: Promise<void>;
}

/** A canvas only the vessel draws into, each draw a frame of its own. */
export interface VesselTarget {
  /** Sizes the canvas, in cells. */
  fit(columns: number, rows: number): void;
  draw(vessel: VesselState): void;
}

const rgb = (c: readonly number[]) => d.vec4f(c[0], c[1], c[2], 1);

const fitTo = (target: web.Surface, across: number, down: number) => {
  if (target.width !== across || target.height !== down)
    target.setSize({ width: across, height: down });
};

/**
 * Compiles the `kind` of vessel and the foil, drawing the vessel into `canvas`; undefined without
 * WebGPU or where it fails. `ground` is the card's ground in bytes.
 */
export async function startGpu(
  canvas: HTMLCanvasElement,
  kind: Vessel,
  colors: Colors,
  ground: Rgb,
): Promise<Gpu | undefined> {
  const root = await pageRoot();
  if (!root) return undefined;

  const format = navigator.gpu.getPreferredCanvasFormat();
  const box = kind === 'box' ? createBox(root, format, colors) : undefined;
  const pouch = kind === 'pack' ? createPouch(root, format) : undefined;
  const foil = createFoil(root, format);
  await Promise.all([
    box?.pipeline.initAsync(),
    pouch?.pipeline.initAsync(),
    foil.pipeline.initAsync(),
  ]);

  const surface = (target: HTMLCanvasElement) =>
    web.surface(root, target, { alphaMode: 'premultiplied', maxPixelRatio: 1 });
  const vesselSurface = surface(canvas);
  const faceCanvas = document.createElement('canvas');
  faceCanvas.className = 'foil';
  faceCanvas.setAttribute('aria-hidden', 'true');
  const faceSurface = surface(faceCanvas);

  const accent = rgb(colors.accent[0]);
  const high = rgb(colors.accent[1]);
  const cardGround = d.vec4f(ground[0] / 255, ground[1] / 255, ground[2] / 255, 1);
  const size = d.vec2f();
  const origin = d.vec2f();
  const art = d.vec4f();
  let finish: Finish | undefined;
  let face: HTMLElement | undefined;
  let shown: { vessel: VesselState | undefined; face: FaceState | undefined } = {
    vessel: undefined,
    face: undefined,
  };

  const writePouch = (input: PouchInput) =>
    pouch?.pouch.write({
      ...input,
      anchor: d.vec2f(...input.anchor),
      body: gpuFrame(input.body),
      strip: gpuFrame(input.strip),
      tint: rgb(input.tint),
      accent,
      high,
      grains: input.grains.map((grain) => d.vec4f(...grain)),
    });

  /** Clears `view` and draws `vessel` into it, if shown, so a sunk vessel leaves nothing behind. */
  const drawVessel = (frame: frames.Frame, view: GPUTextureView, vessel?: VesselState) => {
    if (vessel?.kind === 'box') box?.write(vessel.input, performance.now() / 1000, 1);
    if (vessel?.kind === 'pack') writePouch(vessel.input);
    const pass = frame.encoder.beginRenderPass({
      colorAttachments: { view, clearValue: [0, 0, 0, 0] },
    });
    const drawing = vessel?.kind === 'box' ? box?.pipeline : vessel && pouch?.pipeline;
    if (drawing) {
      pass.setPipeline(drawing);
      pass.draw(3);
    }
    pass.end();
  };

  const loop = frames.create(root, (frame) => {
    drawVessel(frame, vesselSurface.view(frame), shown.vessel);

    const held = shown.face;
    if (!held || !face || !finish) return;
    foil.face.write({
      ground: cardGround,
      size,
      cell: CELL,
      texel: 1,
      origin,
      art,
      tilt: d.vec2f(held.tilt[0], held.tilt[1]),
      sheen: held.sheen,
      flare: d.vec2f(held.sweep, held.flare),
      finish,
    });
    const facePass = frame.encoder.beginRenderPass({
      colorAttachments: { view: faceSurface.view(frame), clearValue: [0, 0, 0, 0] },
    });
    facePass.setPipeline(foil.pipeline);
    facePass.draw(3);
    facePass.end();
  });

  return {
    lost: lostOf(root),
    attach(next, window, cells, nextFinish) {
      face = next;
      finish = nextFinish;
      next.prepend(faceCanvas);
      // Layout sizes, which the card's 3D transform leaves alone. The canvas starts up to a cell
      // before the face, so that its cells fall on the poster's
      const width = next.clientWidth;
      const height = next.clientHeight;
      const [wx, wy] = offsetIn(window, next);
      origin.x = (wx % CELL) - CELL;
      origin.y = (wy % CELL) - CELL;
      const across = Math.ceil((width - origin.x) / CELL);
      const down = Math.ceil((height - origin.y) / CELL);
      faceSurface.setSize({ width: across, height: down });
      Object.assign(faceCanvas.style, {
        left: `${origin.x}px`,
        top: `${origin.y}px`,
        width: `${across * CELL}px`,
        height: `${down * CELL}px`,
      });
      size.x = width;
      size.y = height;
      Object.assign(art, { x: wx, y: wy, z: wx + window.clientWidth, w: wy + window.clientHeight });
      // A card's poster fits in MAX_CELLS each way
      foil.print(cells.pixels, cells.columns, cells.rows);
    },

    detach() {
      face = undefined;
      faceCanvas.remove();
    },

    fit(across, down) {
      fitTo(vesselSurface, across, down);
    },

    draw(vessel, nextFace) {
      shown = { vessel, face: nextFace };
      loop.run(performance.now() / 1000);
    },

    elsewhere(place) {
      const target = surface(place);
      let drawn: VesselState | undefined;
      const own = frames.create(root, (frame) => {
        drawVessel(frame, target.view(frame), drawn);
      });
      return {
        fit: (across, down) => fitTo(target, across, down),
        draw(vessel) {
          drawn = vessel;
          own.run(performance.now() / 1000);
        },
      };
    },
  };
}
