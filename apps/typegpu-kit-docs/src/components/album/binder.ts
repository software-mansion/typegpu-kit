// The album as a binder on the GPU: a sheet for every two sides of the book. A press on a pocket
// picks its card, and a press beside the cards or the keys turn a page. It rises into view from
// below and sinks away again, and tells the overlay where a pocket's card lies on the canvas so the
// card can be taken out. Undefined where the page has no GPU.

import * as frames from '@typegpu/kit/frames';
import * as targets from '@typegpu/kit/targets';
import * as web from '@typegpu/kit/web';
import { d } from 'typegpu';
import { mat4, vec3 } from 'wgpu-matrix';
import { bytesOf, colorsOf, type Rgb } from '../site/dither.ts';
import { lostOf, pageRoot } from '../viz/gpu.ts';
import { cellsOf, CELL, posterImage } from '../pull/print.ts';
import { colorOf, RARITIES, stillMotion, type Entry } from '../pull/pulls.ts';
import { createBinderScene, POSTER } from './binder.scene.ts';
import * as book from './book.ts';
import type { Quad } from './flight.ts';
import { createTurning } from './turning.ts';

/** CSS pixels a texel, as the pull draws. */
const GRAIN = 3;
/**
 * The camera's field of view, and how far it looks down from level: steeper over a single page, so
 * a tall stage holds it larger.
 */
const FOV = 0.46;
const PITCH = { spread: 1.02, page: 1.4 } as const;
/** CSS pixels kept clear above the binder and below it, for the overlay's controls. */
const FRAME = { top: 64, bottom: 84 } as const;
/** Below this width over the clear height, one page shows at a time. */
const SPREAD = 1.15;

const unit = (c: Rgb) => d.vec4f(c[0], c[1], c[2], 1);
const byte = (c: Rgb) => d.vec4f(c[0] / 255, c[1] / 255, c[2] / 255, 1);

/** Whether the point `x`, `y` lies in `quad`, its corners clockwise on screen. */
const within = (quad: Quad, x: number, y: number) =>
  quad.every(([ax, ay], i) => {
    const [bx, by] = quad[(i + 1) % 4];
    return (bx - ax) * (y - ay) >= (by - ay) * (x - ax);
  });

export interface BinderOptions {
  /** The cards by gallery order. */
  readonly entries: readonly Entry[];
  /** Called when a pocket's card is pressed. */
  readonly pick: (key: string) => void;
  /** Called when the pages in view change, with the categories they show. */
  readonly turned: (titles: readonly string[]) => void;
}

export type Binder = NonNullable<Awaited<ReturnType<typeof startBinder>>>;

export async function startBinder(canvas: HTMLCanvasElement, options: BinderOptions) {
  const { entries } = options;
  const pages = book.bookOf(entries.map(({ category }) => category));
  const root = await pageRoot();
  if (!root) return undefined;

  const scene = createBinderScene(root, navigator.gpu.getPreferredCanvasFormat(), {
    sheets: pages.sheets,
    cards: entries.length,
  });
  await scene.initAsync();
  const surface = web.surface(root, canvas, { alphaMode: 'premultiplied', maxPixelRatio: 1 });
  const depth = targets.create(root, surface, {
    depth: { format: 'depth24plus', transient: true },
  });
  const still = stillMotion();
  const indexOf = (key: string | undefined) => entries.findIndex((entry) => entry.key === key);

  // The palette where the canvas is
  const colors = colorsOf(canvas);
  const inks = RARITIES.map(colorOf);
  const view = {
    viewProjection: d.mat4x4f(),
    eye: d.vec4f(),
    grey: colors.grey.map(unit),
    accent: unit(colors.accent[0]),
    high: unit(colors.accent[1]),
    rarity: bytesOf(canvas, inks).map(byte),
    texel: 0,
    posterLevel: 0,
  };

  // The pockets, fixed by the book; the posters, once they have loaded
  const pockets = Array.from({ length: pages.sheets * 2 * book.POCKETS }, () => -1);
  pages.places.forEach(
    ({ side, pocket }, example) => (pockets[side * book.POCKETS + pocket] = example),
  );
  scene.pockets.write(pockets);
  void Promise.all(
    entries.map(({ poster }) =>
      posterImage(poster).then(
        (image) => cellsOf(image, POSTER[0] * CELL, POSTER[1] * CELL).pixels.buffer,
        () => undefined,
      ),
    ),
  ).then((layers) => {
    scene.writePosters(layers);
    wake();
  });

  // The pointer lights the card under it and none between cards; before it moves, or once the keys
  // move, the card the keys mark is lit
  let hovered: number | undefined;
  let marked = -1;
  let out = -1;
  const writeAlbum = () => {
    scene.album.write({ hover: hovered ?? marked, out });
    wake();
  };

  // The camera: down at a whole spread when the canvas is wide, else at one page, panning between
  // the two, with `look` from the left page at -1 to the right at 1
  const turning = createTurning(pages.sheets, still);
  const matrix = mat4.create();
  const eye = vec3.create();
  let width = 1;
  let height = 1;
  let look = 1;
  let lookAim = 1;
  const clear = () => Math.max(height - FRAME.top - FRAME.bottom, 1);
  const single = () => width / clear() < SPREAD;
  const aim = () => {
    const one = single();
    const pitch = one ? PITCH.page : PITCH.spread;
    const halfX = one ? book.WIDTH / 2 + 0.12 : book.REACH + book.COVER.overhang + 0.25;
    const halfZ = book.HEIGHT / 2 + book.COVER.overhang + 0.2;
    const tanY = Math.tan(FOV / 2);
    const distance =
      Math.max((halfX * height) / width, (halfZ * Math.sin(pitch) * height) / clear()) / tanY +
      halfZ * 0.6;
    const target = vec3.create(one ? (book.AXIS + book.WIDTH / 2) * look : 0, 0, 0.12);
    vec3.set(target[0], Math.sin(pitch) * distance, target[2] + Math.cos(pitch) * distance, eye);
    // The binder stands in the band between the clear strips
    const lift = mat4.translation([0, (FRAME.bottom - FRAME.top) / height, 0]);
    mat4.multiply(lift, mat4.perspective(FOV, width / height, 0.5, distance * 4), matrix);
    mat4.multiply(matrix, mat4.lookAt(eye, target, [0, 1, 0]), matrix);
  };
  /** Where `point` falls on the canvas, in CSS pixels. */
  const project = (point: readonly number[]) => {
    const clip = vec3.transformMat4(point, matrix);
    return [((clip[0] + 1) / 2) * width, ((1 - clip[1]) / 2) * height] as const;
  };

  /** The sides in view: both pages of the spread, or the one a single page looks at. */
  const sides = () => {
    const { left, right } = book.sidesOf(turning.spread, pages.sheets);
    return (single() ? [lookAim < 0 ? left : right] : [left, right]).map(
      (side) => pages.sides[side],
    );
  };
  /** The cards on the pages in view, in reading order. */
  const inView = () => sides().flatMap((side) => side?.members.map((i) => entries[i].key) ?? []);
  const turned = () => {
    options.turned(sides().flatMap((side) => side?.title ?? []));
    wake();
  };
  /** Turns a sheet over; a single page shows the spread's other page first, and follows a turn. */
  const turn = (forward: boolean) => {
    if (single() && lookAim > 0 !== forward) lookAim = forward ? 1 : -1;
    else if (!turning.turn(forward)) return;
    else if (single()) lookAim = forward ? -1 : 1;
    turned();
  };

  /** The corners of `key`'s card in its pocket on the canvas, if its page lies open and still. */
  const pocket = (key: string | undefined): Quad | undefined => {
    const place = pages.places[indexOf(key)];
    const { left, right } = book.sidesOf(turning.spread, pages.sheets);
    if (!place || !turning.landed || (place.side !== left && place.side !== right)) return;
    const [x, z] = book.pocketMiddle(place.side, place.pocket);
    const y = turning.sheets[place.side >> 1].height + 0.004;
    // Its corners from the top left, clockwise
    const [w, h] = [book.CARD.width / 2, book.CARD.height / 2];
    const at = (i: number) => project([x + (i % 3 ? w : -w), y, z + (i > 1 ? h : -h)]);
    return [at(0), at(1), at(2), at(3)];
  };

  // A press on a card picks it, and one beside the cards turns toward its side of the stage
  const hitAt = ({ offsetX: x, offsetY: y }: MouseEvent) =>
    inView().find((key) => {
      const quad = pocket(key);
      return quad !== undefined && within(quad, x, y);
    });
  const hover = (at: number | undefined) => {
    canvas.style.cursor = at !== undefined && at >= 0 ? 'pointer' : '';
    if (at === hovered) return;
    hovered = at;
    writeAlbum();
  };
  const idle = () => out < 0 && !rising;
  // A press dragged 40 px sideways turns the page the way it goes, once, and picks no card
  let pressed: number | undefined;
  let swiped = false;
  canvas.addEventListener('pointerdown', (event) => {
    pressed = event.offsetX;
    swiped = false;
  });
  canvas.addEventListener('pointerup', () => (pressed = undefined));
  canvas.addEventListener('pointercancel', () => (pressed = undefined));
  canvas.addEventListener('pointermove', (event) => {
    if (!idle()) return;
    if (pressed !== undefined && !swiped && Math.abs(event.offsetX - pressed) > 40) {
      swiped = true;
      hover(-1);
      return turn(event.offsetX < pressed);
    }
    if (pressed === undefined || !swiped) hover(indexOf(hitAt(event)));
  });
  canvas.addEventListener('pointerleave', () => hover(undefined));
  canvas.addEventListener('click', (event) => {
    if (!idle() || swiped) return;
    const key = hitAt(event);
    if (key) options.pick(key);
    else turn(event.offsetX >= width / 2);
  });

  // Each frame draws the binder as the view and the sheets stand
  const loop = frames.create(root, (frame) => {
    for (let i = 0; i < 16; i++) view.viewProjection.columns[i >> 2][i & 3] = matrix[i];
    view.eye = d.vec4f(eye[0], eye[1], eye[2], 1);
    // A card's width on screen in texels, which the posters' mip level follows
    const [a] = project([0, 0, 0]);
    const [b] = project([book.CARD.width, 0, 0]);
    view.texel = GRAIN / Math.max(Math.abs(b - a), 1e-3);
    view.posterLevel = Math.max(0, Math.log2(POSTER[0] * view.texel));
    scene.view.write(view);
    scene.sheets.write(turning.sheets);
    depth.fit();
    const pass = frame.encoder.beginRenderPass({
      colorAttachments: { view: surface.view(frame), clearValue: [0, 0, 0, 0] },
      depthStencilAttachment: {
        view: depth.views.depth,
        depthClearValue: 1,
        depthLoadOp: 'clear',
        depthStoreOp: 'discard',
      },
    });
    scene.draw(pass);
    pass.end();
  });

  // Draws a frame when anything changes, and runs while the sheets or the view move
  let running = false;
  let rising = false;
  let request = 0;
  let last = 0;
  const tick = (now: number) => {
    request = 0;
    if (!running) return;
    // A frame's time can fall before the wake that asked for it
    const dt = Math.min(Math.max((now - last) / 1000, 0), 1 / 20);
    last = now;
    look += (lookAim - look) * (still ? 1 : 1 - Math.exp(-dt * 7));
    const moving = turning.step(dt) || Math.abs(lookAim - look) > 1e-3;
    aim();
    loop.run(now / 1000);
    if (moving) request = requestAnimationFrame(tick);
  };
  const wake = () => {
    if (!running || request) return;
    last = performance.now();
    request = requestAnimationFrame(tick);
  };
  /** Moves the canvas in from below, or out again, as the binder rises or sinks. */
  const slide = (up: boolean) =>
    canvas
      .animate(
        { translate: up ? ['0 100%', '0 0'] : ['0 0', '0 100%'] },
        {
          duration: still ? 0 : 700,
          easing: up ? 'cubic-bezier(0.3, 1.25, 0.5, 1)' : 'ease-in',
          fill: 'forwards',
        },
      )
      .finished.then(
        () => undefined,
        () => undefined,
      );

  const fit = () => {
    const columns = Math.max(1, Math.ceil((canvas.parentElement?.clientWidth ?? 1) / GRAIN));
    const rows = Math.max(1, Math.ceil((canvas.parentElement?.clientHeight ?? 1) / GRAIN));
    if (columns * GRAIN === width && rows * GRAIN === height) return;
    const wasSingle = single();
    width = columns * GRAIN;
    height = rows * GRAIN;
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    surface.setSize({ width: columns, height: rows });
    if (single() && !wasSingle) look = lookAim = turning.spread >= pages.sheets ? -1 : 1;
    aim();
    if (single() !== wasSingle) turned();
    wake();
  };
  new ResizeObserver(fit).observe(canvas.parentElement ?? canvas);

  return {
    /** Fills the pockets from the album: what is found. */
    fill(found: ReadonlySet<string>) {
      scene.cards.write(
        entries.map((entry) => ({
          rarity: Math.max(
            0,
            RARITIES.findIndex(({ name }) => name === entry.rarity),
          ),
          found: found.has(entry.key) ? 1 : 0,
        })),
      );
      wake();
    },
    /** Lies open at the page holding `key`'s card. */
    show(key: string) {
      const place = pages.places[indexOf(key)];
      if (!place) return;
      turning.turnTo(book.spreadOf(place.side));
      look = lookAim = place.side % 2 === 0 ? 1 : -1;
      turned();
    },
    turn,
    inView,
    pocket,
    /** Draws while the album is open, rising into view; resolves once it has risen. */
    async start() {
      running = true;
      rising = true;
      fit();
      for (const motion of canvas.getAnimations()) motion.cancel();
      await slide(true);
      rising = false;
      wake();
    },
    /** Settles once the device is lost, as iOS loses it for a page in the background. */
    lost: lostOf(root),
    /** Stops drawing, once the album has closed. */
    stop() {
      running = false;
      cancelAnimationFrame(request);
      request = 0;
    },
    /** Sinks out of view; resolves once it is gone. */
    sink() {
      hover(undefined);
      rising = true;
      return slide(false);
    },
    /** Lights `key`'s card as the pointer would, for the keys. */
    mark(key: string | undefined) {
      marked = indexOf(key);
      hovered = undefined;
      writeAlbum();
    },
    /** Leaves `key`'s pocket empty while its card is out, ignoring the pointer, or fills it again. */
    hold(key: string | undefined) {
      out = indexOf(key);
      hover(undefined);
      writeAlbum();
    },
  };
}
