// Simulate the subject once per frame, then draw each visible view in the same encoder

import { clock, frames, type Frame } from '@typegpu/kit';
import * as web from '@typegpu/kit/web';
import type { TgpuRoot } from 'typegpu';
import { bytesOf, colorsOf, type Rgb } from '../site/dither.ts';
import { imageOf, plate } from '../site/lockup.ts';
import { animation, fail, reducedMotion, type Animation } from '../viz/figure.ts';
import { onLost, pageGpu } from '../viz/gpu.ts';
import type { Veil } from '../site/veil.ts';
import { BLOCKS, handIn, highlight, type Part } from './code.ts';
import {
  clamp,
  cornersOf,
  DEG,
  DIAG,
  easeOut,
  evenly,
  fitted,
  mix,
  place,
  pointedIn,
  settle,
  smooth,
  spring,
  stacked,
  uvOn,
} from './geometry.ts';
import { createHome, type Drawn, type Home } from './home.ts';
import { hovering, knob, pointerOn } from './input.ts';
import type { RuleName } from './rules.ts';
import type { Camera, Kind, SheetState, StackView } from './stack.ts';
import { MODULES, type Removable } from './subject.ts';

/** How far a sheet can be dragged out of the stack, in world units; past half, its module goes. */
const OUT = DIAG * 0.34;
/** How far a sheet pointed at is pushed toward the middle, in world units. */
const PUSH = OUT * 0.25;
/** Seconds a sheet takes to dither in as its module comes into the frame, or out as it goes. */
const ARRIVAL = 0.45;
/** Seconds the rule plate takes to dither out, or back in with another rule. */
const SWAP = 0.22;

/** The way to the canvas's middle, in world x and z, from a stack at `anchor`, given the camera's right. */
const towardMiddle = (anchor: readonly [number, number], yaw: number) => {
  const side = anchor[0] < 0.45 ? 1 : -1;
  return [Math.cos(yaw) * side, -Math.sin(yaw) * side] as const;
};

/** The first section's plates, top first: the frame's code builds them from the bottom up. */
const ORDER: readonly Part[] = ['final', 'bloom', 'normals', 'lit', 'shadow', 'grid', 'birds'];

const REMOVABLE: Partial<Record<Part, Removable>> = {
  bloom: 'bloom',
  lit: 'lit',
  shadow: 'shadow',
  grid: 'grid',
};

/** Plates the pointer aims into the frame from, and the grid's, which aims at its own cells. */
const AIMS: Partial<Record<Part, 'frame' | 'grid'>> = {
  final: 'frame',
  bloom: 'frame',
  normals: 'frame',
  lit: 'frame',
  grid: 'grid',
};

/** How far apart the plates of a stack still building can spread, of the whole stack's gap. */
const SPREAD = 1.5;
/** CSS pixels of a block the panel scrolls past its faded edge, to show the block whole. */
const FADE = 30;

/** The rule section's plates, top first. */
const RULE_ORDER: readonly Kind[] = ['final', 'rule', 'grid', 'birds'];

const RULE_LABELS: Record<string, { module: string; holds: string; accent?: boolean }> = {
  final: { module: 'the frame', holds: 'lit and graded, as above' },
  rule: { module: 'your rule', holds: 'its force on each bird', accent: true },
  grid: { module: 'kit.grid', holds: 'the birds in each cell' },
  birds: { module: 'kit.table', holds: 'a row a bird' },
};

/** The stack's look: how much light an empty sheet has, its hairlines and its thickness. */
const LOOK = { base: 0.1, line: 0.6, thick: 0.035 };

/** The scroll, in screens, where the hero gives way, the frame is taken down, and built again. */
const TAKEN = 0.35;
const BUILD = 0.85;
/** Screens of scroll a module's step of the build takes. */
const STEP = 0.5;
/** Where the stack is whole and the reader's to tune. */
const PLAY = BUILD + STEP * MODULES.length;

/** How many of the frame's modules are built at `s` screens of scroll. */
const levelAt = (s: number) => {
  const all = MODULES.length;
  if (s < TAKEN) return all;
  if (s < BUILD) {
    return clamp(all - Math.floor(((s - TAKEN) / (BUILD - TAKEN)) * (all + 1)), 0, all);
  }
  if (s < PLAY) return clamp(Math.floor((s - BUILD) / STEP) + 1, 1, all);
  return all;
};

/** Screens of scroll over which the hero's lockup goes up into the header's mark. */
const HAND = [0.02, 0.3] as const;
/** How fast the stage follows the scroll, per second. */
const SCROLL_RATE = 18;
/** A tap's burst on a touch screen, of a click's. */
const TAP = 0.4;

/** The class on the root that lays the hero and the sections out for the GPU (HomeHero.astro). */
const STAGED = 'home-staged';

/** Reserve header space before logo cells arrive, then reveal the header mark and background. */
function headerOf() {
  const header = document.querySelector<HTMLElement>('.header');
  const bar = header?.querySelector<HTMLElement>('.bar');
  const mark = bar?.querySelector<HTMLElement>('.mark');
  return {
    mark,

    /** Moves the handoff to `s` screens of scroll and says how far it has gone, from 0 to 1. */
    show(s: number) {
      if (!header || !bar || !mark) return 1;
      const hand = smooth((s - HAND[0]) / (HAND[1] - HAND[0]));
      const gap = Number.parseFloat(getComputedStyle(bar).columnGap) || 0;
      // The sections make room before the first cells arrive
      const room = smooth(hand / 0.5);
      mark.style.marginRight = `${(-(mark.offsetWidth + gap) * (1 - room)).toFixed(2)}px`;
      mark.classList.toggle('handed', hand >= 0.999);
      document.documentElement.style.setProperty(
        '--home-bar',
        smooth((s - HAND[1]) / 0.15).toFixed(3),
      );
      return hand;
    },

    /** The bar without the GPU: its mark `handed` while the hero's lockup is off the screen. */
    settle(handed: boolean) {
      if (!mark) return;
      mark.style.marginRight = '';
      mark.classList.toggle('handed', handed);
      document.documentElement.style.removeProperty('--home-bar');
    },
  };
}

/** How a label beside the rule section's stack shows. */
interface LabelState {
  opacity: number;
  pointed: boolean;
  module: string;
  holds: string;
  accent?: boolean;
}

/** CSS pixels a label needs above and below its line, so two never overlap. */
const LABEL_GAP = 34;

/**
 * The labels right of a stack and the hairlines that lead to them, from each sheet's rightmost
 * corner, where the sheets rest.
 */
function labelsFor(stage: HTMLElement) {
  const list = stage.querySelector<HTMLOListElement>('.labels')!;
  const svg = stage.querySelector<SVGSVGElement>('.leaders')!;
  const items: {
    li: HTMLLIElement;
    module: HTMLElement;
    holds: HTMLSpanElement;
    line: SVGLineElement;
    dot: SVGCircleElement;
  }[] = [];

  const item = (i: number) => {
    while (items.length <= i) {
      const li = document.createElement('li');
      const module = document.createElement('b');
      const holds = document.createElement('span');
      li.append(module, holds);
      list.append(li);
      const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      const dot = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
      dot.setAttribute('r', '2');
      svg.append(line, dot);
      items.push({ li, module, holds, line, dot });
    }
    return items[i];
  };

  return {
    update(
      view: StackView,
      sheets: readonly SheetState[],
      resting: readonly SheetState[],
      state: (i: number) => LabelState,
      column: number,
    ) {
      const box = stage.getBoundingClientRect();
      const canvasBox = stage.querySelector('canvas')!.getBoundingClientRect();
      const dx = canvasBox.left - box.left;
      const dy = canvasBox.top - box.top;
      const endOf = (sheet: SheetState) =>
        cornersOf(view, sheet).reduce((a, b) => (b[0] > a[0] ? b : a));
      const ends = sheets.map(endOf);
      const rests = resting.map(endOf);
      svg.setAttribute('viewBox', `0 0 ${box.width} ${box.height}`);

      const placed: number[] = [];
      for (let i = 0; i < Math.max(items.length, sheets.length); i++) {
        const each = item(i);
        const s = i < sheets.length ? state(i) : undefined;
        const y = i < sheets.length ? rests[i][1] + dy : 0;
        const fits =
          s !== undefined &&
          s.opacity > 0.001 &&
          y >= 12 &&
          y <= box.height - 12 &&
          !placed.some((other) => Math.abs(other - y) < LABEL_GAP);
        if (!fits) {
          each.li.style.opacity = '0';
          each.line.style.opacity = '0';
          each.dot.style.opacity = '0';
          continue;
        }
        placed.push(y);
        if (each.module.textContent !== s.module) each.module.textContent = s.module;
        if (each.holds.textContent !== s.holds) each.holds.textContent = s.holds;
        each.li.classList.toggle('pointed', s.pointed);
        each.li.classList.toggle('accent', s.accent === true);
        each.line.classList.toggle('pointed', s.pointed);
        const value = s.opacity.toFixed(3);
        each.li.style.opacity = value;
        each.line.style.opacity = value;
        each.dot.style.opacity = value;
        const [x, sheetY] = ends[i];
        each.li.style.transform = `translate(${(column + dx).toFixed(1)}px, ${y.toFixed(1)}px)`;
        each.line.setAttribute('x1', (x + dx + 3).toFixed(1));
        each.line.setAttribute('y1', (sheetY + dy).toFixed(1));
        each.line.setAttribute('x2', (column + dx - 6).toFixed(1));
        each.line.setAttribute('y2', y.toFixed(1));
        each.dot.setAttribute('cx', (x + dx).toFixed(1));
        each.dot.setAttribute('cy', (sheetY + dy).toFixed(1));
      }
    },
  };
}

/** The hero's parts the runtime moves, when the page has the home hero. */
function heroOf() {
  const hero = document.querySelector<HTMLElement>('[data-home-hero]');
  const title = hero?.querySelector<HTMLElement>('.title');
  const lockup = hero?.querySelector<HTMLElement>('.lockup');
  const svg = lockup?.querySelector('svg');
  const printed = lockup?.querySelector<HTMLCanvasElement>('canvas.plate');
  const badge = hero?.querySelector<SVGSVGElement>('.badge svg') ?? undefined;
  if (!hero || !title || !lockup || !svg || !printed) return undefined;
  return { hero, title, lockup, svg, printed, badge };
}

type Hero = NonNullable<ReturnType<typeof heroOf>>;

/** The page's root. On `?profile`, `gpuDebug` times every pass first, in its panel. */
async function rootOf(): Promise<TgpuRoot | undefined> {
  if (new URLSearchParams(location.search).has('profile')) {
    const { gpuDebug } = await import('@typegpu/kit/debug');
    gpuDebug.install().start();
  }
  return (await pageGpu()).root;
}

/** A view of the stack the loop draws when its section asks. */
interface Viewing extends Drawn {
  readonly element: HTMLElement;
  /** Moves the view on by `dt` seconds and writes it, for the frame about to record. */
  step(dt: number): void;
}

/** What the sections share: the page's root, the frame and its stack, and the loop. */
interface Shared {
  readonly root: TgpuRoot;
  readonly home: Home;
  readonly hero: Hero | undefined;
  readonly header: ReturnType<typeof headerOf>;
  readonly still: boolean;
  readonly format: GPUTextureFormat;
  /** Runs `viewing` on the page's one animation frame, while its section is on screen. */
  run(viewing: Viewing): Animation;
  /** Draws every view again at the next animation frame, as when the frame changes. */
  redraw(): void;
}

/** Every section's animation, which stops when the page unstages. */
const animations = new Set<Animation>();

/** Whether the page is laid out for the GPU. */
const staged = () => document.documentElement.classList.contains(STAGED);

/**
 * The page as its markup lays it out, without the GPU: every animation stopped, the styles the
 * steps wrote cleared, the numbers plain and the hero's lockup printed as the header's.
 */
function unstage() {
  if (!staged()) return;
  document.documentElement.classList.remove(STAGED);
  for (const each of animations) each.destroy();
  animations.clear();
  for (const element of document.querySelectorAll<HTMLElement>(
    '[data-home-hero] [style], [data-home] [style]',
  )) {
    element.removeAttribute('style');
  }
  for (const element of document.querySelectorAll<HTMLElement>('[data-home] [data-pin]')) {
    delete element.dataset.pin;
  }
  for (const control of document.querySelectorAll<HTMLInputElement | HTMLButtonElement>(
    '[data-home="take-it-apart"] .mark, .home-code .knob input',
  )) {
    control.disabled = true;
  }
  const hero = heroOf();
  headerOf().settle(!hero || hero.lockup.getBoundingClientRect().bottom <= 0);
  if (!hero) return;
  hero.hero.classList.remove('live');
  hero.hero.removeAttribute('data-pinned');
  hero.title.classList.remove('tagged');
  void plate(hero.printed, hero.svg, { over: hero.title }).catch(() => {});
}

/** Start one shared loop. Each frame simulates the subject once, then draws requested views. */
async function setUp(): Promise<Shared | undefined> {
  const root = await rootOf();
  if (!root) {
    unstage();
    return undefined;
  }
  // As iOS drops the device of a page in the background
  onLost(unstage);

  const hero = heroOf();
  const image = hero ? await imageOf(hero.svg) : undefined;
  const badgeImage = hero?.badge ? await imageOf(hero.badge) : undefined;
  const reading = new OffscreenCanvas(1, 1).getContext('2d', { willReadFrequently: true })!;
  const readOf = (source: HTMLImageElement | undefined) => (columns: number, rows: number) => {
    reading.canvas.width = columns;
    reading.canvas.height = rows;
    reading.clearRect(0, 0, columns, rows);
    if (source) reading.drawImage(source, 0, 0, columns, rows);
    return reading.getImageData(0, 0, columns, rows).data;
  };
  const [rule, accent, bright] = bytesOf(document.body, [
    'var(--rule-strong)',
    'var(--accent)',
    'var(--accent-high)',
  ]) as [Rgb, Rgb, Rgb];
  const format = navigator.gpu.getPreferredCanvasFormat();
  const home = createHome(root, {
    format,
    colors: colorsOf(document.body),
    lockup: { rule, accent, bright },
    read: readOf(image),
    readBadge: badgeImage && readOf(badgeImage),
  });
  await home.initAsync();

  const still = reducedMotion();

  const wanted = new Map<Viewing, number>();
  let dtNow = 0;
  let queued = false;
  let time = 0;

  const loop = frames.create(root, (frame: Frame) => {
    home.record(frame, dtNow, [...wanted.keys()]);
  });

  const flush = () => {
    queued = false;
    let dt = 0;
    for (const [viewing, step] of wanted) {
      dt = Math.max(dt, step);
      viewing.step(step);
    }
    dtNow = Math.min(dt, 0.1);
    time += dtNow;
    try {
      loop.run(time);
    } catch (error) {
      for (const [viewing] of wanted) fail(viewing.element, error);
      unstage();
    }
    wanted.clear();
  };

  // Held still, the frame shows the flock a few seconds in, gathered
  if (still) {
    dtNow = 1 / 20;
    for (let i = 0; i < 100; i++) {
      time += dtNow;
      loop.run(time);
    }
    dtNow = 0;
  }

  return {
    root,
    home,
    hero,
    header: headerOf(),
    still,
    format,

    run(viewing) {
      let pending = 0;
      const running = animation(viewing.element, {
        step: (dt) => (pending += dt),
        draw() {
          wanted.set(viewing, pending);
          pending = 0;
          if (!queued) {
            queued = true;
            queueMicrotask(flush);
          }
        },
        playing: !still,
        onError(error) {
          fail(viewing.element, error);
          unstage();
        },
      });
      animations.add(running);
      return running;
    },

    redraw() {
      for (const each of animations) each.redraw();
    },
  };
}

/** A section's stage, its canvas and the view of the stack drawn on it. */
function stackIn(section: HTMLElement, shared: Shared) {
  const stage = section.querySelector<HTMLElement>('.stage')!;
  const canvas = stage.querySelector<HTMLCanvasElement>('canvas.stack')!;
  const surface = web.surface(shared.root, canvas, {
    alphaMode: 'premultiplied',
    maxPixelRatio: 2,
  });
  const view = shared.home.view(surface, () => clamp(canvas.clientWidth / 640, 2, 4));
  return { stage, canvas, surface, view };
}

/** Moves a value towards `to` over `dt` seconds at `rate` a second, or straight there when still. */
const easing = (still: boolean, dt: number) => (from: number, to: number, rate: number) =>
  still ? to : from + (to - from) * clock.ease(dt, rate);

function takeItApart(section: HTMLElement, shared: Shared, curtain: Veil | undefined) {
  const { home, hero, header, still } = shared;
  const { subject, ink } = home;
  const { stage, canvas, surface, view } = stackIn(section, shared);
  const left = stage.querySelector<HTMLElement>('.left')!;
  const intro = left.querySelector<HTMLElement>('.intro')!;
  const panel = left.querySelector<HTMLElement>('.panel')!;
  const badges = [...stage.querySelectorAll<HTMLElement>('.badges i')];
  const tag = stage.querySelector<HTMLElement>('.tag')!;
  /** The hero's text, which a plate's tag steps below rather than covering. */
  const heroText = hero
    ? [...hero.title.querySelectorAll<HTMLElement>('.by, h1, .tagline, .install, .actions')]
    : [];

  /** The height in the stage for a tag ending at `right` and centred at `y`, below any hero text it would cover. */
  const below = (right: number, y: number) => {
    const box = stage.getBoundingClientRect();
    const { offsetWidth: width, offsetHeight: height } = tag;
    let centre = y;
    for (const part of heroText) {
      const r = part.getBoundingClientRect();
      if (r.width === 0) continue;
      const top = box.top + centre - height / 2;
      const covers =
        r.right > box.left + right - width &&
        r.left < box.left + right &&
        r.bottom + 8 > top &&
        r.top - 8 < top + height;
      if (covers) centre = r.bottom - box.top + 12 + height / 2;
    }
    return centre;
  };
  const leaders = stage.querySelector<SVGSVGElement>('.leaders')!;
  const leader = leaders.querySelector<SVGPolylineElement>('polyline')!;
  const leaderDot = leaders.querySelector<SVGCircleElement>('circle')!;
  /** The code's blocks in the order the frame builds them, a module each. */
  const blockList = [...panel.querySelectorAll<HTMLElement>('.block')];
  const blocks = new Map(blockList.map((block) => [block.dataset.part as Part, block]));
  const sheets = ORDER.map((part) => {
    const sheet = stacked(part);
    sheet.open = still ? 1 : 0;
    return sheet;
  });
  /** How much room each sheet has made for itself in the stack, 0 to 1. */
  const rooms = sheets.map(() => 1);
  /** The module index of the sheet `i` from the top. */
  const moduleOf = (i: number) => ORDER.length - 1 - i;
  // The sheets placed to hit-test, to draw and to label, written again each frame
  const resting: SheetState[] = [];
  const lifted: SheetState[] = [];
  const marked: SheetState[] = [];
  let building = false;
  /** Where the panel scrolls to, to show a block, until it gets there or the reader scrolls it. */
  let follow: number | undefined;
  let followed: Part | undefined;
  const letGo = () => (follow = undefined);
  panel.addEventListener('wheel', letGo, { passive: true });
  panel.addEventListener('touchstart', letGo, { passive: true });
  const hover = hovering();
  const removable = (i: number) => (i >= 0 && REMOVABLE[ORDER[i]] && !building ? i : -1);
  const pick = (x: number, y: number) => removable(pointedIn(view, resting, x, y, hover.current));
  const pointer = pointerOn(stage, () => running.redraw(), {
    // A socket is put back with a click, not dragged
    grab: (x, y) => {
      const i = pick(x, y);
      return i >= 0 && sheets[i].kind !== 'socket' ? i : -1;
    },
    pick,
    limits: { yaw: 0.3, down: -0.25, up: 0.1 },
    skip: 'a, button, input, label, .left, .tag',
  });

  /** Takes the plate's module out of the frame, or puts it back, dithering the plate. */
  const toggle = (i: number) => {
    const sheet = sheets[i];
    const module = REMOVABLE[ORDER[i]];
    if (!sheet || !module || building) return;
    sheet.out = !sheet.out;
    subject.remove(module, sheet.out);
    showOut();
    shared.redraw();
  };

  /** The module names and struck lines, as the modules stand. */
  const showOut = () => {
    for (const module of ['bloom', 'lit', 'shadow', 'grid'] as const) {
      panel.classList.toggle(`out-${module}`, subject.isOut(module));
    }
    for (const [part, block] of blocks) {
      const module = REMOVABLE[part];
      const mark = block.querySelector<HTMLButtonElement>('.mark');
      if (!module || !mark) continue;
      const out = subject.isOut(module);
      mark.setAttribute('aria-pressed', String(out));
      block.classList.toggle('out', out);
    }
  };

  let blockHover: Part | undefined;
  let knobHeld: Part | undefined;
  for (const [part, block] of blocks) {
    block.addEventListener('pointerenter', () => {
      blockHover = part;
      running.redraw();
    });
    block.addEventListener('pointerleave', () => {
      if (blockHover === part) blockHover = undefined;
      running.redraw();
    });
    const mark = block.querySelector<HTMLButtonElement>('.mark');
    if (mark) {
      mark.disabled = false;
      mark.addEventListener('click', () => toggle(ORDER.indexOf(part)));
    }
  }

  let lastMove = 0;
  addEventListener('pointermove', (event) => {
    if (event.pointerType === 'touch') return;
    const box = canvas.getBoundingClientRect();
    const dt = lastMove ? (event.timeStamp - lastMove) / 1000 : 0;
    lastMove = event.timeStamp;
    ink.point(event.clientX - box.left, event.clientY - box.top, Math.max(dt, 1 / 240));
  });
  document.documentElement.addEventListener('pointerleave', () => {
    lastMove = 0;
    ink.leave();
  });
  const burstAt = (clientX: number, clientY: number, amount: number) => {
    const box = canvas.getBoundingClientRect();
    const x = clientX - box.left;
    const y = clientY - box.top;
    ink.burst(x, y, amount);
    if (ink.near(x, y, 40)) subject.startle();
    running.redraw();
  };
  // A touch bursts the lockup, more softly, once it lifts where it landed, so a scroll doesn't
  let touched: { x: number; y: number; at: number } | undefined;
  hero?.title.addEventListener('pointerdown', (event) => {
    if ((event.target as Element).closest('a, button, input, pre, code')) return;
    if (event.pointerType !== 'touch') burstAt(event.clientX, event.clientY, 1);
    else touched = { x: event.clientX, y: event.clientY, at: event.timeStamp };
  });
  hero?.title.addEventListener('pointerup', (event) => {
    if (!touched || event.pointerType !== 'touch') return;
    const moved = Math.hypot(event.clientX - touched.x, event.clientY - touched.y);
    if (moved < 10 && event.timeStamp - touched.at < 400) {
      burstAt(event.clientX, event.clientY, TAP);
    }
    touched = undefined;
  });
  hero?.title.addEventListener('pointercancel', () => (touched = undefined));

  let elapsed = still ? 10 : 0;
  let yaw = 0;
  let pitch = 0;
  let count = 0;
  let pullFrom = 0;
  let pulling = -1;
  let shownLockup = hero ? 1 : 0;

  /**
   * Screens scrolled into the section, held at its end. A screen is the stage's height, which
   * holds while Safari's toolbars come and go and `innerHeight` with them.
   */
  const scrolled = () => {
    const screen = Math.max(stage.clientHeight, 1);
    const box = section.getBoundingClientRect();
    return clamp(-box.top, 0, Math.max(box.height - screen, 1)) / screen;
  };
  let eased: number | undefined;
  let pace = 0;

  /** Holds the stage fixed while the section is scrolled through, and leaves it at its foot after. */
  const pin = () => {
    const box = section.getBoundingClientRect();
    const at = box.top > 0 ? '' : box.bottom > stage.clientHeight ? 'held' : 'past';
    if (stage.dataset.pin === at) return;
    stage.dataset.pin = at;
    hero?.hero.toggleAttribute('data-pinned', at !== '');
  };

  /** Fades the hero's title out over the first screens of scroll, and says how much shows. */
  const fadeTitle = (s: number) => {
    const shown = 1 - smooth((s - 0.04) / 0.26);
    if (hero) {
      hero.title.style.opacity = shown.toFixed(3);
      hero.title.style.visibility = shown < 0.01 ? 'hidden' : '';
    }
    return shown;
  };

  const OPEN_GAP = 0.74;

  const viewing: Viewing = {
    element: section,
    view,

    get count() {
      return count;
    },

    before(frame, dt) {
      if (shownLockup > 0.001) ink.step(frame, dt, shownLockup, surface, still);
    },

    after(pass) {
      if (shownLockup > 0.001) ink.draw(pass);
    },

    step(dt) {
      elapsed += dt;
      pin();
      // The scroll on a spring: Safari on a phone hands the page its scroll in uneven steps
      const scroll = scrolled();
      // Off screen no frame follows this one, so the spring would stop wherever a jump left it
      const seen = section.getBoundingClientRect();
      const away = seen.bottom <= 0 || seen.top >= innerHeight;
      if (still || away || eased === undefined) [eased, pace] = [scroll, 0];
      else [eased, pace] = spring(eased, pace, scroll, dt, SCROLL_RATE);
      if (Math.abs(eased - scroll) < 1e-4 && Math.abs(pace) < 1e-3) [eased, pace] = [scroll, 0];
      const s = eased;
      const width = canvas.clientWidth;
      const height = canvas.clientHeight;
      const aspect = width / Math.max(height, 1);
      const wide = width >= 900 && aspect >= 1.1;
      const ease = easing(still, dt);
      const open = smooth((s - 0.12) / (BUILD - 0.12));
      const level = levelAt(s);
      subject.built = level;
      building = s >= TAKEN && s < PLAY - 0.05;
      const playing = s >= PLAY - 0.05;
      const step = s >= BUILD && s < PLAY ? level - 1 : -1;
      const gap = mix(0.62, OPEN_GAP, open) * (wide ? 1 : 1.2);

      const turn = still ? 1 : settle((elapsed - 0.9) / 1.7);
      yaw = ease(yaw, pointer.yaw, 8);
      pitch = ease(pitch, pointer.pitch, 8);
      const baseYaw = (30 + 15 * turn) * DEG;
      const basePitch = 48 * DEG;
      const cameraYaw = baseYaw + yaw;
      const cameraPitch = basePitch + pitch;

      if (pointer.clicked >= 0) {
        if (!building) toggle(pointer.clicked);
        pointer.clicked = -1;
      }
      if (pointer.pulling !== pulling) {
        pulling = building ? -1 : pointer.pulling;
        if (pulling >= 0) pullFrom = sheets[pulling].pull;
      }

      const heroScale = wide
        ? Math.max(DIAG / (1.4 * aspect), ((DIAG / 2) * Math.sin(48 * DEG)) / 0.88)
        : 0;
      const columnRight = wide ? left.offsetLeft + left.offsetWidth : 0;
      const bar = header.mark?.closest<HTMLElement>('.header')?.offsetHeight ?? 0;
      const box: [number, number, number, number] = wide
        ? [columnRight + 64, Math.max(height * 0.05, bar + 12), width - 24, height * 0.97]
        : [width * 0.04, height * 0.03, width * 0.96, height * 0.46];
      const fit = fitted(
        baseYaw,
        basePitch,
        evenly(ORDER.length, gap),
        gap * 0.34,
        box,
        width,
        height,
      );
      const heroFit = wide
        ? { scale: heroScale, anchor: [0.655, 0.5] as [number, number] }
        : fitted(
            baseYaw,
            basePitch,
            evenly(ORDER.length, gap),
            0,
            [0, 0, width, height * 0.62],
            width,
            height,
          );
      const scale = mix(heroFit.scale, fit.scale, open);
      const anchor: [number, number] = [
        mix(heroFit.anchor[0], fit.anchor[0], open),
        mix(heroFit.anchor[1], fit.anchor[1], open),
      ];
      const unitsPerPixel = (scale * 2) / Math.max(height, 1);
      const toward = towardMiddle(anchor, cameraYaw);
      // Dragged toward the middle in screen x, whichever side that is
      const drag = toward[0] * Math.cos(cameraYaw) - toward[1] * Math.sin(cameraYaw);

      if (pointer.released >= 0) {
        const sheet = sheets[pointer.released];
        const wasOut = sheet.out;
        sheet.out = sheet.pull > OUT * 0.45;
        pointer.released = -1;
        if (wasOut !== sheet.out) showOut();
      }
      for (const [i, sheet] of sheets.entries()) {
        if (!still) sheet.open = easeOut((elapsed - 0.6 - i * 0.1) / 0.9);
        const built = i === 0 || moduleOf(i) < level;
        // A plate comes once its room has opened, and its room closes once it has gone
        const room = built || sheet.presence > 0.3 ? 1 : 0;
        rooms[i] = still ? room : rooms[i] + clamp(room - rooms[i], -dt / ARRIVAL, dt / ARRIVAL);
        sheet.weight = smooth(rooms[i]);
        const there = built && !sheet.out && rooms[i] > 0.4 ? 1 : 0;
        sheet.presence = still
          ? there
          : sheet.presence + clamp(there - sheet.presence, -dt / ARRIVAL, dt / ARRIVAL);
        sheet.kind = sheet.presence > 0.02 ? ORDER[i] : 'socket';
        sheet.rim = ease(sheet.rim, built ? 1 : 0, 8);
        if (i === pulling) {
          sheet.pull = clamp(pullFrom + pointer.pulled * drag * unitsPerPixel, 0, OUT);
        } else {
          sheet.pull = ease(sheet.pull, 0, 9);
        }
        const module = REMOVABLE[ORDER[i]];
        if (module) {
          const removed = i === pulling ? sheet.pull > OUT * 0.45 : sheet.out;
          if (removed !== subject.isOut(module)) {
            subject.remove(module, removed);
            showOut();
          }
          sheet.keeps = ease(sheet.keeps, removed ? 0.5 : 1, 8);
        }
      }

      // The plates built so far spread out, centred where the whole stack will be
      let whole = 0;
      let has = 0;
      for (let i = 1; i < sheets.length; i++) {
        const sheet = sheets[i];
        whole += sheet.open;
        has += ((sheets[i - 1].weight + sheet.weight) / 2) * sheet.open;
      }
      const spread = Math.min((gap * whole) / Math.max(has, 1e-3), gap * SPREAD);
      const top = -(gap * whole - spread * has) / 2;
      place(sheets, spread, toward, 'resting', top, resting);
      const camera: Camera = {
        yaw: cameraYaw,
        pitch: cameraPitch,
        scale,
        target: [0, 0, 0],
        anchor,
      };
      view.aim(camera);

      // Hit-test resting plates so lifting a plate cannot move it out from under the pointer
      const fromCode = playing ? (knobHeld ?? blockHover) : undefined;
      const raw =
        pointer.over && !pointer.dragging && !building && !fromCode
          ? pointedIn(view, resting, pointer.x, pointer.y, hover.current)
          : -1;
      const hovered = hover.step(raw, dt, still || !pointer.over || fromCode !== undefined);
      const pointed =
        pulling >= 0
          ? pulling
          : fromCode
            ? ORDER.indexOf(fromCode)
            : hovered >= 0
              ? hovered
              : step >= 0
                ? ORDER.length - 1 - step
                : -1;
      for (const [i, sheet] of sheets.entries()) {
        const on = i === pointed ? 1 : 0;
        sheet.hot = ease(sheet.hot, on, 14);
        sheet.lift = ease(sheet.lift, on && step < 0 ? PUSH : 0, 12);
      }
      stage.style.cursor = pointer.dragging
        ? 'grabbing'
        : pulling >= 0
          ? 'ew-resize'
          : pointed >= 0 && hovered === pointed && REMOVABLE[ORDER[pointed]]
            ? 'pointer'
            : 'grab';
      place(sheets, spread, toward, 'drawn', top, lifted);
      place(sheets, spread, toward, 'labelled', top, marked);
      count = view.write(camera, lifted, LOOK);

      const at = hovered >= 0 && pointer.over ? AIMS[ORDER[hovered]] : undefined;
      const uv = at ? uvOn(view, lifted[hovered], pointer.x, pointer.y) : undefined;
      if (at === 'grid') subject.aimFlat(uv);
      else subject.aim(uv);

      const canvasBox = canvas.getBoundingClientRect();
      const hand = header.show(s);
      if (hero) {
        const titleShown = fadeTitle(s);
        shownLockup = hand < 0.999 ? 1 : 0;
        const markBox = header.mark?.getBoundingClientRect();
        if (markBox) ink.hand(markBox.left - canvasBox.left, markBox.top - canvasBox.top, hand);
        const lockup = hero.lockup.getBoundingClientRect();
        const badge = hero.badge?.getBoundingClientRect();
        ink.layout(
          [lockup.left - canvasBox.left, lockup.top - canvasBox.top, lockup.width, lockup.height],
          badge && [
            badge.left - canvasBox.left,
            badge.top - canvasBox.top,
            badge.width,
            badge.height,
          ],
        );
        ink.fade(titleShown);
      }

      const headShown = smooth((s - TAKEN) / 0.3);
      left.style.opacity = headShown.toFixed(3);
      left.style.visibility = headShown < 0.01 ? 'hidden' : 'visible';
      intro.dataset.phase = playing ? 'play' : 'build';
      // The panel builds up a block a step
      const panelShown = smooth((s - (BUILD - 0.1)) / 0.1);
      panel.style.opacity = panelShown.toFixed(3);
      panel.style.visibility = panelShown < 0.01 ? 'hidden' : 'visible';
      panel.classList.toggle('live', playing && panelShown > 0.6);
      const listed = playing ? blockList.length : s >= BUILD ? level : 0;
      for (let k = 0; k < blockList.length; k++) {
        blockList[k].classList.toggle('pending', k >= listed);
        blockList[k].classList.toggle('telling', k === step);
      }

      const stageBox = stage.getBoundingClientRect();
      const dx = canvasBox.left - stageBox.left;
      const dy = canvasBox.top - stageBox.top;
      /** A plate's leftmost corner, where its badge and leader meet it, at rest or pushed. */
      const cornerOf = (i: number) =>
        cornersOf(view, marked[i]).reduce((a, b) => (b[0] < a[0] ? b : a));

      for (const [i, badge] of badges.entries()) {
        const sheet = i < ORDER.length ? marked[i] : undefined;
        // A place without room has no number
        const badgeShown = sheet ? open * sheets[i].weight : 0;
        badge.style.opacity = badgeShown.toFixed(3);
        if (!sheet || badgeShown < 0.01) continue;
        const [x, y] = cornerOf(i);
        badge.style.transform = `translate(${(x + dx).toFixed(1)}px, ${(y + dy).toFixed(1)}px)`;
        badge.classList.toggle('empty', sheet.kind === 'socket');
        badge.classList.toggle('pointed', i === pointed);
      }

      const part = pointed >= 0 ? ORDER[pointed] : undefined;
      const aimed = hovered >= 0 && hovered === pointed;
      for (const [each, block] of blocks) {
        block.classList.toggle('pointed', each === part);
        block.classList.toggle('aimed', aimed && each === part);
      }

      // The panel keeps up with the block the build adds, and shows the one pointed at on the stack
      const wanted = playing && !fromCode ? part : undefined;
      if (step >= 0) {
        const block = blockList[step];
        follow =
          block.offsetHeight > panel.clientHeight ? block.offsetTop - FADE : panel.scrollHeight;
        followed = undefined;
      } else if (wanted !== followed) {
        followed = wanted;
        const block = wanted && blocks.get(wanted);
        if (block) {
          const { offsetTop: from, offsetHeight: tall } = block;
          const { scrollTop: shownFrom, clientHeight: room } = panel;
          if (from < shownFrom || tall > room) follow = from - FADE;
          else if (from + tall > shownFrom + room) follow = from + tall - room + FADE;
        }
      }
      if (follow !== undefined) {
        const to = clamp(follow, 0, Math.max(panel.scrollHeight - panel.clientHeight, 0));
        const next = ease(panel.scrollTop, to, 10);
        const there = Math.abs(next - to) < 0.5;
        panel.scrollTop = there ? to : next;
        if (there) follow = undefined;
      }
      const scrolledBy = panel.scrollTop;
      panel.classList.toggle('above', scrolledBy > 2);
      panel.classList.toggle('below', scrolledBy + panel.clientHeight < panel.scrollHeight - 2);

      const from =
        wide && part && (step >= 0 || playing) ? blocks.get(part)?.querySelector('header') : null;
      leaders.setAttribute('viewBox', `0 0 ${stageBox.width} ${stageBox.height}`);
      const head = from ? from.getBoundingClientRect() : undefined;
      const panelBox = panel.getBoundingClientRect();
      // A header scrolled out of the panel has no leader
      const led =
        head !== undefined &&
        pointed >= 0 &&
        head.top >= panelBox.top - 2 &&
        head.bottom <= panelBox.bottom + 2;
      if (led) {
        const badge = badges[pointed].getBoundingClientRect();
        const [x, y] = cornerOf(pointed);
        const x1 = (head.right - stageBox.left + 8).toFixed(1);
        const y1 = (head.top + head.height / 2 - stageBox.top).toFixed(1);
        const x2 = badge.left - stageBox.left - 4;
        const y2 = (y + dy).toFixed(1);
        const bend = (x2 - 14).toFixed(1);
        leader.setAttribute(
          'points',
          `${x1},${y1} ${bend},${y1} ${bend},${y2} ${x2.toFixed(1)},${y2}`,
        );
        leaderDot.setAttribute('cx', (x + dx).toFixed(1));
        leaderDot.setAttribute('cy', y2);
      }
      leaders.style.opacity = led ? '1' : '0';

      const tagged = hovered >= 0 && hovered === pointed ? hovered : -1;
      if (tagged >= 0) {
        const block = BLOCKS.find((each) => each.part === ORDER[tagged]);
        const module = REMOVABLE[ORDER[tagged]];
        const isOut = module !== undefined && subject.isOut(module);
        tag.querySelector('b')!.textContent = block?.module ?? '';
        tag.querySelector('span')!.textContent = isOut
          ? 'out of the frame · click to put it back'
          : `${block?.says ?? ''}${module ? ' · click to take it out' : ''}`;
        const [x, y] = cornerOf(tagged);
        tag.style.transform = `translate(${(x - 16).toFixed(1)}px, ${below(x - 16, y).toFixed(1)}px)`;
        tag.classList.toggle('out', isOut);
      }
      const tagShown = tagged >= 0 && wide && open < 0.3;
      tag.style.opacity = tagShown ? '1' : '0';
      hero?.title.classList.toggle('tagged', tagShown);
    },
  };

  for (const element of section.querySelectorAll<HTMLElement>('.knob')) {
    knob(
      element,
      (key, value) => {
        subject.knob(key, value);
        shared.redraw();
      },
      (_key, on) => {
        const part = element.closest<HTMLElement>('.block')?.dataset.part as Part | undefined;
        knobHeld = on ? part : undefined;
        running.redraw();
      },
    );
  }
  showOut();

  const running: Animation = shared.run(viewing);
  addEventListener(
    'scroll',
    () => {
      if (!staged()) return;
      pin();
      // On screen, the frame moves the header with the eased scroll
      const box = section.getBoundingClientRect();
      // Off screen, the frame doesn't run: the header and the title follow the scroll itself
      if (box.bottom <= 0 || box.top >= innerHeight) {
        header.show(scrolled());
        fadeTitle(scrolled());
      }
      running.redraw();
    },
    { passive: true },
  );
  pin();
  header.show(scrolled());
  fadeTitle(scrolled());
  section.classList.add('live');
  // The veil lifts once the stack has drawn, then the hero's ink takes over from its SVGs
  requestAnimationFrame(() =>
    requestAnimationFrame(() => {
      void (curtain?.lift() ?? Promise.resolve()).then(() => hero?.hero.classList.add('live'));
    }),
  );
}

/** The rule section: the table, the grid, the reader's rule and the frame. */
function yourRule(section: HTMLElement, shared: Shared) {
  const { home, still } = shared;
  const { subject } = home;
  const { stage, canvas, view } = stackIn(section, shared);
  const side = stage.querySelector<HTMLElement>('.side')!;
  const handed = stage.querySelector<HTMLElement>('.hooked code')!;
  const chips = [...stage.querySelectorAll<HTMLButtonElement>('.rules button')];
  const bodies = [...stage.querySelectorAll<HTMLElement>('pre[data-rule]')];
  const sheets = RULE_ORDER.map((kind) => stacked(kind));
  // Keep the frame above the rule plate so it does not obscure it
  sheets[0].weight = 6.6;
  // The sheets placed to hit-test and to draw, and their heights, written again each frame
  const resting: SheetState[] = [];
  const placed: SheetState[] = [];
  const heights = sheets.map(() => 0);
  const hover = hovering();
  const pointer = pointerOn(stage, () => running.redraw(), {
    limits: { yaw: 0.3, down: -0.2, up: 0.15 },
  });
  const labels = labelsFor(stage);
  let shownRule = '';
  // A rule chosen dithers the rule plate out, takes it, and dithers the plate back in
  const ruleAt = RULE_ORDER.indexOf('rule');
  let pending = subject.rule;
  let swapped = 1;

  const choose = (name: RuleName) => {
    pending = name;
    shared.redraw();
  };
  for (const chip of chips) {
    chip.addEventListener('click', () => choose(chip.dataset.rule as RuleName));
  }

  let yaw = 0;
  let pitch = 0;
  let count = 0;
  let elapsed = 0;

  const viewing: Viewing = {
    element: section,
    view,

    get count() {
      return count;
    },

    step(dt) {
      const ease = easing(still, dt);
      const width = canvas.clientWidth;
      const height = canvas.clientHeight;
      const wide = width >= 900 && width / Math.max(height, 1) >= 1.1;

      const raw =
        pointer.over && !pointer.dragging
          ? pointedIn(view, resting, pointer.x, pointer.y, hover.current)
          : -1;
      const hovered = hover.step(raw, dt, still);
      elapsed += dt;
      for (const [i, sheet] of sheets.entries()) {
        const on = i === hovered ? 1 : 0;
        sheet.hot = ease(sheet.hot, on, 14);
        sheet.lift = ease(sheet.lift, on * PUSH, 12);
      }
      const swapping = pending !== subject.rule;
      if (still) subject.rule = pending;
      else {
        swapped = clamp(swapped + (swapping ? -dt : dt) / SWAP);
        if (swapping && swapped === 0) subject.rule = pending;
      }
      for (const [i, sheet] of sheets.entries()) {
        // The plates dither in from the bottom up the first time the section shows
        const shown = still ? 1 : clamp((elapsed - 0.1 - (sheets.length - 1 - i) * 0.12) / ARRIVAL);
        sheet.presence = i === ruleAt ? Math.min(shown, swapped) : shown;
      }
      stage.style.cursor = pointer.dragging ? 'grabbing' : 'grab';

      const gap = 0.95;
      const baseYaw = 45 * DEG;
      const basePitch = 42 * DEG;
      yaw = ease(yaw, pointer.yaw, 8);
      pitch = ease(pitch, pointer.pitch, 8);
      const yawNow = baseYaw + yaw;
      const pitchNow = basePitch + pitch;
      const column = wide ? side.offsetLeft - 230 : width;
      const box: [number, number, number, number] = wide
        ? [width * 0.03, height * 0.06, column - 48, height * 0.95]
        : [width * 0.05, height * 0.04, width * 0.95, height * 0.96];
      place(sheets, gap, [0, 0], 'resting', 0, resting);
      for (let i = 0; i < sheets.length; i++) heights[i] = resting[i].y;
      const fit = fitted(baseYaw, basePitch, heights, 0.3, box, width, height);
      const toward = towardMiddle(fit.anchor, yawNow);
      place(sheets, gap, toward, 'resting', 0, resting);
      place(sheets, gap, toward, 'drawn', 0, placed);
      count = view.write(
        { yaw: yawNow, pitch: pitchNow, scale: fit.scale, target: [0, 0, 0], anchor: fit.anchor },
        placed,
        LOOK,
      );

      const kind = hovered >= 0 && pointer.over ? RULE_ORDER[hovered] : undefined;
      const uv = kind ? uvOn(view, placed[hovered], pointer.x, pointer.y) : undefined;
      if (kind === 'final') subject.aim(uv);
      else if (kind === 'rule' || kind === 'grid') subject.aimFlat(uv);
      else subject.aim(undefined);

      labels.update(
        view,
        placed,
        resting,
        (i) => {
          const label = RULE_LABELS[RULE_ORDER[i]];
          return {
            opacity: wide ? 1 : 0,
            pointed: sheets[i].hot > 0.5 && i === hovered,
            module: label.module,
            holds: label.holds,
            accent: label.accent === true,
          };
        },
        column,
      );

      if (subject.rule !== shownRule) {
        shownRule = subject.rule;
        for (const body of bodies) body.hidden = body.dataset.rule !== shownRule;
        for (const chip of chips) {
          chip.setAttribute('aria-pressed', String(chip.dataset.rule === shownRule));
        }
        handed.innerHTML = `<span class="line hot new">${highlight(handIn(subject.rule))}</span>`;
      }
    },
  };

  for (const element of section.querySelectorAll<HTMLElement>('.knob')) {
    knob(
      element,
      (key, value) => {
        subject.knob(key, value);
        shared.redraw();
      },
      () => running.redraw(),
    );
  }

  const running: Animation = shared.run(viewing);
  section.classList.add('live');
}

let shared: Promise<Shared | undefined> | undefined;

/** Register a section with the shared runtime. It keeps its static markup if GPU setup fails. */
export function register(
  part: 'take-it-apart' | 'your-rule',
  section: HTMLElement,
  curtain?: Veil,
) {
  shared ??= setUp().catch((error: unknown) => {
    fail(section, error);
    unstage();
    return undefined;
  });
  void shared
    .then((ready) => {
      if (!ready) return;
      if (part === 'take-it-apart') takeItApart(section, ready, curtain);
      else yourRule(section, ready);
    })
    .catch((error: unknown) => {
      fail(section, error);
      unstage();
    });
}
