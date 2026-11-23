import type { Hosted, Stage } from '@typegpu/kit/web';
import type { ExampleScene } from 'typegpu-kit-examples/contract';
import { WARMUP_FRAMES } from 'typegpu-kit-examples/hosts';
import { posterOf } from './posters.ts';
import { veil, type Veil } from './veil.ts';

/** The most cards that hold a scene at once, unless more are on screen: fewer on a phone. */
const MAX_LIVE = matchMedia('(pointer: coarse)').matches ? 3 : 6;

/** Milliseconds a card holds the queue at most, so one slow to load doesn't hold the cards after it. */
const HOLD = 2500;

/** Radians a second that a card's orbit view turns by itself. */
const DRIFT = 0.07;

/** A card's scene, from when it starts until the card lets it go. */
interface Hosting {
  readonly kind: 'hosting';
  readonly hosted: Hosted<ExampleScene>;
  /** The scene once it is up, whose frames the page's animation frame runs. */
  stage: Stage<ExampleScene> | undefined;
  drawn: number;
  clock: number;
}

/** Where a card is: waiting to be asked for or to near the screen, queued, hosting, or failed. */
type Phase =
  | { readonly kind: 'idle' }
  | { readonly kind: 'queued' }
  | Hosting
  | { readonly kind: 'failed' };

interface Card {
  readonly element: HTMLElement;
  readonly canvas: HTMLCanvasElement;
  readonly wait: HTMLElement;
  readonly curtain: Veil;
  /** Builds only when its run button is pressed, one card at a time. */
  readonly asking: boolean;
  /** Over the poster of an asking card until it is pressed. */
  readonly run: HTMLButtonElement;
  readonly key: string;
  readonly options: object;
  phase: Phase;
  /** Always for a card that builds near the screen; for an asking one, from a press until it lets its scene go. */
  asked: boolean;
  near: boolean;
  visible: boolean;
}

const still = matchMedia('(prefers-reduced-motion: reduce)').matches;
const cards: Card[] = [];

/** The asking card that was pressed last, the page's one live preview among them. */
let pressed: Card | undefined;

/** Puts a hidden button (viz.css) reading "Preview" over `screen`, naming `name` to screen readers. */
function runButton(screen: HTMLElement, name: string): HTMLButtonElement {
  const button = document.createElement('button');
  button.type = 'button';
  button.className = 'viz-run';
  button.hidden = true;
  const named = document.createElement('span');
  named.className = 'sr-only';
  named.textContent = ` ${name}`;
  const label = document.createElement('span');
  label.append('Preview', named);
  button.append(label);
  screen.append(button);
  return button;
}

/** Milliseconds without scrolling before a card builds: building blocks the page for a moment. */
const QUIET = 150;
let scrolled = 0;
addEventListener('scroll', () => (scrolled = performance.now()), { passive: true });
const quiet = async () => {
  for (let wait = QUIET; wait > 0; wait = scrolled + QUIET - performance.now()) {
    await new Promise((resume) => setTimeout(resume, wait));
  }
};

const say = (card: Card, text: string) => {
  card.wait.textContent = text;
};

const cardOf = (entry: IntersectionObserverEntry) =>
  cards.find((card) => card.element === entry.target || card.canvas === entry.target);

/** What the cards share once the page has WebGPU: its root, the kit and the examples' scenes. */
async function load() {
  if (!navigator.gpu) return undefined;

  // Imported here, so a browser without WebGPU loads none of it
  const [{ pageRoot, onLost }, { host }] = await Promise.all([
    import('../viz/gpu.ts'),
    import('../../examples/host.ts'),
  ]);
  const root = await pageRoot();
  if (!root) return undefined;

  if (!watching) {
    watching = true;
    onLost(recover);
  }
  return { root, host };
}

let watching = false;

type Shared = NonNullable<Awaited<ReturnType<typeof load>>>;

let shared: Promise<Shared | undefined> | undefined;

/** Lets every card go with the lost device, then builds those near the screen on a new one once the page shows. */
function recover() {
  for (const card of cards) {
    release(card);
    say(card, '');
    if (!card.asked) card.run.hidden = false;
  }
  shared = undefined;
  const rebuild = () => {
    if (document.hidden) return;
    document.removeEventListener('visibilitychange', rebuild);
    shared = load();
    for (const card of cards) if (card.near) void enqueue(card);
  };
  document.addEventListener('visibilitychange', rebuild);
  rebuild();
}

/** Lets the card's scene go, and puts the card back as it was before it neared the screen. */
function release(card: Card) {
  if (card.phase.kind === 'hosting') card.phase.hosted.destroy();
  card.phase = { kind: 'idle' };
  card.element.classList.remove('live');
}

/** Lets the card's scene go and puts it back to its poster, to be pressed again. */
function letGo(card: Card) {
  if (card.phase.kind === 'failed') return;
  release(card);
  if (pressed === card) pressed = undefined;
  card.asked = false;
  card.curtain.rest();
  card.run.hidden = false;
  say(card, '');
}

/** Builds a pressed card, letting the one live before it go. */
function ask(card: Card) {
  card.asked = true;
  card.run.hidden = true;
  card.curtain.rest(false);
  const before = pressed;
  pressed = card;
  if (before && before !== card) letGo(before);
  shared ??= load();
  void shared.then((page) => {
    if (!page) unsupported(cards);
  });
  void enqueue(card);
}

function unsupported(among: readonly Card[]) {
  for (const card of among) {
    card.phase = { kind: 'failed' };
    card.curtain.rest();
    card.run.hidden = true;
    say(card, 'This browser has no WebGPU.');
  }
}

function fail(card: Card, error: unknown) {
  if (card.phase.kind === 'failed') return;
  release(card);
  card.phase = { kind: 'failed' };
  say(card, 'The example stopped. See the console for the error.');
  console.warn(`The ${card.key} card stopped:`, error);
}

/**
 * Lets the scenes off screen go, the farthest first, until at most `MAX_LIVE` cards hold one,
 * `keep` among them.
 */
function cap(keep: Card) {
  const middle = innerHeight / 2;
  const distance = (card: Card) => {
    const { top, bottom } = card.element.getBoundingClientRect();
    return Math.abs((top + bottom) / 2 - middle);
  };

  const hosting = cards.filter((card) => card.phase.kind === 'hosting');
  let live = hosting.length;
  const off = hosting.filter((card) => !card.visible && card !== keep);
  off.sort((a, b) => distance(b) - distance(a));

  for (const card of off) {
    if (live <= MAX_LIVE) return;
    release(card);
    live -= 1;
  }
}

/** Host on the shared root without a schedule; `tick` runs frames. Device errors stop every card because WebGPU does not identify the source scene. */
async function build(card: Card, page: Shared) {
  const { root, host } = page;
  await quiet();
  if (card.phase.kind !== 'queued') return;
  // Queued before the device was lost, or gone from near the screen. An asking card goes back to
  // its poster, to be pressed again
  if (page !== (await shared) || !card.near) {
    if (card.asking) letGo(card);
    else card.phase = { kind: 'idle' };
    return;
  }

  const hosted: Hosted<ExampleScene> = host(card.key, card.canvas, {
    root,
    options: card.options,
    input: false,
    schedule: false,
    maxPixelRatio: 1.25,
    say: (text) => card.phase === hosting && say(card, text),
    onError: (error) => card.phase === hosting && fail(card, error),

    attach: (stage) => {
      hosting.stage = stage;
      wake();
    },

    onFrame: ({ scene, ready }, frame) => {
      const goal = scene.view?.goal as { yaw?: number } | undefined;
      if (!still && goal?.yaw !== undefined) goal.yaw += frame.dt * DRIFT;
      if (!ready) return;

      hosting.drawn += 1;
      if (hosting.drawn === WARMUP_FRAMES) card.element.classList.add('live');
    },
  });
  const hosting: Hosting = { kind: 'hosting', hosted, stage: undefined, drawn: 0, clock: 0 };
  card.phase = hosting;
  cap(card);

  // Rejects for a card let go before its scene was up; an error of its own went to `fail`
  await Promise.race([
    hosted.run(() => {}).catch(() => {}),
    new Promise((resume) => setTimeout(resume, HOLD)),
  ]);
}

let queue = Promise.resolve();

/** Builds the card after every card queued before it: WebKit compiles pipelines serially. */
async function enqueue(card: Card) {
  if (!card.asked || card.phase.kind !== 'idle') return;
  card.phase = { kind: 'queued' };

  const page = await shared;
  if (!page) return;
  queue = queue.then(() => build(card, page).catch((error: unknown) => fail(card, error)));
}

/**
 * Whether a card's scene is up and draws this frame: until it is ready, as it may still be
 * compiling inside its frames, then while on screen, or under reduced motion until it settles.
 */
function draws(
  phase: Phase,
  visible: boolean,
): phase is Hosting & { readonly stage: Stage<ExampleScene> } {
  if (phase.kind !== 'hosting' || !phase.stage) return false;
  const { stage } = phase;
  const resting = still ? phase.drawn >= WARMUP_FRAMES && !stage.scene.settling : !visible;
  return !stage.ready || !resting;
}

let request = 0;
let last = 0;

/** Asks for an animation frame while some card draws, and only then. */
function wake() {
  if (request !== 0 || !cards.some((card) => draws(card.phase, card.visible))) return;
  last = performance.now();
  request = requestAnimationFrame(tick);
}

/**
 * One animation frame drives every card; each records into its own frames and submits once. A
 * card's clock moves only while it draws, so it resumes where it paused.
 */
function tick(now: number) {
  request = 0;
  const dt = Math.min(Math.max(now - last, 0) / 1000, 1 / 20);
  last = now;

  for (const card of cards) {
    const { phase } = card;
    if (!draws(phase, card.visible)) continue;
    phase.clock += dt;
    try {
      phase.stage.loop.run(phase.clock);
    } catch (error) {
      fail(card, error);
    }
  }

  wake();
}

const near = new IntersectionObserver(
  (entries) => {
    for (const entry of entries) {
      const card = cardOf(entry);
      if (!card) continue;
      card.near = entry.isIntersecting;
      if (card.near) void enqueue(card);
    }
  },
  { rootMargin: '240px 0px' },
);

const far = new IntersectionObserver(
  (entries) => {
    for (const entry of entries) {
      const card = cardOf(entry);
      if (!card || entry.isIntersecting || card.phase.kind === 'failed') continue;
      if (!card.asking) release(card);
      else if (card.asked) letGo(card);
    }
  },
  { rootMargin: '200% 0px' },
);

const onScreen = new IntersectionObserver((entries) => {
  for (const entry of entries) {
    const card = cardOf(entry);
    if (!card) continue;
    card.visible = entry.isIntersecting;
    // A card `cap` let go while it was near the screen builds again as it comes into view
    if (card.visible) {
      void enqueue(card);
      wake();
    }
  }
});

/**
 * Makes the cards of a `LiveGrid` live, sharing the page's root, queue and frames with the rest.
 * Cards build as they near the screen. On a phone, or in a grid marked `data-ask`, a card builds
 * only when its button is pressed: building blocks the page for a moment, up to seconds where
 * shaders compile slowly, as on Windows.
 */
export function mount(grid: HTMLElement) {
  const phone = matchMedia('(pointer: coarse) and (hover: none)').matches;
  const asking = Boolean(navigator.gpu) && (phone || grid.dataset.ask !== undefined);
  if (!asking) shared ??= load();

  const mounted: Card[] = [];
  for (const element of grid.querySelectorAll<HTMLElement>('[data-live]')) {
    element.classList.toggle('still', still);
    const screen = element.querySelector<HTMLElement>('.screen') ?? element;
    const curtain = veil(screen, { owner: element, poster: posterOf(element.dataset.live) });
    const card: Card = {
      element,
      canvas: element.querySelector('canvas')!,
      wait: element.querySelector<HTMLElement>('[data-wait]')!,
      curtain,
      asking,
      run: runButton(screen, element.querySelector('.title')?.textContent ?? element.dataset.live!),
      key: element.dataset.live!,
      options: JSON.parse(element.dataset.options ?? '{}') as object,
      phase: { kind: 'idle' },
      asked: !asking,
      near: false,
      visible: false,
    };
    cards.push(card);
    mounted.push(card);
    if (asking) {
      curtain.rest();
      card.run.hidden = false;
      card.run.addEventListener('click', () => ask(card));
    }
    near.observe(element);
    far.observe(element);
    // The canvas, not the card: a card whose caption alone shows needn't draw
    onScreen.observe(card.canvas);
  }

  void shared?.then((page) => {
    if (!page) unsupported(mounted);
  });
}
