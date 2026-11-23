// The album over the dimmed page, only where WebGPU runs. The binder rises from below, open at the
// page of a card just found, and the card flies from where its pull left it into its pocket. A
// press on a pocket takes its card out and up to the middle, face up if found, else face down with
// hints toward its page; a press beside it or Back puts it back. The card in flight is the pull's
// card, laid over the pocket's corners as the camera sees them, so it fades into and out of the
// binder's own drawing of it in place.

import { rooted } from '../site/urls.ts';
import { createCard, fillCard, fitCard, onLeave, printPoster } from '../pull/card.ts';
import { loadEntries, rarityOf, stillMotion, type Entry } from '../pull/pulls.ts';
import { clamp01, inOutCubic } from 'typegpu-kit-pull/timeline.ts';
import { foundCards, onAlbum } from './album.ts';
import type { Binder } from './binder.ts';
import { between, boxOf, onto, type Quad } from './flight.ts';
import { HIDDEN } from './hidden.ts';
import './album.css';

/** Height under the card out for its row of actions and its note. */
const ROW = 132;
/** Milliseconds a card takes out of its pocket and back in, and to fade into the binder's own. */
const OUT = 720;
const IN = 640;
const FADE = 160;

/** The card a button among the keys stands for, from an event on it. */
const keyOf = (event: Event) => (event.target as HTMLElement).dataset.key;

/** The card found last, if any. */
const newest = () => Object.entries(foundCards()).toSorted(([, a], [, b]) => b.at - a.at)[0]?.[0];

const ARROW = (path: string) =>
  `<svg viewBox="0 0 16 16" aria-hidden="true"><path d="${path}" /></svg>`;

const HTML = `
  <div class="binder-stage" data-stage tabindex="-1" autofocus>
    <canvas aria-hidden="true"></canvas>
    <div class="binder-veil" data-veil></div>
    <header class="binder-head">
      <p class="binder-count"><strong>Album</strong><span data-count></span></p>
      <button type="button" data-close aria-label="Close">${ARROW('M4 4l8 8M12 4l-8 8')}</button>
    </header>
    <div class="sr-only" data-keys role="group" aria-label="Cards on these pages"></div>
    <div class="binder-foot">
      <button type="button" data-previous aria-label="Turn back">${ARROW('M10 3l-5 5 5 5')}</button>
      <p class="binder-caption" data-caption aria-hidden="true"></p>
      <button type="button" data-next aria-label="Turn over">${ARROW('M6 3l5 5-5 5')}</button>
    </div>
    <div class="binder-row" data-row>
      <div>
        <a class="binder-open" data-open>Open example<span aria-hidden="true">→</span></a>
        <button type="button" data-back>Back</button>
      </div>
      <p class="binder-note" data-note></p>
    </div>
    <p class="sr-only" aria-live="polite" data-announce></p>
  </div>`;

const pad = (n: number) => String(n).padStart(2, '0');

/** What the album says under a card out: where it was found, or a hint toward its page. */
function noteFor(key: string, source: string | undefined) {
  const on = Object.values(HIDDEN).find(({ page }) => rooted(page) === source);
  if (source === undefined) return [`It waits on a page about ${HIDDEN[key]?.subject}.`];
  if (!on) return ['Your first card.'];
  const a = Object.assign(document.createElement('a'), { href: source, textContent: on.title });
  return ['Found on ', a, '.'];
}

let opened: ((key?: string, from?: DOMRect) => void) | undefined;

/**
 * Opens the album at `key`'s card, where WebGPU runs. `from` is where that card stands on screen,
 * as a pull leaves it, so the binder can fly it into its pocket.
 */
export function openAlbum(key?: string, from?: DOMRect) {
  if (!navigator.gpu) return;
  opened ??= createOverlay();
  opened(key, from);
}

function createOverlay() {
  const dialog = document.createElement('dialog');
  dialog.className = 'binder not-content';
  dialog.setAttribute('aria-label', 'Album');
  dialog.innerHTML = HTML;
  document.body.append(dialog);
  const $ = (name: string) => dialog.querySelector<HTMLElement>(`[data-${name}]`)!;
  const [stage, tally, keys, caption, row, back, note, announce] =
    'stage count keys caption row back note announce'.split(' ').map($);
  const canvas = stage.querySelector('canvas')!;
  const openLink = $('open') as HTMLAnchorElement;
  /** Focuses the button among the keys for `key`'s card. */
  const focusKey = (key: string) =>
    keys.querySelector<HTMLElement>(`[data-key="${key}"]`)?.focus({ preventScroll: true });

  const card = createCard();
  stage.insertBefore(card, row);
  const turn = card.querySelector<HTMLElement>('.turn')!;
  card.querySelector<HTMLElement>('[data-field="title"]')!.tabIndex = -1;
  const still = stillMotion();

  let entries: Entry[] = [];
  let binder: Binder | undefined;
  let loading: Promise<Binder | undefined> | undefined;
  /** Counts the closings; an opening's steps stop once the album has closed since they began. */
  let session = 0;
  /** The card out of its pocket, and whether a step is under way. */
  let out: Entry | undefined;
  let busy = false;

  // Where the card out stands, in CSS pixels of the stage
  let cardW = 1;
  let cardH = 1;
  let slot: [number, number] = [0, 0];
  const layout = () => {
    const width = stage.clientWidth;
    const height = stage.clientHeight;
    [cardW, cardH] = fitCard(card, width, height, ROW, 340);
    slot = [width / 2, Math.max(24 + cardH / 2, (height - ROW) / 2)];
    row.style.top = `${slot[1] + cardH / 2 + 18}px`;
  };

  const entryOf = (key: string | undefined) => entries.find((entry) => entry.key === key);

  /** The card out, facing the viewer in the middle. */
  const shown = () => boxOf(slot[0] - cardW / 2, slot[1] - cardH / 2, cardW, cardH);

  /** Lays the card element `k` of the way from `pocket` to `facing`. */
  const draw = (pocket: Quad, facing: Quad, k: number) =>
    (card.style.transform = onto(between(pocket, facing, k), cardW, cardH));

  /** Runs `frame` from 0 to 1 over `ms`; resolves when done, or at once if the album closes. */
  const animate = (ms: number, frame: (k: number) => void) =>
    new Promise<void>((resolve) => {
      const run = session;
      const begin = performance.now();
      const go = (now: number) => {
        if (run !== session) return resolve();
        const k = still ? 1 : clamp01((now - begin) / ms);
        frame(k);
        if (k < 1) requestAnimationFrame(go);
        else resolve();
      };
      requestAnimationFrame(go);
    });

  /** Makes the card element `entry`'s: face up with its poster if found, else face down. */
  const deal = (entry: Entry) => {
    fillCard(card, entry);
    const found = foundCards()[entry.key];
    turn.style.transform = found ? '' : 'rotateY(180deg)';
    card.style.setProperty('--tell', found ? '0' : '1');
    if (found) void printPoster(card, entry);
    openLink.href = entry.href;
    note.replaceChildren(...noteFor(entry.key, found?.source));
  };

  /**
   * Flies the card out of its pocket to `facing`, fading in over the binder's drawing of it, or
   * from `facing` into its pocket, fading into that drawing as it lands.
   */
  const fly = async (entry: Entry, facing: Quad, outward: boolean) => {
    // The binder it leaves or lands in, even if a lost device drops it mid-flight
    const into = binder;
    const pocket = into?.pocket(entry.key);
    if (!into || !pocket) return;
    card.style.visibility = 'visible';
    await animate(outward ? OUT : IN, (k) => {
      draw(pocket, facing, outward ? inOutCubic(k) : 1 - inOutCubic(k));
      if (!outward) return;
      card.style.opacity = clamp01((k * OUT) / FADE).toFixed(3);
      if (k * OUT >= FADE) into.hold(entry.key);
    });
    if (outward) return;
    into.hold(undefined);
    await animate(FADE, (k) => (card.style.opacity = (1 - k).toFixed(3)));
    card.style.visibility = 'hidden';
  };

  /** Takes `key`'s card out of its pocket and up to the middle. */
  const takeOut = async (key: string) => {
    const entry = entryOf(key);
    if (!binder?.pocket(key) || !entry || out || busy) return;
    const run = session;
    busy = true;
    out = entry;
    deal(entry);
    dialog.toggleAttribute('data-out', true);
    // The stage holds focus while the card travels, as the keys go inert and the row is hidden
    stage.focus({ preventScroll: true });
    keys.inert = true;
    const found = key in foundCards();
    announce.textContent = found
      ? `${entry.title}. ${rarityOf(entry.rarity).title} card. ${entry.line}`
      : `Card ${pad(entry.number)}, not found yet.`;
    card.style.opacity = '0';
    await fly(entry, shown(), true);
    if (run !== session) return;
    busy = false;
    dialog.toggleAttribute('data-shown', true);
    (found ? openLink : back).focus({ preventScroll: true });
  };

  /** Puts the card out back in its pocket. */
  const putBack = async () => {
    const entry = out;
    if (!binder || !entry || busy) return;
    const run = session;
    busy = true;
    stage.focus({ preventScroll: true });
    dialog.removeAttribute('data-shown');
    dialog.removeAttribute('data-out');
    await fly(entry, shown(), false);
    if (run !== session) return;
    keys.inert = false;
    out = undefined;
    busy = false;
    focusKey(entry.key);
  };

  // The cards in view as buttons, for the keys and screen readers
  const listKeys = () => {
    if (!binder) return;
    const found = foundCards();
    const focused = keys.contains(document.activeElement);
    keys.replaceChildren(
      ...binder.inView().map((key) => {
        const entry = entryOf(key)!;
        const button = document.createElement('button');
        button.type = 'button';
        button.dataset.key = key;
        button.textContent =
          key in found
            ? `${pad(entry.number)} ${entry.title}`
            : `${pad(entry.number)}, not found yet`;
        return button;
      }),
    );
    // A turn from the keys leaves them on the new pages
    if (focused) keys.querySelector('button')?.focus({ preventScroll: true });
  };

  const fill = () => {
    const found = foundCards();
    binder?.fill(new Set(Object.keys(found)));
    const have = entries.filter(({ key }) => key in found).length;
    tally.textContent = `${have} of ${entries.length}`;
    listKeys();
  };

  // The binder, loaded and compiled on the first opening
  const load = () =>
    (loading ??= import('./binder.ts')
      .then(({ startBinder }) =>
        startBinder(canvas, {
          entries,
          pick: (key) => void takeOut(key),
          turned: (titles) => {
            announce.textContent = caption.textContent = titles.join(' · ') || 'Album';
            listKeys();
          },
        }),
      )
      .catch(() => undefined)
      .then((loaded) => {
        // A lost device takes the binder with it; the next opening starts another
        void loaded?.lost.then(() => {
          loaded.stop();
          binder = loading = undefined;
          dialog.close();
        });
        return (binder = loaded);
      }));

  /** Closes the album: puts a card out back first, then sinks the binder as the page comes back. */
  const leave = async () => {
    // Before the binder is up, as while it first compiles, the album just closes
    if (!binder || !dialog.hasAttribute('data-ready')) return dialog.close();
    if (busy) return;
    if (out) return putBack();
    const run = session;
    busy = true;
    dialog.toggleAttribute('data-leaving', true);
    // Closes once the binder has sunk and the page has come back from under the dimming
    const fading = dialog.getAnimations({ subtree: true });
    await Promise.all([binder.sink(), ...fading.map(({ finished }) => finished.catch(() => {}))]);
    if (run === session) dialog.close();
  };

  // The keys: focus lights a card's pocket, and a press takes it out
  keys.addEventListener('focusin', (event) => binder?.mark(keyOf(event)));
  keys.addEventListener('focusout', () => binder?.mark(undefined));
  keys.addEventListener('click', (event) => void takeOut(keyOf(event)!));

  const turnBy = (forward: boolean) => {
    if (!out && !busy) binder?.turn(forward);
  };
  $('previous').addEventListener('click', () => turnBy(false));
  $('next').addEventListener('click', () => turnBy(true));
  $('close').addEventListener('click', () => void leave());
  back.addEventListener('click', () => void putBack());
  $('veil').addEventListener('click', () => void putBack());
  onLeave(dialog, () => void leave());
  dialog.addEventListener('keydown', (event) => {
    if (!dialog.hasAttribute('data-ready') || event.target instanceof HTMLAnchorElement) return;
    if (event.key === 'ArrowRight') turnBy(true);
    else if (event.key === 'ArrowLeft') turnBy(false);
    else return;
    event.preventDefault();
  });

  dialog.addEventListener('close', () => {
    session++;
    binder?.stop();
    binder?.hold(undefined);
    card.style.visibility = 'hidden';
    keys.inert = false;
    out = undefined;
    busy = false;
    for (const name of ['data-out', 'data-shown', 'data-ready', 'data-handoff', 'data-leaving'])
      dialog.removeAttribute(name);
  });
  onAlbum(() => {
    if (dialog.open) fill();
  });
  new ResizeObserver(() => {
    layout();
    if (out && !busy) draw(shown(), shown(), 1);
  }).observe(stage);

  /**
   * The binder rises, turns to `key`'s card or the one found last, and the card a pull left at
   * `from` flies into its pocket.
   */
  const open = async (key: string | undefined, from: DOMRect | undefined) => {
    const run = session;
    const all = await loadEntries().catch(() => undefined);
    if (run !== session) return;
    if (!all) return dialog.close();
    entries = all;
    layout();
    // A card from a pull waits where the pull left it while the binder comes
    const arriving = key !== undefined && key in foundCards() ? entryOf(key) : undefined;
    const box = stage.getBoundingClientRect();
    const at =
      from && ([from.left - box.left, from.top - box.top, from.width, from.height] as const);
    if (arriving && at) {
      busy = true;
      deal(arriving);
      draw(boxOf(...at), boxOf(...at), 1);
      card.style.opacity = '1';
      card.style.visibility = 'visible';
    }
    const loaded = await load();
    if (run !== session) return;
    if (!loaded) return dialog.close();
    fill();
    const target = key ?? newest();
    if (target) loaded.show(target);
    if (arriving && at) loaded.hold(arriving.key);
    await loaded.start();
    if (run !== session) return;
    dialog.toggleAttribute('data-ready', true);
    if (arriving && at) {
      await fly(arriving, boxOf(...at), false);
      // The card just filed stays lit, and a press or Enter takes it out to open
      if (run === session) focusKey(arriving.key);
    }
    if (run === session) busy = false;
  };

  return (key?: string, from?: DOMRect) => {
    if (dialog.open) return;
    dialog.toggleAttribute('data-handoff', from !== undefined);
    dialog.showModal();
    void open(key, from);
  };
}
