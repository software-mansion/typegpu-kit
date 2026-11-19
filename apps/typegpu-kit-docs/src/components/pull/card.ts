// The pulled card: its markup, filled from an entry with its printed poster, its size on a stage,
// and the stamped emblem on its back.

import { cardSize, emblem, SEAL } from 'typegpu-kit-pull/card.ts';
import { bayer, bytesOf } from '../site/dither.ts';
import { cellsOf, posterImage, printInto, type Cells } from './print.ts';
import { colorOf, rarityOf, type Entry } from './pulls.ts';
import './card.css';

const HTML = `
  <div class="turn">
    <div class="face">
      <div class="art" data-art><canvas aria-hidden="true"></canvas></div>
      <div class="caption">
        <span class="kicker"><span data-field="number"></span><span data-field="category"></span></span>
        <h2 class="title"><a data-field="title"></a></h2>
        <p data-field="line"></p>
        <ul aria-label="Modules used" data-field="modules"></ul>
      </div>
      <div class="foot">
        <span class="tier"><span class="gem" aria-hidden="true"></span><span data-field="tier"></span></span>
        <span data-field="lines"></span>
      </div>
    </div>
    <div class="back" aria-hidden="true"><div class="emblem"></div></div>
    <span class="edge left" aria-hidden="true"></span>
    <span class="edge right" aria-hidden="true"></span>
  </div>`;

/** A blank card, to fill with `fillCard`. */
export function createCard(): HTMLElement {
  const card = document.createElement('article');
  card.className = 'pull-card';
  card.innerHTML = HTML;
  card.querySelector<HTMLElement>('.emblem')!.style.setProperty('--emblem', emblemMask());
  return card;
}

const pad = (n: number) => String(n).padStart(2, '0');

/** Fills `card` with `entry`. */
export function fillCard(card: HTMLElement, entry: Entry) {
  const field = <T extends HTMLElement>(name: string) =>
    card.querySelector<T>(`[data-field="${name}"]`)!;
  const tier = rarityOf(entry.rarity);
  card.dataset.key = entry.key;
  card.dataset.rarity = entry.rarity;
  card.querySelector<HTMLCanvasElement>('[data-art] canvas')!.width = 0;
  card.style.setProperty('--rarity', colorOf(tier));
  field('number').textContent = pad(entry.number);
  field('category').textContent = entry.category;
  const title = field<HTMLAnchorElement>('title');
  title.textContent = entry.title;
  title.href = entry.href;
  field('line').textContent = entry.line;
  field('modules').replaceChildren(
    ...entry.modules.map((name) => {
      const item = document.createElement('li');
      item.className = 'pill';
      item.textContent = name;
      return item;
    }),
  );
  field('tier').textContent = tier.title;
  field('lines').textContent = `${entry.lines.toLocaleString('en')} lines`;
}

/**
 * Prints `card`'s poster once it has loaded, unless the card holds another entry by then; resolves
 * with the poster's cells.
 */
export async function printPoster(card: HTMLElement, entry: Entry): Promise<Cells | undefined> {
  const image = await posterImage(entry.poster).catch(() => undefined);
  const art = card.querySelector<HTMLElement>('[data-art]')!;
  if (!image || card.dataset.key !== entry.key || !art.clientWidth) return undefined;
  const cells = cellsOf(image, art.clientWidth, art.clientHeight);
  printInto(art.querySelector('canvas')!, cells, bytesOf(card, ['var(--card-ground)'])[0]);
  return cells;
}

/**
 * Sends Escape and other close requests on `dialog` to `leave`. Escape goes from its keydown,
 * because Chrome skips `cancel` on a second Escape with no press between and closes at once.
 */
export function onLeave(dialog: HTMLDialogElement, leave: () => void) {
  dialog.addEventListener('keydown', (event) => {
    if (event.key !== 'Escape') return;
    event.preventDefault();
    leave();
  });
  dialog.addEventListener('cancel', (event) => {
    event.preventDefault();
    leave();
  });
}

/**
 * Sizes `card` as wide as fits a stage `width` by `height`, with `room` under it for its row, and
 * at most `widest`; returns its width and height.
 */
export function fitCard(
  card: HTMLElement,
  width: number,
  height: number,
  room: number,
  widest: number,
) {
  const [w, h] = cardSize(width, height, room, widest);
  card.style.setProperty('--card-w', `${w}px`);
  card.style.setProperty('--u', (w / 340).toFixed(4));
  return [w, h] as const;
}

/** Where `element` is inside `ancestor`, through any positioned box between them. */
export function offsetIn(element: HTMLElement, ancestor: HTMLElement) {
  let [x, y] = [0, 0];
  for (let at: HTMLElement | null = element; at && at !== ancestor;) {
    x += at.offsetLeft;
    y += at.offsetTop;
    at = at.offsetParent as HTMLElement | null;
  }
  return [x, y] as const;
}

let mask: string | undefined;

/** The emblem as a CSS mask of `SEAL` × `SEAL` dithered squares. */
function emblemMask() {
  if (mask) return mask;
  let path = '';
  for (let j = 0; j < SEAL; j++) {
    let run = 0;
    for (let i = 0; i <= SEAL; i++) {
      const x = ((i + 0.5) / SEAL) * 2 - 1;
      const y = ((j + 0.5) / SEAL) * 2 - 1;
      if (i < SEAL && emblem(x, y) > bayer(i, j)) run++;
      else if (run) {
        path += `M${i - run} ${j}h${run}v1h-${run}z`;
        run = 0;
      }
    }
  }
  const svg = `<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 ${SEAL} ${SEAL}' shape-rendering='crispEdges'><path d='${path}'/></svg>`;
  mask = `url("data:image/svg+xml,${encodeURIComponent(svg)}")`;
  return mask;
}
