// Whether the reader plays the cards: the boxes, the packs and the album. They are on where
// WebGPU runs unless the reader turns them off. Head.astro marks the page before its first paint
// with `data-cards` on the root, so that nothing of them shows, lays out or loads while they are off.

const KEY = 'typegpu-kit:cards';
const root = document.documentElement;

export const cardsOn = () => 'cards' in root.dataset;

let waiting: (() => void)[] = [];

/** Calls `start` now if the cards are on, else once the reader turns them on. */
export function whenCards(start: () => void) {
  if (cardsOn()) start();
  else waiting.push(start);
}

/** Turns the cards off on this page alone, where the GPU can't draw them; the footer's switch goes too. */
export const cardsFailed = () => {
  root.removeAttribute('data-cards');
  root.toggleAttribute('data-cards-failed', true);
};

/** Turns the cards on or off, here and on every page from now on. */
export function setCards(on: boolean) {
  try {
    if (on) localStorage.removeItem(KEY);
    else localStorage.setItem(KEY, 'off');
  } catch {
    // Without storage the choice holds for this page
  }
  root.toggleAttribute('data-cards', on);
  if (!on) return;
  const ready = waiting;
  waiting = [];
  for (const start of ready) start();
}
