// The way into a pull: a box on its pool of light or a foil pack, resting on the page, where WebGPU
// runs. It shows once compiled, answers the pointer, and hands itself to the overlay when pressed.
// A vessel hidden in the docs holds one card, and is gone once that card is in the album; any other
// is the first one, gone once the album holds a card.

import { count, has, onAlbum } from '../album/album.ts';
import { cardsFailed } from '../album/cards.ts';
import { colorsOf } from '../site/dither.ts';
import { restingBox, restOf } from 'typegpu-kit-pull/box.ts';
import type { Vessel, VesselState, VesselTarget } from './gpu.ts';
import { grainsAt, poseOf as pouchPose } from './pouch.ts';
import { CELL } from './print.ts';
import { overlayOf } from './pull.ts';
import { loadEntries, stillMotion } from './pulls.ts';

/** Milliseconds the light takes to run along the pack's seam. */
const RUN = 1300;

const now = () => performance.now();

/** Wires the vessels on the page, once. */
export function mountResting() {
  for (const button of document.querySelectorAll<HTMLButtonElement>('[data-pull-object]'))
    if (!('resting' in button.dataset)) rest(button);
}

function rest(button: HTMLButtonElement) {
  button.dataset.resting = '';
  const vessel = button.dataset.pullObject as Vessel;
  const overlay = overlayOf(vessel);
  const card = button.dataset.card;
  /** Hides it once its card, or for the first vessel any card, is in the album. */
  const check = () => {
    button.hidden = card === undefined ? count() > 0 : has(card);
  };
  check();
  const lean = Number(button.dataset.lean ?? 0);
  const canvas = button.querySelector('canvas')!;
  const accent = colorsOf(button).accent[0];
  const still = stillMotion();
  let target: VesselTarget | undefined;

  // Layout, in CSS pixels of the button: the vessel's ground and unit; the pack about 3.5 units tall
  // from its foot to its crimp, standing on the button's foot
  let anchor: [number, number] = [0, 0];
  let unit = 1;
  // Hidden, or with the cards turned off, it has no size to lay out
  const layout = () => {
    const width = button.clientWidth;
    const height = button.clientHeight;
    if (!width) return;
    if (vessel === 'box') {
      const spot = restOf(width, height);
      unit = spot.unit;
      anchor = [...spot.anchor];
    } else {
      unit = Math.min(width / 2.5, height / 3.9);
      anchor = [width / 2, height + 0.2 * unit];
    }
    const columns = Math.ceil(width / CELL);
    const rows = Math.ceil(height / CELL);
    canvas.style.width = `${columns * CELL}px`;
    canvas.style.height = `${rows * CELL}px`;
    target?.fit(columns, rows);
  };

  // What it answers: the pointer over it and pressing it, each eased toward its aim, and when the
  // reader came over it
  let over = 0;
  let overAim = 0;
  let press = 0;
  let pressAim = 0;
  let overAt = -1e9;
  let away = false;

  const stateAt = (at: number): VesselState => {
    if (vessel === 'box') {
      return { kind: 'box', input: restingBox(anchor, unit, over, press, accent) };
    }
    const frames = pouchPose({
      yaw: 0,
      roll: lean,
      lift: 0.25 + 0.06 * over - 0.05 * press,
      tear: 0,
      fly: 0,
    });
    // The light along its seam, where to tear it, as the pointer comes over it
    const run = at - overAt;
    const input = {
      anchor,
      unit,
      time: at / 1000,
      body: frames.body,
      strip: frames.strip,
      tear: 0,
      hint: run >= 0 && run < RUN ? run / RUN : -1,
      mouth: 0,
      sink: 0,
      gone: 0,
      tint: accent,
      grains: grainsAt(() => -1, frames.body, anchor, unit),
    };
    return { kind: 'pack', input };
  };

  const draw = (at: number) => target?.draw(stateAt(at));

  let frame = 0;
  const tick = (at: number) => {
    frame = 0;
    if (away) return;
    over += (overAim - over) * 0.16;
    press += (pressAim - press) * 0.3;
    draw(at);
    if (Math.abs(over - overAim) + Math.abs(press - pressAim) > 0.002 || at - overAt < RUN) wake();
  };
  const wake = () => {
    if (target && !still) frame ||= requestAnimationFrame(tick);
  };

  // Compiled when the page is idle after it first comes near, and drawn from then on
  const compile = () => {
    void overlay.warm().then((gpu) => {
      if (!gpu) return cardsFailed();
      if (target) return;
      target = gpu.elsewhere(canvas);
      // A lost device takes it too, until it compiles on the next one once the page is back in
      // view, as iOS loses it for a page in the background
      void gpu.lost.then(() => {
        target = undefined;
        button.removeAttribute('data-gpu');
        const back = () => {
          if (document.hidden) return;
          document.removeEventListener('visibilitychange', back);
          compile();
        };
        document.addEventListener('visibilitychange', back);
        back();
      });
      layout();
      draw(now());
      button.toggleAttribute('data-gpu', true);
    });
  };
  new IntersectionObserver(
    ([entry], observer) => {
      if (!entry?.isIntersecting) return;
      observer.disconnect();
      loadEntries().catch(() => {});
      if ('requestIdleCallback' in window) requestIdleCallback(compile, { timeout: 2000 });
      else setTimeout(compile, 600);
    },
    { rootMargin: '300px' },
  ).observe(button);

  // Laid out and drawn again at every size it takes
  new ResizeObserver(() => {
    if (!button.clientWidth) return;
    layout();
    draw(now());
  }).observe(button);

  // The overlay takes it from where it stands, and gives it back when it closes
  button.addEventListener('click', () => {
    const box = button.getBoundingClientRect();
    overlay.open(card, {
      anchor: [box.left + anchor[0], box.top + anchor[1]],
      unit,
      yaw: vessel === 'box' ? 0.1 * over : 0,
      roll: vessel === 'box' ? 0 : lean,
      shown: () => {
        away = true;
        button.toggleAttribute('data-away', true);
      },
      closed: () => {
        away = false;
        button.removeAttribute('data-away');
        check();
        over = overAim = press = pressAim = 0;
        draw(now());
      },
    });
  });

  // The pointer over it and pressing it
  const ease = (aims: { over?: number; press?: number }) => {
    overAim = aims.over ?? overAim;
    pressAim = aims.press ?? pressAim;
    wake();
  };
  button.addEventListener('pointerenter', () => {
    if (now() - overAt > RUN) overAt = now();
    ease({ over: 1 });
  });
  button.addEventListener('pointerleave', () => ease({ over: 0, press: 0 }));
  button.addEventListener('focus', () => ease({ over: 1 }));
  button.addEventListener('blur', () => ease({ over: 0 }));
  button.addEventListener('pointerdown', (event) => ease({ press: event.button === 0 ? 1 : 0 }));
  button.addEventListener('pointerup', () => ease({ press: 0 }));
  button.addEventListener('pointercancel', () => ease({ press: 0 }));

  // A card found in another tab, or the album emptied, hides it or brings it back
  onAlbum(() => {
    if (away) return;
    check();
    draw(now());
  });

  layout();
  draw(now());
}
