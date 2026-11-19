// A pull over the page, in an overlay. A box knocks and pops its lid, and the card leaves it face
// down. A foil pack waits for the reader to tear it across the top, then drops away from the card
// inside. Either way the card is tossed up and turned over into its place, and goes into the
// reader's album. The vessel comes from its place on the page. A vessel hidden in the docs holds
// its own card; the first one a random common card.

import { addCard, count, has } from '../album/album.ts';
import { setCards } from '../album/cards.ts';
import { bytesOf, colorsOf, type Colors, type Rgb } from '../site/dither.ts';
import { boxStage, pulledBox } from 'typegpu-kit-pull/box.ts';
import { cardAt, FINISHES } from 'typegpu-kit-pull/card.ts';
import type { Rgb01 } from 'typegpu-kit-pull/dither.ts';
import { draw } from 'typegpu-kit-pull/rarity.ts';
import {
  clamp01,
  EXIT,
  flightAt,
  FLIGHT,
  handoffAt,
  HAND,
  inOutQuad,
  LAUNCH,
  lerp,
  packAt,
  TEAR,
} from 'typegpu-kit-pull/timeline.ts';
import { createCard, fillCard, fitCard, onLeave, printPoster } from './card.ts';
import type { FaceState, Gpu, Vessel, VesselState } from './gpu.ts';
import {
  ELEVATION,
  grainsAt,
  mouthOf,
  POUCH,
  poseOf as pouchPose,
  type PouchInput,
} from './pouch.ts';
import { CELL, type Cells } from './print.ts';
import {
  colorOf,
  loadEntries,
  RARITIES,
  rarityOf,
  stillMotion,
  type Entry,
  type Rarity,
} from './pulls.ts';

/** Height under the card for its button and its note, at least. */
const ROW = 92;
/** The pack's half-width in the card's widths, so the card fits inside it. */
const PACK = 0.53;
/** A tear dragged this far finishes itself when the pointer lets go. */
const ENOUGH = 0.4;

const COS = Math.cos(ELEVATION);

const HTML = `
  <div class="pull-stage" data-stage tabindex="-1" autofocus>
    <canvas class="pull-vessel" aria-hidden="true"></canvas>
    <p class="pull-hint" data-hint aria-hidden="true">Drag across the top to tear it open</p>
    <button type="button" class="sr-only" data-tear hidden>Tear the pack open</button>
    <div class="pull-row" data-row>
      <button type="button" class="pull-primary" data-album>Into the album</button>
      <p class="pull-note" data-note></p>
    </div>
    <button type="button" class="pull-close" data-close aria-label="Close">
      <svg viewBox="0 0 16 16" aria-hidden="true"><path d="M4 4l8 8M12 4l-8 8" /></svg>
    </button>
    <p class="sr-only" aria-live="polite" data-announce></p>
  </div>`;

/**
 * A vessel on the page that the overlay takes over, so the pull starts where it stands: its ground
 * and unit in CSS pixels of the viewport, and its turn there beyond the overlay's own.
 */
export interface Handoff {
  readonly anchor: readonly [number, number];
  readonly unit: number;
  readonly yaw: number;
  readonly roll: number;
  /** Called once the overlay has drawn the vessel over its place, and once it closes. */
  readonly shown: () => void;
  readonly closed: () => void;
}

export interface Overlay {
  /** Pulls `card`, or a random common card, from the vessel handed over by `from`. */
  open(card: string | undefined, from: Handoff): void;
  /** Compiles the vessel ahead of a pull, if the page can; resolves with it. */
  warm(): Promise<Gpu | undefined>;
}

/**
 * A pack's tear: how far it has been dragged, when it began to tear itself and from where, when it
 * was finished, and when it passed each point along the seam, in milliseconds since the pull.
 */
const untorn = () => ({
  dragged: 0,
  autoAt: undefined as number | undefined,
  autoFrom: 0,
  tornAt: undefined as number | undefined,
  history: [] as [number, number][],
  dragging: undefined as { x: number; moved: number } | undefined,
});

const overlays = new Map<Vessel, Overlay>();
/** The overlay for `vessel`, made on first use. */
export const overlayOf = (vessel: Vessel) => {
  if (!overlays.has(vessel)) overlays.set(vessel, createOverlay(vessel));
  return overlays.get(vessel)!;
};

function createOverlay(vessel: Vessel): Overlay {
  const dialog = document.createElement('dialog');
  dialog.className = 'pull not-content';
  dialog.setAttribute('aria-label', vessel === 'box' ? 'Open a box' : 'Open a pack');
  dialog.innerHTML = HTML;
  document.body.append(dialog);

  const $ = <T extends Element>(selector: string) => dialog.querySelector<T>(selector)!;
  const stage = $<HTMLElement>('[data-stage]');
  const canvas = $<HTMLCanvasElement>('.pull-vessel');
  const row = $<HTMLElement>('[data-row]');
  const toAlbum = $<HTMLButtonElement>('[data-album]');
  const note = $<HTMLElement>('[data-note]');
  const close = $<HTMLButtonElement>('[data-close]');
  const announce = $<HTMLElement>('[data-announce]');
  const hintText = $<HTMLElement>('[data-hint]');
  const tearButton = $<HTMLButtonElement>('[data-tear]');
  tearButton.addEventListener('click', () => tearOn());

  const card = createCard();
  stage.insertBefore(card, row);
  const turn = card.querySelector<HTMLElement>('.turn')!;
  const face = card.querySelector<HTMLElement>('.face')!;
  const art = card.querySelector<HTMLElement>('[data-art]')!;
  const title = card.querySelector<HTMLAnchorElement>('[data-field="title"]')!;

  const still = stillMotion();
  const colors: Colors = colorsOf(dialog);
  const accent = colors.accent[0];
  const cardGround: Rgb = bytesOf(card, ['var(--card-ground)'])[0];
  const tints = bytesOf(dialog, RARITIES.map(colorOf)).map(([r, g, b]): Rgb01 => [
    r / 255,
    g / 255,
    b / 255,
  ]);
  const tintOf = (rarity: Rarity) => tints[RARITIES.indexOf(rarityOf(rarity))];
  /** Tells the reader what the pull found, `ms` from now, unless it has been pulled again. */
  const tell = (ms: number) => {
    const entry = winner;
    setTimeout(() => {
      if (entry && winner === entry)
        announce.textContent = `${rarityOf(entry.rarity).title}: ${entry.title}. ${entry.line}`;
    }, ms);
  };

  // The pull, and its card for the album: put in once it has turned over, and filed in its pocket
  // when the overlay closes
  let winner: Entry | undefined;
  let keeping: Entry | undefined;
  let finding: Entry | undefined;
  let direction = 1;
  let phase = 0;
  let started = 0;
  let cells: Cells | undefined;
  let settled = false;

  // The pack's tear
  let seam = untorn();
  /** Whether the pack tears by itself, or is torn. */
  const tearing = () => seam.autoAt !== undefined || seam.tornAt !== undefined;
  const tearAt = (elapsed: number) => {
    if (seam.tornAt !== undefined && elapsed >= seam.tornAt) return 1;
    if (seam.autoAt === undefined) return seam.dragged;
    const run = TEAR * (1 - seam.autoFrom);
    return lerp(seam.autoFrom, 1, inOutQuad(clamp01((elapsed - seam.autoAt) / Math.max(run, 1))));
  };
  const ageOf = (elapsed: number) => (along: number) => {
    const passed = seam.history.find(([, progress]) => progress >= along);
    return passed ? elapsed - passed[0] : -1;
  };

  // The GPU, once compiled; the vessel shows from then on
  let gpu: Gpu | undefined;
  let compiling: Promise<Gpu | undefined> | undefined;
  /** The vessel's canvas, in cells. */
  let grid: [number, number] = [1, 1];
  const askGpu = () =>
    (compiling ??= !navigator.gpu
      ? Promise.resolve(undefined)
      : import('./gpu.ts')
          .then(({ startGpu }) => startGpu(canvas, vessel, colors, cardGround))
          .then(
            (compiled) => {
              if (!compiled) return undefined;
              // A lost device takes the vessel with it; the next press compiles it again
              void compiled.lost.then(() => {
                compiled.detach();
                gpu = compiling = undefined;
                dialog.removeAttribute('data-gpu');
                dialog.close();
              });
              gpu = compiled;
              gpu.fit(...grid);
              dialog.toggleAttribute('data-gpu', true);
              attachFoil();
              wake();
              return compiled;
            },
            (error: unknown) => {
              console.warn('The pull did not start on the GPU:', error);
              return undefined;
            },
          ));

  // The vessel on the page this pull came from, told when it is shown and closed
  let from: Handoff | undefined;
  let unshown: (() => void) | undefined;
  const attachFoil = () => {
    if (!gpu) return;
    if (winner && cells && winner.rarity !== 'common')
      gpu.attach(face, art, cells, FINISHES[winner.rarity]);
    else gpu.detach();
  };

  // Layout, in CSS pixels of the stage: the card, its slot, the vessel's ground and unit, the
  // pack's seam, and where the stage is in the viewport
  let cardW = 1;
  let cardH = 1;
  let slot: [number, number] = [0, 0];
  let ground: [number, number] = [0, 0];
  let size = 50;
  let offset: [number, number] = [0, 0];
  let mouthY = 0;
  const layout = () => {
    const width = stage.clientWidth;
    const height = stage.clientHeight;
    // The pack stands about 1.7 card widths tall with the card inside it, its top under the mouth
    const room = Math.max(ROW, row.offsetHeight + 24);
    const tall = vessel === 'pack' ? (height - 96) / 1.75 : Infinity;
    [cardW, cardH] = fitCard(card, width, height, room, Math.min(360, tall));
    const box = stage.getBoundingClientRect();
    offset = [box.left, box.top];
    if (vessel === 'box') {
      // The box stands where the card will be, which leaves it and grows over it as it goes
      const stood = boxStage(width, height, cardH, room);
      slot = [...stood.slot];
      size = stood.size;
      ground = [...stood.ground];
    } else {
      size = cardW * PACK;
      mouthY = (height - 40) / 2 - POUCH.seam * size;
      slot = [width / 2, mouthY + cardH / 2 + 3];
      ground = [width / 2, mouthY + (POUCH.top + 0.3 + POUCH.seam) * COS * size];
      hintText.style.top = `${ground[1] - 0.3 * COS * size + 18}px`;
    }
    grid = [Math.ceil(width / CELL), Math.ceil(height / CELL)];
    canvas.style.width = `${grid[0] * CELL}px`;
    canvas.style.height = `${grid[1] * CELL}px`;
    gpu?.fit(...grid);
    row.style.top = `${slot[1] + cardH / 2 + 18}px`;
    if (winner)
      void printPoster(card, winner).then((printed) => {
        if (!printed) return;
        cells = printed;
        attachFoil();
        wake();
      });
  };

  /**
   * The vessel `t` milliseconds into the pull, or after the pack's tear: what to draw, where the
   * card leaves it and how wide it is there, and the lip the card is hidden below.
   */
  const vesselAt = (elapsed: number, t: number, tear: number, tint: Rgb01) => {
    // From its place on the page to its own, at its size there and turned as it was
    const hand = from!;
    const { anchor, unit, yaw, roll } = handoffAt(
      elapsed,
      { ...hand, anchor: [hand.anchor[0] - offset[0], hand.anchor[1] - offset[1]] },
      ground,
      size,
    );
    if (vessel === 'box') {
      const { input, ...exit } = pulledBox(t, phase, anchor, unit, yaw, accent, tint);
      return { state: { kind: 'box', input } as VesselState, ...exit };
    }
    const p = packAt(t);
    const waiting = t < 0 && seam.autoAt === undefined && tear < 0.02;
    // Square to the viewer, floating and swaying a little, stilled by the tear
    const calm = 1 - tear;
    const frames = pouchPose({
      yaw: 0.08 * Math.sin(elapsed / 900) * calm + yaw,
      roll: 0.015 * Math.sin(elapsed / 700) * calm + 0.02 * (seam.dragging ? 1 : 0) + roll,
      lift: 0.25 + 0.04 * Math.sin(elapsed / 420) * calm - 0.05 * p.press - 4.6 * p.drop,
      tear,
      fly: p.fly,
    });
    // While it waits, a light runs along the seam now and then, where to tear it
    const run = (elapsed - HAND - 500) % 2600;
    const input: PouchInput = {
      anchor,
      unit,
      time: elapsed / 1000,
      body: frames.body,
      strip: frames.strip,
      tear,
      hint: waiting && elapsed > HAND + 500 && run < 1300 ? run / 1300 : -1,
      mouth: p.mouth,
      sink: p.sink,
      gone: p.gone,
      tint,
      grains: grainsAt(ageOf(elapsed), frames.body, anchor, unit),
    };
    const mouth = mouthOf(frames.body, anchor, unit);
    const lip = [-0.5, 0.5].map(
      (side) =>
        [
          mouth.middle[0] + mouth.right[0] * mouth.width * side,
          mouth.middle[1] + mouth.right[1] * mouth.width * side,
        ] as const,
    );
    return { state: { kind: 'pack', input } as VesselState, mouth: mouth.middle, span: cardW, lip };
  };

  /** Puts the card where it is `t` milliseconds into the pull; returns its face's look. */
  const placeCard = (t: number, where: ReturnType<typeof vesselAt>): FaceState => {
    const flare = FINISHES[winner?.rarity ?? 'common'].flare;
    const pose = cardAt(
      vessel,
      t,
      where,
      { width: cardW, height: cardH, slot, size },
      direction,
      flare,
    );
    const { x, y, z, scale, rx, ry, rz, rim } = pose;
    card.style.visibility = pose.visible ? 'visible' : 'hidden';
    if (rim) {
      const points = [
        [-cardW, -cardH],
        [cardW * 2, -cardH],
        [cardW * 2, rim[1][1]],
        ...rim.toReversed(),
        [-cardW, rim[0][1]],
      ];
      card.style.clipPath = `polygon(${points.map(([px, py]) => `${px!.toFixed(1)}px ${py!.toFixed(1)}px`).join(', ')})`;
    } else {
      card.style.clipPath = 'none';
    }

    card.style.transform = `translate3d(${(x - cardW / 2).toFixed(2)}px, ${(y - cardH / 2).toFixed(2)}px, ${z.toFixed(1)}px) rotateZ(${rz.toFixed(3)}deg) rotateX(${rx.toFixed(3)}deg) scale(${scale.toFixed(4)})`;
    turn.style.transform = `rotateY(${ry.toFixed(3)}deg)`;

    // Lit from the upper left: the face darkens as it turns away, and its foil follows its normal
    card.style.setProperty('--shade', pose.shade.toFixed(3));
    card.style.setProperty('--back-shade', pose.backShade.toFixed(3));
    card.style.setProperty('--lift', pose.lift.toFixed(3));
    card.style.setProperty('--tell', pose.tell.toFixed(3));
    row.style.opacity = pose.row.toFixed(3);
    row.inert = pose.row < 0.5;
    return { tilt: pose.tilt, sheen: pose.sheen, sweep: pose.sweep, flare: pose.flare };
  };

  /** Milliseconds since the pull began, and since it began or the pack's tear was finished. */
  const clockAt = (now: number) => {
    const elapsed = now - started;
    if (still) return { elapsed, t: LAUNCH[vessel] + FLIGHT, tear: 1 };
    if (vessel === 'box') return { elapsed, t: elapsed, tear: 0 };
    const tear = tearAt(elapsed);
    if (tear > (seam.history.at(-1)?.[1] ?? 0)) seam.history.push([elapsed, tear]);
    if (tear >= 1 && seam.tornAt === undefined) {
      seam.tornAt = elapsed;
      tell(LAUNCH.pack + 600);
    }
    return { elapsed, t: seam.tornAt === undefined ? -1e6 : elapsed - seam.tornAt, tear };
  };

  const render = (now: number) => {
    const { elapsed, t, tear } = clockAt(now);
    const entry = winner;
    const where = vesselAt(elapsed, t, tear, entry ? tintOf(entry.rarity) : tints[0]);
    const look = placeCard(t, where);
    const waiting = vessel === 'pack' && seam.tornAt === undefined && !still;
    const asking = waiting && seam.autoAt === undefined;
    dialog.toggleAttribute('data-waiting', asking);
    // A button tears it for the keys and screen readers while it waits, and hands focus back after
    if (!asking && document.activeElement === tearButton) stage.focus({ preventScroll: true });
    tearButton.hidden = !asking;
    hintText.style.opacity = waiting
      ? (clamp01((elapsed - HAND) / 400) * (1 - clamp01(tear * 5))).toFixed(3)
      : '0';

    gpu?.draw(
      t > LAUNCH[vessel] + 800 ? undefined : where.state,
      entry && entry.rarity !== 'common' ? look : undefined,
    );
    settled = t >= LAUNCH[vessel] + FLIGHT;
    // Filed once its face turns toward the reader
    if (t >= LAUNCH[vessel] && flightAt(t - LAUNCH[vessel]).turn >= 0.5) keep();
    unshown?.();
    unshown = undefined;
    return settled;
  };

  /** Puts the pulled card in the album, once it shows its face. */
  const keep = () => {
    if (!keeping) return;
    addCard(keeping.key, location.pathname);
    keeping = undefined;
  };

  /** Tears the pack the rest of the way by itself, from where the drag left it. */
  const tearOn = () => {
    if (vessel !== 'pack' || tearing() || !winner) return;
    seam.autoAt = performance.now() - started;
    seam.autoFrom = seam.dragged;
    wake();
  };

  // The loop runs while anything moves
  let frame = 0;
  const tick = (now: number) => {
    frame = 0;
    if (dialog.open && !render(now)) wake();
  };
  const wake = () => {
    frame ||= requestAnimationFrame(tick);
  };

  const pull = (held: string | undefined) => {
    void loadEntries()
      .then((all) => {
        const commons = all.filter(({ rarity }) => rarity === 'common');
        const found = all.find(({ key }) => key === held);
        ({ item: winner, direction, phase } = draw(found ? [found] : commons));
        keeping = finding = winner;
        started = performance.now();
        settled = false;
        cells = undefined;
        seam = untorn();
        fillCard(card, winner);
        card.style.visibility = 'hidden';
        title.tabIndex = -1;
        noteOn(count() === 0, all.length);
        gpu?.detach();
        layout();
        wake();
        if (vessel === 'box' || still) tell(still ? 0 : EXIT[1] + 600);
      })
      .catch(() => dialog.close());
  };

  // The first card welcomes the reader to the album, and lets them turn the cards off
  const off = document.createElement('button');
  off.type = 'button';
  off.className = 'pull-off';
  off.textContent = 'Turn them off';
  off.addEventListener('click', () => {
    finding = undefined;
    setCards(false);
    dialog.close();
  });

  /** The line under the button: what the card means for the album. */
  const noteOn = (first: boolean, total: number) => {
    note.toggleAttribute('data-first', first);
    if (!first) {
      note.textContent = `${count() + (winner && has(winner.key) ? 0 : 1)} of ${total} found.`;
      return;
    }
    const lines = [
      ['Your first card'],
      [`${total - 1} more hide in these docs, one on a page, to find as you read.`],
      ['They are only for fun. ', off, ' here or at the foot of any page.'],
    ].map((parts) => {
      const line = document.createElement('span');
      line.append(...parts);
      return line;
    });
    note.replaceChildren(...lines);
  };

  // The pack tears under a drag across it, any way and in any strokes; a press tears it outright
  const onPack = (event: PointerEvent) => {
    const x = event.clientX - offset[0];
    const y = event.clientY - offset[1];
    return (
      Math.abs(x - ground[0]) < size * 1.1 && y > mouthY - 0.5 * size && y < ground[1] - 0.2 * size
    );
  };
  stage.addEventListener('pointerdown', (event) => {
    if (vessel !== 'pack' || tearing() || still) return;
    if (!onPack(event)) return;
    seam.dragging = { x: event.clientX, moved: 0 };
    stage.setPointerCapture(event.pointerId);
    wake();
  });
  stage.addEventListener('pointermove', (event) => {
    if (!seam.dragging || tearing()) return;
    const dx = Math.abs(event.clientX - seam.dragging.x);
    seam.dragging.x = event.clientX;
    seam.dragging.moved += dx;
    seam.dragged = Math.min(1, seam.dragged + dx / (size * 2 * 0.85));
    wake();
  });
  const letGo = () => {
    if (!seam.dragging) return;
    const { moved } = seam.dragging;
    seam.dragging = undefined;
    if (moved < 6 || seam.dragged >= ENOUGH) tearOn();
    wake();
  };
  stage.addEventListener('pointerup', letGo);
  stage.addEventListener('pointercancel', letGo);
  stage.addEventListener('keydown', (event) => {
    if (event.target === stage && (event.key === 'Enter' || event.key === ' ')) {
      event.preventDefault();
      tearOn();
    }
  });

  /**
   * Closes the overlay; a card shown flies from here into its pocket in the album, so every find
   * ends there. The overlay stays until the album shows over it, so the page never shows undimmed
   * between them.
   */
  let leaving: Promise<void> | undefined;
  const leave = () => {
    if (leaving) return;
    const filed = settled ? finding : undefined;
    finding = undefined;
    if (!filed) return dialog.close();
    const at = card.getBoundingClientRect();
    leaving = import('../album/open.ts')
      .then(({ openAlbum }) => openAlbum(filed.key, at))
      .finally(() => {
        leaving = undefined;
        dialog.close();
      });
  };
  toAlbum.addEventListener('click', leave);
  close.addEventListener('click', leave);
  // Once the card has settled, a press on the ground around it closes the overlay, as Escape does
  dialog.addEventListener('click', (event) => {
    if (settled && (event.target === stage || event.target === dialog)) leave();
  });
  onLeave(dialog, leave);
  // Closed before its face shows, the card stays in its vessel
  dialog.addEventListener('close', () => {
    keeping = undefined;
    cancelAnimationFrame(frame);
    frame = 0;
    from?.closed();
    from = undefined;
  });
  new ResizeObserver(() => {
    if (!dialog.open) return;
    layout();
    wake();
  }).observe(stage);

  return {
    open(held, handoff) {
      void askGpu();
      from = handoff;
      unshown = handoff.shown;
      if (!dialog.open) dialog.showModal();
      pull(held);
    },
    warm: askGpu,
  };
}
