import { bayer, bytesOf, plain, type Rgb } from './dither.ts';

/** Seconds a sweep takes to cross the veil. */
const LIFT = 0.6;

/** Seconds the light takes to cross the veil while it waits. */
const PASS = 2.6;

export interface Veil {
  /** Sweeps the veil off; resolves once it is gone, or covered again first. */
  lift(): Promise<void>;
  /** Puts the veil back, at once, settling a lift under way. */
  cover(): void;
  /**
   * Stops the waiting light, for a screen that won't go live or waits to be asked; `rest(false)`
   * starts it again. A loaded poster shows no light.
   */
  rest(resting?: boolean): void;
}

/** Cells deep the printed edge around a poster is fully covered. */
const SOLID = 6;

/** Cells further in over which the printed edge thins to nothing. */
const FADE = 5;

/** CSS pixels wide below which a screen is a small thumbnail, whose printed edge is half as deep. */
const SMALL = 480;

/** `value`, a byte, printed in six levels through the threshold `threshold`. */
const print = (value: number, threshold: number) => {
  const level = (value / 255) * 6;
  const low = Math.floor(level);
  return ((low + (level - low > threshold ? 1 : 0)) / 6) * 255;
};

/**
 * Veils `screen` with a canvas of cells `cell` CSS pixels wide, put over its first canvas and under
 * what follows it, on the screen's own background, or showing `poster`, an image's address, as it
 * covers the screen: sharp, inside an edge printed in the cells unless the reader turned the dither
 * off. When `owner` is given, the veil lifts as it gains the `live` class and comes back as it
 * loses it.
 */
export function veil(
  screen: HTMLElement,
  {
    owner,
    cell = 3,
    poster,
  }: {
    owner?: HTMLElement;
    cell?: number;
    poster?: string | undefined;
  } = {},
): Veil {
  const still = matchMedia('(prefers-reduced-motion: reduce)');
  const canvas = document.createElement('canvas');
  canvas.className = 'veil';
  canvas.setAttribute('aria-hidden', 'true');
  Object.assign(canvas.style, {
    position: 'absolute',
    top: '0',
    left: '0',
    pointerEvents: 'none',
    imageRendering: 'pixelated',
  });
  // A sharp poster at the screen's own resolution, under the cells, which print its edge and sweep
  const plate = poster ? document.createElement('canvas') : undefined;
  if (plate) {
    plate.setAttribute('aria-hidden', 'true');
    Object.assign(plate.style, {
      position: 'absolute',
      top: '0',
      left: '0',
      pointerEvents: 'none',
    });
  }
  const content = screen.querySelector(':scope > canvas');
  if (content) content.after(...(plate ? [plate, canvas] : [canvas]));
  else screen.prepend(...(plate ? [plate, canvas] : [canvas]));

  const [base, raised, higher, highest, accent, bright] = bytesOf(screen, [
    getComputedStyle(screen).backgroundColor,
    'var(--ground-3)',
    'var(--ground-4)',
    'var(--rule-strong)',
    'var(--accent)',
    'var(--accent-high)',
  ]) as [Rgb, Rgb, Rgb, Rgb, Rgb, Rgb];
  const ramp = [base, raised, higher, highest];

  const context = canvas.getContext('2d')!;
  let image = context.createImageData(1, 1);
  // Where the sweep has passed or is passing, the cells the poster is cut out of
  let cut = new ImageData(1, 1);
  const cutCanvas = new OffscreenCanvas(1, 1);
  let resized = true;
  let picture: HTMLImageElement | undefined;
  // The poster at a pixel a cell, cropped as `object-fit: cover` would, for its printed edge
  let pixels: Uint8ClampedArray | undefined;
  // Plate pixels a CSS pixel
  let pixel = 1;
  // How deep the printed edge is, against a large screen's
  let edge = 1;
  // Bottom rows a cell of the edge's depth there
  let stretch = 1;
  const size = () => {
    if (!resized) return;
    resized = false;
    const { width, height } = screen.getBoundingClientRect();
    edge = width < SMALL ? 0.5 : 1;
    const columns = Math.max(1, Math.ceil(width / cell));
    const rows = Math.max(1, Math.ceil(height / cell));
    // A screen with a caption over it sets `--veil-bottom`, the share of its height the bottom edge
    // reaches, stretching that edge's solid and fading parts alike
    const bottom = Number.parseFloat(getComputedStyle(screen).getPropertyValue('--veil-bottom'));
    stretch = bottom > 0 ? (bottom * rows) / ((SOLID + FADE) * edge) : 1;
    if (canvas.width !== columns || canvas.height !== rows) {
      canvas.width = columns;
      canvas.height = rows;
      canvas.style.width = `${columns * cell}px`;
      canvas.style.height = `${rows * cell}px`;
      image = context.createImageData(columns, rows);
      cut = new ImageData(columns, rows);
      cutCanvas.width = columns;
      cutCanvas.height = rows;
    }
    if (!picture || !plate) return;
    if (!plain) {
      const shrunk = new OffscreenCanvas(columns, rows).getContext('2d', {
        willReadFrequently: true,
      })!;
      cover(shrunk, picture);
      pixels = shrunk.getImageData(0, 0, columns, rows).data;
    }
    pixel = devicePixelRatio;
    plate.width = Math.max(1, Math.round(width * pixel));
    plate.height = Math.max(1, Math.round(height * pixel));
    plate.style.width = `${width}px`;
    plate.style.height = `${height}px`;
  };

  // Loaded as the screen nears the viewport
  let requested = false;
  const request = () => {
    if (requested || !poster) return;
    requested = true;
    const loading = new Image();
    loading.src = poster;
    loading.decode().then(
      () => {
        picture = loading;
        resized = true;
        run();
      },
      () => {},
    );
  };

  // The poster, cropped as `object-fit: cover` would, less the cells the sweep has reached
  const paint = (sweeping: boolean) => {
    if (!plate || !picture) return;
    const plating = plate.getContext('2d')!;
    plating.globalCompositeOperation = 'source-over';
    plating.imageSmoothingEnabled = true;
    cover(plating, picture);
    if (!sweeping) return;
    cutCanvas.getContext('2d')!.putImageData(cut, 0, 0);
    plating.globalCompositeOperation = 'destination-out';
    plating.imageSmoothingEnabled = false;
    plating.drawImage(cutCanvas, 0, 0, cut.width * cell * pixel, cut.height * cell * pixel);
  };

  let resting = false;

  // Where the waiting light is, and where the sweep's front is, from -0.3 to 1.3 across
  const draw = (light: number, front: number) => {
    const { width: columns, height: rows, data } = image;
    const gone = cut.data;
    for (let j = 0; j < rows; j++) {
      const y = (j + 0.5) / rows;
      for (let i = 0; i < columns; i++) {
        const x = (i + 0.5) / columns;
        const threshold = bayer(i, j);
        const swept = (front - x * 0.85 - y * 0.15) * 9;
        const at = (j * columns + i) * 4;
        if (plate) gone[at + 3] = swept > threshold ? 255 : 0;
        // Full in the outer `SOLID` cells, thinning to none over the next `FADE`, both by `edge`
        const depth = Math.min(i, j, columns - 1 - i, (rows - 1 - j) / stretch);
        const edged = pixels && threshold < 1 - (depth - SOLID * edge + 1) / (FADE * edge + 1);
        if (swept > threshold + 0.6 || (picture && swept <= threshold && !edged)) {
          data[at + 3] = 0;
          continue;
        }

        let color: Rgb;
        if (swept > threshold + 0.25) color = bright;
        else if (swept > threshold) color = accent;
        else if (pixels) {
          color = [
            Math.max(base[0], print(pixels[at], threshold)),
            Math.max(base[1], print(pixels[at + 1], threshold)),
            Math.max(base[2], print(pixels[at + 2], threshold)),
          ];
        } else {
          const across = (x - light) * 3.4;
          const glow = Math.exp(-across * across) * (0.55 + 0.45 * y);
          const level = 2.3 * glow;
          const low = Math.min(Math.floor(level), ramp.length - 2);
          color = ramp[level - low > threshold ? low + 1 : low];
        }
        data[at] = color[0];
        data[at + 1] = color[1];
        data[at + 2] = color[2];
        data[at + 3] = 255;
      }
    }
    context.putImageData(image, 0, 0);
  };

  let phase:
    | { kind: 'covered' }
    | { kind: 'lifting'; start: number; done: (() => void)[] }
    | { kind: 'gone' } = { kind: 'covered' };
  // Gone is off the screen; not `hidden`, which viz.css reads as the figure's canvas having failed
  const show = (shown: boolean) => {
    const before = phase;
    phase = { kind: shown ? 'covered' : 'gone' };
    canvas.style.display = shown ? '' : 'none';
    if (plate) plate.style.display = shown ? '' : 'none';
    if (before.kind === 'lifting') for (const done of before.done) done();
  };

  let frame = 0;
  // Drawn once at first; it animates once the observer says it is on screen
  let seen = false;

  const tick = (now: number) => {
    frame = 0;
    size();
    const light = still.matches || resting ? -1 : ((now / 1000 / PASS) % 1) * 1.6 - 0.3;
    if (phase.kind === 'lifting') {
      const k = Math.min((now - phase.start) / 1000 / LIFT, 1);
      draw(light, -0.3 + 1.9 * (1 - (1 - k) ** 2));
      if (k === 1) return show(false);
      paint(true);
    } else {
      draw(light, -1);
      paint(false);
    }
    // A poster holds still until the sweep
    const waiting = !picture && !(still.matches || resting);
    if (seen && (phase.kind === 'lifting' || waiting)) frame = requestAnimationFrame(tick);
  };
  const run = () => {
    if (!frame && phase.kind !== 'gone') frame = requestAnimationFrame(tick);
  };

  new IntersectionObserver(([entry]) => {
    seen = entry?.isIntersecting ?? true;
    if (seen) run();
  }).observe(screen);

  const result: Veil = {
    lift: () =>
      new Promise<void>((done) => {
        if (phase.kind === 'gone') return done();
        if (still.matches) {
          show(false);
          return done();
        }
        if (phase.kind === 'covered')
          phase = { kind: 'lifting', start: performance.now(), done: [] };
        phase.done.push(done);
        run();
      }),
    cover() {
      show(true);
      run();
    },
    rest(on = true) {
      resting = on;
      run();
    },
  };

  if (poster) {
    new IntersectionObserver(([entry]) => entry?.isIntersecting && request(), {
      rootMargin: '400px 0px',
    }).observe(screen);
  }

  new ResizeObserver(() => {
    resized = true;
    run();
  }).observe(screen);

  if (owner) {
    new MutationObserver(() => {
      if (owner.classList.contains('live')) void result.lift();
      else if (phase.kind !== 'covered') result.cover();
    }).observe(owner, { attributes: true, attributeFilter: ['class'] });
  }
  run();
  return result;
}

/** Draws `picture` over all of `context`'s canvas, cropped as `object-fit: cover` would. */
function cover(
  context: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D,
  picture: HTMLImageElement,
) {
  const { width, height } = context.canvas;
  context.clearRect(0, 0, width, height);
  context.imageSmoothingQuality = 'high';
  const scale = Math.max(width / picture.naturalWidth, height / picture.naturalHeight);
  const [w, h] = [picture.naturalWidth * scale, picture.naturalHeight * scale];
  context.drawImage(picture, (width - w) / 2, (height - h) / 2, w, h);
}
