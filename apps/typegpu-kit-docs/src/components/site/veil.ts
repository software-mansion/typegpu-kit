import { bayer, bytesOf, type Rgb } from './dither.ts';

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
   * Stops the waiting light and shows the poster undimmed, for a screen that won't go live or waits
   * to be asked; `rest(false)` starts the light again.
   */
  rest(resting?: boolean): void;
}

/** How bright a poster prints while it waits, before the passing light raises it. */
const DIM = 0.6;

/** `value`, a byte, printed in six levels through the threshold `threshold`. */
const print = (value: number, threshold: number) => {
  const level = (value / 255) * 6;
  const low = Math.floor(level);
  return ((low + (level - low > threshold ? 1 : 0)) / 6) * 255;
};

/**
 * Veils `screen` with a canvas of cells `cell` CSS pixels wide, put over its first canvas and under
 * what follows it, on the screen's own background, or printing `poster`, an image's address, as it
 * covers the screen. When `owner` is given, the veil lifts as it gains the `live` class and comes
 * back as it loses it.
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
  const content = screen.querySelector(':scope > canvas');
  if (content) content.after(canvas);
  else screen.prepend(canvas);

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
  let resized = true;
  // The poster at a pixel a cell, cropped as `object-fit: cover` would, once it has loaded
  let picture: HTMLImageElement | undefined;
  let pixels: Uint8ClampedArray | undefined;
  const size = () => {
    if (!resized) return;
    resized = false;
    const { width, height } = screen.getBoundingClientRect();
    const columns = Math.max(1, Math.ceil(width / cell));
    const rows = Math.max(1, Math.ceil(height / cell));
    if (canvas.width !== columns || canvas.height !== rows) {
      canvas.width = columns;
      canvas.height = rows;
      canvas.style.width = `${columns * cell}px`;
      canvas.style.height = `${rows * cell}px`;
      image = context.createImageData(columns, rows);
    }
    if (!picture) return;
    const scale = Math.max(columns / picture.naturalWidth, rows / picture.naturalHeight);
    const shrunk = new OffscreenCanvas(columns, rows).getContext('2d', {
      willReadFrequently: true,
    })!;
    shrunk.imageSmoothingQuality = 'high';
    const [w, h] = [picture.naturalWidth * scale, picture.naturalHeight * scale];
    shrunk.drawImage(picture, (columns - w) / 2, (rows - h) / 2, w, h);
    pixels = shrunk.getImageData(0, 0, columns, rows).data;
  };
  if (poster) {
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
  }

  let resting = false;

  // Where the waiting light is, and where the sweep's front is, from -0.3 to 1.3 across
  const draw = (light: number, front: number) => {
    const { width: columns, height: rows, data } = image;
    for (let j = 0; j < rows; j++) {
      const y = (j + 0.5) / rows;
      for (let i = 0; i < columns; i++) {
        const x = (i + 0.5) / columns;
        const threshold = bayer(i, j);
        const swept = (front - x * 0.85 - y * 0.15) * 9;
        const at = (j * columns + i) * 4;
        if (swept > threshold + 0.6) {
          data[at + 3] = 0;
          continue;
        }

        let color: Rgb;
        if (swept > threshold + 0.25) color = bright;
        else if (swept > threshold) color = accent;
        else {
          const across = (x - light) * 3.4;
          const glow = Math.exp(-across * across) * (0.55 + 0.45 * y);
          if (pixels) {
            const k = resting ? 1 : DIM + (1 - DIM) * glow;
            color = [
              Math.max(base[0], print(pixels[at] * k, threshold)),
              Math.max(base[1], print(pixels[at + 1] * k, threshold)),
              Math.max(base[2], print(pixels[at + 2] * k, threshold)),
            ];
          } else {
            const level = 2.3 * glow;
            const low = Math.min(Math.floor(level), ramp.length - 2);
            color = ramp[level - low > threshold ? low + 1 : low];
          }
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
    } else draw(light, -1);
    if (seen && (phase.kind === 'lifting' || !(still.matches || resting)))
      frame = requestAnimationFrame(tick);
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
