import { createGraph } from './graph.ts';
import { theme } from './graphLayout.ts';
import { requestUntrackedFrame } from './instrument.ts';
import { type Measurement, type MeasureOptions, measureText } from './measure.ts';
import {
  captureText,
  type Detail,
  detailLabels,
  details,
  formatMs,
  type FrameCapture,
  type Recorder,
} from './recorder.ts';

interface Layout {
  left: number;
  top: number;
  width: number;
  height: number;
  collapsed: boolean;
  detail: Detail;
  /** The setting before `detail`, read from older layouts. */
  split?: boolean;
}

const storageKey = 'typegpu-kit';
const defaultSize = { width: 440, height: 280 };
const minSize = { width: 360, height: 160 };
const compactWidth = 420;
const margin = 16;
const titleHeight = 26;
const statsInterval = 250;

const styles = {
  window: {
    position: 'fixed',
    zIndex: '9999',
    borderRadius: '6px',
    overflow: 'hidden',
    background: 'rgba(22, 24, 29, 0.94)',
    border: '1px solid rgba(255, 255, 255, 0.08)',
    boxShadow: '0 10px 30px rgba(0, 0, 0, 0.45)',
    color: theme.text,
    font: theme.font,
    userSelect: 'none',
  },
  titleBar: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    height: `${titleHeight}px`,
    padding: '0 8px',
    background: 'rgba(255, 255, 255, 0.06)',
    cursor: 'grab',
    fontSize: '12px',
  },
  title: { flex: 'none' },
  stats: {
    flex: '1',
    minWidth: '0',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
    color: theme.muted,
    fontVariantNumeric: 'tabular-nums',
  },
  button: {
    flex: 'none',
    font: 'inherit',
    color: '#aab',
    background: 'none',
    border: '0',
    outline: 'none',
    padding: '0 2px',
    whiteSpace: 'nowrap',
    cursor: 'pointer',
  },
  canvas: { display: 'block', width: '100%', cursor: 'crosshair' },
  capture: {
    maxHeight: '220px',
    overflow: 'auto',
    padding: '6px 10px 10px',
    borderTop: '1px solid rgba(255, 255, 255, 0.08)',
    font: '500 11px/1.5 ui-monospace, monospace',
    userSelect: 'text',
  },
  captureHeader: { display: 'flex', justifyContent: 'space-between', color: theme.muted },
  captureDetail: { color: theme.muted, paddingLeft: '16px' },
  wgsl: { margin: '2px 0 6px', color: theme.text, font: 'inherit', whiteSpace: 'pre' },
  grip: {
    position: 'absolute',
    right: '0',
    bottom: '0',
    width: '12px',
    height: '12px',
    background: `linear-gradient(135deg, transparent 50%, ${theme.accent} 50%)`,
    cursor: 'nwse-resize',
  },
} satisfies Record<string, Partial<CSSStyleDeclaration>>;

function element<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  style: Partial<CSSStyleDeclaration> = {},
  text = '',
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  node.textContent = text;
  Object.assign(node.style, style);
  return node;
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(value, max));
}

function loadLayout(): Partial<Layout> {
  try {
    return JSON.parse(localStorage.getItem(storageKey) ?? '{}') as Partial<Layout>;
  } catch {
    return {};
  }
}

function saveLayout(layout: Layout) {
  try {
    localStorage.setItem(storageKey, JSON.stringify(layout));
  } catch {
    // Ignore storage failures so the panel works when persistent storage is disabled
  }
}

function download(filename: string, data: unknown) {
  const url = URL.createObjectURL(new Blob([JSON.stringify(data)], { type: 'application/json' }));
  Object.assign(document.createElement('a'), { href: url, download: filename }).click();
  URL.revokeObjectURL(url);
}

function draggable(
  handle: HTMLElement,
  begin: () => (dx: number, dy: number) => void,
  end: () => void,
) {
  handle.addEventListener('pointerdown', (down) => {
    if (down.button !== 0 || (down.target as HTMLElement).closest('button')) return;

    down.preventDefault();
    const drag = begin();
    const move = (event: PointerEvent) => {
      drag(event.clientX - down.clientX, event.clientY - down.clientY);
    };
    const stop = () => {
      handle.removeEventListener('pointermove', move);
      handle.removeEventListener('lostpointercapture', stop);
      end();
    };
    handle.addEventListener('pointermove', move);
    handle.addEventListener('lostpointercapture', stop);
    handle.setPointerCapture(down.pointerId);
  });
}

function captureRows({ commands }: FrameCapture, colorOf: (name: string) => string): HTMLElement[] {
  return commands.map((command) => {
    const row = element('div');
    row.append(
      element('span', { color: theme.muted }, `#${command.index} `),
      element('span', {}, command.ms === undefined ? '' : `${formatMs(command.ms)} `),
      element(
        'span',
        { color: command.ms === undefined ? theme.muted : colorOf(command.displayName) },
        `${command.displayName}${command.calls > 1 ? ` ×${command.calls}` : ''}`,
      ),
    );
    const reads = command.after.map(
      ({ from, resource }) =>
        `${from === undefined ? 'earlier frame' : `#${from}`} ${resource.label}`,
    );
    const writes = command.uses
      .filter(({ access }) => access !== 'read')
      .map(({ resource }) => resource.label);
    for (const [arrow, labels] of [
      ['←', reads],
      ['→', writes],
    ] as const) {
      if (labels.length > 0) {
        row.append(
          element('div', styles.captureDetail, `${arrow} ${[...new Set(labels)].join(', ')}`),
        );
      }
    }
    if (command.wgsl.length > 0) {
      const shader = element('details', styles.captureDetail);
      shader.append(
        element('summary', { cursor: 'pointer' }, 'WGSL'),
        element('pre', styles.wgsl, command.wgsl.join('\n')),
      );
      row.append(shader);
    }
    return row;
  });
}

/**
 * Mounts the draggable, resizable panel on the page. `measure` runs for its measure button.
 * Returns a function that removes it.
 */
export function mountPanel(
  recorder: Recorder,
  measure: (options?: MeasureOptions) => Promise<Measurement>,
): () => void {
  const panel = element('div', styles.window);
  const titleBar = element('div', styles.titleBar);
  const toggle = element('button', styles.button);
  const title = element('span', styles.title, 'GPU debug');
  const stats = element('span', styles.stats);
  const detailButton = element('button', styles.button);
  const measureButton = element('button', styles.button, 'Measure');
  const captureButton = element('button', styles.button, 'Capture');
  const logButton = element('button', styles.button, 'Log');
  const traceButton = element('button', styles.button, 'Save trace');
  const body = element('div');
  const canvas = element('canvas', styles.canvas);
  const capture = element('div', styles.capture);
  const grip = element('div', styles.grip);
  const graph = createGraph(canvas);
  let collapsed = false;
  let frameId = 0;
  let statsTime = -Infinity;

  titleBar.append(
    toggle,
    title,
    stats,
    detailButton,
    measureButton,
    captureButton,
    traceButton,
    logButton,
  );
  body.append(canvas, capture);
  panel.append(titleBar, body, grip);
  document.body.append(panel);

  // Keep the requested size so resizing the window can restore it after fitting to a smaller
  // viewport
  const stored = loadLayout();
  const wanted = {
    width: stored.width ?? defaultSize.width,
    height: stored.height ?? defaultSize.height,
  };
  const fit = () => {
    const widest = window.innerWidth - 2 * margin;
    const width = clamp(wanted.width, Math.min(minSize.width, widest), widest);
    const highest = Math.max(minSize.height, window.innerHeight - titleHeight - 2 * margin);
    panel.style.width = `${width}px`;
    canvas.style.height = `${clamp(wanted.height, minSize.height, highest)}px`;
    title.hidden = width < compactWidth;
  };
  const place = (left: number, top: number) => {
    panel.style.left = `${clamp(left, 0, window.innerWidth - panel.offsetWidth)}px`;
    panel.style.top = `${clamp(top, 0, window.innerHeight - titleBar.offsetHeight)}px`;
  };
  const keepInView = () => {
    fit();
    const { left, top } = panel.getBoundingClientRect();
    place(left, top);
  };
  const setCollapsed = (value: boolean) => {
    collapsed = value;
    body.hidden = value;
    grip.hidden = value;
    toggle.textContent = value ? '▸' : '▾';
  };
  const setDetail = (detail: Detail) => {
    recorder.setDetail(detail);
    detailButton.textContent = detailLabels[detail];
  };
  const save = () => {
    const { left, top } = panel.getBoundingClientRect();
    saveLayout({ left, top, ...wanted, collapsed, detail: recorder.detail() });
  };
  const show = (heading: string, rows: HTMLElement[]) => {
    const close = element('button', styles.button, '×');
    close.addEventListener('click', () => {
      capture.hidden = true;
    });
    const header = element('div', styles.captureHeader, heading);
    header.append(close);
    capture.replaceChildren(header, ...rows);
    capture.hidden = false;
  };

  fit();
  capture.hidden = true;
  setCollapsed(stored.collapsed ?? false);
  setDetail(
    details.includes(stored.detail as Detail)
      ? stored.detail!
      : stored.split === false
        ? 'pass'
        : 'dispatch',
  );
  // Place the panel at bottom-left to avoid lil-gui and example captions, and keep saved
  // positions within the viewport
  const lowest = window.innerHeight - panel.offsetHeight - margin;
  place(stored.left ?? margin, Math.max(0, Math.min(stored.top ?? lowest, lowest)));

  draggable(
    titleBar,
    () => {
      const { left, top } = panel.getBoundingClientRect();
      return (dx, dy) => place(left + dx, top + dy);
    },
    save,
  );
  draggable(
    grip,
    () => {
      const { left, width } = panel.getBoundingClientRect();
      const height = canvas.offsetHeight;
      return (dx, dy) => {
        wanted.width = Math.min(width + dx, window.innerWidth - left);
        wanted.height = height + dy;
        fit();
      };
    },
    save,
  );

  toggle.addEventListener('click', () => {
    setCollapsed(!collapsed);
    save();
  });

  detailButton.addEventListener('click', () => {
    setDetail(details[(details.indexOf(recorder.detail()) + 1) % details.length]);
    save();
  });

  measureButton.addEventListener('click', async () => {
    measureButton.disabled = true;
    measureButton.textContent = 'Measuring';
    try {
      const result = await measure();
      const text = measureText(result);
      console.log(`[gpu debug] measure\n${text}`, result);
      const [heading = '', ...lines] = text.split('\n');
      show(
        heading,
        lines.map((line) => element('div', {}, line)),
      );
    } catch (error) {
      show(`measure failed: ${error instanceof Error ? error.message : String(error)}`, []);
    } finally {
      measureButton.disabled = false;
      measureButton.textContent = 'Measure';
    }
  });

  traceButton.addEventListener('click', () => download('gpu-trace.json', recorder.trace()));
  logButton.addEventListener('click', () => console.log(`[gpu debug]\n${recorder.summary()}`));

  captureButton.addEventListener('click', async () => {
    captureButton.disabled = true;
    try {
      const result = await recorder.capture();
      console.log(`[gpu debug] capture\n${captureText(result)}`, result);

      const timings = result.timings === 'complete' ? '' : ` · timings ${result.timings}`;
      show(`frame ${result.frame}${timings}`, captureRows(result, graph.colorOf));
    } catch {
      // Stopping rejects the capture and removes the panel
    } finally {
      captureButton.disabled = false;
    }
  });

  const draw = (time: number) => {
    frameId = requestUntrackedFrame(draw);

    if (time - statsTime > statsInterval) {
      const fps = recorder.fps();
      stats.textContent = fps > 0 ? `${fps.toFixed(0)} fps · ${formatMs(1000 / fps)}` : '';
      statsTime = time;
    }
    if (!collapsed) {
      graph.draw(
        recorder.bars(),
        recorder.timestampQuery() === false
          ? 'The device does not support timestamp-query.'
          : undefined,
      );
    }
  };
  frameId = requestUntrackedFrame(draw);
  window.addEventListener('resize', keepInView);

  return () => {
    cancelAnimationFrame(frameId);
    window.removeEventListener('resize', keepInView);
    panel.remove();
  };
}
