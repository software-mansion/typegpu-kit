// All visible views share one encoder and one submit per frame

import type { Frame, Surface } from '@typegpu/kit';
import type { TgpuRoot } from 'typegpu';
import { COLORS, type Colors, type Rgb } from '../site/dither.ts';
import { createInk, type ReadLockup } from './ink.ts';
import { createStacks, type StackView } from './stack.ts';
import { createSubject } from './subject.ts';

export interface HomeOptions {
  /** The canvases' format. */
  readonly format: GPUTextureFormat;
  /** The stack's colours. Default the site's tokens as they stand. */
  readonly colors?: Colors;
  /** The lockup's colours, as the header prints it. Default from `colors`. */
  readonly lockup?: { readonly rule: Rgb; readonly accent: Rgb; readonly bright: Rgb };
  /** The lockup's pixels at a size. */
  readonly read: ReadLockup;
  /** The badge beside the lockup, read the same way. */
  readonly readBadge?: ReadLockup | undefined;
}

const bytes = (c: Rgb): Rgb => [c[0] * 255, c[1] * 255, c[2] * 255];

/** A view to draw this frame: its stack as its last `write` left it, and what it adds. */
export interface Drawn {
  readonly view: StackView;
  /** Sheets its last `write` wrote. */
  readonly count: number;
  /** Records what the view adds before any stack's passes. */
  before?(frame: Frame, dt: number): void;
  /** Draws over the stack in its last pass. */
  readonly after?: Parameters<StackView['record']>[2];
}

export function createHome(root: TgpuRoot, options: HomeOptions) {
  const colors = options.colors ?? COLORS;
  const subject = createSubject(root);
  const stacks = createStacks(root, subject, colors, options.format);
  const ink = createInk(root, {
    colors: options.lockup ?? {
      rule: bytes(colors.grey[2]),
      accent: bytes(colors.accent[0]),
      bright: bytes(colors.accent[1]),
    },
    format: options.format,
    read: options.read,
    readBadge: options.readBadge,
  });
  const views = new Set<StackView>();

  return {
    subject,
    stacks,
    ink,

    /** A view of the stack on `surface`, with cells `cellCss` CSS pixels across. */
    view(surface: Surface, cellCss: () => number) {
      const view = stacks.view(surface, cellCss);
      views.add(view);
      return view;
    },

    async initAsync() {
      await Promise.all([subject.initAsync(), stacks.initAsync(), ink.initAsync()]);
    },

    /** Records `dt` seconds of the frame, then each of `drawn`. */
    record(frame: Frame, dt: number, drawn: readonly Drawn[]) {
      subject.record(frame, dt);
      for (const each of drawn) each.before?.(frame, dt);
      for (const each of drawn) each.view.record(frame, each.count, each.after);
    },

    destroy() {
      for (const view of views) view.destroy();
      views.clear();
      ink.destroy();
      stacks.destroy();
      subject.destroy();
    },
  };
}

export type Home = ReturnType<typeof createHome>;
