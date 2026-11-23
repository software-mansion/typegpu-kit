import { passes, targets, text, type Font, type Frame } from '@typegpu/kit';
import type { TgpuRoot } from 'typegpu';

const SIZE = { width: 512, height: 256 };
/** Digits and the percent sign, in pixels of the label texture. */
const DIGITS = { em: 180, x: 330, y: 128 };
const PERCENT = { em: 140, x: 340, y: 138 };

/** The slider's value as text, drawn into a texture only when it changes. */
export function createLabel(root: TgpuRoot, font: Font) {
  const page = targets.create(root, SIZE, { ink: { format: 'rgba8unorm', sampled: true } });
  const tags = text.create(root, font, { capacity: 2, length: 3, place: text.screen });
  const paint = passes.create(root, page, { ink: passes.color });
  const drawn = paint.add(tags.drawing);

  tags.write(1, '%', { em: PERCENT.em, align: 'left' });
  tags.labels.patch({
    0: { position: [DIGITS.x, DIGITS.y, 0.5], color: [1, 1, 1, 1] },
    1: { position: [PERCENT.x, PERCENT.y, 0.5], color: [1, 1, 1, 1] },
  });
  let shown = -1;

  return {
    view: page.textures.ink.createView(),
    initAsync: () => paint.initAsync(),
    /** Draws `value` into the texture unless it already shows it. */
    show(frame: Frame, value: number) {
      if (value === shown) return;
      shown = value;
      tags.write(0, String(value), { em: DIGITS.em, align: 'right' });
      paint.record(frame, drawn);
    },
    destroy() {
      page.destroy();
      tags.destroy();
    },
  };
}
