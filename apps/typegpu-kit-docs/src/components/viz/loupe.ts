import type { Frame, Surface } from '@typegpu/kit';
import * as web from '@typegpu/kit/web';
import { common, d, std, tgpu, type TgpuRoot } from 'typegpu';
import { clamp } from '../home/geometry.ts';
import { onTap } from './figure.ts';

/** Pixels across the square a loupe magnifies, unless it says otherwise. */
export const TEXELS = 24;

const layout = tgpu.bindGroupLayout({ square: { texture: d.texture2d(d.f32) } });

/** Each pixel of the square as a block, nearest-neighbour. */
const blocks = tgpu.fragmentFn({ in: { uv: d.vec2f }, out: d.vec4f })(({ uv }) => {
  'use gpu';
  const size = d.vec2i(std.textureDimensions(layout.$.square));
  const texel = std.min(d.vec2i(uv * d.vec2f(size)), size - 1);
  return std.textureLoad(layout.$.square, texel, 0);
});

/** The first pixel of the `texels` wide run whose middle is nearest `at`, kept within `size`. */
export const cornerOf = (at: number, size: number, texels: number) =>
  Math.max(0, Math.min(size - texels, Math.round(at - texels / 2)));

export interface Lens extends Disposable {
  /**
   * Copies the square of `source` whose middle is nearest `x`, `y`, in pixels, kept inside it, and
   * draws it into `target`, in the frame's encoder after the pass that drew `source`. A source
   * smaller than the square draws nothing.
   */
  draw(frame: Frame, source: GPUTexture, target: Surface, x: number, y: number): void;
  /** Destroys the square it copies into. */
  destroy(): void;
}

/**
 * A loupe's GPU part: a `texels` wide square of a texture of `format`, which must be copyable,
 * drawn nearest-neighbour into a surface of the same format.
 */
export function create(root: TgpuRoot, format: GPUTextureFormat, texels = TEXELS): Lens {
  const square = root.createTexture({ size: [texels, texels], format }).$usage('sampled');
  const copied = root.unwrap(square);
  const drawing = root
    .createRenderPipeline({
      vertex: common.fullScreenTriangle,
      fragment: blocks,
      targets: { format },
    })
    .with(root.createBindGroup(layout, { square }));

  const destroy = () => square.destroy();

  return {
    draw(frame, source, target, x, y) {
      if (source.width < texels || source.height < texels) return;

      const origin = {
        x: cornerOf(x, source.width, texels),
        y: cornerOf(y, source.height, texels),
      };
      root
        .unwrap(frame.encoder)
        .copyTextureToTexture({ texture: source, origin }, { texture: copied }, [texels, texels]);

      const pass = frame.encoder.beginRenderPass({
        colorAttachments: { view: target.view(frame) },
      });
      pass.setPipeline(drawing);
      pass.draw(3);
      pass.end();
    },

    destroy,
    [Symbol.dispose]: destroy,
  };
}

export interface Placement {
  /** Where it looks first, as fractions of the canvas from its top left. Default the middle. */
  readonly at: readonly [number, number];
}

/** What a `Loupe.astro` element asks for, in its `data-loupe`. */
export const placementOf = (element: HTMLElement): Placement => ({
  at: [0.5, 0.5],
  ...(JSON.parse(element.dataset.loupe || '{}') as Partial<Placement>),
});

export interface Loupe {
  /** Copies and draws the square, after the scene's frame drew the canvas. */
  draw(frame: Frame): void;
  destroy(): void;
}

/**
 * Runs the `Loupe.astro` element `element` over `canvas`, which `surface` draws and whose texture
 * can be copied: its canvas shows the square, and its outline marks it on `canvas`. It calls
 * `redraw` when it moves, so a paused figure shows the new square.
 */
export function attach(
  element: HTMLElement,
  canvas: HTMLCanvasElement,
  root: TgpuRoot,
  surface: web.Surface,
  redraw: () => void,
): Loupe {
  const lensCanvas = element.querySelector('canvas');
  const outline = element.querySelector<HTMLElement>('[data-outline]');
  if (!lensCanvas || !outline) throw new Error('The loupe has no canvas or outline.');

  const { at } = placementOf(element);
  const texels = TEXELS;
  const target = web.surface(root, lensCanvas, { format: surface.format });
  const lens = create(root, surface.format, texels);
  let point = [...at];

  const listening = new AbortController();
  const { signal } = listening;
  const move = (event: PointerEvent) => {
    point = [clamp(event.offsetX / canvas.clientWidth), clamp(event.offsetY / canvas.clientHeight)];
    redraw();
  };

  // It follows the pointer as it hovers, and goes where the canvas is tapped
  canvas.addEventListener('pointermove', (event) => event.buttons === 0 && move(event), { signal });
  onTap(canvas, move, signal);

  return {
    draw(frame) {
      const source = surface.context.getCurrentTexture();
      const x = point[0] * source.width;
      const y = point[1] * source.height;
      lens.draw(frame, source, target, x, y);

      // The outline sits on the square the copy took, in CSS pixels
      const scale = canvas.clientWidth / source.width;
      const side = `${texels * scale}px`;
      outline.style.width = side;
      outline.style.height = side;
      outline.style.translate = `${cornerOf(x, source.width, texels) * scale}px ${cornerOf(y, source.height, texels) * scale}px`;
      element.classList.toggle('left', point[0] > 0.5);
    },

    destroy() {
      listening.abort();
      lens.destroy();
      target.destroy();
    },
  };
}
