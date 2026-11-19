// Draws the profiler graph's shapes with WebGPU, for the React Native panel.

import { d, tgpu, type TgpuRoot } from 'typegpu';
import type { Frame } from '../frames.ts';
import { blend } from '../pipelines.ts';
import { rectFloats } from './graphLayout.ts';

const Rect = d.struct({ box: d.vec4f, color: d.vec4f, shift: d.vec2f });

const layout = tgpu.bindGroupLayout({
  rects: { storage: d.arrayOf(Rect), access: 'readonly' },
  view: { uniform: d.vec2f },
});

const vertex = tgpu.vertexFn({
  in: { vertexIndex: d.builtin.vertexIndex, instanceIndex: d.builtin.instanceIndex },
  out: { position: d.builtin.position, color: d.vec4f },
})(({ vertexIndex, instanceIndex }) => {
  'use gpu';
  const rect = Rect(layout.$.rects[instanceIndex]);
  const corner = d.vec2f(d.f32(vertexIndex & 1), d.f32((vertexIndex >>> 1) & 1));
  // The right edge's top and bottom move down by `shift`, for the legend's links
  const shift = (rect.shift.x + (rect.shift.y - rect.shift.x) * corner.y) * corner.x;
  const point = rect.box.xy + corner * rect.box.zw + d.vec2f(0, shift);
  const ndc = (point / layout.$.view) * d.vec2f(2, -2) + d.vec2f(-1, 1);
  return { position: d.vec4f(ndc, 0, 1), color: rect.color };
});

const fragment = tgpu.fragmentFn({ in: { color: d.vec4f }, out: d.vec4f })(({ color }) => {
  'use gpu';
  return color;
});

/** Draws shapes laid out by `graphLayout` into a pass. */
export interface GraphDrawing {
  /**
   * Writes `count` shapes and draws them over a cleared `view`, in a graph `width` by `height`
   * points.
   */
  draw(
    frame: Frame,
    view: GPUTextureView,
    rects: Float32Array<ArrayBuffer>,
    count: number,
    width: number,
    height: number,
  ): void;
  destroy(): void;
}

/**
 * Creates a pipeline drawing up to `capacity` premultiplied shapes into `format` targets, with
 * a buffer for them.
 */
export function createGraphDrawing(
  root: TgpuRoot,
  format: GPUTextureFormat,
  capacity: number,
): GraphDrawing {
  const rects = root.createBuffer(d.arrayOf(Rect, capacity)).$usage('storage');
  const view = root.createBuffer(d.vec2f).$usage('uniform');
  const gpuRects = root.unwrap(rects);
  const group = root.createBindGroup(layout, { rects, view });
  const pipeline = root.createRenderPipeline({
    vertex,
    fragment,
    targets: { format, blend: blend.over },
    primitive: { topology: 'triangle-strip' },
  });
  const drawing = pipeline.with(group);
  const size = new Float32Array(2);

  return {
    draw(frame, target, data, count, width, height) {
      // Write only the drawn shapes, without copying them out of the layout's array
      if (count > 0) root.device.queue.writeBuffer(gpuRects, 0, data, 0, count * rectFloats);
      size[0] = width;
      size[1] = height;
      view.write(size.buffer);

      const pass = frame.encoder.beginRenderPass({
        colorAttachments: [
          { view: target, loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 0] },
        ],
      });
      if (count > 0) {
        pass.setPipeline(drawing);
        pass.draw(4, count);
      }
      pass.end();
    },

    destroy() {
      rects.destroy();
      view.destroy();
    },
  };
}
