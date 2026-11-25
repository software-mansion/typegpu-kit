import { d, std, tgpu, type TgpuRoot } from 'typegpu';
import { afterAll, beforeAll, describe, expect, test } from 'vitest';
import type { GLDevice } from '../../src/device/index.ts';
import {
  compare,
  compareRoots,
  errorsOf,
  hasWebGpu,
  pixel,
  readTexture,
  roots,
} from './compare.ts';

let gpu: TgpuRoot;
let gl: TgpuRoot;

beforeAll(async () => {
  if (!hasWebGpu) return;
  ({ gpu, gl } = await roots());
});

afterAll(() => {
  gpu?.destroy();
  gl?.destroy();
});

const SIZE = 16;

const fullScreen = tgpu.vertexFn({
  in: { index: d.builtin.vertexIndex },
  out: { position: d.builtin.position },
})(({ index }) => {
  'use gpu';
  const corners = [d.vec2f(-1, -1), d.vec2f(3, -1), d.vec2f(-1, 3)];
  return { position: d.vec4f(corners[index]!, 0, 1) };
});

/** A triangle over the bottom-right corner of the target. */
const corner = tgpu.vertexFn({
  in: { index: d.builtin.vertexIndex },
  out: { position: d.builtin.position },
})(({ index }) => {
  'use gpu';
  const corners = [d.vec2f(0, -1), d.vec2f(1, -1), d.vec2f(1, 0)];
  return { position: d.vec4f(corners[index]!, 0, 1) };
});

const solid = (color: d.v4f) =>
  tgpu.fragmentFn({ out: d.vec4f })(() => {
    'use gpu';
    return color;
  });

/** A triangle over the left half of the target at depth 0.25. */
const left = tgpu.vertexFn({
  in: { index: d.builtin.vertexIndex },
  out: { position: d.builtin.position },
})(({ index }) => {
  'use gpu';
  const corners = [d.vec2f(-1, -1), d.vec2f(0, -1), d.vec2f(-1, 3)];
  return { position: d.vec4f(corners[index]!, 0.25, 1) };
});

/** A full-screen triangle at depth 0.5. */
const middle = tgpu.vertexFn({
  in: { index: d.builtin.vertexIndex },
  out: { position: d.builtin.position },
})(({ index }) => {
  'use gpu';
  const corners = [d.vec2f(-1, -1), d.vec2f(3, -1), d.vec2f(-1, 3)];
  return { position: d.vec4f(corners[index]!, 0.5, 1) };
});

/**
 * A pass writes depth and stencil 1 over the left half, discarding one aspect and storing the
 * other, and a second pass draws green where the stored aspect lets it.
 */
function oneAspectStored(keep: 'depth' | 'stencil') {
  return (root: TgpuRoot) => {
    const color = root
      .createTexture({ size: [SIZE, SIZE], format: 'rgba8unorm' })
      .$usage('render', 'sampled');
    const depth = root
      .createTexture({ size: [SIZE, SIZE], format: 'depth24plus-stencil8' })
      .$usage('render');
    root
      .createRenderPipeline({
        vertex: left,
        fragment: solid(d.vec4f(1, 0, 0, 1)),
        targets: { format: 'rgba8unorm' },
        depthStencil: {
          format: 'depth24plus-stencil8',
          depthCompare: 'always',
          depthWriteEnabled: true,
          stencilFront: { compare: 'always', passOp: 'replace' },
          stencilBack: { compare: 'always', passOp: 'replace' },
        },
      })
      .withColorAttachment({ view: color, clearValue: [0, 0, 0, 1] })
      .withDepthStencilAttachment({
        view: depth,
        depthClearValue: 1,
        depthLoadOp: 'clear',
        depthStoreOp: keep === 'depth' ? 'store' : 'discard',
        stencilClearValue: 0,
        stencilLoadOp: 'clear',
        stencilStoreOp: keep === 'stencil' ? 'store' : 'discard',
      })
      .withStencilReference(1)
      .draw(3);
    root
      .createRenderPipeline({
        vertex: middle,
        fragment: solid(d.vec4f(0, 1, 0, 1)),
        targets: { format: 'rgba8unorm' },
        depthStencil: {
          format: 'depth24plus-stencil8',
          depthCompare: keep === 'depth' ? 'less' : 'always',
          depthWriteEnabled: false,
          stencilFront: { compare: keep === 'stencil' ? 'equal' : 'always' },
          stencilBack: { compare: keep === 'stencil' ? 'equal' : 'always' },
        },
      })
      .withColorAttachment({ view: color, loadOp: 'load' })
      .withDepthStencilAttachment({
        view: depth,
        depthClearValue: 1,
        depthLoadOp: keep === 'depth' ? 'load' : 'clear',
        depthStoreOp: 'store',
        stencilClearValue: 0,
        stencilLoadOp: keep === 'stencil' ? 'load' : 'clear',
        stencilStoreOp: 'store',
      })
      .withStencilReference(1)
      .draw(3);
    return color;
  };
}

/** Rows of red, green, blue and white, top to bottom, as an `ImageBitmap` and a canvas. */
async function rows() {
  const canvas = new OffscreenCanvas(4, 4);
  const context = canvas.getContext('2d')!;
  ['#ff0000', '#00ff00', '#0000ff', '#ffffff'].forEach((style, row) => {
    context.fillStyle = style;
    context.fillRect(0, row, 4, 1);
  });
  return { canvas, bitmap: await createImageBitmap(canvas) };
}

/** Copies the top two rows of `source`, flipped, into a 4x2 texture and reads it back. */
async function flippedRegion(device: GPUDevice, source: ImageBitmap | OffscreenCanvas) {
  const texture = device.createTexture({
    size: [4, 2],
    format: 'rgba8unorm',
    usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.COPY_SRC | GPUTextureUsage.RENDER_ATTACHMENT,
  });
  device.queue.copyExternalImageToTexture(
    { source, origin: [0, 0], flipY: true },
    { texture },
    [4, 2],
  );
  const read = await readTexture(device, texture);
  texture.destroy();
  return read;
}

/**
 * A pass fills the target red, a kernel the barrier keeps on the CPU stores green into its top
 * half, and a pass with the same attachments loads it and draws blue over a corner.
 */
function kernelBetweenPasses(root: TgpuRoot) {
  const target = root
    .createTexture({ size: [SIZE, SIZE], format: 'rgba8unorm' })
    .$usage('render', 'storage');
  const storage = target.createView(d.textureStorage2d('rgba8unorm', 'write-only'));
  const store = tgpu.computeFn({ in: { id: d.builtin.globalInvocationId }, workgroupSize: [4, 4] })(
    ({ id }) => {
      'use gpu';
      std.workgroupBarrier();
      if (id.y < d.u32(SIZE / 2)) {
        std.textureStore(storage.$, d.vec2i(id.xy), d.vec4f(0, 1, 0, 1));
      }
    },
  );
  root
    .createRenderPipeline({
      vertex: fullScreen,
      fragment: solid(d.vec4f(1, 0, 0, 1)),
      targets: { format: 'rgba8unorm' },
    })
    .withColorAttachment({ view: target })
    .draw(3);
  root.createComputePipeline({ compute: store }).dispatchWorkgroups(SIZE / 4, SIZE / 4);
  root
    .createRenderPipeline({
      vertex: corner,
      fragment: solid(d.vec4f(0, 0, 1, 1)),
      targets: { format: 'rgba8unorm' },
    })
    .withColorAttachment({ view: target, loadOp: 'load' })
    .draw(3);
  return target;
}

describe.skipIf(!hasWebGpu)('render passes keep WebGPU semantics', () => {
  test('a pass that loads its target sees what a CPU kernel stored since the last pass', async () => {
    const result = await compareRoots(gpu, gl, kernelBetweenPasses);
    expect((gl.device as GLDevice).stats.fallbacks.get('store')).toMatch(/^CPU only/);
    expect(pixel(result.actual, 2, 2)).toEqual([0, 255, 0, 255]);
    expect(pixel(result.actual, 2, SIZE - 2)).toEqual([255, 0, 0, 255]);
    expect(pixel(result.actual, SIZE - 2, SIZE - 2)).toEqual([0, 0, 255, 255]);
    expect(result.first).toBeUndefined();
  });
  test.each(['depth', 'stencil'] as const)(
    'a pass that discards one aspect of a depth-stencil target keeps the %s it stored',
    async (keep) => {
      const result = await compareRoots(gpu, gl, oneAspectStored(keep));
      expect(pixel(result.expected, 2, 2)).toEqual(
        keep === 'depth' ? [255, 0, 0, 255] : [0, 255, 0, 255],
      );
      expect(pixel(result.expected, SIZE - 2, 2)).toEqual(
        keep === 'depth' ? [0, 255, 0, 255] : [0, 0, 0, 255],
      );
      expect(result.first).toBeUndefined();
    },
  );

  test.each(['bitmap', 'canvas'] as const)(
    'a flipped copy of part of an image flips within that part, from a %s',
    async (kind) => {
      const images = await rows();
      const source = images[kind];
      const expected = await flippedRegion(gpu.device, source);
      const actual = await flippedRegion(gl.device, source);
      // Rows 0 and 1, red and green, the bottom one first
      expect(pixel(expected, 0, 0)).toEqual([0, 255, 0, 255]);
      expect(pixel(expected, 0, 1)).toEqual([255, 0, 0, 255]);
      expect(compare(expected, actual).first).toBeUndefined();
    },
  );
});

/** A triangle over the whole target from vertex positions, in a pipeline with `multisample`. */
function maskedDraw(mask: number) {
  return (root: TgpuRoot) => {
    const color = root
      .createTexture({ size: [SIZE, SIZE], format: 'rgba8unorm' })
      .$usage('render', 'sampled');
    root
      .createRenderPipeline({
        vertex: fullScreen,
        fragment: solid(d.vec4f(1, 0, 0, 1)),
        targets: { format: 'rgba8unorm' },
        multisample: { count: 1, mask },
      })
      .withColorAttachment({ view: color, clearValue: [0, 0, 1, 1] })
      .draw(3);
    return color;
  };
}

describe.skipIf(!hasWebGpu)('sample masks', () => {
  test('a pipeline whose sample mask clears sample 0 writes nothing', async () => {
    const result = await compareRoots(gpu, gl, maskedDraw(0));
    expect(pixel(result.expected, 2, 2)).toEqual([0, 0, 255, 255]);
    expect(result.first).toBeUndefined();
    const kept = await compareRoots(gpu, gl, maskedDraw(1));
    expect(pixel(kept.expected, 2, 2)).toEqual([255, 0, 0, 255]);
    expect(kept.first).toBeUndefined();
  });

  test('a multisampled pipeline that masks samples is an error that names the mask', async () => {
    const messages = await errorsOf(gl.device, () =>
      gl
        .createRenderPipeline({
          vertex: fullScreen,
          fragment: solid(d.vec4f(1, 0, 0, 1)),
          targets: { format: 'rgba8unorm' },
          multisample: { count: 4, mask: 0b0101 },
        })
        .initSync(),
    );
    expect(messages.join('\n')).toMatch(/multisample\.mask/);
  });
});

const Point = d.struct({ position: d.vec2f });
const points = tgpu.vertexLayout(d.arrayOf(Point));

/** Draws vertices 0 to 2 as indices 1 to 3 with a base vertex of -1. */
function negativeBaseVertex(root: TgpuRoot) {
  const color = root
    .createTexture({ size: [SIZE, SIZE], format: 'rgba8unorm' })
    .$usage('render', 'sampled');
  const vertices = root
    .createBuffer(d.arrayOf(Point, 4), [
      { position: d.vec2f(-1, -1) },
      { position: d.vec2f(3, -1) },
      { position: d.vec2f(-1, 3) },
      { position: d.vec2f(0, 0) },
    ])
    .$usage('vertex');
  const indices = root.createBuffer(d.arrayOf(d.u16, 4), [1, 2, 3, 0]).$usage('index');
  const vertex = tgpu.vertexFn({
    in: { position: d.vec2f },
    out: { position: d.builtin.position },
  })(({ position }) => {
    'use gpu';
    return { position: d.vec4f(position, 0, 1) };
  });
  root
    .createRenderPipeline({
      attribs: points.attrib,
      vertex,
      fragment: solid(d.vec4f(1, 0, 0, 1)),
      targets: { format: 'rgba8unorm' },
    })
    .with(points, vertices)
    .withIndexBuffer(indices)
    .withColorAttachment({ view: color, clearValue: [0, 0, 1, 1] })
    .drawIndexed(3, 1, 0, -1);
  return color;
}

describe.skipIf(!hasWebGpu)('base vertices', () => {
  test('a negative base vertex before the start of a vertex buffer is an error that names it', async () => {
    const expected = await compareRoots(gpu, gpu, negativeBaseVertex);
    expect(pixel(expected.expected, 2, 2)).toEqual([255, 0, 0, 255]);
    const messages = await errorsOf(gl.device, () => negativeBaseVertex(gl));
    expect(messages.join('\n')).toMatch(/baseVertex -1/);
  });
});

describe.skipIf(!hasWebGpu)('vertex strides', () => {
  test('a vertex buffer stride above 255 bytes is an error that names it', async () => {
    const Wide = d.struct({
      position: d.vec2f,
      ...Object.fromEntries(Array.from({ length: 16 }, (_, i) => [`pad${i}`, d.vec4f])),
    });
    const wide = tgpu.vertexLayout(d.arrayOf(Wide));
    const vertex = tgpu.vertexFn({
      in: { position: d.vec2f },
      out: { position: d.builtin.position },
    })(({ position }) => {
      'use gpu';
      return { position: d.vec4f(position, 0, 1) };
    });
    const messages = await errorsOf(gl.device, () =>
      gl
        .createRenderPipeline({
          attribs: { position: wide.attrib.position },
          vertex,
          fragment: solid(d.vec4f(1, 0, 0, 1)),
          targets: { format: 'rgba8unorm' },
        })
        .initSync(),
    );
    expect(messages.join('\n')).toMatch(/arrayStride of 272/);
  });
});

/** A pass draws red and discards it, and a pass that loads it draws blue over a corner. */
function discardedThenRead(read: 'copy' | 'load') {
  return (root: TgpuRoot) => {
    const target = root
      .createTexture({ size: [SIZE, SIZE], format: 'rgba8unorm' })
      .$usage('render', 'sampled');
    root
      .createRenderPipeline({
        vertex: fullScreen,
        fragment: solid(d.vec4f(1, 0, 0, 1)),
        targets: { format: 'rgba8unorm' },
      })
      .withColorAttachment({ view: target, storeOp: 'discard' })
      .draw(3);
    if (read === 'load') {
      root
        .createRenderPipeline({
          vertex: corner,
          fragment: solid(d.vec4f(0, 0, 1, 1)),
          targets: { format: 'rgba8unorm' },
        })
        .withColorAttachment({ view: target, loadOp: 'load' })
        .draw(3);
    }
    return target;
  };
}

describe.skipIf(!hasWebGpu)('discarded attachments', () => {
  test.each(['copy', 'load'] as const)(
    'an attachment a pass discarded reads as zeros, by a %s',
    async (read) => {
      const result = await compareRoots(gpu, gl, discardedThenRead(read));
      expect(pixel(result.expected, 2, 2)).toEqual([0, 0, 0, 0]);
      expect(result.first).toBeUndefined();
    },
  );
});
