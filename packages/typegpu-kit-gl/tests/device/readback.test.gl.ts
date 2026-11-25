import { d, tgpu } from 'typegpu';
import { afterEach, expect, test, vi } from 'vitest';
import { glRoot } from './compare.ts';

const COUNT = 64;

afterEach(() => vi.restoreAllMocks());

/** A storage buffer a GPU form fills with `value`, and a buffer to map copies of it into. */
function filled() {
  const root = glRoot();
  const { device } = root;
  const value = root.createUniform(d.u32);
  const target = root.createMutable(d.arrayOf(d.u32, COUNT));
  const fill = tgpu.computeFn({
    workgroupSize: [COUNT],
    in: { gid: d.builtin.globalInvocationId },
  })(({ gid }) => {
    'use gpu';
    target.$[gid.x] = value.$ + gid.x;
  });
  const pipeline = root.createComputePipeline({ compute: fill });
  const staging = device.createBuffer({
    size: COUNT * 4,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  const run = (v: number, copy: boolean) => {
    value.write(v);
    pipeline.dispatchWorkgroups(1);
    if (!copy) return;
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(root.unwrap(target.buffer), 0, staging, 0, COUNT * 4);
    device.queue.submit([encoder.finish()]);
  };
  return { root, staging, run };
}

const expected = (v: number) => Array.from({ length: COUNT }, (_, i) => v + i);

test('a mapped copy holds the bytes from when it was copied, not later writes', async () => {
  const { root, staging, run } = filled();
  try {
    run(100, true);
    run(200, false);
    await staging.mapAsync(GPUMapMode.READ);
    expect([...new Uint32Array(staging.getMappedRange())]).toEqual(expected(100));
    staging.unmap();
  } finally {
    root.destroy();
  }
});

test('copying what GL holds into a mappable buffer does not wait for the GPU', async () => {
  const { root, staging, run } = filled();
  try {
    const read = vi.spyOn(WebGL2RenderingContext.prototype, 'readPixels');
    run(7, true);
    // Every read went into a pack buffer, whose last argument is an offset
    expect(read.mock.calls.every((call) => typeof call.at(-1) === 'number')).toBe(true);
    expect(read).toHaveBeenCalled();
    const mapped = staging.mapAsync(GPUMapMode.READ);
    expect(() => staging.getMappedRange()).toThrow();
    await mapped;
    expect([...new Uint32Array(staging.getMappedRange())]).toEqual(expected(7));
  } finally {
    root.destroy();
  }
});

test('unmapping or destroying a buffer while its map waits rejects the map cleanly', async () => {
  const { root, staging, run } = filled();
  try {
    run(1, true);
    const unmapped = staging.mapAsync(GPUMapMode.READ);
    staging.unmap();
    await expect(unmapped).rejects.toThrow();
    // The copy still lands for the next map
    await staging.mapAsync(GPUMapMode.READ);
    expect([...new Uint32Array(staging.getMappedRange())]).toEqual(expected(1));
    staging.unmap();
    run(2, true);
    const destroyed = staging.mapAsync(GPUMapMode.READ);
    staging.destroy();
    await expect(destroyed).rejects.toThrow();
  } finally {
    root.destroy();
  }
});

test('one pixel of a render target reads back as the picker reads it, from when it was copied', async () => {
  const root = glRoot();
  const { device } = root;
  try {
    const texture = device.createTexture({
      size: [8, 8],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    });
    const staging = device.createBuffer({
      size: 256,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const clear = (color: GPUColor, copy: boolean) => {
      const encoder = device.createCommandEncoder();
      encoder
        .beginRenderPass({
          colorAttachments: [
            { view: texture.createView(), loadOp: 'clear', storeOp: 'store', clearValue: color },
          ],
        })
        .end();
      if (copy) {
        encoder.copyTextureToBuffer(
          { texture, origin: [5, 3] },
          { buffer: staging, bytesPerRow: 256 },
          [1, 1],
        );
      }
      device.queue.submit([encoder.finish()]);
    };
    clear([1, 0, 0, 1], true);
    clear([0, 0, 1, 1], false);
    await staging.mapAsync(GPUMapMode.READ);
    expect([...new Uint8Array(staging.getMappedRange(), 0, 4)]).toEqual([255, 0, 0, 255]);
  } finally {
    root.destroy();
  }
});
