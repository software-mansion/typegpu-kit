import { common, d, std, tgpu } from 'typegpu';
import { describe, expect, it } from 'vitest';
import { frames, memo, pipelines, targets } from '../src/index.ts';
import { offscreenSurface, recordingRoot } from '../src/testing.ts';

/** A recording device, a way to make buffers on it, and the buffer usages. */
function double() {
  const { device } = recordingRoot().root;
  const buffer = (usage: number, size = 16) => device.createBuffer({ size, usage });

  return { device, buffer, ...GPUBufferUsage };
}

describe('recordingRoot', () => {
  it.each([
    [
      'a buffer that maps for reading and is also storage',
      ({ buffer, MAP_READ, STORAGE }) => buffer(MAP_READ | STORAGE),
      'Buffer usages MAP_READ|STORAGE are invalid. MAP_READ permits only COPY_DST alongside it.',
    ],
    [
      'a copy from a buffer without COPY_SRC',
      ({ device, buffer, STORAGE, COPY_DST }) =>
        device
          .createCommandEncoder()
          .copyBufferToBuffer(buffer(STORAGE), 0, buffer(COPY_DST), 0, 16),
      `[Buffer ""] usage (STORAGE) doesn't include COPY_SRC.`,
    ],
    [
      'a write at an offset that is not a multiple of 4',
      ({ device, buffer, COPY_DST }) =>
        device.queue.writeBuffer(buffer(COPY_DST), 2, new Uint8Array(4)),
      'Write offset (2) is not a multiple of 4 bytes.',
    ],
    [
      'a write of a size that is not a multiple of 4',
      ({ device, buffer, COPY_DST }) =>
        device.queue.writeBuffer(buffer(COPY_DST), 0, new Uint8Array(6)),
      'Write size (6) is not a multiple of 4 bytes.',
    ],
    [
      'a copy past the end of its destination',
      ({ device, buffer, COPY_SRC, COPY_DST }) =>
        device
          .createCommandEncoder()
          .copyBufferToBuffer(buffer(COPY_SRC), 0, buffer(COPY_DST, 8), 0, 16),
      'Copy destination range (offset: 0, size: 16) does not fit in [Buffer ""] with size (8).',
    ],
    [
      'a write to a destroyed buffer',
      ({ device, buffer, COPY_DST }) => {
        const destroyed = buffer(COPY_DST);
        destroyed.destroy();
        device.queue.writeBuffer(destroyed, 0, new Uint32Array(1));
      },
      'Cannot submit work that uses destroyed [Buffer ""].',
    ],
    [
      'getMappedRange on a buffer that is not mapped',
      ({ buffer, MAP_READ, COPY_DST }) => buffer(MAP_READ | COPY_DST).getMappedRange(),
      '[Buffer ""] is not mapped.',
    ],
    [
      'a second mapAsync while one is pending',
      ({ buffer, MAP_READ, COPY_DST }) => {
        const staging = buffer(MAP_READ | COPY_DST);
        void staging.mapAsync(GPUMapMode.READ);
        void staging.mapAsync(GPUMapMode.READ);
      },
      '[Buffer ""] already has a pending map. Wait for it before mapping again.',
    ],
    [
      'a render pass into a texture that is not a render attachment',
      ({ device }) => {
        const usage = GPUTextureUsage.TEXTURE_BINDING;
        const texture = device.createTexture({ size: [4, 4], format: 'rgba8unorm', usage });
        device.createCommandEncoder().beginRenderPass({
          colorAttachments: [{ view: texture.createView(), loadOp: 'clear', storeOp: 'store' }],
        });
      },
      `The color attachment [Texture ""] usage (TEXTURE_BINDING) doesn't include RENDER_ATTACHMENT.`,
    ],
    [
      'a bind group with a buffer that lacks the usage its entry needs',
      ({ device, buffer, STORAGE }) => {
        const layout = device.createBindGroupLayout({
          entries: [{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: {} }],
        });
        device.createBindGroup({
          layout,
          entries: [{ binding: 0, resource: { buffer: buffer(STORAGE) } }],
        });
      },
      `Binding usage (STORAGE) of [Buffer ""] doesn't match expected usage (UNIFORM).`,
    ],
    [
      'a submit of commands that use a destroyed buffer',
      ({ device, buffer, COPY_SRC, COPY_DST }) => {
        const source = buffer(COPY_SRC);
        const encoder = device.createCommandEncoder();
        encoder.copyBufferToBuffer(source, 0, buffer(COPY_DST), 0, 16);
        source.destroy();
        device.queue.submit([encoder.finish()]);
      },
      'Cannot submit work that uses destroyed [Buffer ""].',
    ],
  ] satisfies [string, (on: ReturnType<typeof double>) => unknown, string][])(
    'throws on %s, as a device would reject it',
    (_, misuse, message) => {
      expect(() => misuse(double())).toThrow(message);
    },
  );

  it('maps a buffer in turn, and unmapping a pending map rejects it', async () => {
    const { buffer, MAP_READ, COPY_DST } = double();
    const staging = buffer(MAP_READ | COPY_DST);

    const mapped = staging.mapAsync(GPUMapMode.READ);
    expect(staging.mapState).toBe('pending');
    await mapped;
    expect(staging.mapState).toBe('mapped');
    expect(staging.getMappedRange().byteLength).toBe(16);

    staging.unmap();
    const aborted = staging.mapAsync(GPUMapMode.READ);
    staging.unmap();
    await expect(aborted).rejects.toThrow('before mapping was resolved');
    expect(staging.mapState).toBe('unmapped');
  });

  it('counts what a frame creates, so a test can show that a steady frame creates nothing', () => {
    const { root, log } = recordingRoot();
    const size = { width: 64, height: 32 };
    const screen = targets.create(root, size, {
      hdr: { format: 'rgba16float', sampled: true },
      output: { format: 'rgba8unorm' },
    });

    const layout = tgpu.bindGroupLayout({ hdr: { texture: d.texture2d() } });
    const blit = root.createRenderPipeline({
      vertex: common.fullScreenTriangle,
      fragment: tgpu.fragmentFn({ in: { position: d.builtin.position }, out: d.vec4f })(
        ({ position }) => {
          'use gpu';
          return std.textureLoad(layout.$.hdr, d.vec2i(position.xy), 0);
        },
      ),
      targets: { format: 'rgba8unorm' },
    });
    const group = memo((hdr: typeof screen.textures.hdr) => root.createBindGroup(layout, { hdr }));

    const loop = frames.create(root, ({ encoder }) => {
      screen.fit();
      const pass = encoder.beginRenderPass({ colorAttachments: { view: screen.views.output } });
      pass.setPipeline(blit);
      pass.setBindGroup(group(screen.textures.hdr));
      pass.draw(3);
      pass.end();
    });

    loop.run(0);
    loop.run(1);
    const before = log.counts();

    loop.run(2);
    loop.run(3);
    expect(log.counts()).toEqual(before);

    size.width = 128;
    loop.run(4);
    expect(log.textures).toBe(before.textures + 2);
    expect(log.bindGroups).toBe(before.bindGroups + 1);
  });

  it('counts the pipelines something waited for, and none compiled ahead', async () => {
    const { root, log } = recordingRoot();
    const filled = () =>
      root.createRenderPipeline({
        vertex: common.fullScreenTriangle,
        fragment: tgpu.fragmentFn({ out: d.vec4f })(() => {
          'use gpu';
          return d.vec4f(1);
        }),
        targets: { format: 'rgba8unorm' },
      });
    const [ahead, late] = [filled(), filled()];

    await pipelines.compile([ahead]);
    root.unwrap(ahead);
    expect([log.pipelines, log.blocking]).toEqual([1, 0]);

    root.unwrap(late);
    expect([log.pipelines, log.blocking]).toEqual([2, 1]);
  });

  it('lists what is alive until it is destroyed, so a test can show that a destroy frees everything', () => {
    const { root, log } = recordingRoot({ features: ['timestamp-query'] });
    const buffer = root.createBuffer(d.u32).$usage('storage').$name('counter');
    const texture = root.createTexture({ size: [4, 4], format: 'rgba8unorm' }).$name('canvas');
    const querySet = root.createQuerySet('timestamp', 2);

    expect(log.alive()).toEqual([]);

    root.unwrap(buffer);
    root.unwrap(texture);
    root.unwrap(querySet);
    expect(log.alive()).toEqual(['[Buffer "counter"]', '[Texture "canvas"]', '[QuerySet ""]']);

    buffer.destroy();
    texture.destroy();
    querySet.destroy();
    expect(log.alive()).toEqual([]);
  });

  it('replays a render bundle into the pass that executes it', () => {
    const { root, log } = recordingRoot();
    const { device } = root;
    const indirect = device.createBuffer({ size: 16, usage: GPUBufferUsage.INDIRECT });
    const bundler = device.createRenderBundleEncoder({ colorFormats: ['rgba8unorm'] });
    bundler.setPipeline({} as GPURenderPipeline);
    bundler.draw(3);
    bundler.drawIndirect(indirect, 0);
    const bundle = bundler.finish();
    indirect.destroy();

    const texture = device.createTexture({
      size: [4, 4],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [{ view: texture.createView(), loadOp: 'clear', storeOp: 'store' }],
    });
    pass.executeBundles([bundle, bundle]);
    pass.end();

    expect(log.events).toEqual([
      'begin-render',
      'bundles',
      'draw',
      'draw-indirect',
      'draw',
      'draw-indirect',
      'end-render',
    ]);
    expect(log.state).toEqual(['set-pipeline', 'set-pipeline']);
    expect(() => device.queue.submit([encoder.finish()])).toThrow(
      'Cannot submit work that uses destroyed [Buffer ""].',
    );
  });

  it("reports WebGPU's default limits, and names a device member it lacks", () => {
    const { device } = recordingRoot({ limits: { maxBufferSize: 1024 } }).root;

    expect(device.limits.maxStorageBufferBindingSize).toBe(134217728);
    expect(device.limits.maxBufferSize).toBe(1024);
    expect(() => device.importExternalTexture).toThrow(
      "recordingRoot doesn't implement GPUDevice.importExternalTexture; run this test on gpuRoot from @typegpu/kit/testing/dawn.",
    );
  });
});

describe('offscreenSurface', () => {
  it('gives frames one view per texture, and a texture of the new size when resized', () => {
    const { root, log } = recordingRoot();
    const surface = offscreenSurface(root, { width: 64, height: 32, format: 'bgra8unorm' });

    const views: GPUTextureView[] = [];
    const loop = frames.create(root, (frame) => {
      views.push(surface.view(frame));
      const pass = frame.encoder.beginRenderPass({
        colorAttachments: { view: surface.view(frame) },
      });
      pass.end();
    });

    loop.run(0);
    loop.run(1);
    expect(views[1]).toBe(views[0]);
    expect(log).toMatchObject({ textures: 1, views: 1 });

    surface.setSize({ width: 128, height: 64 });
    loop.run(2);
    expect([surface.width, surface.height]).toEqual([128, 64]);
    expect(surface.pixelRatio).toBe(1);
    expect(views[2]).not.toBe(views[0]);
    expect(log).toMatchObject({ textures: 2, views: 2 });
  });
});
