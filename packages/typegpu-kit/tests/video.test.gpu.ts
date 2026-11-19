import { gpuRoot, type GpuRoot } from '@typegpu/kit/testing/dawn';
import { common, tgpu, d, type TgpuRoot } from 'typegpu';
import { expect, it } from 'vitest';
import { bindings, video, type Frame } from '../src/index.ts';

const SIDE = 64;
const RED = [255, 0, 0, 255];
const GREEN = [0, 255, 0, 255];
const BLUE = [0, 0, 255, 255];
const WHITE = [255, 255, 255, 255];
const NOTHING = [0, 0, 0, 0];
/** A live view reads nothing of the frame it updates in. */
const stub = {} as Frame;

/** A 2×2 picture, red and green on top, blue and white below. */
function picture(root: TgpuRoot) {
  const texture = root.createTexture({ size: [2, 2], format: 'rgba8unorm' }).$usage('sampled');
  texture.write(new Uint8Array([...RED, ...GREEN, ...BLUE, ...WHITE]) as unknown as ArrayBuffer);
  // WebGPU binds a 2D view where an external texture goes, which stands in for a camera frame
  return root.unwrap(texture).createView() as unknown as GPUExternalTexture;
}

/** The top-left, top-right, bottom-left and bottom-right pixels of `texture`. */
async function corners(root: GpuRoot, texture: GPUTexture) {
  const readback = root.device.createBuffer({
    size: SIDE * 4 * SIDE,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const copy = root.device.createCommandEncoder();
  copy.copyTextureToBuffer({ texture }, { buffer: readback, bytesPerRow: SIDE * 4 }, [SIDE, SIDE]);
  root.device.queue.submit([copy.finish()]);
  await readback.mapAsync(GPUMapMode.READ);
  const bytes = new Uint8Array(readback.getMappedRange().slice(0));
  const at = (x: number, y: number) => [
    ...bytes.subarray((y * SIDE + x) * 4, (y * SIDE + x) * 4 + 4),
  ];
  return [at(0, 0), at(SIDE - 1, 0), at(0, SIDE - 1), at(SIDE - 1, SIDE - 1)];
}

async function viewing(options: video.LiveOptions = {}) {
  const root = await gpuRoot();
  const live = video.live(root, options);
  const shared = bindings.share(root, { camera: live.part });
  const show = tgpu.fragmentFn({ in: { uv: d.vec2f }, out: d.vec4f })(({ uv }) => {
    'use gpu';
    return video.sample(uv);
  });
  const pipeline = root.pipe(live.inject()).createRenderPipeline({
    vertex: common.fullScreenTriangle,
    fragment: show,
    targets: { format: 'rgba8unorm' },
  });
  const target = root.createTexture({ size: [SIDE, SIDE], format: 'rgba8unorm' }).$usage('render');
  const draw = async () => {
    const encoder = root.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [{ view: target.createView('render'), loadOp: 'clear', storeOp: 'store' }],
    });
    pass.setBindGroup(shared.layout, shared.group());
    pass.setPipeline(pipeline);
    pass.draw(3);
    pass.end();
    encoder.submit();
    return corners(root, root.unwrap(target));
  };

  const texture = picture(root);
  const frame = { texture, width: 2, height: 4, rotation: 90 as const, mirrored: false, serial: 1 };
  const source = { latest: () => frame };
  return { live, draw, frame, source };
}

it('samples the newest frame in place, turned upright and mirrored', async () => {
  const { live, draw, frame, source } = await viewing();

  expect(await draw()).toEqual([NOTHING, NOTHING, NOTHING, NOTHING]);

  live.attach(source);
  expect(live.update(stub)).toBe(true);
  expect(live.frameSize).toEqual([4, 2]);
  // Turned a quarter clockwise, the left column becomes the top row
  expect(await draw()).toEqual([BLUE, RED, WHITE, GREEN]);

  frame.mirrored = true;
  expect(live.update(stub)).toBe(false);
  expect(live.frames).toBe(1);
  expect(await draw()).toEqual([RED, BLUE, GREEN, WHITE]);

  live.attach(undefined);
  expect(await draw()).toEqual([NOTHING, NOTHING, NOTHING, NOTHING]);
});

it('crops the frame as a feed of the same size and fit would', async () => {
  // The upright frame is twice as wide as tall, so a square contained in it leaves the top and
  // bottom quarters uncovered
  const { live, draw, source } = await viewing({ crop: { size: [2, 2], fit: 'contain' } });
  live.attach(source);
  live.update(stub);
  expect(await draw()).toEqual([NOTHING, NOTHING, NOTHING, NOTHING]);
});
