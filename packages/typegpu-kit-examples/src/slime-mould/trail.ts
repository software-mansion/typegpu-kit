import { pipelines } from '@typegpu/kit';
import { d, std, tgpu, type TgpuComputePass, type TgpuRoot } from 'typegpu';

/** Integer units per deposit. Atomics require integer additions. */
const PARTS = 1024;
const WORKGROUP = 16;

/** Trail retention per step and blur strength, both from 0 to 1. */
export const Spread = d.struct({ keep: d.f32, blur: d.f32 });

export const TRAIL = { decay: 0.07, blur: 1 };

/**
 * Stores trail density in a square grid. Agents deposit with atomics. Each step blurs and fades
 * the trail, then adds deposits. Texture sampling interpolates between cells.
 */
export function createTrail(root: TgpuRoot, options: { size: number }) {
  const { size } = options;
  const texture = () =>
    root.createTexture({ size: [size, size], format: 'rgba16float' }).$usage('storage', 'sampled');
  const trail = texture();
  const spare = texture();
  const reading = trail.createView(d.texture2d(d.f32));
  const writing = trail.createView(d.textureStorage2d('rgba16float', 'write-only'));
  const spareReading = spare.createView(d.texture2d(d.f32));
  const spareWriting = spare.createView(d.textureStorage2d('rgba16float', 'write-only'));
  const smooth = root.createSampler({ magFilter: 'linear', minFilter: 'linear' });
  const dropped = root.createMutable(d.arrayOf(d.atomic(d.u32), size * size));
  const spread = root.createUniform(Spread, { keep: 1 - TRAIL.decay, blur: TRAIL.blur });

  /** Atomically adds `amount` to a cell, allowing concurrent deposits. */
  const deposit = (cell: d.v2i, amount: number) => {
    'use gpu';
    std.atomicAdd(dropped.$[cell.y * size + cell.x], d.u32(amount * PARTS));
  };

  /** Samples the trail at grid coordinates `uv` from 0 to 1, with interpolation between cells. */
  const at = (uv: d.v2f) => {
    'use gpu';
    return std.textureSampleLevel(reading.$, smooth.$, uv, 0).x;
  };

  const inside = (cell: d.v2i) => {
    'use gpu';
    return std.clamp(cell, d.vec2i(0), d.vec2i(size - 1));
  };

  // Reading neighbours while writing the same texture would depend on thread order. Blur across
  // into a spare texture, then down into the trail texture
  const across = tgpu.computeFn({
    workgroupSize: [WORKGROUP, WORKGROUP],
    in: { gid: d.builtin.globalInvocationId },
  })(({ gid }) => {
    'use gpu';
    if (gid.x >= size || gid.y >= size) return;
    const cell = d.vec2i(gid.xy);
    const here = std.textureLoad(reading.$, cell, 0).x;
    const left = std.textureLoad(reading.$, inside(cell - d.vec2i(1, 0)), 0).x;
    const right = std.textureLoad(reading.$, inside(cell + d.vec2i(1, 0)), 0).x;
    const blurred = std.mix(here, (left + here * 2 + right) / 4, spread.$.blur);
    std.textureStore(spareWriting.$, gid.xy, d.vec4f(blurred, 0, 0, 1));
  });

  const down = tgpu.computeFn({
    workgroupSize: [WORKGROUP, WORKGROUP],
    in: { gid: d.builtin.globalInvocationId },
  })(({ gid }) => {
    'use gpu';
    if (gid.x >= size || gid.y >= size) return;
    const cell = d.vec2i(gid.xy);
    const here = std.textureLoad(spareReading.$, cell, 0).x;
    const up = std.textureLoad(spareReading.$, inside(cell - d.vec2i(0, 1)), 0).x;
    const below = std.textureLoad(spareReading.$, inside(cell + d.vec2i(0, 1)), 0).x;
    const blurred = std.mix(here, (up + here * 2 + below) / 4, spread.$.blur);

    const index = gid.y * size + gid.x;
    const fresh = d.f32(std.atomicLoad(dropped.$[index])) / PARTS;
    std.atomicStore(dropped.$[index], 0);
    std.textureStore(writing.$, gid.xy, d.vec4f(blurred * spread.$.keep + fresh, 0, 0, 1));
  });

  const blurringAcross = root.createComputePipeline({ compute: across });
  const blurringDown = root.createComputePipeline({ compute: down });
  const groups = Math.ceil(size / WORKGROUP);

  return {
    /** Grid width and height in cells. */
    size,
    deposit,
    at,
    settings: spread,
    initAsync: () => pipelines.compile([blurringAcross, blurringDown]),

    spread(pass: TgpuComputePass) {
      pass.setPipeline(blurringAcross);
      pass.dispatchWorkgroups(groups, groups);
      pass.setPipeline(blurringDown);
      pass.dispatchWorkgroups(groups, groups);
    },

    clear() {
      trail.clear();
      dropped.buffer.clear();
    },

    destroy() {
      trail.destroy();
      spare.destroy();
      dropped.buffer.destroy();
    },
  };
}

export type Trail = ReturnType<typeof createTrail>;
