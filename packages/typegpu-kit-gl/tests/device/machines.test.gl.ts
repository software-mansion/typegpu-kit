import { machines } from '@typegpu/kit';
import { d, tgpu, type TgpuRoot } from 'typegpu';
import { afterAll, beforeAll, expect, it } from 'vitest';
import type { GLDevice } from '../../src/device/index.ts';
import { roots } from './compare.ts';

let gpu: TgpuRoot;
let gl: TgpuRoot;
beforeAll(async () => {
  // Placement would move the sends to the CPU, which these runs read back: they test the form
  ({ gpu, gl } = await roots({ placement: false }));
});
afterAll(() => {
  gpu.destroy();
  gl.destroy();
});

const ROWS = 256;
const STEP = 1 / 60;

/**
 * Rows that cycle off, on and broken, sent events by invocations that are not theirs, many to
 * one row, over rounds of a sending dispatch and a step: each row's state and age after each.
 */
async function rounds(root: TgpuRoot) {
  const Lamp = machines.define('Lamp', {
    events: ['flip', 'kick'],
    states: {
      off: { next: [{ to: 'on', on: 'flip' }] },
      on: {
        next: [
          { to: 'broken', on: 'kick' },
          { to: 'off', on: 'flip' },
        ],
      },
      broken: { next: [{ to: 'off', on: 'flip' }] },
    },
  });
  const lamps = machines.create(root, Lamp, { capacity: ROWS, step: STEP });
  const round = root.createUniform(d.u32);
  const sending = tgpu.computeFn({
    workgroupSize: [64],
    in: { gid: d.builtin.globalInvocationId },
  })(({ gid }) => {
    'use gpu';
    const i = gid.x;
    if (i >= ROWS) return;
    const r = round.$;
    // Many invocations send to one row, at rows that change with the round
    if ((i + r) % 3 === 0) Lamp.send.flip((i * 7 + r) % ROWS);
    if ((i + r) % 5 === 1) Lamp.send.kick(((i >>> 3) + r * 11) % ROWS);
  });
  const pipeline = root
    .pipe(lamps.inject())
    .createComputePipeline({ compute: sending })
    .$name('sending');
  const seen: string[] = [];
  for (let r = 0; r < 5; r++) {
    round.write(r);
    const encoder = root.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.dispatchWorkgroups(ROWS / 64);
    lamps.step(pass);
    pass.end();
    encoder.submit();
    const rows = await lamps.rows.read();
    seen.push(rows.map((row) => `${row.state}:${row.age}`).join(' '));
  }
  lamps.destroy();
  return seen;
}

it('takes events kernels send to rows other invocations share, as WebGPU does', async () => {
  const actual = await rounds(gl);
  const { stats, gl: context } = gl.device as unknown as GLDevice;
  expect(stats.fallbacks.get('sending')).toBeUndefined();
  expect([...stats.kernels.keys()]).toContain('sending (gpu)');
  expect(context.getError()).toBe(0);
  expect(actual).toEqual(await rounds(gpu));
  // The states moved, so the comparison is not of rows left alone
  expect(
    new Set(
      actual
        .at(-1)
        ?.split(' ')
        .map((row) => row.split(':')[0]),
    ).size,
  ).toBe(3);
});
