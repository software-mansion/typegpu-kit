import { meshPool } from '@typegpu/kit';
import { meshes } from '@typegpu/kit/geometry';
import { d, tgpu, type TgpuRoot } from 'typegpu';
import { expect, test } from 'vitest';
import { glRoot } from './compare.ts';
import { diffScene } from './diff.ts';

/**
 * Kernel results the CPU backend gets wrong, found by comparing whole examples with WebGPU. Each is
 * `test.fails` until the backend is fixed, when it starts failing and the `.fails` comes off.
 */

const COUNT = 128;

/** Writes the positions of a cylinder's vertices, as the mesh pool's bake does. */
async function cylinder(root: TgpuRoot) {
  const shape = meshes.cylinder({ radius: 0.55, height: 0.4 });
  const out = root.createBuffer(d.arrayOf(d.vec4f, COUNT)).$usage('storage');
  const written = out.as('mutable');
  const kernel = tgpu.computeFn({
    workgroupSize: [64],
    in: { gid: d.builtin.globalInvocationId },
  })(({ gid }) => {
    'use gpu';
    const i = gid.x;
    if (i < shape.vertexCount) {
      written.$[i] = d.vec4f(shape.vertexAt(i).position, 1);
    }
  });
  root.createComputePipeline({ compute: kernel }).dispatchWorkgroups(COUNT / 64);
  return (await out.read()).map((v) => [v.x, v.y, v.z].map((c) => Number(c.toFixed(4))));
}

// The top cap's ring, vertices 67 to 98, comes out at x = z = 0 on the CPU
test.fails("a cylinder's cap vertices from a CPU kernel", { timeout: 60_000 }, async () => {
  const gpu = await tgpu.init();
  const gl = glRoot({ gpuCompute: false });
  try {
    expect(await cylinder(gl)).toEqual(await cylinder(gpu));
  } finally {
    gpu.destroy();
    gl.destroy();
  }
});

/** Bakes a box into a mesh pool and reads back the bounds the bake's atomics wrote. */
async function boxBounds(root: TgpuRoot) {
  const pool = meshPool.create(root, { capacity: 2 });
  pool.add(meshes.box({ width: 2, height: 1, depth: 0.5 }));
  const out = root.createBuffer(d.arrayOf(d.vec4f, 2)).$usage('storage');
  const written = out.as('mutable');
  const kernel = tgpu.computeFn({ workgroupSize: [1] })(() => {
    'use gpu';
    const box = meshPool.boundsOf(0);
    written.$[0] = d.vec4f(box.center, 0);
    written.$[1] = d.vec4f(box.half, 0);
  });
  root.pipe(pool.inject()).createComputePipeline({ compute: kernel }).dispatchWorkgroups(1);
  const result = (await out.read()).map((v) => [v.x, v.y, v.z].map((c) => Number(c.toFixed(4))));
  pool.destroy();
  return result;
}

// The bake's `workgroup.reduce` and `atomicMin`/`atomicMax` leave every box at zero size, so culling
// drops meshes that WebGPU keeps: robot-arm's table and wall
test.fails("a mesh pool's baked bounds from CPU kernels", { timeout: 60_000 }, async () => {
  const gpu = await tgpu.init();
  const gl = glRoot({ gpuCompute: false });
  try {
    expect(await boxBounds(gl)).toEqual(await boxBounds(gpu));
  } finally {
    gpu.destroy();
    gl.destroy();
  }
});

// The pose kernels leave words of `Pose_device` at 0 where WebGPU has 1 and -0.44, and 39 % of
// `poses_joints` differs after 4 s, so skeletons stay collapsed on the floor. Every kernel runs on
// the CPU here and both runs draw the same seeded random numbers.
test.fails("skeletons' poses from CPU kernels", { timeout: 280_000 }, async () => {
  const gpu = await tgpu.init();
  const gl = glRoot({ gpuCompute: false });
  try {
    const { buffers } = await diffScene(gpu, gl, 'skeletons', { frames: 100 });
    const poses = buffers.filter(
      (buffer) => buffer.label.startsWith('Pose_device') || buffer.label.startsWith('poses_joints'),
    );
    expect(poses).toEqual([]);
  } finally {
    gpu.destroy();
    gl.destroy();
  }
});
