import { meshPool } from '@typegpu/kit';
import { meshes } from '@typegpu/kit/geometry';
import { d, tgpu, type TgpuRoot } from 'typegpu';

export async function boxBounds(root: TgpuRoot): Promise<number[]> {
  const pool = meshPool.create(root, { capacity: 2 });
  pool.add(meshes.box({ width: 2, height: 1, depth: 0.5 }));
  const output = root.createMutable(d.arrayOf(d.vec4f, 2));
  const kernel = tgpu.computeFn({ workgroupSize: [1] })(() => {
    'use gpu';
    const box = meshPool.boundsOf(0);
    output.$[0] = d.vec4f(box.center, 0);
    output.$[1] = d.vec4f(box.half, 0);
  });
  root.pipe(pool.inject()).createComputePipeline({ compute: kernel }).dispatchWorkgroups(1);
  const result = (await output.read()).flatMap((v) => [v.x, v.y, v.z, v.w]);
  pool.destroy();
  return result;
}

export async function cylinder(root: TgpuRoot): Promise<number[]> {
  const shape = meshes.cylinder({ radius: 0.55, height: 0.4 });
  const output = root.createMutable(d.arrayOf(d.vec4f, shape.vertexCount));
  const kernel = tgpu.computeFn({ workgroupSize: [64], in: { id: d.builtin.globalInvocationId } })(
    ({ id }) => {
      'use gpu';
      if (id.x < shape.vertexCount) output.$[id.x] = d.vec4f(shape.vertexAt(id.x).position, 1);
    },
  );
  root
    .createComputePipeline({ compute: kernel })
    .dispatchWorkgroups(Math.ceil(shape.vertexCount / 64));
  return (await output.read()).flatMap((v) => [v.x, v.y, v.z, v.w]);
}
