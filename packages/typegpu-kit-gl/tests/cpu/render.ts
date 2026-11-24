import { Pose } from '../../../typegpu-kit/src/render/skinning.ts';
import { bend, leg } from '../../../typegpu-kit/tests/render/leg.ts';
import { meshPool, poses } from '@typegpu/kit';
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

export async function posePhase(root: TgpuRoot): Promise<number[]> {
  const figures = poses.create(
    root,
    leg,
    {
      short: bend,
      long: {
        ...bend,
        name: 'long',
        duration: 8 / 3,
        channels: bend.channels.map((channel) => ({ ...channel, times: [0, 8 / 3] })),
      },
    },
    { capacity: 1 },
  );
  const id = figures.add('short', 0);
  const output = root.createMutable(d.arrayOf(d.f32, 3));
  const start = tgpu.computeFn({ workgroupSize: [1] })(() => {
    'use gpu';
    poses.once(id, figures.clips.long, 0, 0);
  });
  const read = tgpu.computeFn({ workgroupSize: [1] })(() => {
    'use gpu';
    const track = Pose.track0(Pose.slotOf(d.u32(id)));
    output.$[0] = track.playing.offset;
    output.$[1] = track.playing.speed;
    output.$[2] = d.f32(track.playing.clip);
  });
  const starting = root.pipe(figures.inject()).createComputePipeline({ compute: start });
  const reading = root.pipe(figures.inject()).createComputePipeline({ compute: read });
  let encoder = root.createCommandEncoder();
  let pass = encoder.beginComputePass();
  figures.animate(pass, 0);
  pass.end();
  encoder.submit();
  encoder = root.createCommandEncoder();
  pass = encoder.beginComputePass();
  pass.setPipeline(starting);
  pass.dispatchWorkgroups(1);
  figures.animate(pass, 2.075);
  pass.setPipeline(reading);
  pass.dispatchWorkgroups(1);
  pass.end();
  encoder.submit();
  const result = await output.read();
  figures.destroy();
  return result;
}
