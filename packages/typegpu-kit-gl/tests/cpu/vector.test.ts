import { recordingRoot } from '@typegpu/kit/testing';
import { d, tgpu } from 'typegpu';
import { expect, it } from 'vitest';
import { CpuGenerator } from '../../src/cpu/generator.ts';
import { compile } from '../../src/cpu/index.ts';

it('updates struct fields through aliases with independent vector values', () => {
  const { root } = recordingRoot();
  const Particle = d.struct({ position: d.vec3f, velocity: d.vec3f });
  const particles = root.createMutable(d.arrayOf(Particle, 4));
  const kernel = tgpu.computeFn({ workgroupSize: [4], in: { id: d.builtin.globalInvocationId } })(({
    id,
  }) => {
    'use gpu';
    const particle = particles.$[id.x];
    particle.velocity = d.vec3f(1, 2, 3);
    let copy = d.vec3f(particle.velocity);
    copy.x = 9;
    particle.position = particle.velocity * 2 + copy;
  });
  const module = compile(tgpu.resolve([kernel], { unstable_shaderGenerator: new CpuGenerator() }));
  const bytes = new ArrayBuffer(d.sizeOf(d.arrayOf(Particle, 4)));
  module.dispatch(1, 1, 1, {
    buffer: () => ({ bytes, offset: 0, size: bytes.byteLength }),
    texture: (): never => {
      throw new Error('unexpected texture');
    },
    sampler: (): never => {
      throw new Error('unexpected sampler');
    },
  });
  const data = new Float32Array(bytes);
  for (let i = 0; i < 4; i++) {
    expect(Array.from(data.subarray(i * 8, i * 8 + 3))).toEqual([11, 6, 9]);
    expect(Array.from(data.subarray(i * 8 + 4, i * 8 + 7))).toEqual([1, 2, 3]);
  }
  root.destroy();
});
