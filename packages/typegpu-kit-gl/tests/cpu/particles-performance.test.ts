import { writeFile } from 'node:fs/promises';
import { d, tgpu } from 'typegpu';
import { it } from 'vitest';
import { cpuRoot } from './device.ts';

it.runIf(process.env.CPU_BENCH === '1')(
  'measures 100k composite particle rows with thirty vector operations',
  async () => {
    const root = cpuRoot();
    const count = 100_000;
    const Particle = d.struct({ position: d.vec3f, velocity: d.vec3f, life: d.f32, mass: d.f32 });
    const data = root.createMutable(d.arrayOf(Particle, count));
    const kernel = tgpu.computeFn({
      workgroupSize: [100],
      in: { id: d.builtin.globalInvocationId },
    })(({ id }) => {
      'use gpu';
      const particle = data.$[id.x];
      let position = d.vec3f(particle.position);
      const velocity = d.vec3f(particle.velocity);
      for (let i = 0; i < 10; i++) position = position * 0.999 + velocity * 0.01;
      particle.position = d.vec3f(position);
      particle.life -= 0.01;
    });
    const pipeline = root.createComputePipeline({ compute: kernel });
    pipeline.dispatchWorkgroups(count / 100);
    const start = performance.now();
    for (let i = 0; i < 3; i++) pipeline.dispatchWorkgroups(count / 100);
    const milliseconds = performance.now() - start;
    await writeFile(
      new URL('./particles-performance-results.json', import.meta.url),
      `${JSON.stringify({ rows: count, iterations: 3, vectorOperationsPerInvocation: 30, milliseconds, invocationsPerSecond: ((count * 3) / milliseconds) * 1000 }, null, 2)}\n`,
    );
    root.destroy();
  },
);
