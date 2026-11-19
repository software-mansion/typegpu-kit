import { meshPool } from '@typegpu/kit';
import { gpuRoot } from '@typegpu/kit/testing/dawn';
import { d, tgpu } from 'typegpu';
import { expect, it } from 'vitest';
import { createTerrain } from '../src/spider/terrain.ts';

it('finds the height of the drawn triangle under every point of the ground', async () => {
  const root = await gpuRoot();
  const pool = meshPool.create(root, { capacity: 1 });
  const ground = await createTerrain(root, pool);
  const { baseVertex, vertexCount, firstIndex, indexCount } = pool.meshes[0];

  const positions = root.createMutable(d.arrayOf(d.vec3f, vertexCount));
  root
    .pipe(pool.inject())
    .createComputePipeline({
      compute: tgpu.computeFn({ workgroupSize: [64], in: { gid: d.builtin.globalInvocationId } })(
        ({ gid }) => {
          'use gpu';
          if (gid.x >= vertexCount) return;
          positions.$[gid.x] = meshPool.positionAt(baseVertex + gid.x);
        },
      ),
    })
    .dispatchWorkgroups(Math.ceil(vertexCount / 64));
  const vertices = await positions.read();
  const indices = (await pool.indices.read()).slice(firstIndex, firstIndex + indexCount);

  // A triangle's centroid lies inside it, so `heightAt` must read that triangle there
  let worst = 0;
  for (let t = 0; t < indices.length; t += 3) {
    const a = vertices[indices[t]];
    const b = vertices[indices[t + 1]];
    const c = vertices[indices[t + 2]];
    const height = ground.heightAt((a.x + b.x + c.x) / 3, (a.z + b.z + c.z) / 3);
    worst = Math.max(worst, Math.abs(height - (a.y + b.y + c.y) / 3));
  }
  expect(indices.length).toBe(200 * 200 * 6);
  expect(worst).toBeLessThan(1e-4);

  positions.buffer.destroy();
  ground.destroy();
  pool.destroy();
  root.destroy();
});
