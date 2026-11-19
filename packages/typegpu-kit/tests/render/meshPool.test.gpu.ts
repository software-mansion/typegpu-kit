import { gpuRoot } from '@typegpu/kit/testing/dawn';
import { meshes } from '@typegpu/kit/geometry';
import { d, tgpu } from 'typegpu';
import { expect, it } from 'vitest';
import { meshPool } from '../../src/index.ts';

it('bakes one shape added twice, around another, into each mesh its own ranges and box', async () => {
  const root = await gpuRoot();
  const brick = meshes.box({ width: 2, height: 3, depth: 4 });
  const pool = meshPool.create(root, { vertices: 200, indices: 400, capacity: 4 });
  const first = pool.add(brick);
  pool.add(meshes.plane());
  const second = pool.add(brick);

  const { vertexCount } = brick;
  const Baked = d.struct({ first: d.vec3f, second: d.vec3f, shape: d.vec3f });
  const output = root.createBuffer(d.arrayOf(Baked, vertexCount)).$usage('storage');
  const boxes = root.createBuffer(d.arrayOf(meshPool.Box, 2)).$usage('storage');
  const out = output.as('mutable');
  const bounds = boxes.as('mutable');
  const kernel = tgpu.computeFn({ workgroupSize: [1], in: { gid: d.builtin.globalInvocationId } })(
    ({ gid }) => {
      'use gpu';
      const i = gid.x;
      out.$[i] = Baked({
        first: meshPool.positionAt(first.baseVertex + i),
        second: meshPool.positionAt(second.baseVertex + i),
        shape: brick.vertexAt(i).position,
      });
      if (i === 0) {
        bounds.$[0] = meshPool.boundsOf(first.index);
        bounds.$[1] = meshPool.boundsOf(second.index);
      }
    },
  );
  const reading = root.pipe(pool.inject()).createComputePipeline({ compute: kernel });
  reading.dispatchWorkgroups(vertexCount);

  for (const { first: a, second: b, shape } of await output.read()) {
    expect([...a]).toEqual([...shape]);
    expect([...b]).toEqual([...shape]);
  }
  for (const { center, half } of await boxes.read()) {
    expect([...center]).toEqual([0, 0, 0]);
    expect([...half]).toEqual([1, 1.5, 2]);
  }
  const indices = await pool.indices.read();
  const own = (mesh: meshPool.Mesh) =>
    indices.slice(mesh.firstIndex, mesh.firstIndex + mesh.indexCount);
  expect(own(second)).toEqual(own(first));
  expect(new Set(own(first)).size).toBe(vertexCount);
  root.destroy();
});
