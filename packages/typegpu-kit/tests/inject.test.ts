import { d, tgpu } from 'typegpu';
import { expect, it } from 'vitest';
import { byStage } from '../src/inject.ts';
import { recordingRoot } from '../src/testing.ts';

const Rows = d.arrayOf(d.f32);

it('binds storage read-write in compute and read-only in a vertex stage', () => {
  const { root } = recordingRoot();
  const rows = tgpu.accessor(Rows);
  const buffer = root.createBuffer(Rows(4)).$usage('storage');

  const write = tgpu.computeFn({ workgroupSize: [1] })(() => {
    'use gpu';
    rows.$[0] = 1;
  });
  const read = tgpu.vertexFn({ out: { position: d.builtin.position } })(() => {
    'use gpu';
    return { position: d.vec4f(rows.$[0], 0, 0, 1) };
  });
  const fragment = tgpu.fragmentFn({ out: d.vec4f })(() => {
    'use gpu';
    return d.vec4f(1);
  });

  const cfg = root.with(rows, byStage(buffer));
  const compute = tgpu.resolve([cfg.createComputePipeline({ compute: write })]);
  const render = tgpu.resolve([
    cfg.createRenderPipeline({ vertex: read, fragment, targets: { format: 'rgba8unorm' } }),
  ]);

  expect(compute).toContain('var<storage, read_write>');
  expect(render).toContain('var<storage, read>');
  expect(render).not.toContain('read_write');
});
