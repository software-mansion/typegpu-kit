import {
  cameras,
  dof,
  environment,
  exposure,
  impostors,
  meshPool,
  passes,
  picker,
} from '@typegpu/kit';
import { meshes } from '@typegpu/kit/geometry';
import { d, std, tgpu, type TgpuBuffer, type TgpuRoot } from 'typegpu';
import { createCulling } from '../../../typegpu-kit/src/render/culling.ts';

export async function cullingValues(root: TgpuRoot): Promise<number[]> {
  const count = 900,
    capacity = 1000,
    meshCount = 4;
  const rows = [
    ...Array.from({ length: 600 }, (_, i) => i),
    ...Array.from({ length: 300 }, (_, i) => 899 - i),
  ];
  const listed = root.createReadonly(d.arrayOf(d.u32, capacity + meshCount), [
    4,
    604,
    604,
    0,
    ...rows,
    ...Array<number>(100).fill(0),
  ]);
  const entries = root.createUniform(d.u32, count);
  const cull = createCulling(root, root, {
    capacity,
    meshes: meshCount,
    layers: 2,
    firstInstance: false,
    listed,
    entries,
    meshOf: (row) => {
      'use gpu';
      return std.select(d.u32(2), d.u32(0), row < 600);
    },
    seen: (row) => {
      'use gpu';
      if (passes.layer.$ === 0) return row % 3 !== 0;
      return row % 2 === 0;
    },
    active: () => true,
  });
  for (let frame = 0; frame < 2; frame++) {
    const encoder = root.createCommandEncoder();
    const pass = encoder.beginComputePass();
    cull.record(pass, count);
    pass.end();
    encoder.submit();
  }
  const result = [...(await cull.culled.read()), ...(await cull.draws.read())];
  cull.destroy();
  return result;
}

export async function pickerValues(root: TgpuRoot): Promise<number[]> {
  const view = cameras.orbit(root, {
    size: { width: 200, height: 100 },
    target: [0, 0, 0],
    distance: 10,
    pitch: 0,
  });
  const positions = root.createReadonly(
    d.arrayOf(d.vec3f, 300),
    Array.from({ length: 300 }, (_, i) =>
      i === 3 ? d.vec3f(3, 0, 0) : i === 150 ? d.vec3f(5, 0, 0) : d.vec3f(0, 0, 50 + i),
    ),
  );
  const hits = picker.create(root, view, {
    capacity: 300,
    size: { width: 200, height: 100 },
    position: (i) => {
      'use gpu';
      return d.vec3f(positions.$[i]);
    },
  });
  const out = root.createMutable(d.u32);
  const probe = root.pipe(hits.inject()).createComputePipeline({
    compute: tgpu.computeFn({ workgroupSize: [1] })(() => {
      'use gpu';
      out.$ = hits.found();
    }),
  });
  const results: number[] = [];
  for (const point of [
    [3, 0, 0],
    [5, 0, 0],
    [0, 0, 0],
  ]) {
    const ndc = [0, 0, 0];
    view.project(point, ndc);
    const encoder = root.createCommandEncoder();
    const pass = encoder.beginComputePass();
    hits.pick(pass, (ndc[0]! + 1) / 2, (1 - ndc[1]!) / 2);
    pass.setPipeline(probe);
    pass.dispatchWorkgroups(1);
    pass.end();
    encoder.submit();
    results.push(await out.read());
  }
  hits.destroy();
  return results;
}

export async function exposureValues(root: TgpuRoot): Promise<number[]> {
  const image = root.createTexture({ size: [16, 16], format: 'rgba16float' }).$usage('sampled');
  const eye = exposure.create(root, { source: () => image });
  const out = root.createMutable(d.f32);
  const probe = root.pipe(eye.inject()).createComputePipeline({
    compute: tgpu.computeFn({ workgroupSize: [1] })(() => {
      'use gpu';
      out.$ = exposure.current.$;
    }),
  });
  const result: number[] = [];
  for (const half of [0x4000, 0x2c00, 0x4000]) {
    const data = new Uint16Array(16 * 16 * 4).fill(half);
    root.device.queue.writeTexture(
      { texture: root.unwrap(image) },
      data,
      { bytesPerRow: 16 * 8 },
      [16, 16],
    );
    const encoder = root.createCommandEncoder();
    eye.record(encoder, { dt: 0.5 });
    const pass = encoder.beginComputePass();
    pass.setPipeline(probe);
    pass.dispatchWorkgroups(1);
    pass.end();
    encoder.submit();
    result.push(await out.read());
  }
  eye.destroy();
  image.destroy();
  return result;
}

export async function environmentValues(root: TgpuRoot): Promise<number[]> {
  const sky = environment.create(root, { size: 16 });
  const source = sky.add((direction: d.v3f) => {
    'use gpu';
    return d.vec3f(0.5) + std.max(direction, d.vec3f());
  });
  const encoder = root.createCommandEncoder();
  sky.bake(encoder, source);
  encoder.submit();
  const axes = root.createReadonly(d.arrayOf(d.vec3f, 6), [
    d.vec3f(1, 0, 0),
    d.vec3f(-1, 0, 0),
    d.vec3f(0, 1, 0),
    d.vec3f(0, -1, 0),
    d.vec3f(0, 0, 1),
    d.vec3f(0, 0, -1),
  ]);
  const out = root.createMutable(d.arrayOf(d.vec3f, 24));
  const probe = tgpu.computeFn({ workgroupSize: [6], in: { id: d.builtin.localInvocationIndex } })(
    ({ id }) => {
      'use gpu';
      const axis = axes.$[id];
      out.$[id * 4] = environment.irradiance(axis);
      out.$[id * 4 + 1] = environment.reflection(axis, d.f32(0));
      out.$[id * 4 + 2] = environment.reflection(axis, 0.5);
      out.$[id * 4 + 3] = environment.reflection(axis, d.f32(1));
    },
  );
  root.pipe(sky.inject()).createComputePipeline({ compute: probe }).dispatchWorkgroups(1);
  const result = (await out.read()).flatMap((v) => [v.x, v.y, v.z]);
  sky.destroy();
  return result;
}

export async function focusValues(root: TgpuRoot): Promise<number[]> {
  const view = cameras.orbit(root, {
    size: { width: 16, height: 16 },
    distance: 4,
    target: [0, 0, 0],
  });
  const image = root.createTexture({ size: [16, 16], format: 'rgba16float' }).$usage('sampled');
  const depth = root
    .createTexture({ size: [16, 16], format: 'depth32float' })
    .$usage('sampled', 'render');
  // The moved focus is in the settings only where `dof` copies it there, so read its own buffer
  const created: TgpuBuffer<d.BaseData>[] = [];
  const createBuffer = root.createBuffer;
  root.createBuffer = ((...args: Parameters<typeof createBuffer>) => {
    const buffer = createBuffer.apply(root, args);
    created.push(buffer as TgpuBuffer<d.BaseData>);
    return buffer;
  }) as typeof createBuffer;
  const blur = dof.create(root, view, {
    source: () => image,
    depth: () => depth,
    pull: 1,
    settings: { focus: 10 },
  });
  root.createBuffer = createBuffer;
  const state = created.find(
    ({ dataType }) => 'propTypes' in dataType && 'goal' in (dataType.propTypes as object),
  ) as TgpuBuffer<d.WgslStruct<{ focus: d.F32 }>>;
  const results: number[] = [];
  for (const dt of [0.5, 0.5]) {
    const encoder = root.createCommandEncoder();
    encoder
      .beginRenderPass({
        colorAttachments: [],
        depthStencilAttachment: {
          view: depth.createView('render'),
          depthClearValue: 0.5,
          depthLoadOp: 'clear',
          depthStoreOp: 'store',
        },
      })
      .end();
    blur.aim(0.5, 0.5);
    blur.record(encoder, { dt });
    encoder.submit();
    results.push((await state.read()).focus);
  }
  blur.destroy();
  image.destroy();
  depth.destroy();
  return results;
}

export async function impostorValues(root: TgpuRoot): Promise<number[]> {
  const created: GPUBuffer[] = [];
  const original = root.device.createBuffer.bind(root.device);
  root.device.createBuffer = (descriptor) => {
    const buffer = original(descriptor);
    created.push(buffer);
    return buffer;
  };
  try {
    const pool = meshPool.create(root, { capacity: 4 });
    const box = pool.add(meshes.box({ width: 2, height: 1, depth: 0.5 }));
    const card = pool.add(impostors.CARD).meshes[0];
    const atlas = impostors.create(root, pool, [{ mesh: box, position: [2, 3, 4] }], {
      card,
      views: 2,
      size: 16,
    });
    const encoder = root.createCommandEncoder();
    atlas.record(encoder);
    encoder.submit();
    const buffer = created.find((value) => value.label === 'subjectBuffer');
    if (!buffer)
      throw new Error(`Missing subjectBuffer: ${created.map((value) => value.label).join(', ')}`);
    const staging = original({
      size: buffer.size,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const copy = root.device.createCommandEncoder();
    copy.copyBufferToBuffer(buffer, 0, staging, 0, buffer.size);
    root.device.queue.submit([copy.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const result = [...new Float32Array(staging.getMappedRange().slice(0))];
    staging.unmap();
    staging.destroy();
    atlas.destroy();
    pool.destroy();
    return result;
  } finally {
    root.device.createBuffer = original;
  }
}
