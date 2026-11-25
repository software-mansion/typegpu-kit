import { recordingRoot } from '@typegpu/kit/testing';
import { d, tgpu } from 'typegpu';
import { expect, it } from 'vitest';
import { KitGenerator } from '../../src/generator.ts';

const kitRoot = () =>
  tgpu.initFromDevice({
    device: recordingRoot().root.device,
    unstable_shaderGeneratorClass: KitGenerator,
  });

it('refuses a compute entry written as a WGSL string by name', () => {
  const root = kitRoot();
  const out = root.createMutable(d.arrayOf(d.u32, 4));
  const kernel = tgpu.computeFn({
    workgroupSize: [1],
    in: { gid: d.builtin.globalInvocationId },
  })`{ out[in.gid.x] = 1u; }`.$uses({ out });
  expect(() => root.unwrap(root.createComputePipeline({ compute: kernel }))).toThrow(
    "@typegpu/kit-gl: A function written as a WGSL string ('kernel') is not available in WebGL 2 shaders.",
  );
});

it('refuses a fragment written as a WGSL string by name', () => {
  const root = kitRoot();
  const vertex = tgpu.vertexFn({ out: { pos: d.builtin.position } })(() => {
    'use gpu';
    return { pos: d.vec4f() };
  });
  const tint = tgpu.fragmentFn({ out: d.vec4f })`{ return vec4f(1); }`;
  expect(() =>
    root.unwrap(
      root.createRenderPipeline({ vertex, fragment: tint, targets: { format: 'rgba8unorm' } }),
    ),
  ).toThrow("A function written as a WGSL string ('tint')");
});
