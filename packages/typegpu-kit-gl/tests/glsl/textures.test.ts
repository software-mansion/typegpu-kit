import { common, d, std, tgpu } from 'typegpu';
import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import type { RenderModule } from '../../src/glsl/renderModule.ts';
import { glslRoot } from './utils.ts';
import { problems, validator, type Validator } from './validator.ts';

let gl: Validator;
beforeAll(async () => {
  gl = await validator();
});
afterAll(() => gl.close());

async function expectValid(modules: readonly RenderModule[]) {
  const reports = await gl.check(modules);
  for (const [i, report] of reports.entries()) {
    expect(problems(report), `${modules[i]?.fragment}`).toEqual([]);
  }
}

const colors = tgpu.bindGroupLayout({
  flat: { texture: d.texture2d(d.f32) },
  signed: { texture: d.texture2d(d.i32) },
  unsigned: { texture: d.texture2d(d.u32) },
  layers: { texture: d.texture2dArray(d.f32) },
  cube: { texture: d.textureCube(d.f32) },
  volume: { texture: d.texture3d(d.f32) },
  stored: { storageTexture: d.textureStorage2d('rgba8unorm', 'read-only') },
  linear: { sampler: 'filtering' },
  nearest: { sampler: 'non-filtering' },
});

const depths = tgpu.bindGroupLayout({
  depth: { texture: d.textureDepth2d() },
  depthLayers: { texture: d.textureDepth2dArray() },
  depthCube: { texture: d.textureDepthCube() },
  compare: { sampler: 'comparison' },
  plain: { sampler: 'non-filtering' },
});

/** Takes a texture and a sampler, so each call site with other globals gets its own copy. */
const blurred = (map: d.texture2d<d.F32>, through: d.sampler, uv: d.v2f) => {
  'use gpu';
  const offset = d.vec2f(0.01, 0);
  return (
    (std.textureSample(map, through, uv + offset) + std.textureSample(map, through, uv - offset)) *
    0.5
  );
};

/** Takes a depth array, as shadow filters do, and samples it with a global comparison sampler. */
const shadowed = (map: d.textureDepth2dArray, uv: d.v2f, layer: number, depth: number) => {
  'use gpu';
  return (
    std.textureSampleCompareLevel(map, depths.$.compare, uv, layer, depth) +
    std.textureSampleCompareLevel(map, depths.$.compare, uv + d.vec2f(0.01), layer, depth)
  );
};

describe('textures', () => {
  it('sample, load and measure every color texture kind', async () => {
    const { root, modules } = glslRoot();
    const pipeline = root.createRenderPipeline({
      vertex: common.fullScreenTriangle,
      fragment: tgpu.fragmentFn({ in: { uv: d.vec2f }, out: d.vec4f })(({ uv }) => {
        'use gpu';
        const at = d.vec2i(uv * 4);
        let sum = std.textureSample(colors.$.flat, colors.$.linear, uv);
        sum += std.textureSample(colors.$.flat, colors.$.linear, uv, d.vec2i(1, -1));
        sum += std.textureSampleBias(colors.$.flat, colors.$.linear, uv, 0.5);
        sum += std.textureSampleLevel(colors.$.flat, colors.$.nearest, uv, 2);
        sum += std.textureSampleGrad(
          colors.$.flat,
          colors.$.linear,
          uv,
          d.vec2f(0.1),
          d.vec2f(0.1),
        );
        sum += blurred(colors.$.flat, colors.$.linear, uv);
        sum += blurred(colors.$.flat, colors.$.nearest, uv);
        sum += d.vec4f(std.textureLoad(colors.$.signed, at, 0));
        sum += d.vec4f(std.textureLoad(colors.$.unsigned, at, 1));
        sum += std.textureSample(colors.$.layers, colors.$.linear, uv, 2);
        sum += std.textureLoad(colors.$.layers, at, 1, 0);
        sum += std.textureSampleLevel(colors.$.cube, colors.$.linear, d.vec3f(uv, 1), 0);
        sum += std.textureSample(colors.$.volume, colors.$.linear, d.vec3f(uv, 0.5));
        sum += std.textureLoad(colors.$.volume, d.vec3i(at, 0), 0);
        sum += std.textureLoad(colors.$.stored, at);
        const size = std.textureDimensions(colors.$.flat);
        const layers = std.textureNumLayers(colors.$.layers);
        const levels = std.textureNumLevels(colors.$.flat);
        const volume = std.textureDimensions(colors.$.volume, 1);
        return sum + d.vec4f(d.f32(size.x + layers + levels + volume.z));
      }),
      targets: { format: 'rgba8unorm' },
    });
    root.unwrap(pipeline);
    const [module] = modules();
    expect(module?.levels).toEqual([{ group: 0, binding: 0, uniform: 'flat_1_levels' }]);
    expect(
      module?.textures.filter((texture) => texture.group === 0 && texture.binding === 0),
    ).toHaveLength(2);
    await expectValid(modules());
  });

  it('compare and load depth textures', async () => {
    const { root, modules } = glslRoot();
    const pipeline = root.createRenderPipeline({
      vertex: common.fullScreenTriangle,
      fragment: tgpu.fragmentFn({ in: { uv: d.vec2f }, out: d.vec4f })(({ uv }) => {
        'use gpu';
        let lit = std.textureSampleCompare(depths.$.depth, depths.$.compare, uv, 0.5);
        lit += std.textureSampleCompareLevel(depths.$.depth, depths.$.compare, uv, 0.5);
        lit += std.textureSampleCompare(depths.$.depthLayers, depths.$.compare, uv, 1, 0.5);
        lit += shadowed(depths.$.depthLayers, uv, 2, 0.5);
        lit += std.textureSampleCompareLevel(
          depths.$.depthCube,
          depths.$.compare,
          d.vec3f(uv, 1),
          0.5,
        );
        lit += std.textureLoad(depths.$.depth, d.vec2i(uv * 4), 0);
        lit += std.textureLoad(depths.$.depthLayers, d.vec2u(uv * 4), 1, 0);
        lit += std.textureSample(depths.$.depth, depths.$.plain, uv);
        lit += std.textureSampleLevel(depths.$.depthLayers, depths.$.plain, uv, 1, 0);
        return d.vec4f(lit);
      }),
      targets: { format: 'rgba8unorm' },
    });
    root.unwrap(pipeline);
    const [module] = modules();
    expect(
      module?.textures.map(({ uniform, dimension, sampleType, comparison, sampler }) => [
        uniform,
        dimension,
        sampleType,
        comparison,
        sampler?.binding,
      ]),
    ).toEqual(
      expect.arrayContaining([
        ['depth_compare', '2d', 'depth', true, 3],
        ['depth_1', '2d', 'depth', false, undefined],
        ['depthLayers_compare', '2d-array', 'depth', true, 3],
        ['depthCube_compare', 'cube', 'depth', true, 3],
        ['depth_plain', '2d', 'depth', false, 4],
      ]),
    );
    await expectValid(modules());
  });
});
