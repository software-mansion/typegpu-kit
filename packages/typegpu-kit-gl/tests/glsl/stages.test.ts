import { common, d, std, tgpu } from 'typegpu';
import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { isRenderModule, parse, type RenderModule } from '../../src/glsl/renderModule.ts';
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
    expect(problems(report), `${modules[i]?.vertex}\n-----\n${modules[i]?.fragment}`).toEqual([]);
  }
}

const Instance = d.struct({ offset: d.vec3f, tag: d.vec2u });
const vertices = tgpu.vertexLayout(d.arrayOf(d.vec3f));
const instances = tgpu.vertexLayout(d.arrayOf(Instance), 'instance');

const settings = tgpu.bindGroupLayout({ tint: { uniform: d.vec4f }, scale: { uniform: d.f32 } });

/** Reads derivatives, which only a fragment stage has. */
const edge = (value: number) => {
  'use gpu';
  return std.fwidth(value) + std.abs(std.dpdx(value));
};

/** Both stages call this one. */
const halved = (value: d.v3f) => {
  'use gpu';
  return value * 0.5;
};

describe('stage IO', () => {
  it('reads attributes and builtins and writes several typed targets', async () => {
    const { root, modules } = glslRoot();
    const pipeline = root.createRenderPipeline({
      attribs: { position: vertices.attrib, ...instances.attrib },
      vertex: tgpu.vertexFn({
        in: {
          position: d.vec3f,
          offset: d.vec3f,
          tag: d.vec2u,
          instance: d.builtin.instanceIndex,
          index: d.builtin.vertexIndex,
        },
        out: {
          position: d.builtin.position,
          tag: d.interpolate('flat', d.vec2u),
          shade: d.interpolate('linear', d.f32),
          normal: d.interpolate('perspective, centroid', d.vec3f),
          picked: d.interpolate('flat', d.f32),
        },
      })(({ position, offset, tag, instance, index }) => {
        'use gpu';
        return {
          position: d.vec4f(halved(position + offset), 1),
          tag: tag + d.vec2u(instance, index),
          shade: settings.$.scale,
          normal: std.normalize(position),
          picked: d.f32(instance),
        };
      }),
      fragment: tgpu.fragmentFn({
        in: {
          tag: d.interpolate('flat', d.vec2u),
          shade: d.interpolate('linear', d.f32),
          normal: d.interpolate('perspective, centroid', d.vec3f),
          picked: d.interpolate('flat', d.f32),
          front: d.builtin.frontFacing,
          pixel: d.builtin.position,
        },
        out: {
          color: d.location(0, d.vec4f),
          id: d.location(2, d.vec4u),
          coverage: d.location(1, d.vec4f),
          depth: d.builtin.fragDepth,
          mask: d.builtin.sampleMask,
        },
      })(({ tag, shade, normal, picked, front, pixel }) => {
        'use gpu';
        if (shade < 0) std.discard();
        const facing = std.select(d.f32(-1), d.f32(1), front);
        return {
          color: settings.$.tint * shade * facing + d.vec4f(halved(normal), edge(pixel.x)),
          id: d.vec4u(tag, d.u32(picked), 0),
          coverage: d.vec4f(shade),
          depth: pixel.z * 0.5,
          mask: tag.x,
        };
      }),
      targets: {
        color: { format: 'rgba8unorm' },
        coverage: { format: 'r32float' },
        id: { format: 'rgba32uint' },
      },
    });
    root.unwrap(pipeline);
    const [module] = modules();
    expect(module?.vertex).toContain('layout(location = 0) in vec3 kit_attr0;');
    expect(module?.vertex).toContain('in uvec2 kit_attr2;');
    expect(module?.vertex).toContain('uint(gl_InstanceID + kit_firstInstance)');
    expect(module?.vertex).toContain('flat out uvec2 kit_vary');
    expect(module?.vertex).toContain('centroid out vec3 kit_vary');
    expect(module?.vertex).toMatch(/kit_vary\d \*= gl_Position.w;/);
    expect(module?.fragment).toContain('layout(location = 2) out uvec4 kit_out2;');
    expect(module?.fragment).toContain('layout(location = 1) out vec4 kit_out1;');
    expect(module?.fragment).toContain('gl_FragDepth = ');
    expect(module?.fragment).toContain('if ((kit_sampleMask & 1u) == 0u) discard;');
    expect(module?.fragment).toMatch(/\(kit_vary\d \* gl_FragCoord.w\)/);
    // Derivatives stay out of the vertex stage, the helper both call goes to both
    expect(module?.vertex).not.toContain('fwidth');
    expect(module?.fragment).toContain('fwidth');
    expect(module?.vertex).toContain('vec3 halved(');
    expect(module?.fragment).toContain('vec3 halved(');
    expect(module?.vertex).not.toContain('tint');
    expect(module?.blocks?.map((block) => block.block).toSorted()).toEqual([
      'scale_block',
      'tint_block',
    ]);
    await expectValid(modules());
  });

  it('leaves the fragment stage out of a depth-only module', async () => {
    const { root, codes } = glslRoot();
    const pipeline = root.createRenderPipeline({
      attribs: { position: vertices.attrib },
      vertex: tgpu.vertexFn({ in: { position: d.vec3f }, out: { position: d.builtin.position } })(
        ({ position }) => {
          'use gpu';
          return { position: d.vec4f(position, 1) };
        },
      ),
      depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'less' },
    });
    root.unwrap(pipeline);
    const [code] = codes();
    expect(code && isRenderModule(code)).toBe(true);
    const module = parse(code as string, { defines: ['#define KIT_CLIP_ZERO_TO_ONE'] });
    expect(module.fragment).toBeUndefined();
    expect(module.vertex.split('\n').slice(0, 3)).toEqual([
      '#version 300 es',
      '',
      '#define KIT_CLIP_ZERO_TO_ONE',
    ]);
    await expectValid([module]);
  });

  it('draws a full-screen triangle written in WGSL', async () => {
    const { root, modules } = glslRoot();
    const pipeline = root.createRenderPipeline({
      vertex: common.fullScreenTriangle,
      fragment: tgpu.fragmentFn({ in: { uv: d.vec2f }, out: d.vec4f })(({ uv }) => {
        'use gpu';
        return d.vec4f(uv, 0, 1);
      }),
      targets: { format: 'rgba8unorm' },
    });
    root.unwrap(pipeline);
    await expectValid(modules());
  });
});
