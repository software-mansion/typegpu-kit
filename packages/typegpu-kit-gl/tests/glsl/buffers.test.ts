import { d, std, tgpu } from 'typegpu';
import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { glslRoot } from './utils.ts';
import { problems, validator, type Validator } from './validator.ts';

let gl: Validator;
beforeAll(async () => {
  gl = await validator();
});
afterAll(() => gl.close());

const Particle = d.struct({
  position: d.vec3f,
  age: d.f32,
  life: d.f32,
  color: d.vec4f,
  id: d.u32,
});

describe('buffers', () => {
  it('reads struct members of array elements', async () => {
    const { root, modules } = glslRoot();
    const pool = root.createBuffer(d.arrayOf(Particle, 16)).$usage('storage');
    const params = root
      .createBuffer(d.struct({ scale: d.f32, tint: d.vec4f, m: d.mat4x4f }))
      .$usage('uniform');
    const poolRead = pool.as('readonly');
    const paramsRead = params.as('uniform');
    const pipeline = root.createRenderPipeline({
      vertex: tgpu.vertexFn({
        in: { i: d.builtin.instanceIndex },
        out: { pos: d.builtin.position, color: d.vec4f },
      })(({ i }) => {
        'use gpu';
        const p = poolRead.$[i];
        const alive = poolRead.$[i].age < poolRead.$[i].life;
        const scaled = std.length(p.position) * paramsRead.$.scale;
        return {
          pos: paramsRead.$.m * d.vec4f(p.position * scaled, 1),
          color: std.select(paramsRead.$.tint, p.color, alive),
        };
      }),
      fragment: tgpu.fragmentFn({ in: { color: d.vec4f }, out: d.vec4f })(({ color }) => {
        'use gpu';
        return color;
      }),
      targets: { format: 'rgba8unorm' },
    });
    root.unwrap(pipeline);
    const [module] = modules();
    if (process.env.SHOW) console.log(module?.vertex);
    expect(module?.buffers.map((b) => [b.texture, b.access])).toEqual([['pool', 'readonly']]);
    expect(module?.blocks?.map((b) => [b.block, b.size])).toEqual([['params_block', 96]]);
    const reports = await gl.check(modules());
    expect(problems(reports[0]!)).toEqual([]);
  });
});
