import { d, std, tgpu } from 'typegpu';
import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { glslRoot } from './utils.ts';
import { problems, validator, type Validator } from './validator.ts';

let gl: Validator;
beforeAll(async () => {
  gl = await validator();
});
afterAll(() => gl.close());

async function expectValid(modules: ReturnType<ReturnType<typeof glslRoot>['modules']>) {
  const reports = await gl.check(modules);
  for (const [i, report] of reports.entries()) {
    expect(problems(report), `${modules[i]?.vertex}\n-----\n${modules[i]?.fragment}`).toEqual([]);
  }
}

describe('render modules', () => {
  it('emits both stages of a triangle in one module', async () => {
    const { root, modules } = glslRoot();
    const pipeline = root.createRenderPipeline({
      vertex: tgpu.vertexFn({
        in: { i: d.builtin.vertexIndex },
        out: { pos: d.builtin.position, uv: d.vec2f },
      })(({ i }) => {
        'use gpu';
        const p = [d.vec2f(-1, -1), d.vec2f(3, -1), d.vec2f(-1, 3)];
        return { pos: d.vec4f(p[i] as d.v2f, 0, 1), uv: p[i] as d.v2f };
      }),
      fragment: tgpu.fragmentFn({ in: { uv: d.vec2f }, out: d.vec4f })(({ uv }) => {
        'use gpu';
        return d.vec4f(uv, std.fract(uv.x), 1);
      }),
      targets: { format: 'rgba8unorm' },
    });
    root.unwrap(pipeline);
    const [module] = modules();
    if (process.env.SHOW) console.log(module?.vertex, '\n=====\n', module?.fragment);
    expect(module?.vertex).toContain('gl_Position.y = -gl_Position.y;');
    await expectValid(modules());
  });
});
