import { d, std, tgpu, type TgpuRoot } from 'typegpu';
import { encodeHalf } from '../../src/cpu/textures.ts';

export const cornerDirections = Array.from({ length: 8 }, (_, i) =>
  d.vec3f(i & 1 ? -1 : 1, i & 2 ? -1 : 1, i & 4 ? -1 : 1),
);
export const cornerValues = [3, 6, 12, 24, 48, 96];
export function cornerData(size: number) {
  return new Uint16Array(
    Array.from({ length: size * size * 12 * 4 }, (_, word) => {
      const layer = Math.floor(word / (size * size * 4));
      return encodeHalf(
        word % 4 === 3 ? 1 : cornerValues[layer % 6]! + Math.floor(layer / 6) * 192,
      );
    }),
  );
}
export async function sampleCorners(root: TgpuRoot, size: number) {
  const texture = root
    .createTexture({ size: [size, size, 12], format: 'rgba16float' })
    .$usage('sampled');
  texture.write(cornerData(size));
  const cube = texture.createView(d.textureCubeArray(d.f32));
  const sampler = root.createSampler({ magFilter: 'linear', minFilter: 'linear' });
  const directions = root.createReadonly(d.arrayOf(d.vec3f, 8), cornerDirections);
  const output = root.createMutable(d.arrayOf(d.vec4f, 16));
  root
    .createComputePipeline({
      compute: tgpu.computeFn({ workgroupSize: [8], in: { id: d.builtin.globalInvocationId } })(
        ({ id }) => {
          'use gpu';
          output.$[id.x] = std.textureSampleLevel(
            cube.$,
            sampler.$,
            directions.$[id.x % 8],
            std.intdiv(id.x, d.u32(8)),
            0,
          );
        },
      ),
    })
    .dispatchWorkgroups(2);
  return (await output.read()).map((v) => [...v]);
}
