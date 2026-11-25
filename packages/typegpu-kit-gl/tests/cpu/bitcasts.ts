import { d, std, tgpu, type TgpuRoot } from 'typegpu';

export async function bitcasts(root: TgpuRoot): Promise<number[]> {
  const floats = root.createReadonly(d.vec4f, d.vec4f(-0, 0.1, -123.5, 2 ** -100));
  const integers = root.createReadonly(d.vec4i, d.vec4i(-2147483648, 1, -123, 2147483647));
  const words = root.createReadonly(
    d.vec4u,
    d.vec4u(0x80000000, 0x3dcccccd, 0xc2f70000, 0x0d800000),
  );
  const output = root.createMutable(d.arrayOf(d.vec4u, 5));
  const kernel = tgpu.computeFn({ workgroupSize: [1] })(() => {
    'use gpu';
    output.$[0] = std.bitcast(d.vec4f, d.vec4u)(floats.$);
    output.$[1] = std.bitcast(d.vec4i, d.vec4u)(integers.$);
    const decoded = std.bitcast(d.vec4u, d.vec4f)(words.$);
    output.$[2] = std.bitcast(d.vec4f, d.vec4u)(decoded);
    const scalar = std.bitcast(d.u32, d.f32)(words.$.z);
    output.$[3] = d.vec4u(
      std.bitcast(d.f32, d.u32)(scalar),
      std.bitcast(d.i32, d.u32)(integers.$.z),
      std.bitcast(d.f32, d.u32)(floats.$.x),
      std.bitcast(d.f32, d.u32)(floats.$.y * 2),
    );
    output.$[4] = std.bitcastF32toU32(std.bitcastU32toF32(words.$));
  });
  root.createComputePipeline({ compute: kernel }).dispatchWorkgroups(1);
  return (await output.read()).flatMap((row) => [...row]);
}
