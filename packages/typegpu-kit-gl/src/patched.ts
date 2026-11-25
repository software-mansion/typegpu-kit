/**
 * Whether TypeGPU carries the patch the fallback needs. Unpatched, TypeGPU writes WGSL for samplers
 * and textures from bind group layouts straight into the shader instead of asking the generator,
 * so the GLSL the device compiles fails far from the cause. A laid-out sampler resolved once shows
 * which TypeGPU this is.
 */

import { tgpu } from 'typegpu';
import { WgslGenerator } from 'typegpu/~internal';

const README =
  'https://github.com/software-mansion/typegpu-kit/tree/main/packages/typegpu-kit-gl#using-it-outside-this-repository';

let patched: boolean | undefined;

function check(): boolean {
  let declared = false;
  class Probe extends WgslGenerator {
    override declareGlobalVar(...args: Parameters<WgslGenerator['declareGlobalVar']>) {
      if (args[0].scope === 'handle') declared = true;
      return super.declareGlobalVar(...args);
    }
  }
  const layout = tgpu.bindGroupLayout({ probe: { sampler: 'filtering' } });
  // The laid-out sampler is the layout's own; reading `layout.$` works only inside a shader
  const key = Object.getOwnPropertySymbols(layout).find((symbol) =>
    symbol.description?.endsWith('$internal'),
  );
  const items = key ? (layout as unknown as Record<symbol, unknown>)[key] : undefined;
  // A TypeGPU this probe cannot read is not judged
  if (!Array.isArray(items) || items.length !== 1) return true;
  tgpu.resolve([items[0] as Parameters<typeof tgpu.resolve>[0][number]], {
    unstable_shaderGenerator: new Probe(),
  });
  return declared;
}

/** Throws one error naming the patch where TypeGPU lacks it. */
export function assertPatched(): void {
  patched ??= check();
  if (!patched) {
    throw new Error(
      '@typegpu/kit-gl needs typegpu@0.12.11 with the patch it ships, ' +
        "'@typegpu/kit-gl/patches/typegpu@0.12.11.patch', registered in your package manager. " +
        `See 'Using it outside this repository' in its README: ${README}`,
    );
  }
}
