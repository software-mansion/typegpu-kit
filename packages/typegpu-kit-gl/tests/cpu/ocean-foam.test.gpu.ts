import { gpuRoot } from '@typegpu/kit/testing/dawn';
import { tgpu, type TgpuRoot } from 'typegpu';
import { expect, it } from 'vitest';
import { createFoam } from '../../../typegpu-kit-examples/src/ocean/foam.ts';
import { compile, type CpuModule } from '../../src/cpu/index.ts';
import { cpuRoot } from './device.ts';
import { SweepGenerator } from './scene-generator.ts';
import { observeScene } from './scene-observer.ts';
import { readTexture } from './example-comparison.ts';
import { recordResults } from './results.ts';

async function tile(root: TgpuRoot, observer: ReturnType<typeof observeScene>) {
  const foam = createFoam(root, { direction: Math.PI / 2 });
  await foam.initAsync();
  const encoder = root.createCommandEncoder(),
    pass = encoder.beginComputePass();
  foam.bake(pass);
  pass.end();
  encoder.submit();
  await root.device.queue.onSubmittedWorkDone();
  const resource = observer.resources.find((item) => item.label === 'tile')!;
  const cpu = observer.cpuTextures.get(resource.handle);
  const levels = cpu
    ? cpu.levels.map((level) =>
        new Uint16Array(
          level.data.buffer,
          level.data.byteOffset,
          level.data.byteLength / 2,
        ).slice(),
      )
    : (await readTexture(root.device, resource.handle as GPUTexture)).map(
        (level) => new Uint16Array(level.bytes),
      );
  foam.destroy();
  return levels;
}
it.runIf(process.env.CPU_OCEAN_FOAM === '1')(
  'separates ocean foam-tile half-store drift from mip arithmetic',
  async () => {
    const dawn = await gpuRoot(),
      gpuObserver = observeScene();
    const gpu = tgpu.initFromDevice({ device: gpuObserver.device(dawn.device) });
    try {
      const expected = await tile(gpu, gpuObserver);
      const results = [];
      for (const mode of ['original', 'equal-base', 'truncate-stores'] as const) {
        const observer = observeScene(),
          cache = new Map<string, CpuModule>();
        const cpu = cpuRoot({
          generator: SweepGenerator,
          observeDevice: observer.device,
          onTexture: (texture, data) => observer.cpuTextures.set(texture, data),
          beforeDispatch(code, _label, dispatch) {
            if (mode === 'original') return;
            let writesTile = false;
            for (const use of dispatch.module.bindings) {
              const entry = Array.from(dispatch.groups.get(use.group)!.entries).find(
                (item) => item.binding === use.binding,
              )!;
              if (observer.resourceOf(entry.resource)?.label !== 'tile') continue;
              if (use.kind === 'texture') {
                const level = dispatch.resources.texture(use.group, use.binding).levels[0]!;
                if (level.width === 1024) (level.data as Uint16Array).set(expected[0]!);
              }
              writesTile ||= use.kind === 'storageTexture';
            }
            if (mode === 'truncate-stores' && writesTile) {
              let module = cache.get(code);
              if (!module) {
                module = compile(
                  `const originalStore=$textures.textureStore;$textures={...$textures,textureStore:(t,c,v,a)=>{const convert=values=>values.map(value=>{let bits=$pack([value,0],16,'float')&65535;const rounded=$unpack(bits,16,'float')[0];if(Math.abs(rounded)>Math.abs(value))bits--;return $unpack(bits,16,'float')[0];});return typeof v==='number'?originalStore(t,c,v,convert(a)):originalStore(t,c,convert(v));}};\n${code}`,
                );
                cache.set(code, module);
              }
              dispatch.module.dispatch = module.dispatch;
            }
          },
        });
        try {
          const actual = await tile(cpu, observer);
          const levels = actual.map((values, level) => {
            let differences = 0,
              maximumUlps = 0;
            for (let i = 0; i < values.length; i++) {
              if (values[i] !== expected[level]![i]) differences++;
              maximumUlps = Math.max(maximumUlps, Math.abs(values[i]! - expected[level]![i]!));
            }
            return { level, differences, maximumUlps };
          });
          results.push({ mode, levels });
          recordResults('oceanFoamTile', results);
          if (mode === 'truncate-stores')
            expect(levels.every((level) => level.differences === 0)).toBe(true);
        } finally {
          cpu.destroy();
        }
      }
    } finally {
      gpu.destroy();
      dawn.destroy();
    }
  },
  300000,
);
