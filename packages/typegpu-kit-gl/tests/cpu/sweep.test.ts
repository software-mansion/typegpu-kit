import { writeFile } from 'node:fs/promises';
import { frames } from '@typegpu/kit';
import { offscreenSurface, recordingRoot } from '@typegpu/kit/testing';
import { tgpu } from 'typegpu';
import {
  WgslGenerator,
  type FunctionDefinitionOptions,
  type ResolutionCtx,
} from 'typegpu/~internal';
import { it } from 'vitest';
import { examples } from '../../../typegpu-kit-examples/src/index.ts';
import { filesOf } from '../../../typegpu-kit-examples/src/hosts.ts';
import { fromDisk, driven } from '../../../typegpu-kit-examples/tests/utils/examples.ts';
import { CpuGenerator } from '../../src/cpu/generator.ts';
import { compile, isCpuModule } from '../../src/cpu/index.ts';

class SweepGenerator extends WgslGenerator {
  override languageKey = 'wgsl' as const;
  constructor() {
    super();
    let backend: WgslGenerator | undefined;
    let context: ResolutionCtx;
    return new Proxy(this, {
      get(_target, key) {
        if (key === 'initGenerator')
          return (ctx: ResolutionCtx) => {
            context = ctx;
          };
        if (key === 'functionDefinition')
          return (options: FunctionDefinitionOptions) => {
            if (!backend) {
              backend =
                options.functionType === 'compute' ? new CpuGenerator() : new WgslGenerator();
              backend.initGenerator(context);
            }
            return backend.functionDefinition(options);
          };
        if (key === 'languageKey') return backend?.languageKey ?? 'wgsl';
        if (!backend) {
          backend =
            context.topFunctionScope?.functionType === 'compute'
              ? new CpuGenerator()
              : new WgslGenerator();
          backend.initGenerator(context);
        }
        const member = Reflect.get(backend, key);
        return typeof member === 'function' ? member.bind(backend) : member;
      },
    });
  }
}

it.runIf(process.env.CPU_SWEEP === '1')(
  'records per-example CPU compilation coverage',
  async () => {
    const results: { example: string; status: string; modules: number; error?: string }[] = [];
    for (const example of examples) {
      const recording = recordingRoot();
      const root = tgpu.initFromDevice({
        device: recording.root.device,
        unstable_shaderGeneratorClass: SweepGenerator,
      });
      const surface = offscreenSurface(root, { width: 64, height: 36, format: 'bgra8unorm' });
      let scene:
        | Awaited<ReturnType<Awaited<ReturnType<typeof example.scene>>['createScene']>>
        | undefined;
      let loop: ReturnType<typeof frames.create> | undefined;
      try {
        const { createScene } = await example.scene();
        scene = await createScene(root, surface, await filesOf(example.key, fromDisk));
        loop = frames.create(root, scene.frame);
        const driver = driven(loop, scene.ready);
        await driver.warmUp();
        await driver.run(2);
        const modules = recording.log.shaders.filter(isCpuModule);
        modules.forEach(compile);
        results.push({ example: example.key, status: 'pass', modules: modules.length });
      } catch (error) {
        results.push({
          example: example.key,
          status: 'fail',
          modules: recording.log.shaders.filter(isCpuModule).length,
          error: String(error),
        });
      } finally {
        loop?.destroy();
        scene?.destroy?.();
        surface.destroy();
        root.destroy();
      }
    }
    await writeFile(
      new URL('./sweep-results.json', import.meta.url),
      `${JSON.stringify(results, null, 2)}\n`,
    );
  },
  120000,
);
