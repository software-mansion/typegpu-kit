import { tgpu, type TgpuRoot } from 'typegpu';
import { expect, onTestFinished } from 'vitest';
import { create, globals } from 'webgpu';

// Retain each Dawn instance because dawn.node frees it when its JS object is collected
const instances = new Map<string, GPU>();
const instanceOf = (dawn: readonly string[]) => {
  const key = dawn.join('\n');
  let gpu = instances.get(key);
  if (!gpu) instances.set(key, (gpu = create([...dawn])));
  return gpu;
};

let users = 0;
let restoreGlobals: (() => void) | undefined;

/** Installs Dawn's WebGPU globals for the current test, before any device is requested. */
function useDawnGlobals(): void {
  onTestFinished(() => {
    if (--users === 0) {
      restoreGlobals?.();
      restoreGlobals = undefined;
    }
  });
  if (users++ > 0) return;

  const navigator = globalThis.navigator;
  const gpu = Object.getOwnPropertyDescriptor(navigator, 'gpu');
  const descriptors = Object.keys(globals).map((name) => ({
    name,
    descriptor: Object.getOwnPropertyDescriptor(globalThis, name),
  }));
  restoreGlobals = () => {
    for (const { name, descriptor } of descriptors) {
      if (descriptor) Object.defineProperty(globalThis, name, descriptor);
      else Reflect.deleteProperty(globalThis, name);
    }
    if (gpu) Object.defineProperty(navigator, 'gpu', gpu);
    else Reflect.deleteProperty(navigator, 'gpu');
  };
  Object.assign(globalThis, globals);
  // TypeGPU reads WGSL language features such as immediates from navigator.gpu
  Object.defineProperty(navigator, 'gpu', {
    get: () => instanceOf([]),
    configurable: true,
  });
}

export interface GpuRoot extends TgpuRoot {
  /**
   * Waits for the queue and returns pending uncaptured WebGPU errors, then clears them. Use this
   * for expected errors in a test.
   */
  takeErrors(): Promise<string[]>;
}

export interface GpuRootOptions {
  /** Requires device features. `gpuRoot` throws if the adapter lacks any. Default none. */
  readonly features?: readonly GPUFeatureName[];
  /**
   * Enables features when the adapter supports them. Check availability in `device.features`.
   * Default none.
   */
  readonly optionalFeatures?: readonly GPUFeatureName[];
  /**
   * Sets required device limits, as in `requestDevice.requiredLimits`. Default the adapter
   * defaults.
   */
  readonly limits?: Record<string, GPUSize64>;
  /**
   * Passes flags to `webgpu.create`, such as `'backend=null'` to validate without a GPU. Each
   * set of flags uses a separate Dawn instance. Default none.
   */
  readonly dawn?: readonly string[];
}

/**
 * Creates a root on a dawn.node device. Call this inside a test. At test completion, it fails on
 * uncaptured WebGPU errors not consumed by `takeErrors` and destroys the device. Calling
 * `root.destroy()` leaves device destruction to this cleanup. The first call creates the Dawn
 * instance. WebGPU globals stay installed until the last test using them finishes.
 *
 * @param options.features features the device must have, which throw when the adapter lacks one
 * @returns a root whose device the test's end destroys
 */
export async function gpuRoot(options: GpuRootOptions = {}): Promise<GpuRoot> {
  useDawnGlobals();
  const { features = [], optionalFeatures = [], limits, dawn = [] } = options;
  const adapter = await instanceOf(dawn).requestAdapter();
  if (!adapter) throw new Error('No WebGPU adapter.');

  const missing = features.filter((feature) => !adapter.features.has(feature));
  if (missing.length > 0) {
    throw new Error(
      `The adapter lacks ${missing.join(', ')}. Ask for it in optionalFeatures to run without it.`,
    );
  }

  const device = await adapter.requestDevice({
    requiredFeatures: [
      ...features,
      ...optionalFeatures.filter((feature) => adapter.features.has(feature)),
    ],
    ...(limits && { requiredLimits: limits }),
  });
  const errors: string[] = [];
  device.addEventListener('uncapturederror', (event) => {
    errors.push((event as GPUUncapturedErrorEvent).error.message);
  });

  const takeErrors = async () => {
    await device.queue.onSubmittedWorkDone();
    return errors.splice(0);
  };

  onTestFinished(async () => {
    try {
      expect(await takeErrors(), 'uncaptured WebGPU errors').toEqual([]);
    } finally {
      device.destroy();
    }
  });

  return Object.assign(tgpu.initFromDevice({ device }), { takeErrors });
}
