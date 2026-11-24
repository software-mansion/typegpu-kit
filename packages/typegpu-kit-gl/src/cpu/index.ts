import type { CpuModule } from './types.ts';

export type * from './types.ts';

/** Reports whether `code` is a module `CpuGenerator` wrote. */
export function isCpuModule(code: string): boolean {
  return code.includes('//#kit-gl compute');
}

/** Compiles a compute module's JavaScript into a kernel the device dispatches. */
export function compile(_code: string): CpuModule {
  throw new Error('@typegpu/kit-gl: CPU compute is not implemented yet');
}
