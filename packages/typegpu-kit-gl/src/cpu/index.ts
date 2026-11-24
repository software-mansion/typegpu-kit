import { bufferValue, type Layout } from './memory.ts';
import type { CpuBindingUse, CpuModule, CpuResources } from './types.ts';
export type * from './types.ts';

export function isCpuModule(code: string): boolean {
  return code.split('\n').some((line) => line.trim() === '//#kit-gl compute');
}

interface Entry {
  name: string;
  size: [number, number, number];
  inputs: string[];
}

type Kernel = (...args: unknown[]) => void;
type ViewConstructor = typeof Uint32Array | typeof Int32Array | typeof Float32Array;

export function compile(code: string): CpuModule {
  if (!isCpuModule(code)) throw new Error('@typegpu/kit-gl CPU: missing compute marker');
  const entryLine = code.split('\n').find((line) => line.startsWith('//#entry '));
  if (!entryLine) throw new Error('@typegpu/kit-gl CPU: missing entry metadata');
  const entry = JSON.parse(entryLine.slice(9)) as Entry;
  if (
    !/^[a-zA-Z_$][\w$]*$/.test(entry.name) ||
    entry.size.length !== 3 ||
    !entry.size.every((n) => Number.isInteger(n) && n > 0)
  )
    throw new Error('@typegpu/kit-gl CPU: invalid entry metadata');
  const bindings = code
    .split('\n')
    .filter((line) => line.startsWith('//#binding '))
    .map((line) => {
      const value = JSON.parse(line.slice(11)) as CpuBindingUse;
      return { ...value, group: Number(value.group) };
    });
  const makeKernel = new Function('$buffer', '$memory', `${code}\nreturn ${entry.name};`) as (
    buffer: (group: number, binding: number, ctor: ViewConstructor) => ArrayBufferView,
    memory: (group: number, binding: number, layout: Layout) => unknown,
  ) => Kernel;
  return {
    kind: 'compute',
    workgroupSize: entry.size,
    bindings,
    dispatch(x: number, y: number, z: number, resources: CpuResources): void {
      for (const count of [x, y, z])
        if (!Number.isInteger(count) || count < 0)
          throw new Error('@typegpu/kit-gl CPU: invalid dispatch size');
      const kernel = makeKernel(
        (group, binding, ctor) => {
          const buffer = resources.buffer(group, binding);
          return new ctor(buffer.bytes, buffer.offset, buffer.size / 4);
        },
        (group, binding, layout) => bufferValue(resources.buffer(group, binding), layout),
      );
      const global = { x: 0, y: 0, z: 0 };
      const local = { x: 0, y: 0, z: 0 };
      const workgroup = { x: 0, y: 0, z: 0 };
      const builtins: Record<string, unknown> = {
        global_invocation_id: global,
        local_invocation_id: local,
        workgroup_id: workgroup,
        num_workgroups: { x, y, z },
        local_invocation_index: 0,
      };
      const args: unknown[] = [];
      for (workgroup.z = 0; workgroup.z < z; workgroup.z++)
        for (workgroup.y = 0; workgroup.y < y; workgroup.y++)
          for (workgroup.x = 0; workgroup.x < x; workgroup.x++) {
            let index = 0;
            for (local.z = 0; local.z < entry.size[2]; local.z++)
              for (local.y = 0; local.y < entry.size[1]; local.y++)
                for (local.x = 0; local.x < entry.size[0]; local.x++) {
                  global.x = workgroup.x * entry.size[0] + local.x;
                  global.y = workgroup.y * entry.size[1] + local.y;
                  global.z = workgroup.z * entry.size[2] + local.z;
                  builtins.local_invocation_index = index++;
                  for (let i = 0; i < entry.inputs.length; i++)
                    args[i] = builtins[entry.inputs[i]!];
                  kernel(...args);
                }
          }
    },
  };
}
