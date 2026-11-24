import { ref } from './references.ts';
import { construct, vector } from './values.ts';
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
  generator?: boolean;
}

type Kernel = (...args: unknown[]) => void | Generator<unknown, void>;
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
  const makeKernel = new Function(
    '$buffer',
    '$memory',
    '$construct',
    '$ref',
    '$shared',
    '$bits',
    `const arrayLength = (value) => value.$.length;\n${code}\nreturn ${entry.name};`,
  ) as (
    buffer: (group: number, binding: number, ctor: ViewConstructor) => ArrayBufferView,
    memory: (group: number, binding: number, layout: Layout) => unknown,
    constructValue: typeof construct,
    referenceValue: typeof ref,
    sharedValue: (name: string, make: () => unknown) => unknown,
    bits: Record<string, (value: number) => number>,
  ) => Kernel;
  return {
    kind: 'compute',
    workgroupSize: entry.size,
    bindings,
    dispatch(x: number, y: number, z: number, resources: CpuResources): void {
      for (const count of [x, y, z])
        if (!Number.isInteger(count) || count < 0)
          throw new Error('@typegpu/kit-gl CPU: invalid dispatch size');
      const scratch = new ArrayBuffer(4);
      const views = {
        u32: new Uint32Array(scratch),
        i32: new Int32Array(scratch),
        f32: new Float32Array(scratch),
      };
      const bits: Record<string, (value: number) => number> = {};
      for (const from of ['u32', 'i32', 'f32'] as const)
        for (const to of ['u32', 'i32', 'f32'] as const)
          bits[`${from}_${to}`] = (value) => {
            views[from][0] = value;
            return views[to][0]!;
          };
      const shared = new Map<string, unknown>();
      const instantiate = () =>
        makeKernel(
          (group, binding, ctor) => {
            const buffer = resources.buffer(group, binding);
            return new ctor(buffer.bytes, buffer.offset, buffer.size / 4);
          },
          (group, binding, layout) => bufferValue(resources.buffer(group, binding), layout),
          construct,
          ref,
          (name, make) => {
            if (!shared.has(name)) shared.set(name, make());
            return shared.get(name);
          },
          bits,
        );
      let kernel = instantiate();
      const global = vector([0, 0, 0]) as unknown as { x: number; y: number; z: number };
      const local = vector([0, 0, 0]) as unknown as { x: number; y: number; z: number };
      const workgroup = vector([0, 0, 0]) as unknown as { x: number; y: number; z: number };
      const builtins: Record<string, unknown> = {
        global_invocation_id: global,
        local_invocation_id: local,
        workgroup_id: workgroup,
        num_workgroups: vector([x, y, z]),
        local_invocation_index: 0,
      };
      const args: unknown[] = [];
      for (workgroup.z = 0; workgroup.z < z; workgroup.z++)
        for (workgroup.y = 0; workgroup.y < y; workgroup.y++)
          for (workgroup.x = 0; workgroup.x < x; workgroup.x++) {
            shared.clear();
            if (code.includes('//#scope workgroup')) kernel = instantiate();
            const invocations: Generator<unknown, void>[] = [];
            let index = 0;
            for (local.z = 0; local.z < entry.size[2]; local.z++)
              for (local.y = 0; local.y < entry.size[1]; local.y++)
                for (local.x = 0; local.x < entry.size[0]; local.x++) {
                  global.x = workgroup.x * entry.size[0] + local.x;
                  global.y = workgroup.y * entry.size[1] + local.y;
                  global.z = workgroup.z * entry.size[2] + local.z;
                  builtins.local_invocation_index = index++;
                  for (let i = 0; i < entry.inputs.length; i++)
                    args[i] =
                      entry.generator && Array.isArray(builtins[entry.inputs[i]!])
                        ? vector([...(builtins[entry.inputs[i]!] as number[])])
                        : builtins[entry.inputs[i]!];
                  if (code.includes('//#scope private')) kernel = instantiate();
                  const invocation = kernel(...args);
                  if (invocation) invocations.push(invocation);
                }
            if (invocations.length) {
              for (;;) {
                const states = invocations.map((invocation) => invocation.next());
                if (states.every((state) => state.done)) break;
                if (states.some((state) => state.done))
                  throw new Error('@typegpu/kit-gl CPU: divergent workgroup barrier');
              }
            }
          }
    },
  };
}
