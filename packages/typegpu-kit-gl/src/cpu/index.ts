import { compileJavascript } from './javascript.ts';
import { pack, unpack } from './packing.ts';
import { textureOps } from './textures.ts';
import { ref } from './references.ts';
import { construct, vector } from './values.ts';
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
  phases?: number;
}

type Kernel = (...args: unknown[]) => void | Generator<unknown, void>;
interface Execution {
  kernel: Kernel;
  phases?: () => ((...args: unknown[]) => boolean | void)[];
  capture(values?: unknown[]): unknown[];
  restore(values: unknown[]): void;
  resetWorkgroup(): void;
  bind(resources: CpuResources): void;
  release(): void;
}
type ViewConstructor = typeof Uint32Array | typeof Int32Array | typeof Float32Array;

const ASSIGNS = /^(?:=(?![=>])|(?:\+|-|\*\*?|\/|%|&&?|\|\|?|\^|<<|>>>?|\?\?)=|\+\+|--)/;

/**
 * Whether a mutable buffer binding is only ever read, so a dispatch leaves GL's copy current:
 * every use of the typed arrays declared over it is an element read, `name[…]` neither assigned
 * nor stepped. Any other use, such as an atomic helper or a pointer taking the array whole, counts
 * as a write, so the answer errs towards one.
 */
function onlyRead(code: string, use: CpuBindingUse): boolean {
  const declared = new RegExp(
    `^const ([A-Za-z_$][\\w$]*) = \\$buffer\\(${use.group}, ?${use.binding}, ?`,
    'gm',
  );
  const names = [...code.matchAll(declared)].map((match) => match[1]!);
  if (names.length === 0) return false;
  const body = code
    .split('\n')
    .filter((line) => !line.startsWith('//') && !/^const [A-Za-z_$][\w$]* = \$buffer\(/.test(line))
    .join('\n');
  for (const name of names) {
    const uses = new RegExp(`(?<![\\w$.])${name.replaceAll('$', '\\$')}(?![\\w$])`, 'g');
    for (const match of body.matchAll(uses)) {
      const at = match.index;
      if (/(?:\+\+|--)[\s(]*$/.test(body.slice(Math.max(0, at - 8), at))) return false;
      let i = at + name.length;
      if (body[i] !== '[') return false;
      for (let depth = 0; i < body.length; i++) {
        if (body[i] === '[') depth++;
        else if (body[i] === ']' && --depth === 0) break;
      }
      i++;
      while (i < body.length && /[\s)]/.test(body[i]!)) i++;
      if (ASSIGNS.test(body.slice(i, i + 4))) return false;
    }
  }
  return true;
}

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
      const use = { ...value, group: Number(value.group) };
      return use.kind === 'buffer' && use.writes && onlyRead(code, use)
        ? { ...use, writes: false }
        : use;
    });
  const privateNames = [...code.matchAll(/^\/\/#private (\w+)$/gm)].map((match) => match[1]!);
  const workgroupState = code.includes('//#scope workgroup');
  const workgroupVariables = [...code.matchAll(/^\/\/#workgroup (.*)$/gm)].map(
    (match) => JSON.parse(match[1]!) as { name: string; initial: string; reset?: string },
  );
  const reusable =
    code.split('\n').includes('//#kit-gl reusable') &&
    (!workgroupState || workgroupVariables.length > 0);
  const caches = [...code.matchAll(/^\/\/#dispatch-cache (.*)$/gm)].flatMap(
    (match) => JSON.parse(match[1]!) as string[],
  );
  const resourceBindings: { name: string; value: string }[] = [];
  code = code.replace(
    /^const ([A-Za-z_$][\w$]*) = (\$(?:buffer|resources\.(?:texture|sampler))\([^;\n]*\));$/gm,
    (_declaration, name: string, value: string) => {
      resourceBindings.push({ name, value });
      return `let ${name} = ${value};`;
    },
  );
  const resetCaches = caches.map((name) => `${name}=undefined;`).join('');
  const makeKernel = compileJavascript(
    [
      '$buffer',
      '$construct',
      '$ref',
      '$shared',
      '$bits',
      '$vector',
      '$resources',
      '$textures',
      '$pack',
      '$unpack',
    ],
    `${code}\nconst arrayLength = (value) => value.$.length;\nreturn {bind:(resources)=>{$resources=resources;${resourceBindings.map(({ name, value }) => `${name}=${value};`).join('')}${resetCaches}},release:()=>{$resources=undefined;${resourceBindings.map(({ name }) => `${name}=undefined;`).join('')}${resetCaches}},kernel:${entry.name},${entry.phases ? `phases:${entry.name}_phases,` : ''}capture:(values=[])=>{${privateNames.map((name, i) => `values[${i}]=${name};`).join('')}return values;},restore:(values)=>{${privateNames.map((name, i) => `${name}=values[${i}];`).join('')}},resetWorkgroup:()=>{${workgroupVariables.map(({ name, initial, reset }) => reset ?? `${name}=${initial};`).join('')}}};`,
  ) as (
    buffer: (group: number, binding: number, ctor: ViewConstructor) => ArrayBufferView,
    constructValue: typeof construct,
    referenceValue: typeof ref,
    sharedValue: (name: string, make: () => unknown) => unknown,
    bits: Record<string, (value: number) => number>,
    vectorValue: typeof vector,
    resourcesValue: CpuResources,
    texturesValue: typeof textureOps,
    packValue: typeof pack,
    unpackValue: typeof unpack,
  ) => Execution;
  let boundResources: CpuResources | undefined;
  let heldExecution: Execution | undefined;
  const scratch = new ArrayBuffer(4);
  const u32 = new Uint32Array(scratch);
  const i32 = new Int32Array(scratch);
  const f32 = new Float32Array(scratch);
  const bits = {
    u32_u32: (value: number) => value >>> 0,
    i32_i32: (value: number) => value | 0,
    f32_f32: Math.fround,
    u32_i32: (value: number) => value | 0,
    i32_u32: (value: number) => value >>> 0,
    u32_f32: (value: number) => {
      u32[0] = value;
      return f32[0]!;
    },
    i32_f32: (value: number) => {
      i32[0] = value;
      return f32[0]!;
    },
    f32_u32: (value: number) => {
      f32[0] = value;
      return u32[0]!;
    },
    f32_i32: (value: number) => {
      f32[0] = value;
      return i32[0]!;
    },
  };
  const shared = new Map<string, unknown>();
  const instantiate = () =>
    makeKernel(
      (group, binding, ctor) => {
        const buffer = boundResources!.buffer(group, binding);
        return new ctor(buffer.bytes, buffer.offset, buffer.size / 4);
      },
      construct,
      ref,
      (name, make) => {
        if (!shared.has(name)) shared.set(name, make());
        return shared.get(name);
      },
      bits,
      vector,
      boundResources!,
      textureOps,
      pack,
      unpack,
    );
  return {
    kind: 'compute',
    workgroupSize: entry.size,
    bindings,
    entry: entry.name,
    dispatch(x: number, y: number, z: number, resources: CpuResources): void {
      for (const count of [x, y, z])
        if (!Number.isInteger(count) || count < 0)
          throw new Error('@typegpu/kit-gl CPU: invalid dispatch size');
      boundResources = resources;
      shared.clear();
      let execution!: Execution;
      try {
        if (reusable && heldExecution) {
          execution = heldExecution;
          execution.bind(resources);
        } else {
          execution = instantiate();
          if (reusable) heldExecution = execution;
        }
        const global = [0, 0, 0] as [number, number, number];
        const local = [0, 0, 0] as [number, number, number];
        const workgroup = [0, 0, 0] as [number, number, number];
        const builtins: Record<string, unknown> = {
          global_invocation_id: global,
          local_invocation_id: local,
          workgroup_id: workgroup,
          num_workgroups: vector([x, y, z]),
          local_invocation_index: 0,
        };
        const args: unknown[] = [];
        const phaseLanes = execution.phases
          ? Array.from({ length: entry.size[0] * entry.size[1] * entry.size[2] }, () => ({
              phases: execution.phases!(),
              inputs: entry.inputs.map((name) =>
                Array.isArray(builtins[name]) ? [0, 0, 0] : 0,
              ) as unknown[],
              privateValues: [] as unknown[],
            }))
          : undefined;
        for (workgroup[2] = 0; workgroup[2] < z; workgroup[2]++)
          for (workgroup[1] = 0; workgroup[1] < y; workgroup[1]++)
            for (workgroup[0] = 0; workgroup[0] < x; workgroup[0]++) {
              shared.clear();
              if (workgroupVariables.length) execution.resetWorkgroup();
              else if (workgroupState) execution = instantiate();
              const invocations: {
                iterator: Generator<unknown, void>;
                privateValues?: unknown[];
              }[] = [];
              let index = 0;
              let phaseFinished = 0;
              for (local[2] = 0; local[2] < entry.size[2]; local[2]++)
                for (local[1] = 0; local[1] < entry.size[1]; local[1]++)
                  for (local[0] = 0; local[0] < entry.size[0]; local[0]++) {
                    global[0] = workgroup[0] * entry.size[0] + local[0];
                    global[1] = workgroup[1] * entry.size[1] + local[1];
                    global[2] = workgroup[2] * entry.size[2] + local[2];
                    builtins.local_invocation_index = index++;
                    if (phaseLanes) {
                      const lane = phaseLanes[index - 1]!;
                      for (let i = 0; i < entry.inputs.length; i++) {
                        const value = builtins[entry.inputs[i]!];
                        if (Array.isArray(value)) {
                          const held = lane.inputs[i] as number[];
                          held[0] = value[0];
                          held[1] = value[1];
                          held[2] = value[2];
                        } else lane.inputs[i] = value;
                      }
                      if (lane.phases[0]!(...lane.inputs) === true) phaseFinished++;
                      if (privateNames.length) execution.capture(lane.privateValues);
                      continue;
                    }
                    for (let i = 0; i < entry.inputs.length; i++)
                      args[i] =
                        entry.generator && Array.isArray(builtins[entry.inputs[i]!])
                          ? vector([...(builtins[entry.inputs[i]!] as number[])])
                          : builtins[entry.inputs[i]!];
                    const invocation = execution.kernel(...args);
                    if (invocation) invocations.push({ iterator: invocation });
                  }
              if (phaseLanes) {
                for (let phase = 1; phase < entry.phases!; phase++) {
                  if (phaseFinished === phaseLanes.length) break;
                  if (phaseFinished)
                    throw new Error('@typegpu/kit-gl CPU: divergent workgroup barrier');
                  phaseFinished = 0;
                  for (const lane of phaseLanes) {
                    if (privateNames.length) execution.restore(lane.privateValues);
                    if (lane.phases[phase]!() === true) phaseFinished++;
                    if (privateNames.length) execution.capture(lane.privateValues);
                  }
                }
              }
              if (invocations.length) {
                for (;;) {
                  let finished = 0;
                  for (const invocation of invocations) {
                    if (invocation.privateValues) execution.restore(invocation.privateValues);
                    const state = invocation.iterator.next();
                    if (privateNames.length)
                      invocation.privateValues = execution.capture(invocation.privateValues);
                    if (state.done) finished++;
                  }
                  if (finished === invocations.length) break;
                  if (finished) throw new Error('@typegpu/kit-gl CPU: divergent workgroup barrier');
                }
              }
            }
      } finally {
        execution?.release();
        boundResources = undefined;
        shared.clear();
      }
    },
  };
}
