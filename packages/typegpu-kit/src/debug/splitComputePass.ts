type Call = [method: string, args: unknown[]];

function snapshot(args: unknown[]): unknown[] {
  return args.map((arg) => (arg instanceof Uint32Array || Array.isArray(arg) ? arg.slice() : arg));
}

function replay(pass: GPUComputePassEncoder, [method, args]: Call) {
  Reflect.apply(Reflect.get(pass, method), pass, args);
}

/**
 * Creates a compute-pass wrapper that records state and replays it into a separate pass per
 * dispatch. This gives each dispatch its own timestamps. Dispatches within a pass are ordered,
 * so splitting changes only timing. `begin` receives the debug groups open around the dispatch.
 * A pass without dispatches still begins one empty pass, so split and whole frames have the same
 * passes.
 */
export function splitComputePass(
  label: string,
  begin: (label: string, groups: readonly string[]) => GPUComputePassEncoder,
): GPUComputePassEncoder {
  const state = new Map<string, Call>();
  const debugGroups: string[] = [];
  let dispatched = false;

  const remember = (key: string, method: string, args: unknown[]) => {
    state.set(key, [method, snapshot(args)]);
  };

  const dispatch = (call: Call) => {
    dispatched = true;
    const pass = begin(label, debugGroups.slice());
    for (const group of debugGroups) pass.pushDebugGroup(group);
    for (const stateCall of state.values()) replay(pass, stateCall);

    replay(pass, call);

    for (const _ of debugGroups) pass.popDebugGroup();
    pass.end();
  };

  const overrides: Record<string, (...args: unknown[]) => void> = {
    setPipeline: (...args) => remember('setPipeline', 'setPipeline', args),
    pushDebugGroup: (group) => {
      debugGroups.push(group as string);
    },
    popDebugGroup: () => {
      debugGroups.pop();
    },
    insertDebugMarker: () => undefined,
    end: () => {
      if (!dispatched) begin(label, []).end();
    },
    dispatchWorkgroups: (...args) => dispatch(['dispatchWorkgroups', args]),
    dispatchWorkgroupsIndirect: (...args) => dispatch(['dispatchWorkgroupsIndirect', args]),
  };

  // State methods use their first argument as a slot key, for example `setBindGroup(0, …)`
  const prototype = GPUComputePassEncoder.prototype;
  const properties: PropertyDescriptorMap = {
    label: {
      get: () => label,
      set: (value: string) => {
        label = value;
      },
    },
  };

  for (const name of Object.getOwnPropertyNames(prototype)) {
    const method = Object.getOwnPropertyDescriptor(prototype, name)?.value;
    if (name === 'constructor' || typeof method !== 'function') continue;

    properties[name] = {
      value:
        overrides[name] ??
        ((...args: unknown[]) => remember(`${name}:${String(args[0])}`, name, args)),
    };
  }

  return Object.create(prototype, properties) as GPUComputePassEncoder;
}
