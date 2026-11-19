// What the figures' programs read off the recording root they run on

/** The floats of a recorded buffer's bytes. Writing them writes the buffer. */
export const floatsOf = (bytes: Uint8Array) =>
  new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);

/**
 * What a bind group binds, each buffer, view or sampler with its layout's entry. The recording
 * device keeps the descriptors it is given, so a group holds its entries and its layout's.
 */
export function boundBy(group: GPUBindGroup | null) {
  const { layout, entries = [] } = (group ?? {}) as Partial<GPUBindGroupDescriptor>;
  const slots = Array.from(
    (layout as Partial<GPUBindGroupLayoutDescriptor> | undefined)?.entries ?? [],
  );
  return Array.from(entries, ({ binding, resource }) => ({
    object: 'buffer' in resource ? resource.buffer : resource,
    entry: slots.find((slot) => slot.binding === binding),
  }));
}

/** Calls `each` with every bind group `pass` is given, before the pass sets it. */
export function onBindGroups<P extends GPUBindingCommandsMixin>(
  pass: P,
  each: (group: GPUBindGroup | null) => void,
): P {
  const setBindGroup = pass.setBindGroup.bind(pass) as (...args: unknown[]) => void;
  (pass as { setBindGroup: unknown }).setBindGroup = (
    index: number,
    group: GPUBindGroup | null,
    ...rest: unknown[]
  ) => {
    each(group);
    setBindGroup(index, group, ...rest);
  };
  return pass;
}
