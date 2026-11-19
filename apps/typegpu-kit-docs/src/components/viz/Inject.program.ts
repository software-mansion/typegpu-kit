import { cameras, table, type Inject } from '@typegpu/kit';
import { recordingRoot } from '@typegpu/kit/testing';
import { d, tgpu, type TgpuComputeFn } from 'typegpu';
import { boundBy, onBindGroups } from './program.ts';

export const KINDS = [
  { value: 'member', label: 'Members' },
  { value: 'definition', label: 'Definition' },
  { value: 'hook', label: 'Hook' },
] as const;
export type Kind = (typeof KINDS)[number]['value'];

/** What is piped into the pipeline's root: both instances in either order, the first, or none. */
export const PIPES = ['ab', 'ba', 'a', 'none'] as const;
export type Pipes = (typeof PIPES)[number];

export interface Example {
  readonly names: readonly [string, string];
  /** What each instance is. */
  readonly what: string;
  /** The accessors the shader reads, by the instance that made them; one for both as a placeholder. */
  readonly accessors: readonly string[];
  readonly reads: string;
}

export interface Outcome {
  /** Whose resources the pipeline binds: 0 for the first instance, 1 for the second. */
  readonly bound: readonly number[];
  readonly error?: string;
}

export const examples: Record<Kind, Example> = {
  member: {
    names: ['near', 'far'],
    what: 'table of Block',
    accessors: ['near.gpu', 'far.gpu'],
    reads: 'near.gpu.shape(i) + far.gpu.shape(i)',
  },
  definition: {
    names: ['near', 'far'],
    what: 'table of Block',
    accessors: ['Block_host'],
    reads: 'Block.shape(i)',
  },
  hook: {
    names: ['player', 'map'],
    what: 'camera',
    accessors: ['cameras.current'],
    reads: 'cameras.current.$.viewProjection',
  },
};

/** The words for each of PIPES, given the two instances' names. */
export const piping = ([a, b]: readonly [string, string]): Record<Pipes, string> => ({
  ab: `${a}, then ${b}`,
  ba: `${b}, then ${a}`,
  a: `${a} only`,
  none: 'nothing',
});

/** Resolves every pairing of a kind and what is piped, up front. */
export function createProgram() {
  const { root } = recordingRoot();
  const device = root.device;

  let bound: object[] = [];
  const createCommandEncoder = device.createCommandEncoder.bind(device);
  device.createCommandEncoder = (descriptor) => {
    const encoder = createCommandEncoder(descriptor);
    const beginComputePass = encoder.beginComputePass.bind(encoder);
    encoder.beginComputePass = (passDescriptor) =>
      onBindGroups(beginComputePass(passDescriptor), (group) => {
        for (const { object } of boundBy(group)) bound.push(object);
      });
    return encoder;
  };

  const Block = table.define('Block', { host: { shape: d.u32 } });
  const near = table.create(root, Block, { capacity: 64 });
  const far = table.create(root, Block, { capacity: 64 });
  const size = { width: 1, height: 1 };
  const player = cameras.flat(root, { size, height: 2 });
  const map = cameras.flat(root, { size, height: 20 });

  const counts = root.createMutable(d.arrayOf(d.u32, 1));
  const values = root.createMutable(d.arrayOf(d.f32, 1));
  const input = { gid: d.builtin.globalInvocationId };

  const kernels: Record<Kind, TgpuComputeFn> = {
    member: tgpu.computeFn({ workgroupSize: [1], in: input })(({ gid }) => {
      'use gpu';
      counts.$[0] = near.gpu.shape(gid.x) + far.gpu.shape(gid.x);
    }),
    definition: tgpu.computeFn({ workgroupSize: [1], in: input })(({ gid }) => {
      'use gpu';
      counts.$[0] = Block.shape(gid.x);
    }),
    hook: tgpu.computeFn({ workgroupSize: [1], in: input })(() => {
      'use gpu';
      values.$[0] = cameras.current.$.viewProjection.columns[0].x;
    }),
  };

  const pairs: Record<Kind, readonly [{ inject(): Inject }, { inject(): Inject }]> = {
    member: [near, far],
    definition: [near, far],
    hook: [player, map],
  };

  const owners = new Map<object, number>([
    [root.unwrap(near.buffers.host!), 0],
    [root.unwrap(far.buffers.host!), 1],
    [root.unwrap(player.uniform.buffer), 0],
    [root.unwrap(map.uniform.buffer), 1],
  ]);

  const outcome = (kind: Kind, pipes: Pipes): Outcome => {
    const [first, second] = pairs[kind];
    const order = { ab: [first, second], ba: [second, first], a: [first], none: [] }[pipes];
    const cfg = order.reduce(
      (piped, instance) => piped.pipe(instance.inject()),
      root.pipe((unpiped) => unpiped),
    );

    bound = [];
    try {
      const pipeline = cfg.createComputePipeline({ compute: kernels[kind] });
      const encoder = root.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline);
      pass.dispatchWorkgroups(1);
      pass.end();
    } catch (error) {
      const message = String((error as Error).message);
      const unfilled = /Nothing filled [^:]*/.exec(message)?.[0];
      return { bound: [], error: unfilled?.replaceAll('`', '') ?? message };
    }

    const whose = bound.map((object) => owners.get(object)).filter((owner) => owner !== undefined);
    return { bound: [...new Set(whose)].toSorted() };
  };

  // Every pairing up front: each resolves a pipeline, and the figure only switches between them
  const outcomes = new Map<string, Outcome>();
  for (const { value: kind } of KINDS) {
    for (const pipes of PIPES) outcomes.set(`${kind} ${pipes}`, outcome(kind, pipes));
  }

  return {
    /** What the pipeline of `kind` binds with `pipes` piped into its root. */
    outcome: (kind: Kind, pipes: Pipes) => outcomes.get(`${kind} ${pipes}`)!,
  };
}
