import {
  cameras,
  clock,
  frames,
  passes as described,
  readback,
  targets,
  particles,
  bloom,
  finish,
} from '@typegpu/kit';
import { recordingRoot } from '@typegpu/kit/testing';
import { createFlowField } from 'typegpu-kit-examples/flow-field/flow-field.ts';
import { boundBy, onBindGroups } from './program.ts';

export interface Event {
  /** A recording log event, `write` for a queue write, or `map` for a staging buffer's mapAsync. */
  readonly kind: string;
  /** The call that recorded it, such as `field.step`; `you` for the program's own calls. */
  readonly by: string;
}

/** A buffer or texture, with the module whose code created it. */
export interface Resource {
  readonly id: number;
  readonly label: string;
  readonly kind: 'buffer' | 'texture';
  /** `field`, `particles`, `readback`, `bloom`, `finish`, `camera`, `screen` or `you`. */
  readonly owner: string;
  /** Bound as a uniform and nothing else. */
  readonly uniform: boolean;
}

export type Access = 'read' | 'write';

/** A pass, a copy between passes, or the map after the submit, as the frame graph shows it. */
export interface Pass {
  readonly kind: 'compute' | 'render' | 'copy' | 'map';
  /** The call that began it; `you` for the program. */
  readonly by: string;
  /** The calls that recorded into it, in order, and how many commands each recorded. */
  readonly calls: { by: string; commands: number }[];
  /** What it reads and writes: bound resources by their layout, attachments as writes. */
  readonly uses: Map<Resource, Set<Access>>;
}

export interface Trace {
  readonly events: readonly Event[];
  readonly passes: readonly Pass[];
  readonly submitted: boolean;
  readonly index: number;
  readonly time: number;
  readonly dt: number;
}

export interface Include {
  field: boolean;
  particles: boolean;
  readback: boolean;
  /** Queues a burst before the frame, so `particles.spawn` has work. */
  burst: boolean;
  /** Throws at the end of the recorder. */
  fail: boolean;
}

const SIZE = { width: 640, height: 360 };
const RATE = 60;

/** `particles.step` is a call of `particles`. */
const moduleOf = (by: string) => by.split('.')[0];

/** The modules the frame graph turns on and off, and its buttons' words for them. */
export const MODULES = {
  field: 'Flow field',
  particles: 'Particles',
  readback: 'Readback',
  bloom: 'Bloom and finish',
} as const;
export type Module = keyof typeof MODULES;

export interface Options {
  /** Draws the scene into an HDR target, with bloom and the finish pass. Default false. */
  readonly bloomed?: boolean;
}

export function createProgram({ bloomed = false }: Options = {}) {
  const { root, log } = recordingRoot();
  const device = root.device;
  const canvasFormat = 'bgra8unorm';
  const sceneFormat = bloomed ? 'rgba16float' : canvasFormat;

  let current = 'you';
  let creating = 'you';
  let passes: Pass[] = [];

  const resources = new WeakMap<object, Resource>();
  let count = 0;
  const register = (object: object, label: string, kind: Resource['kind'], uniform: boolean) => {
    // TypeGPU creates a buffer when it is first used, so a call that uses one first owns it
    const owner = current === 'you' ? creating : moduleOf(current);
    const resource = { id: count++, label, kind, owner, uniform };
    resources.set(object, resource);
    return resource;
  };

  const use = (pass: Pass | undefined, object: unknown, access: Access) => {
    const resource = object ? resources.get(object as object) : undefined;
    if (!pass || !resource) return;
    const accesses = pass.uses.get(resource) ?? new Set();
    accesses.add(access);
    pass.uses.set(resource, accesses);
  };

  const begin = (kind: Pass['kind'], by = current) => {
    const pass: Pass = { kind, by, calls: [], uses: new Map() };
    passes.push(pass);
    return pass;
  };

  const command = (pass: Pass) => {
    const last = pass.calls.at(-1);
    if (last?.by === current) last.commands += 1;
    else pass.calls.push({ by: current, commands: 1 });
  };

  // Queue writes and staging maps join the log, so they show up in call order
  const writeBuffer = device.queue.writeBuffer.bind(device.queue);
  device.queue.writeBuffer = (...args: Parameters<GPUQueue['writeBuffer']>) => {
    log.events.push('write');
    writeBuffer(...args);
  };

  const createBuffer = device.createBuffer.bind(device);
  device.createBuffer = (descriptor) => {
    const buffer = createBuffer(descriptor);
    const { usage } = descriptor;
    const uniform = (usage & GPUBufferUsage.UNIFORM) !== 0 && !(usage & GPUBufferUsage.STORAGE);
    const resource = register(buffer, descriptor.label ?? '', 'buffer', uniform);

    if (usage & GPUBufferUsage.MAP_READ) {
      const mapAsync = buffer.mapAsync.bind(buffer);
      buffer.mapAsync = (...args) => {
        log.events.push('map');
        const map = begin('map', 'readback.capture');
        map.calls.push({ by: 'readback.capture', commands: 1 });
        map.uses.set(resource, new Set(['read']));
        return mapAsync(...args);
      };
    }
    return buffer;
  };

  const createTexture = device.createTexture.bind(device);
  device.createTexture = (descriptor) => {
    const texture = createTexture(descriptor);
    const resource = register(texture, descriptor.label ?? '', 'texture', false);

    const createView = texture.createView.bind(texture);
    texture.createView = (viewDescriptor) => {
      const view = createView(viewDescriptor);
      resources.set(view, resource);
      return view;
    };
    return texture;
  };

  // What each bind group reads and writes, from its layout's entries
  const commands = <P extends GPUComputePassEncoder | GPURenderPassEncoder>(raw: P, pass: Pass) =>
    onBindGroups(raw, (group) => {
      for (const { object, entry } of boundBy(group)) {
        use(pass, object, 'read');
        const writes =
          entry?.buffer?.type === 'storage' ||
          (entry?.storageTexture !== undefined && entry.storageTexture.access !== 'read-only');
        if (writes) use(pass, object, 'write');
      }
    });

  const counted =
    <A extends unknown[]>(pass: Pass, call: (...args: A) => void, read?: number) =>
    (...args: A) => {
      if (read !== undefined) use(pass, args[read], 'read');
      command(pass);
      call(...args);
    };

  const createCommandEncoder = device.createCommandEncoder.bind(device);
  device.createCommandEncoder = (descriptor) => {
    const encoder = createCommandEncoder(descriptor);

    const beginRenderPass = encoder.beginRenderPass.bind(encoder);
    encoder.beginRenderPass = (passDescriptor) => {
      const pass = begin('render');
      for (const attachment of passDescriptor.colorAttachments) {
        use(pass, attachment?.view, 'write');
        use(pass, attachment?.resolveTarget, 'write');
      }
      use(pass, passDescriptor.depthStencilAttachment?.view, 'write');

      const raw = beginRenderPass(passDescriptor);
      return {
        ...commands(raw, pass),
        setVertexBuffer(slot: number, buffer: GPUBuffer | null, ...rest: number[]) {
          use(pass, buffer, 'read');
          raw.setVertexBuffer(slot, buffer, ...rest);
        },
        setIndexBuffer(buffer: GPUBuffer, ...rest: [GPUIndexFormat, number?, number?]) {
          use(pass, buffer, 'read');
          raw.setIndexBuffer(buffer, ...rest);
        },
        draw: counted(pass, raw.draw),
        drawIndexed: counted(pass, raw.drawIndexed),
        drawIndirect: counted(pass, raw.drawIndirect, 0),
        drawIndexedIndirect: counted(pass, raw.drawIndexedIndirect, 0),
      } as GPURenderPassEncoder;
    };

    const beginComputePass = encoder.beginComputePass.bind(encoder);
    encoder.beginComputePass = (passDescriptor) => {
      const pass = begin('compute');
      const raw = beginComputePass(passDescriptor);
      return {
        ...commands(raw, pass),
        dispatchWorkgroups: counted(pass, raw.dispatchWorkgroups),
        dispatchWorkgroupsIndirect: counted(pass, raw.dispatchWorkgroupsIndirect, 0),
      } as GPUComputePassEncoder;
    };

    const copyBufferToBuffer = encoder.copyBufferToBuffer.bind(encoder) as (
      ...args: unknown[]
    ) => void;
    encoder.copyBufferToBuffer = ((...args: unknown[]) => {
      const pass = begin('copy');
      pass.calls.push({ by: current, commands: 1 });
      use(pass, args[0], 'read');
      use(pass, args[2], 'write');
      copyBufferToBuffer(...args);
    }) as GPUCommandEncoder['copyBufferToBuffer'];

    return encoder;
  };

  /** Attributes the module's allocations to it in the figure. */
  const attribute = <T>(owner: string, build: () => T) => {
    creating = owner;
    const built = build();
    creating = 'you';
    return built;
  };

  const sim = clock.fixedStep({ step: 1 / RATE });
  const camera = attribute('camera', () => cameras.flat(root, { size: SIZE, height: 2 }));

  const field = attribute('field', () =>
    createFlowField(root, { count: 4096, step: sim.step, format: sceneFormat, size: SIZE }),
  );
  // TypeGPU's encoder takes a texture where it takes a view
  const scene = described.create(root, {
    format: sceneFormat,
    ...SIZE,
    view: () => (screen?.textures.hdr ?? canvas) as never,
  });
  const { sparks, drawn } = attribute('particles', () => {
    const system = particles.create(root, camera, { capacity: 256, step: sim.step, dimensions: 2 });
    root.unwrap(system.pool); // the pool is the particles', though the readback uses it first
    return { sparks: system, drawn: scene.add(system.drawing) };
  });
  // At the display's rate, so every frame captures
  const probe = attribute('readback', () => readback.create(root, sparks.pool, { hz: RATE }));

  const canvas = root
    .createTexture({ size: [SIZE.width, SIZE.height], format: canvasFormat })
    .$usage('render');
  const screen = bloomed
    ? attribute('screen', () =>
        targets.create(root, SIZE, { hdr: { format: 'rgba16float', sampled: true } }),
      )
    : undefined;
  const source = () => screen!.textures.hdr;
  const glow = screen && attribute('bloom', () => bloom.create(root, { source, levels: 3 }));
  const final =
    glow && attribute('finish', () => finish.create(root, { format: canvasFormat, source, glow }));

  let include: Include = { field: true, particles: true, readback: true, burst: true, fail: false };
  const owners: string[] = [];
  let shown = { index: 0, time: 0, dt: 0 };

  // Events since the last claim were recorded by `by`
  const claim = (by: string) => {
    while (owners.length < log.events.length) owners.push(by);
  };
  const as = (by: string, record: () => void) => {
    claim('you');
    current = by;
    record();
    current = 'you';
    claim(by);
  };

  const loop = frames.create(root, (frame) => {
    claim('loop.run');
    shown = { index: frame.index, time: frame.time, dt: frame.dt };
    const { encoder, dt } = frame;

    camera.update(dt);
    claim('camera.update');

    const steps = sim.update(dt);
    if (include.field || include.particles) {
      const compute = encoder.beginComputePass();
      if (include.particles) as('particles.spawn', () => sparks.spawn(compute));

      for (let i = 0; i < steps; i++) {
        if (include.field) as('field.step', () => field.step(compute));
        if (include.particles) as('particles.step', () => sparks.step(compute));
      }

      if (include.particles) as('particles.compact', () => sparks.compact(compute, sim.ahead));
      compute.end();
    }

    if (include.readback) as('readback.capture', () => probe.capture(frame));

    const pass = scene.begin(frame);
    if (include.field) as('field.draw', () => field.draw(pass));
    if (include.particles) as('particles.draw', () => drawn.draw(pass));
    pass.end();

    if (glow && final) {
      as('bloom.record', () => glow.record(encoder));

      const last = encoder.beginRenderPass({ colorAttachments: { view: canvas } });
      as('finish.draw', () => final.draw(last));
      last.end();
    }

    claim('you');
    if (include.fail) throw new Error('The recorder threw.');
  });

  let seconds = 0;

  /** Records and submits the next frame, one display interval after the last. */
  const run = (next: Include): Trace => {
    include = next;
    log.events.length = 0;
    owners.length = 0;
    passes = [];

    if (include.burst) sparks.burst({ count: 64, position: [0, 0, 0], scatter: 0.5 });

    let submitted = false;
    try {
      // A hair over 1/60 s, so the fixed step takes exactly one step a frame
      submitted = loop.run((seconds += 1 / RATE + 1e-7));
    } catch {
      seconds -= 1 / RATE + 1e-7;
    }

    claim('loop.run');
    const events = log.events.map((kind, i) => ({
      kind,
      by: kind === 'map' ? 'readback.capture' : owners[i],
    }));

    return { events, passes, submitted, ...shown };
  };

  // The first frame compiles every pipeline and takes no step, so the figure starts on the second
  run({ ...include, burst: false });

  return { run };
}

export type Program = ReturnType<typeof createProgram>;
