import { watchFrames } from '../animate.ts';
import {
  type GpuResourceUse,
  meaningfulLabel,
  pipelineName,
  pipelineWgsl,
  trackMetadata,
  usesOf,
  usesOfAttachments,
  usesOfBindGroup,
} from './metadata.ts';
import { patches, type Patch } from './patch.ts';
import { splitComputePass } from './splitComputePass.ts';

export interface GpuCommand {
  kind: 'compute' | 'render' | 'copy' | 'upload';
  name: string;
  label: string | undefined;
  /** Names the compute pass a split dispatch came from, by its label. */
  group: string | undefined;
  /** Lists the debug groups open around it, outermost first, on its encoder and its pass. */
  groups: readonly string[];
  /**
   * Identifies the pass it records, shared by the dispatches of one split compute pass.
   * Undefined for copies and uploads.
   */
  pass: number | undefined;
  /** Indicates a dispatch timed in its own pass, split from a compute pass. */
  split: boolean;
  pipelines: string[];
  /** Gives the WGSL of the shader modules its pipelines use, each once. */
  wgsl: string[];
  calls: number;
  uses: GpuResourceUse[];
  frame: number;
  start?: bigint;
  end?: bigint;
  /**
   * Indicates whether timestamps were written, dropped because all query sets awaited readback,
   * or left to the pass's own `timestampWrites`, which leave it untimed. It is unset on devices
   * without timestamp queries.
   */
  timestamps?: 'written' | 'dropped' | 'own';
}

/** These hooks receive instrumented WebGPU events. The recorder implements them. */
export interface InstrumentHooks {
  /** Reports whether the current frame is being captured. */
  capturing(): boolean;
  /** Asked as each compute pass begins, so a recorder can split only some frames. */
  splitComputePasses(): boolean;
  /** Notes a requested device and whether it has `timestamp-query`. */
  deviceRequested(timestampQuery: boolean): void;
  /** Starts a frame at `timestamp`, in milliseconds. */
  frameStarted(timestamp: number): void;
  /** Adds `ms` of CPU time under `name` to the current frame. */
  callbackTimed(name: string, ms: number): void;
  /** Reports submitted commands before their timestamps arrive. */
  submitted(commands: GpuCommand[]): void;
  /**
   * Reports submitted commands after filling their `start` and `end` times, or without them when
   * reading the timestamps failed.
   */
  timed(commands: GpuCommand[]): void;
}

export interface InstrumentOptions {
  /**
   * Selects what starts a frame. `'animation-frame'` wraps `requestAnimationFrame` and video frame
   * callbacks. `'animate'` wraps the frames that the kit's `animate` runs, as on a React Native
   * worker, where the host paces frames. Default `'animation-frame'` where `window` exists,
   * otherwise `'animate'`.
   */
  readonly frames?: 'animation-frame' | 'animate';
  /**
   * Patches these interface objects besides WebGPU's own, for a device whose objects are not
   * WebGPU's, such as the WebGL fallback's `webgl.interfaces`. Default none.
   */
  readonly interfaces?: Interfaces;
}

/** The interface objects whose prototypes the instrumentation patches, by WebGPU's names. */
export interface Interfaces {
  readonly GPUDevice: { readonly prototype: GPUDevice };
  readonly GPUQueue: { readonly prototype: GPUQueue };
  readonly GPUCommandEncoder: { readonly prototype: GPUCommandEncoder };
  readonly GPUComputePassEncoder: { readonly prototype: GPUComputePassEncoder };
  readonly GPURenderPassEncoder: { readonly prototype: GPURenderPassEncoder };
  readonly GPUTexture: { readonly prototype: GPUTexture };
}

export interface DriveOptions {
  /** Sets the milliseconds between the timestamps that driven frames receive. */
  readonly interval: number;
  /** Runs frames for this many milliseconds before reporting them, while the GPU's clock rises. */
  readonly warmup: number;
  /** Runs frames for this many milliseconds after `warmup`, reporting them to the hooks. */
  readonly duration: number;
}

export interface Instrumentation {
  /**
   * Starts reporting to `hooks`. Encoders created after this call record commands and time
   * passes with these hooks.
   */
  start(hooks: InstrumentHooks): void;
  /**
   * Stops recording new encoders. Passes already begun keep their timestamps and report to their
   * original hooks after submission. Stops a drive, which rejects.
   */
  stop(): void;
  /**
   * Runs the page's frames back to back, unpaced by the display, and reports them to `hooks` in
   * place of the started hooks. Frames run in bursts with timestamps `interval` apart, each burst
   * queued on the GPU while the next records, and draw into a texture in place of the canvas's.
   * On the web, the page's animation frame callbacks run in bursts; with `frames: 'animate'`, the
   * kit's `animate` frames do. Requires `start`.
   *
   * @returns milliseconds per frame from the GPU finishing the reported bursts. Rejects if no
   * frame comes within two seconds, a frame throws or profiling stops
   */
  drive(hooks: InstrumentHooks, options: DriveOptions): Promise<number>;
  /** Stops and removes the patches. */
  [Symbol.dispose](): void;
}

type PassEncoder = GPUComputePassEncoder | GPURenderPassEncoder;
type PassDescriptor = GPUComputePassDescriptor | GPURenderPassDescriptor;
type TimestampSlot = {
  device: GPUDevice;
  hooks: InstrumentHooks;
  querySet: GPUQuerySet;
  resolve: GPUBuffer;
  readback: GPUBuffer;
  passes: GpuCommand[];
  batches: TimestampBatch[];
  closed: boolean;
  /** Counts the slot's reads, so a batch from before the last one is left alone. */
  generation: number;
};
/** The passes of one encoder in a slot, in order, and how far that encoder got. */
type TimestampBatch = {
  slot: TimestampSlot;
  generation: number;
  first: number;
  passes: GpuCommand[];
  state: 'recording' | 'finished' | 'submitted';
};
type EncoderState = {
  device: GPUDevice;
  hooks: InstrumentHooks;
  commands: GpuCommand[];
  batches: TimestampBatch[];
  /** Debug groups open on the encoder, outermost first. */
  groups: string[];
};
type Driving = {
  readonly hooks: InstrumentHooks;
  reject: (error: unknown) => void;
  live: InstrumentHooks | undefined;
  /** The latest frame `animate` offered, which the drive runs at its own timestamps. */
  run: ((milliseconds: number) => void) | undefined;
  /** The display's latest timestamp, where driven timestamps start. */
  now: number | undefined;
};
type Held = { id: number; callback: FrameRequestCallback };

/** Frames a burst can grow to while driving. */
const largestBurst = 64;
/** Milliseconds of GPU work a burst aims for, so one is queued while the next records. */
const burstTarget = 8;

const ignore = () => undefined;

/** Reports nothing, for frames that warm the GPU up before a drive reports. */
const inert: InstrumentHooks = {
  capturing: () => false,
  splitComputePasses: () => false,
  deviceRequested() {},
  frameStarted() {},
  callbackTimed() {},
  submitted() {},
  timed() {},
};

const passesPerSlot = 256;
const bytesPerPass = 16;
/** Reserve query sets for the app within the Metal limit of 32 per device. */
const slotsPerDevice = 16;

/** Calls the original `requestAnimationFrame` so panel frames do not enter recordings. */
export const requestUntrackedFrame = globalThis.requestAnimationFrame?.bind(globalThis);

const encoders = new WeakMap<GPUCommandEncoder, EncoderState>();
const commandBuffers = new WeakMap<GPUCommandBuffer, EncoderState>();
const passes = new WeakMap<PassEncoder, GpuCommand>();
const slotPools = new WeakMap<GPUDevice, TimestampSlot[]>();
const slotsOf = new WeakMap<GPUDevice, TimestampSlot[]>();
const openSlots = new WeakMap<GPUDevice, TimestampSlot>();

/** Numbers passes so the dispatches split from one pass share its number. */
let passCount = 0;

function command(
  kind: GpuCommand['kind'],
  name: string,
  label?: string,
  group?: string,
): GpuCommand {
  return {
    kind,
    name,
    label: meaningfulLabel(label),
    group: meaningfulLabel(group),
    groups: [],
    pass: undefined,
    split: false,
    pipelines: [],
    wgsl: [],
    calls: 0,
    uses: [],
    frame: 0,
  };
}

function timeable(state: EncoderState, descriptor: PassDescriptor | undefined): boolean {
  return !descriptor?.timestampWrites && state.device.features.has('timestamp-query');
}

function acquireSlot({ device, hooks }: EncoderState): TimestampSlot | undefined {
  const pool = slotPools.get(device) ?? [];
  slotPools.set(device, pool);

  const free = pool.pop();
  if (free) return Object.assign(free, { hooks });

  const slots = slotsOf.get(device) ?? [];
  slotsOf.set(device, slots);
  if (slots.length === slotsPerDevice) return undefined;

  const size = passesPerSlot * bytesPerPass;
  const slot: TimestampSlot = {
    device,
    hooks,
    querySet: device.createQuerySet({ type: 'timestamp', count: passesPerSlot * 2 }),
    resolve: device.createBuffer({
      size,
      usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
    }),
    readback: device.createBuffer({
      size,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    }),
    passes: [],
    batches: [],
    closed: false,
    generation: 0,
  };
  slots.push(slot);
  return slot;
}

/**
 * Reads the slot's timestamps and gives it back to the pool, also when the read fails. It reports
 * the passes of submitted encoders, without times if the read failed. The passes of encoders not
 * yet submitted are dropped, as they can no longer resolve their timestamps.
 */
function readTimestamps(slot: TimestampSlot) {
  const timed = slot.passes;
  const size = timed.length * bytesPerPass;
  const reported: GpuCommand[] = [];
  for (const batch of slot.batches) {
    if (batch.state === 'submitted') reported.push(...batch.passes);
    else for (const pass of batch.passes) pass.timestamps = 'dropped';
  }
  slot.batches = [];
  slot.generation++;

  const release = () => {
    Object.assign(slot, { passes: [], closed: false });
    slotPools.get(slot.device)?.push(slot);
  };
  if (reported.length === 0) return release();

  const report = () => {
    release();
    slot.hooks.timed(reported);
  };
  slot.readback.mapAsync(GPUMapMode.READ, 0, size).then(() => {
    const stamps = new BigUint64Array(slot.readback.getMappedRange(0, size));
    timed.forEach((pass, index) => {
      pass.start = stamps[index * 2];
      pass.end = stamps[index * 2 + 1];
    });
    slot.readback.unmap();
    report();
  }, report);
}

function settle(slot: TimestampSlot) {
  const { closed, batches } = slot;
  if (closed && batches.length > 0 && batches.every(({ state }) => state === 'submitted')) {
    readTimestamps(slot);
  }
}

/**
 * Reads the closed slots whose unsubmitted encoders were all left unfinished, so they cannot hold
 * every slot. An encoder finished but not yet submitted may still be, so it holds its slot.
 */
function retireAbandoned(device: GPUDevice) {
  for (const slot of slotsOf.get(device) ?? []) {
    const { closed, batches } = slot;
    if (!closed || !batches.some(({ state }) => state === 'recording')) continue;
    if (batches.some(({ state }) => state === 'finished')) continue;
    readTimestamps(slot);
  }
}

function closeSlot(slot: TimestampSlot) {
  if (slot.closed) return;

  if (openSlots.get(slot.device) === slot) openSlots.delete(slot.device);
  slot.closed = true;
  settle(slot);
}

/**
 * Share a query set across encoders created in one task so a frame can submit multiple encoders
 * without consuming a set per encoder.
 */
function openSlot(state: EncoderState): TimestampSlot | undefined {
  const open = openSlots.get(state.device);
  if (open?.hooks === state.hooks) return open;
  // Keep each query slot tied to one profiling session, even if profiling restarts in the same
  // task
  if (open) closeSlot(open);

  const slot = acquireSlot(state);
  if (!slot) {
    retireAbandoned(state.device);
    return undefined;
  }

  openSlots.set(state.device, slot);
  queueMicrotask(() => closeSlot(slot));
  return slot;
}

function withTimestamps<D extends PassDescriptor>(
  state: EncoderState,
  pass: GpuCommand,
  descriptor: D,
): D {
  if (descriptor.timestampWrites) pass.timestamps = 'own';
  if (!timeable(state, descriptor)) return descriptor;

  const slot = openSlot(state);

  if (!slot) {
    pass.timestamps = 'dropped';
    return descriptor;
  }

  // Record interleaved runs as separate batches so each encoder resolves only its own passes
  const index = slot.passes.push(pass) - 1;
  let batch = state.batches.at(-1);
  if (batch?.slot !== slot || batch.first + batch.passes.length !== index) {
    const { generation } = slot;
    batch = { slot, generation, first: index, passes: [], state: 'recording' };
    state.batches.push(batch);
    slot.batches.push(batch);
  }

  batch.passes.push(pass);
  pass.timestamps = 'written';
  if (slot.passes.length === passesPerSlot) closeSlot(slot);

  return {
    ...descriptor,
    timestampWrites: {
      querySet: slot.querySet,
      beginningOfPassWriteIndex: index * 2,
      endOfPassWriteIndex: index * 2 + 1,
    },
  };
}

function trackPass<P extends PassEncoder>(state: EncoderState, entry: GpuCommand, pass: P): P {
  state.commands.push(entry);
  passes.set(pass, entry);
  return pass;
}

function beginComputePass(
  state: EncoderState,
  descriptor: GPUComputePassDescriptor,
  next: (descriptor: GPUComputePassDescriptor) => GPUComputePassEncoder,
) {
  const entry = command('compute', 'compute pass', descriptor.label);
  entry.groups = state.groups.slice();
  entry.pass = ++passCount;
  return trackPass(state, entry, next(withTimestamps(state, entry, descriptor)));
}

function beginSplitPass(
  state: EncoderState,
  descriptor: GPUComputePassDescriptor,
  next: (descriptor: GPUComputePassDescriptor) => GPUComputePassEncoder,
  prototype: GPUComputePassEncoder,
) {
  const pass = ++passCount;
  const outer = state.groups.slice();
  return splitComputePass(
    descriptor.label ?? '',
    (label, groups) => {
      const entry = command('compute', 'compute pass', undefined, label);
      entry.groups = [...outer, ...groups];
      entry.pass = pass;
      entry.split = true;
      return trackPass(state, entry, next(withTimestamps(state, entry, {})));
    },
    prototype,
  );
}

function countCall(pass: PassEncoder) {
  const entry = passes.get(pass);
  if (entry) entry.calls++;
}

function addPipeline(
  pass: PassEncoder,
  pipeline: GPUComputePipeline | GPURenderPipeline,
  capturing: boolean | undefined,
) {
  const entry = passes.get(pass);
  if (!entry) return;
  const name = pipelineName(pipeline);
  if (!entry.pipelines.includes(name)) entry.pipelines.push(name);
  // Only a capture shows the WGSL
  if (!capturing) return;
  for (const code of pipelineWgsl(pipeline)) {
    if (!entry.wgsl.includes(code)) entry.wgsl.push(code);
  }
}

/** The encoder's state while its frame is captured, which records copies. */
function captured(encoder: GPUCommandEncoder): EncoderState | undefined {
  const state = encoders.get(encoder);
  return state?.hooks.capturing() ? state : undefined;
}

const copy = (name: string, ...uses: GpuResourceUse[][]): GpuCommand => ({
  ...command('copy', name),
  uses: uses.flat(),
});

// Track installation globally because WebGPU prototypes are global
let instrumented = false;
let untracked = 0;

/**
 * Runs `fn` without recording the encoders it creates or the uploads it makes, so a panel drawing
 * with WebGPU on the profiled runtime stays out of its own measurements.
 */
export function withoutTracking<T>(fn: () => T): T {
  untracked++;
  try {
    return fn();
  } finally {
    untracked--;
  }
}

/**
 * Installs WebGPU prototype patches without starting recording. Call this before requesting a
 * device. It requests `timestamp-query` and tracks pipeline names, bind groups, views and shader
 * modules even while stopped. `finish` and `submit` keep reporting passes timed before a stop.
 * Other patched calls run their original method when recording is stopped.
 *
 * After `start`, it times passes, derives names from pipelines and reports frames, submissions
 * and timings to hooks. Without WebGPU, it installs no patches and reports nothing. Only one
 * instrumentation can be installed per runtime at a time; dispose of it before installing another.
 * On a React Native worker, install it on that runtime: each runtime has its own WebGPU
 * prototypes.
 *
 * @returns patches you start with hooks, and dispose of to remove
 */
export function instrument(options: InstrumentOptions = {}): Instrumentation {
  const sets: Interfaces[] = [];
  if (typeof GPUCommandEncoder !== 'undefined') sets.push(globalThis);
  if (options.interfaces) sets.push(options.interfaces);
  if (sets.length === 0) {
    return {
      start() {},
      stop() {},
      drive: () => Promise.reject(new Error('WebGPU is unavailable.')),
      [Symbol.dispose]() {},
    };
  }
  if (instrumented) {
    throw new Error(
      'Dispose of the existing WebGPU instrumentation before installing another instance.',
    );
  }
  instrumented = true;
  const frames = options.frames ?? (typeof window === 'undefined' ? 'animate' : 'animation-frame');
  let disposed = false;
  const { patch, restore } = patches();

  let hooks: InstrumentHooks | undefined;
  let timestampQuery: boolean | undefined;
  const active = () => hooks !== undefined;
  const whileActive: Patch = (target, key, handler) => patch(target, key, handler, active);

  // The device of the latest tracked encoder, whose queue tells when a driven burst finishes
  let lastDevice: GPUDevice | undefined;
  let driving: Driving | undefined;
  // While frames are driven, the animation frames the page requests wait here to run in a burst
  const held: Held[] = [];
  // Held requests released to the browser, by the id the page holds
  const aliases = new Map<number, number>();
  let heldCount = 0;
  let requestFrame: ((callback: FrameRequestCallback) => number) | undefined;
  // The timestamp of the latest animation frame
  let latest: number | undefined;

  function frameCallback<A extends unknown[]>(
    callback: (now: number, ...rest: A) => void,
    fallbackName: string,
  ) {
    return (now: number, ...rest: A) => {
      latest = now;
      const current = hooks;
      current?.frameStarted(now);
      const start = performance.now();
      callback(now, ...rest);
      current?.callbackTimed(callback.name || fallbackName, performance.now() - start);
    };
  }

  function runFrame(run: (now: number) => void, name: string, now: number) {
    const current = hooks;
    current?.frameStarted(now);
    const start = performance.now();
    try {
      run(now);
    } finally {
      current?.callbackTimed(name, performance.now() - start);
    }
  }

  function releaseHeld() {
    for (const { id, callback } of held.splice(0)) {
      const live = frameCallback(callback, 'animation frame');
      aliases.set(
        id,
        requestFrame!((now) => {
          aliases.delete(id);
          live(now);
        }),
      );
    }
  }

  function endDrive(drive: Driving, error?: unknown) {
    if (driving !== drive) return;
    driving = undefined;
    hooks = drive.live;
    if (requestFrame) releaseHeld();
    for (const texture of standIns.values()) texture.destroy();
    standIns.clear();
    if (error !== undefined) drive.reject(error);
  }

  const nextFrame = () =>
    new Promise<void>((resolve) => {
      if (requestFrame) requestFrame(() => resolve());
      else setTimeout(resolve, 4);
    });

  /**
   * Runs bursts of frames, each `interval` after the last in the timestamps they get, keeping one
   * burst queued on the GPU while it records the next. `frame` runs one at the timestamp it gets
   * and returns true, or returns false with nothing to run. A burst aims for `burstTarget`
   * milliseconds of the GPU's time per frame so far.
   *
   * @returns milliseconds per frame from the GPU finishing the reported bursts
   */
  async function runDrive(
    drive: Driving,
    { interval, warmup, duration }: DriveOptions,
    frame: (at: number) => boolean,
  ): Promise<number> {
    const timeout = setTimeout(() => {
      endDrive(
        drive,
        new Error('Profiling timed out after two seconds while waiting for an animation frame.'),
      );
    }, 2000);
    let started: number | undefined;
    let time: number | undefined;
    let size = 2;
    let queued: Promise<number> | undefined;
    let queuedFrames = 0;
    let finished: number | undefined;
    let first: number | undefined;
    let last: number | undefined;
    let counted = 0;

    try {
      for (;;) {
        if (driving !== drive) break;
        const clock = performance.now();
        if (started !== undefined && clock >= started + warmup + duration) break;
        time ??= drive.now;
        if (time === undefined) {
          await nextFrame();
          continue;
        }

        const reported = started !== undefined && clock >= started + warmup;
        hooks = reported ? drive.hooks : inert;
        let count = 0;
        let at = time + interval;
        while (count < size && frame(at)) {
          time = at;
          at += interval;
          count++;
        }
        if (count === 0) {
          await nextFrame();
          continue;
        }
        if (started === undefined) {
          started = clock;
          clearTimeout(timeout);
        }

        // Wait for the burst before this one, so this one stays queued while the next records
        const done = (lastDevice?.queue.onSubmittedWorkDone() ?? Promise.resolve()).then(
          () => performance.now(),
          () => performance.now(),
        );
        const before = queued;
        const beforeFrames = queuedFrames;
        queued = done;
        queuedFrames = reported ? count : 0;
        if (!before) continue;

        const end = await before;
        // Bursts finish in order, so each one's end after another's times it
        if (finished !== undefined && beforeFrames > 0) {
          first ??= finished;
          last = end;
          counted += beforeFrames;
          const perFrame = (end - finished) / beforeFrames;
          size = Math.round(Math.min(Math.max(burstTarget / perFrame, 1), largestBurst));
        }
        finished = end;
      }

      if (queued) {
        const end = await queued;
        if (finished !== undefined && queuedFrames > 0) {
          first ??= finished;
          last = end;
          counted += queuedFrames;
        }
      }
    } finally {
      clearTimeout(timeout);
      endDrive(drive);
    }
    return first !== undefined && last !== undefined && counted > 0
      ? (last - first) / counted
      : Number.NaN;
  }

  /** Runs every held animation frame callback at `at`, as one animation frame does. */
  function runHeld(at: number): boolean {
    const callbacks = held.splice(0);
    for (const { callback } of callbacks)
      runFrame(callback, callback.name || 'animation frame', at);
    return callbacks.length > 0;
  }

  function knowTimestamps(device: GPUDevice) {
    if (timestampQuery !== undefined) return;
    timestampQuery = device.features.has('timestamp-query');
    hooks?.deviceRequested(timestampQuery);
  }

  /** The pass's command while its frame is captured, which records what the pass uses. */
  function capturedPass(pass: PassEncoder): GpuCommand | undefined {
    return hooks?.capturing() ? passes.get(pass) : undefined;
  }

  function recordUpload(name: string, target: GPUBuffer | GPUTexture) {
    if (untracked > 0 || !hooks?.capturing()) return;
    hooks.submitted([{ ...command('upload', name), uses: usesOf(target, 'write') }]);
  }

  function tracked(encoder: GPUCommandEncoder): EncoderState | undefined {
    const state = encoders.get(encoder);
    return state?.hooks === hooks ? state : undefined;
  }

  for (const set of sets) trackMetadata(patch, set);

  // Driven frames draw into a texture of the canvas's size and configuration in place of the
  // canvas's own, so the browser doesn't hand the canvas's texture between the GPU and the screen
  // for each of them. The screen shows the last frame before the drive until it ends
  const standIns = new Map<GPUCanvasContext, GPUTexture>();
  if (typeof GPUCanvasContext !== 'undefined') {
    patch(
      GPUCanvasContext.prototype,
      'getCurrentTexture',
      (self, args, next) => {
        const configuration = self.getConfiguration?.();
        if (!configuration) return next(...args);
        const { width, height } = self.canvas;
        let texture = standIns.get(self);
        if (!texture || texture.width !== width || texture.height !== height) {
          texture?.destroy();
          texture = configuration.device.createTexture({
            label: 'driven canvas',
            size: [width, height],
            format: configuration.format,
            usage: configuration.usage ?? GPUTextureUsage.RENDER_ATTACHMENT,
            viewFormats: configuration.viewFormats ?? [],
          });
          standIns.set(self, texture);
        }
        return texture;
      },
      () => driving !== undefined,
    );
  }

  // On React Native the JS thread requests the device, so encoders report its features instead
  if (typeof GPUAdapter !== 'undefined') {
    patch(GPUAdapter.prototype, 'requestDevice', (self, [descriptor], next) => {
      timestampQuery = self.features.has('timestamp-query');
      const features = new Set(descriptor?.requiredFeatures);
      if (timestampQuery) features.add('timestamp-query');

      hooks?.deviceRequested(timestampQuery);
      return next({ ...descriptor, requiredFeatures: [...features] });
    });
  }

  if (frames === 'animate') {
    watchFrames((milliseconds, run) => {
      if (!hooks) return run(milliseconds);
      // A drive runs the frames after its first at its own timestamps
      if (driving?.now !== undefined) {
        driving.run = run;
        return;
      }
      runFrame(run, 'frame', milliseconds);
      if (driving) {
        driving.run = run;
        driving.now = milliseconds;
      }
    });
  } else if (typeof window !== 'undefined') {
    requestFrame = window.requestAnimationFrame.bind(window);
    whileActive(window, 'requestAnimationFrame', (_self, [callback], next) => {
      if (!driving) return next(frameCallback(callback, 'animation frame'));
      // Driven frames follow the frame that requested this one
      driving.now ??= latest ?? performance.now();
      const id = -++heldCount;
      held.push({ id, callback });
      return id;
    });
    // Held ids are negative, apart from the browser's
    patch(
      window,
      'cancelAnimationFrame',
      (_self, [id], next) => {
        if (typeof id !== 'number' || id >= 0) return next(id);
        const index = held.findIndex((each) => each.id === id);
        if (index >= 0) held.splice(index, 1);
        const real = aliases.get(id);
        aliases.delete(id);
        if (real !== undefined) next(real);
      },
      () => held.length > 0 || aliases.size > 0,
    );
  }

  if (frames === 'animation-frame' && typeof HTMLVideoElement !== 'undefined') {
    whileActive(
      HTMLVideoElement.prototype,
      'requestVideoFrameCallback',
      (_self, [callback], next) => next(frameCallback(callback, 'video frame')),
    );
  }

  /** Patches one set of interface objects, which shadow WebGPU's own names in here. */
  function patchInterfaces({
    GPUDevice,
    GPUQueue,
    GPUCommandEncoder,
    GPUComputePassEncoder,
    GPURenderPassEncoder,
  }: Interfaces) {
    // Use the original encoder method to keep timestamp copies out of recorded commands
    const copyBufferToBuffer = GPUCommandEncoder.prototype.copyBufferToBuffer as (
      source: GPUBuffer,
      sourceOffset: number,
      destination: GPUBuffer,
      destinationOffset: number,
      size: number,
    ) => undefined;

    whileActive(GPUDevice.prototype, 'createCommandEncoder', (self, args, next) => {
      const encoder = next(...args);
      knowTimestamps(self);
      if (hooks && untracked === 0) {
        encoders.set(encoder, { device: self, hooks, commands: [], batches: [], groups: [] });
        lastDevice = self;
      }
      return encoder;
    });

    whileActive(
      GPUCommandEncoder.prototype,
      'beginComputePass',
      (self, [descriptor = {}], next) => {
        const state = tracked(self);
        if (!state) return next(descriptor);

        if (state.hooks.splitComputePasses() && timeable(state, descriptor)) {
          return beginSplitPass(state, descriptor, next, GPUComputePassEncoder.prototype);
        }

        return beginComputePass(state, descriptor, next);
      },
    );

    whileActive(GPUCommandEncoder.prototype, 'beginRenderPass', (self, [descriptor], next) => {
      const state = tracked(self);
      if (!state) return next(descriptor);

      const entry = command('render', 'render pass', descriptor.label);
      entry.groups = state.groups.slice();
      entry.pass = ++passCount;
      if (state.hooks.capturing()) entry.uses = usesOfAttachments(descriptor);

      return trackPass(state, entry, next(withTimestamps(state, entry, descriptor)));
    });

    whileActive(GPUCommandEncoder.prototype, 'pushDebugGroup', (self, [group], next) => {
      tracked(self)?.groups.push(group);
      return next(group);
    });

    whileActive(GPUCommandEncoder.prototype, 'popDebugGroup', (self, args, next) => {
      tracked(self)?.groups.pop();
      return next(...args);
    });

    // Optional chaining skips building the copy's command unless its frame is captured
    whileActive(GPUCommandEncoder.prototype, 'copyBufferToBuffer', (self, args, next) => {
      captured(self)?.commands.push(
        copy(
          'copyBufferToBuffer',
          usesOf(args[0], 'read'),
          usesOf((typeof args[1] === 'number' ? args[2] : args[1]) as GPUBuffer, 'write'),
        ),
      );
      return next(...args);
    });

    whileActive(GPUCommandEncoder.prototype, 'copyBufferToTexture', (self, args, next) => {
      captured(self)?.commands.push(
        copy(
          'copyBufferToTexture',
          usesOf(args[0].buffer, 'read'),
          usesOf(args[1].texture, 'write'),
        ),
      );
      return next(...args);
    });

    whileActive(GPUCommandEncoder.prototype, 'copyTextureToBuffer', (self, args, next) => {
      captured(self)?.commands.push(
        copy(
          'copyTextureToBuffer',
          usesOf(args[0].texture, 'read'),
          usesOf(args[1].buffer, 'write'),
        ),
      );
      return next(...args);
    });

    whileActive(GPUCommandEncoder.prototype, 'copyTextureToTexture', (self, args, next) => {
      captured(self)?.commands.push(
        copy(
          'copyTextureToTexture',
          usesOf(args[0].texture, 'read'),
          usesOf(args[1].texture, 'write'),
        ),
      );
      return next(...args);
    });

    whileActive(GPUCommandEncoder.prototype, 'clearBuffer', (self, args, next) => {
      captured(self)?.commands.push(copy('clearBuffer', usesOf(args[0], 'write')));
      return next(...args);
    });

    // Resolve passes timed before a stop even while profiling is stopped, so submission releases
    // their query sets
    patch(GPUCommandEncoder.prototype, 'finish', (self, args, next) => {
      const state = encoders.get(self);
      // Resolve at offset 0 to meet the 256-byte alignment requirement, then copy each batch to
      // its destination
      for (const batch of state?.batches ?? []) {
        const { slot, first, passes: timed } = batch;
        if (batch.generation !== slot.generation) continue;
        batch.state = 'finished';
        self.resolveQuerySet(slot.querySet, first * 2, timed.length * 2, slot.resolve, 0);
        copyBufferToBuffer.call(
          self,
          slot.resolve,
          0,
          slot.readback,
          first * bytesPerPass,
          timed.length * bytesPerPass,
        );
      }

      const buffer = next(...args);
      if (state) commandBuffers.set(buffer, state);
      return buffer;
    });

    whileActive(GPUComputePassEncoder.prototype, 'setPipeline', (self, [pipeline], next) => {
      addPipeline(self, pipeline, hooks?.capturing());
      return next(pipeline);
    });

    whileActive(GPURenderPassEncoder.prototype, 'setPipeline', (self, [pipeline], next) => {
      addPipeline(self, pipeline, hooks?.capturing());
      return next(pipeline);
    });

    for (const Pass of [GPUComputePassEncoder, GPURenderPassEncoder] as const) {
      whileActive(Pass.prototype as PassEncoder, 'setBindGroup', (self, args, next) => {
        capturedPass(self)?.uses.push(...usesOfBindGroup(args[1]));
        return next(...args);
      });
    }

    whileActive(GPUComputePassEncoder.prototype, 'dispatchWorkgroups', (self, args, next) => {
      countCall(self);
      return next(...args);
    });

    whileActive(
      GPUComputePassEncoder.prototype,
      'dispatchWorkgroupsIndirect',
      (self, args, next) => {
        countCall(self);
        capturedPass(self)?.uses.push(...usesOf(args[0], 'read'));
        return next(...args);
      },
    );

    whileActive(GPURenderPassEncoder.prototype, 'setVertexBuffer', (self, args, next) => {
      capturedPass(self)?.uses.push(...usesOf(args[1], 'read'));
      return next(...args);
    });

    whileActive(GPURenderPassEncoder.prototype, 'setIndexBuffer', (self, args, next) => {
      capturedPass(self)?.uses.push(...usesOf(args[0], 'read'));
      return next(...args);
    });

    whileActive(GPURenderPassEncoder.prototype, 'draw', (self, args, next) => {
      countCall(self);
      return next(...args);
    });

    whileActive(GPURenderPassEncoder.prototype, 'drawIndexed', (self, args, next) => {
      countCall(self);
      return next(...args);
    });

    whileActive(GPURenderPassEncoder.prototype, 'drawIndirect', (self, args, next) => {
      countCall(self);
      capturedPass(self)?.uses.push(...usesOf(args[0], 'read'));
      return next(...args);
    });

    whileActive(GPURenderPassEncoder.prototype, 'drawIndexedIndirect', (self, args, next) => {
      countCall(self);
      capturedPass(self)?.uses.push(...usesOf(args[0], 'read'));
      return next(...args);
    });

    whileActive(GPURenderPassEncoder.prototype, 'executeBundles', (self, args, next) => {
      countCall(self);
      return next(...args);
    });

    whileActive(GPUQueue.prototype, 'writeBuffer', (_self, args, next) => {
      recordUpload('writeBuffer', args[0]);
      return next(...args);
    });

    whileActive(GPUQueue.prototype, 'writeTexture', (_self, args, next) => {
      recordUpload('writeTexture', args[0].texture);
      return next(...args);
    });

    whileActive(GPUQueue.prototype, 'copyExternalImageToTexture', (_self, args, next) => {
      recordUpload('copyExternalImageToTexture', args[1].texture);
      return next(...args);
    });

    patch(GPUQueue.prototype, 'submit', (_self, [buffers], next) => {
      // Read an iterable once, and hand the queue the array read
      const list = Array.isArray(buffers) ? buffers : [...buffers];
      const result = next(list);

      for (const buffer of list) {
        const state = commandBuffers.get(buffer);
        if (!state) continue;

        state.hooks.submitted(state.commands);
        for (const batch of state.batches) {
          if (batch.generation !== batch.slot.generation) continue;
          batch.state = 'submitted';
          settle(batch.slot);
        }
      }

      return result;
    });
  }

  for (const set of sets) patchInterfaces(set);

  return {
    start(started) {
      if (driving) driving.live = started;
      else hooks = started;
      if (timestampQuery !== undefined) started.deviceRequested(timestampQuery);
    },
    stop() {
      if (driving)
        endDrive(driving, new Error('Profiling stopped before the requested frames finished.'));
      hooks = undefined;
    },
    drive(measuring, driveOptions) {
      if (!hooks) return Promise.reject(new Error('Start profiling before driving frames.'));
      if (driving) return Promise.reject(new Error('Frames are already being driven.'));

      const drive: Driving = {
        hooks: measuring,
        reject: ignore,
        live: hooks,
        run: undefined,
        now: undefined,
      };
      const stopped = new Promise<never>((_, reject) => {
        drive.reject = reject;
      });
      driving = drive;

      const frame = requestFrame
        ? runHeld
        : (at: number) => {
            if (drive.run) runFrame(drive.run, 'frame', at);
            return drive.run !== undefined;
          };
      return Promise.race([runDrive(drive, driveOptions, frame), stopped]);
    },
    [Symbol.dispose]() {
      if (disposed) return;
      disposed = true;
      if (driving) endDrive(driving, new Error('The instrumentation was disposed.'));
      hooks = undefined;
      if (frames === 'animate') watchFrames(undefined);
      restore();
      instrumented = false;
    },
  };
}
