/**
 * Exports the recorder, the WebGPU patches and the frame hook for reading frames without
 * mounting the panel.
 *
 * @module
 * @unstable
 * @kind tool
 * @platform web
 * @flat
 * @requires timestamp-query optional it asks for it on devices requested after it patches;
 * without it, no timings
 * @owns up to 16 query sets, 32 buffers, a texture per canvas: per device, while it times passes:
 * timestamps, a buffer they resolve into and one to map; while it drives frames, a texture each
 * canvas's frames draw into
 */

export { type FrameWatch, watchFrames } from '../animate.ts';
export {
  instrument,
  type GpuCommand,
  type InstrumentHooks,
  type InstrumentOptions,
  type Instrumentation,
} from './instrument.ts';
export type { Access, GpuResource, GpuResourceUse } from './metadata.ts';
export {
  createRecorder,
  type Bar,
  type CapturedCommand,
  type Dependency,
  type FrameCapture,
  type Recorder,
  type Segment,
  type Timings,
} from './recorder.ts';
