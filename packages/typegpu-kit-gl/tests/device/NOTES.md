# Device notes for the orchestrator

## Where the time goes on slow examples

`device.stats` (on a `GLDevice`) accumulates milliseconds per compute pipeline label, texture readback
into CPU copies and uploads of CPU copies; `stats.reset()` starts over. Profiles of 800×600 frames in
headless Chromium, after warm-up:

| Example      | Frame   | CPU kernels                                       | Device                                                                 |
| ------------ | ------- | ------------------------------------------------- | ---------------------------------------------------------------------- |
| ember-mirror | 124 ms  | `shedder` 80, `stepPipeline` 36, `live_compact` 8 | readback 3.2 (feed texture, half the frames)                           |
| night-market | 106 ms  | `stepPipeline` 76, `moving` 22                    | readback 22 (one texture a frame, mostly the GPU sync of `readPixels`) |
| forest       | 2990 ms | culling `marking` 2770, `placing` 220             | under 0.2                                                              |
| quiver-trees | 1023 ms | culling `marking` 950, `placing` 71               | under 0.3                                                              |
| jump-flood   | 7340 ms | `pipeline` 7140, `neighboursPipeline` 180         | under 0.3                                                              |

GL calls for a whole frame of night-market (591 draws) total about 1 ms. The device is not the
bottleneck anywhere measured; the CPU kernels are. The instance culling in
`packages/typegpu-kit/src/render/culling.ts` is the largest single cost: on `backends.of(root).compute
=== 'cpu'` it could skip culling and draw every row, as `forest` does with `cull: false`.

A kernel that reads a texture a pass rendered this frame forces a synchronous `readPixels`, which
waits for the GPU. One-channel textures read back as `RED`/`FLOAT` to keep it small; the wait
remains.

## Failures that are not the device's

From `scripts/examples.ts --backend webgl` at device commit 3c4aa4c1:

- benthos: `computeFn:mainCompute ... is written in WGSL, which the GLSL generator cannot translate.`
- fireworks: `The compute pipeline 'spawnPipeline' failed on the CPU: trail is not defined`
- jellies: `The compute pipeline 'trailing' failed on the CPU: cpuAddress_4 is not defined`
- platformer, robot-arm, skeletons, night-market (`finding`): `@typegpu/kit-gl CPU: Invalid
  destructuring assignment target` (robot-arm passed in the device's own harness earlier the same
  day, so this is recent)
- censer, classroom, falling-sand, ocean: the page stops answering; fishtank, flow-field,
  jump-flood, kinetic, jelly-world: no frame before the timeout. jump-flood measures 7.3 s a frame
  in CPU kernels; the others are compute-heavy simulations of the same kind.
- forest, quiver-trees, night-market, ember-mirror: ready but slow, as profiled above, so the
  screenshot shows the loading cover.
- solar-system: Saturn's ring is missing (the WebGPU reference shows it). The ring is the second
  mesh of a pool, drawn by instance records; vertex pulling with a base vertex, direct and indirect,
  matches WebGPU on the device (`roots.test.gl.ts`), so the records the CPU kernels write are the
  likelier cause.

## Compute as fragment passes

The device runs a compute pipeline's GPU form (`src/glsl/computeModule.ts`) when its module carries
one and the dispatch fits, and the CPU module otherwise. A buffer target's pass renders a new data
texture that replaces the old one; the buffer's CPU copy is then stale and `sync()` reads it back
before any CPU use (mapping, `writeBuffer`, copies, indirect arguments, CPU kernels, vertex, index
and uniform copies). A texture target's pass renders each layer into a new texture, blitted into
the target after every pass ran. `device.stats.kernels` names GPU dispatches `<label> (gpu)`, and
`device.stats.fallbacks` holds why a pipeline runs on the CPU, including the generator's `cpu-only`
reason. In jump-flood every kernel stays on the CPU: `pipeline` and `clearPipeline` write cells
other than their own, `placePipeline` and `neighboursPipeline` use atomics, and `computePipeline`
writes from a helper function. In classroom `scattering` runs as fragment passes in 1 ms, but
`summing` stays on the CPU (930 ms a frame; it stores to texels other than its own) and reads the 3D
texture `scattering` wrote, a 64-slice readback of 60 ms each frame.

## Contract notes

- `RenderModule.blocks` is bound: each block gets binding point `n` in declaration order and a
  `UNIFORM_BUFFER` copy of the buffer. A range whose offset is off `UNIFORM_BUFFER_OFFSET_ALIGNMENT`
  or which runs past the buffer binds a copy of its own. More blocks than
  `MAX_UNIFORM_BUFFER_BINDINGS` (24 or more) is a validation error on the pipeline; per-stage limits
  (`MAX_VERTEX_UNIFORM_BLOCKS`, `MAX_FRAGMENT_UNIFORM_BLOCKS`, at least 12) show up as link errors.
- Buffer and texture bindings share texture units: buffers first, then textures, at most
  `MAX_COMBINED_TEXTURE_IMAGE_UNITS - 1` (one unit is kept for uploads). Exceeding it is a
  validation error naming the counts.
