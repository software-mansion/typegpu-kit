# Device notes for the orchestrator

## Where the time goes on slow examples

`device.stats` (on a `GLDevice`) accumulates milliseconds per compute pipeline label (GPU forms as
`<label> (gpu)`), texture readback into CPU copies and uploads of CPU copies; `stats.reset()` starts
over. Milliseconds per 800×600 frame in headless Chromium at eb33f1e7, after warm-up. A kernel's
time includes the readback it triggers; GL is the time inside every WebGL call, `readPixels`
included.

| Example      | Frame | CPU kernels (largest)                | Readback | GL (calls)  |
| ------------ | ----- | ------------------------------------ | -------- | ----------- |
| classroom    | 157   | `summing` 153 (incl. readback)       | 25       | 25.5 (2022) |
| forest       | 99    | `placing` 61, `marking` 38           | 0        | 0.1 (281)   |
| quiver-trees | 49    | `placing` 31, `marking` 18           | 0        | 0.1 (655)   |
| fishtank     | 35    | `steerPipeline` 25, `live_compact` 5 | 0        | 5 (349)     |
| fireworks    | 30    | `exposure` 25 (incl. readback 23)    | 23       | 24.5 (895)  |
| skeletons    | 26    | `finding` 12, `marking` 7            | 0        | 7 (743)     |
| kinetic      | 20    | `exposure` 19 (incl. readback 17)    | 17       | 16.6 (320)  |
| jelly-world  | 13    | `culling` 12                         | 0        | 0.2 (140)   |
| night-market | 11    | `live_compact` 2.5                   | 0        | 8 (4387)    |
| ember-mirror | 11    | `shedder` 6, `live_compact` 4.5      | 2.2      | 3.7 (245)   |
| benthos      | 6     | `live_compact` 6                     | 0        | 3.1 (229)   |

Uploads are under 0.3 ms everywhere. The device's own share is the readback that a CPU kernel
reading a rendered texture forces: `exposure` reads the HDR target each frame, a synchronous
`readPixels` that waits for the frame's GPU work, and classroom's `summing` reads a 64-slice 3D
texture. Levels a kernel never touches are not read back (eb33f1e7), and half-float textures read
back as half floats. Draw overhead is small: night-market's 4387 GL calls take 8 ms. The rest of
each frame is CPU kernels; the instance culling in `packages/typegpu-kit/src/render/culling.ts`
(`marking`, `placing`) is the largest, and could skip work on `backends.of(root).compute === 'cpu'`.

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

## Rendering bugs traced to CPU kernels

`tests/device/diff.ts` runs a scene on WebGPU and on the device and compares every texture and
buffer both create, in creation order; `glRoot({ gpuCompute: false })` keeps every kernel on the
CPU. Cases the CPU backend gets wrong are `test.fails` in `kernels.test.gl.ts`, so each flips once
fixed:

- robot-arm, the missing table, wall and lamp light: the mesh pool's bake leaves every bounding box
  at zero size (`workgroup.reduce` with `atomicMin`/`atomicMax`, case "a mesh pool's baked bounds"),
  so culling drops meshes WebGPU keeps. It also writes a cylinder's cap ring at x = z = 0 (case "a
  cylinder's cap vertices"; the cap's `vertexAt` inside `concat` and `transform`).
- classroom: dark on the docs page up to ea61c2ea and correct from 4158329c ("preserve composite
  compute values across calls", a CPU backend fix), bisected on a clean checkout's docs server.
- skeletons stay collapsed: with every kernel on the CPU and seeded `Math.random`, the pose
  kernels leave words of `Pose_device` at 0 where WebGPU has 1 and -0.44 (words 166, 167, 171,
  537, 538 after 100 frames), and 39 % of `poses_joints` differs (case "skeletons' poses"). Clips
  and poses live in buffers, which the device keeps on the CPU, so no device copy is involved. The
  arena walls are missing for the same reason as robot-arm's table.

`diffScene` seeds `Math.random` the same for both runs; row ids still differ by design, since
each table made from a definition starts its generations elsewhere.

## Browsers without WebGPU

`scripts/examples.ts --backend nogpu` deletes `navigator.gpu` and every `GPU*` global before the
page loads, so the page falls back by itself; `--browser webkit|firefox` picks the engine.
`tests/device/nogpu.test.gl.ts` runs an example the same way, and
`tests/device/vitest.browsers.config.mts` runs the device tests in WebKit or Firefox (`BROWSER=`),
skipping the WebGPU comparisons there.

At 4775cb9e (WebKit 26.5 is Playwright's build; Firefox 151 cannot be driven in this environment,
its automation channel never connects, so it is untested):

| Example      | Chromium, no WebGPU | WebKit, no WebGPU                                          |
| ------------ | ------------------- | ---------------------------------------------------------- |
| afterimage   | 120                 | 60                                                         |
| benthos      | 120                 | 60                                                         |
| blockfall    | 120                 | 60                                                         |
| censer       | hung                | hung                                                       |
| classroom    | 7 (CPU summing)     | 60                                                         |
| cloister     | 120                 | 60                                                         |
| dominoes     | 120                 | 60                                                         |
| ember-mirror | 100                 | 60                                                         |
| falling-sand | 0.6                 | 0.9                                                        |
| fireworks    | 27                  | 31                                                         |
| fishtank     | 63                  | 60                                                         |
| flow-field   | 120                 | 60                                                         |
| forest       | 11                  | 61                                                         |
| hex-isle     | 120                 | 60                                                         |
| jellies      | 120                 | 60                                                         |
| jelly-world  | 79                  | 60                                                         |
| jump-flood   | 12                  | 8                                                          |
| kinetic      | 48                  | 51                                                         |
| lattice      | 120                 | 60                                                         |
| night-market | 111                 | 61                                                         |
| ocean        | hung                | hung                                                       |
| others       | ready               | ready (60 except slime-mould 2, sph-sculpt 19, stampede 9) |

WebKit's frame rate is capped at 60 and the Chromium column was swept before the culling and
`summing` GPU forms, which explains forest and classroom. Every screenshot matches the Chromium
fallback's apart from animation and simulation state; WebKit's WebGL 2 (ANGLE on Metal) exposes the
same extensions the device uses except `EXT_disjoint_timer_query_webgl2`, which the device does
not need. `navigator.gpu` is absent in that WebKit build, so plain URLs fall back there too.

## Ocean on SwiftShader

`node scripts/examples.ts --backend nogpu --flag=--use-angle=swiftshader ocean` fails with "The GPU
device was lost". With `--enable-logging=stderr`, Chromium's GPU process receives `signal 11
SEGV_ACCERR 000000000000` about 2.4 s after load, during ocean's first frame. The frame bakes the
environment and lasts about 2 s on SwiftShader; ocean is still behind its cover. Every other example
survives SwiftShader. Measured at 89f8043f:

- It happens offscreen too (`runScene` on `glRoot()` under the flag), at 800×600, 160×120 and 8×8,
  so it is neither the canvas compositor nor the surface size.
- `flush`, `finish` or `getError` after every GL call does not prevent it, so it is not command
  batching. An earlier note that a round trip after every call avoided it was wrong: the loss is
  only reported after the next idle tick, which that probe misread.
- Ruled out by removing each in turn: `deleteBuffer` (the only call in the idle tick where the loss
  shows; a no-op `deleteBuffer` still crashes), the `FinalizationRegistry` frees, blits,
  `KHR_parallel_shader_compile` (all 43 programs link synchronously without loss) and
  `EXT_texture_filter_anisotropic`. `--disable-angle-features=warmUpPipelineCacheAtLink` and
  `enableParallelCompileAndLink` change nothing either, if Chromium applies them.
- Deleting a uniform buffer that is bound, rebound or unbound, and an anisotropic sampler with a
  mip chain, survive on their own.
- With a `finish` after every call and timestamps against Chromium's log, the process dies
  about 0.9 s into the `finish` after ocean's samplers are created (`TEXTURE_MAX_LOD` 32), which a
  synchronous call cannot explain unless SwiftShader is still working through earlier commands.
  Loss is reported to the page asynchronously, so a call log alone cannot name the culprit.

Narrowing further means bisecting ocean's own passes, which would need changes to the example.

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

## Grouped placement (`kit-gl/placement-chain`, not merged)

What it does: a GPU form whose outputs the CPU reads back, but whose inputs other GPU forms rewrite before each of its dispatches, cannot move to the CPU alone. `placement.ts` moves it with those forms, transitively, as one group of at most 4 small forms, judges the group over three windows of every member's dispatches, and sends it back as one: for good when it costs more than it saves, or counting a return on a dispatch too large for the CPU, an outside GPU writer of a member's input, a leader that stopped dispatching against its own cadence, or a member whose pipeline was collected. A single form with CPU-current inputs decides exactly as on main.

Measured with `node packages/typegpu-kit-gl/scripts/examples.ts --base <server> --backend webgl --stats fireworks platformer skeletons`, main (4b3f492d) against the branch, two runs each, per frame:

- fireworks: readback 0.28–0.33 → 0–0.002 ms, readbacks 0.55 → 0.03. Groups: `shellsFollowed` with `starsFollowed` and `embers_step`; `shells_spawn` with `shells_advance`. Offscreen 7.8 → 2.6 ms a frame.
- platformer: readback 1.29–1.65 → 0 ms, readbacks 2 → 0, CPU kernel 1.3–1.7 → 0.05 ms (`live_gather` with `live_mark`).
- skeletons: readback 1.65–1.71 → 0.60 ms, readbacks 3 → 1, CPU kernel 1.7 → 0.9 ms (`cullFlagging` with `marching`). Frame time equal: both hold 120 Hz.

Review findings fixed, each with a test in `placement.test.ts` that fails on the code before it: an oversized dispatch of a member (sends the group back before it runs); a leader that stops dispatching, and one that dispatches regularly but rarely (13:1 keeps its group); the GPU baseline of a member whose window just closed; credit only for waits charged to the members' own outputs, and no more than the device as a whole stopped waiting (a wait that moves to another readback is not credited); blocker history cleared at the move so the group's weak references hold.

Open or unproven:

- Finding 1 is only partly addressed. The credit is capped by the device-wide drop in waits per leader dispatch, taken from the window before the move against each judged span. Load that changes between the two still moves the device-wide figure; the reviewer's suggestion to re-take the "before" side by returning to the GPU for a window is not built.
- A collected member breaks the group for the survivors, but no test forces a collection (no `gc` in the unit project).
- `stats.placement` is keyed by label, and several pipelines share labels such as `cullFlagging`, so a group's messages can be overwritten by a namesake.
- On the page, the second fireworks group can take about 5 s to form, until enough of the shells' spawns are read back.
- My dev servers on 4471 and 4472 stopped twice during test runs in this session; I restarted them and did not find why.
