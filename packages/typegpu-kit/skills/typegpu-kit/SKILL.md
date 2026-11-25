---
name: typegpu-kit
description: >-
  Use when building a WebGPU game, 3D scene or demo with @typegpu/kit, or changing code that imports it or one of its subpaths: frames and surfaces, scenes and hosts, passes and drawings, tables, cameras, lighting and shadows, instances, glTF, post-processing, particles, text, the neighbour grid, picking, readback, testing. TypeGPU Kit is a set of GPU modules on TypeGPU that borrow the caller's root and record into the caller's encoder or passes, on the web and React Native. Use it alongside the typegpu skill, which covers TypeGPU itself.
metadata:
  library: '@typegpu/kit'
---

# Building on TypeGPU Kit

Read this file before writing GPU code and after compaction. Before each feature, read [Find it before you build it](#find-it-before-you-build-it). Links are relative to this file.

- Docs: Markdown in [docs](../../docs/index.md). Open each module's page, most at `docs/kit/<name-in-kebab-case>.md`, before using it; its card lists every option and member. [modules.md](../../docs/modules.md) gives each module's import, factories, options, ownership, recording methods and bindings; search it before the kit's source. Each declaration in `dist/` opens with its page and examples.
- Examples: [docs/examples](../../docs/examples/README.md) holds every example's source; each exports `createScene` for a host. The harness files its README names are the docs site's, not kit API. Copy an example's modules, frame and file layout, not its look: each picks techniques for its own scene. Before calling a look done, compare it with [the finished look](../../docs/recipes/lit-models.md) and try each stage it adds.
- TypeGPU itself: the `typegpu` skill, at `node_modules/typegpu/skills/typegpu/SKILL.md` in the project. It ships with TypeGPU and matches the installed version.
- Versions: [what a release keeps](../../docs/getting-started.md#versions). A module whose line says **stability** unstable can change in any way in a minor.

## Find it before you build it

Look up each feature here before writing it. Each line names its modules and a file to start from.

- Start
  - A project, a scene on a page: [set up a project](../../docs/getting-started.md#setting-up-a-project), [host a scene](../../docs/kit/hosting.md).
  - A scene on React Native: `useScene`, or `host` without React; [on React Native](../../docs/kit/hosting.md#on-react-native).
  - GPU work on a React Native worker, and values between runtimes: `runtimes`; [the runtime pieces](../../docs/kit/hosting.md#the-runtime-pieces).
  - A host of your own: `hosting`; [a host of your own](../../docs/kit/hosting.md#a-host-of-your-own).
  - A browser without WebGPU: `web.host`'s `fallback` from `@typegpu/kit-gl`, and `backends.of(root)` to pick kernels or sizes where compute runs on the CPU; [where WebGPU is missing](../../docs/kit/hosting.md#where-webgpu-is-missing).
- Look
  - Many models, finished: shadows, occlusion, TAA and bloom in one file to copy, [look.ts](../../docs/recipes/lit-models/look.ts), which [the finished look](../../docs/recipes/lit-models.md) explains.
  - Sun shadows: `lighting` with `shadow`, read in shaders through `shading`; lamp shadows: `pointShadow.define`, given to `lighting.create`; [robot-arm](../../docs/examples/robot-arm/).
  - A still lamp's shadow layers, kept while no caster near it moves: [`redraw`](../../docs/kit/lighting.md#lights-that-hold-still), through `prepare` before culling.
  - Darker sky light in corners, creases and where things meet: `ao`; [robot-arm](../../docs/examples/robot-arm/).
  - `ao` under an MSAA scene: it reads a single-sampled depth prepass you draw; [dominoes](../../docs/examples/dominoes/).
  - Smooth edges: `taa`; [survey-model](../../docs/examples/survey-model/).
  - Smooth edges for less work: `fxaa`; [robot-arm](../../docs/examples/robot-arm/).
  - Glow, tone and colour: `bloom`, `tone`, `finish` and its `grade`.
  - Light that adapts to the picture: `exposure`.
  - An effect of your own between them: `post`.
  - What surfaces are made of: `materials`, from colours or a model's textures.
  - Sky, stars, light from the surroundings: `environment`; [kinetic](../../docs/examples/kinetic/).
  - Lamps, headlights, many small lights: `lighting` points, with [candidates](../../docs/kit/lighting.md#only-the-lights-that-reach) for many; [night-market](../../docs/examples/night-market/), [dark scenes](../../docs/recipes/dark-scenes.md).
  - How much light reaches a point, asked on the CPU: [`shading.falloff` and `shading.beam`](../../docs/kit/lighting.md#point-lights).
  - Fog: `shading.fog`; [spider/terrain.ts](../../docs/examples/spider/terrain.ts).
  - A lighting hook for only the renderers that read it: `renderer.pipe((cfg) => cfg.with(shading.fog, fog))`.
  - Depth of field: `dof`; [hex-isle](../../docs/examples/hex-isle/).
  - Water: [fireworks/lake.ts](../../docs/examples/fireworks/lake.ts).
- Things in the scene
  - Many models with per-object colour or state: `meshPool`, `instances` with a kind; [hex-isle/isle.ts](../../docs/examples/hex-isle/isle.ts).
  - An instance hidden: its kind's `visible` hook; freed: `remove`.
  - Meshes that grow, sway or topple: a kind's `deform`, with `bounds` when culling; [forest/trees.ts](../../docs/examples/forest/trees.ts).
  - Thousands of objects, most off screen: `renderer({ cull: true })`; [forest](../../docs/examples/forest/).
  - Distant trees, crowds or props as cards: `impostors`; [quiver-trees](../../docs/examples/quiver-trees/).
  - A model whose files sit beside it: [`gltf.load(url)`](../../docs/kit/gltf.md#models-in-several-files).
  - People, animals, walking characters: `poses`; [night-market/people.ts](../../docs/examples/night-market/people.ts).
  - A posed figure's joint on the CPU: [`jointAt`](../../docs/kit/poses.md#where-a-joint-is).
  - Wires, cables, outlines, paths: `lines`; [survey-model/contours.ts](../../docs/examples/survey-model/contours.ts).
  - Lines, joins and circles in a shader of your own: `polylineVariableWidth`, `lineVariableWidth`, `circle`; [lines in your own shaders](../../docs/kit/lines.md#lines-in-your-own-shaders).
  - Trails: `trails`; [solar-system/comets.ts](../../docs/examples/solar-system/comets.ts).
  - Smoke, sparks, dust, rain: `particles`; [night-market/rain.ts](../../docs/examples/night-market/rain.ts).
  - Lit particles: a `shade` piped with `light.inject()`; [night-market/rain.ts](../../docs/examples/night-market/rain.ts).
  - Labels in the world: `text`, with `font`; [star-map/label.ts](../../docs/examples/star-map/label.ts).
  - Text laid out on the CPU: `typeset`; [solar-system/labels.ts](../../docs/examples/solar-system/labels.ts).
  - A font baked at build time: `bake` from `@typegpu/kit/fontBaker`; [baking a font](../../docs/kit/text.md#baking-a-font).
  - A mesh or shader of your own: `meshes` and [drawings](../../docs/kit/passes.md#drawings).
  - Buffers and textures of your own: [`bindings`](../../docs/kit/passes.md#buffers-and-textures-of-your-own), `images` and `targets`.
  - A texture baked from a function of position: `baker`.
  - A texture's mip levels, or farthest depths for occlusion: `mipmaps`.
  - A camera feed: `video`; [ember-mirror](../../docs/examples/ember-mirror/).
- Input
  - Turn, pan and zoom: `cameras.orbit` and `web.orbitControls`, or `cameras.turn(goal, dx, dy)` and `cameras.pan(goal, dx, dy, { height })` from any drag.
  - Click or hover an object: `picker`; [dominoes/topple.ts](../../docs/examples/dominoes/topple.ts).
  - A pick by an object's real shape: a `picker` `rank` that tests `candidate.ray`; [exact shapes](../../docs/kit/picker.md#exact-shapes).
  - Everything inside a dragged box: `liveLists` with a region test as `alive`; [many at once](../../docs/kit/picker.md#many-at-once).
  - The ground under the pointer: `web.pointer`, then `view.rayAt` and `cameras.hitHeight`.
  - DOM over a world point: `view.projectAt`, 0 to 1 across the surface, +y down, as `view.rayAt` takes.
- Simulation
  - Fixed steps: `clock.fixedStep`; [slime-mould](../../docs/examples/slime-mould/).
  - Motion at any frame rate: `frame.dt` and `frame.time`, eased with `clock.ease(dt, rate)` or `clock.spring`.
  - A sequence over frames: `coroutines`.
  - Meshes moved by kernels: `instances`; [dominoes/dominoes.ts](../../docs/examples/dominoes/dominoes.ts).
  - Rows of your own on the GPU: `table`; [platformer/critters.ts](../../docs/examples/platformer/critters.ts).
  - A varying list on the GPU: `liveLists`.
  - States and transitions per object: `machines`.
  - Neighbours, flocks, crowds: `grid`; [fishtank/school.ts](../../docs/examples/fishtank/school.ts).
  - Results on the CPU: `counters`, `readback`.
  - Sums, bounds or packed offsets across a workgroup's invocations: `workgroup`.
  - Keys sorted with a payload, such as splats back to front: `sorter`.
  - An effect you can scrub: compute it from seed and age, as in [scrubbed effects](../../docs/recipes/scrubbed-effects.md).
- Running well
  - A loading screen: `pipelines.compile`.
  - A frame budget: `dynamicResolution`; [jelly-world](../../docs/examples/jelly-world/).
  - A value cached per key: `memo`.
  - Part of a large array uploaded: `buffer.write(typed.buffer, { endOffset })`.
  - Optional collection of many resources: [`owner`](../../docs/concepts/modules.md#collecting-with-an-owner).
  - GPU time per pass, read in the program: `profiler`.
  - Every pass and dispatch timed on the page, the program unchanged: `gpuDebug`, on React Native `GpuDebugPanel`; [GPU debug panel](../../docs/kit/gpu-debug.md).
  - The same on the WebGL 2 fallback: `gpuDebug.install({ interfaces: webgl.interfaces })`; [GPU debug panel](../../docs/kit/gpu-debug.md).
  - Frames timed from code: `instrument`, `createRecorder`, `watchFrames`; [without the panel](../../docs/kit/gpu-debug.md#without-the-panel).
  - Pipelines, their bindings per stage and late compiles, counted: `census`; [census](../../docs/kit/pipelines.md#census).

## The shape of a program

A program owns a root, a surface (`web.surface` or `native.surface`), a loop (`frames.create`) and `frames.animate`, which runs it:

```ts
const root = await tgpu.init();
const surface = web.surface(root, canvas);
// modules, buffers and pipelines: made once, here
const loop = frames.create(root, (frame) => {
  // passes and modules' recording methods, in order
  const pass = frame.encoder.beginRenderPass({ colorAttachments: { view: surface.view(frame) } });
  pass.end();
});
frames.animate(loop);
```

The loop submits once. Modules take the pass, encoder or frame their methods name and never submit.

Export `createScene(root, surface, options)` for `web.host` or `useScene`. Return `frame`, drawing into `surface.view(frame)`, and optional `destroy`. DOM and input stay in the host. Use one `pipelines.compile([...])` list and one `for (const made of [...]) made.destroy()` loop for your textures, non-uniform buffers and modules.

Leave uniforms, samplers, bind groups, layouts and pipelines to the garbage collector.

## Structure your program

Split by system into files of 100 to 400 lines, as the examples do.

- `scene.ts` makes the shared pieces, such as the mesh pool, camera, lighting, passes and post-processing, adds each system's drawings and records the frame in order.
- Give each system, such as roads, lamps, traffic or water, its own file: a factory `create(root, …)` that takes the shared pieces it reads and returns its renderers or drawings, an `update` for the frame, and `destroy` if needed.
- Keep simulation and game rules free of GPU imports. A system writes instance rows and buffers when its state changes, not every frame.
- Put things that shade alike in one instance set, with per-object values in its kind's fields.
- Type a fill that several renderers share as `Inject`: `const fill: Inject = (cfg) => cfg.with(dayTime, timeUniform)`.

A module others reuse goes further: it borrows the root, takes required inputs by position and the rest as options, returns resources, `'use gpu'` functions and recording methods, and frees only what it made. Accessors made in `create` default to the instance's own resources; `inject()` fills the placeholders others read, each with an `unfilled` default naming that `inject()`. A swappable resource other modules read goes in a `bindings.layout` part. [Write a module](../../docs/tutorials/write-a-module.md) builds one.

## GPU code

- Write kernels as `tgpu.computeFn` with `workgroupSize` and a bounds check, then `createComputePipeline` and `dispatchWorkgroups`, rather than guarded compute pipelines.
- Your recording methods read `frame.encoder` on each call.
- Factories allocate, `pipelines.compile` compiles, frames record.
- Take randomness and noise from `@typegpu/noise` (`randf`, `perlin2d`, `perlin3d`); never hand-write a hash.
- Store per-object values in buffers; captured constants create separate pipelines.
- Split uniforms by update rate and write settings only when they change. Keep evolving state on the GPU.
- An instance's members, such as `enemies.gpu`, `emit` or `add`, read it in any pipeline, two tables of one definition in one kernel. Pipe fills only into placeholders, hooks such as the camera and a definition's members: `root.pipe(module.inject())`, `passes.create(...).pipe(view.inject())`, `tgpu.fn(f).pipe(other.inject())`.
- Describe scene outputs once with `passes.create`, shadow passes with depth alone; `scene.add(drawing)` builds a drawing for every output. Bind a custom group on the pass as the passes page shows: `cfg.with(group)` and `root.with(group)` bind none.
- Keep vertex-moving state in `history`, or give its table or instances `history: true`; outputs reading last frame's vertices use `history.previously`.
- Read vertex-stage storage read-only: `buffer.as('readonly')`, table members like `boids.gpu.position(i)`, or `byStage(buffer)`.
- Stay within 8 storage and 12 uniform buffers per stage. modules.md lists each module's; over budget, `pipelines.compile` throws.
- A `grid` search that starts in a corner and stops early biases dense groups toward it.
- Each lit point visits every point-light slot unless `points.candidates`, from a grid whose `sight` covers every light's range, narrows them.
- Write vector arithmetic as `a * b`, not `std.mul` and the like. Operators work only in `'use gpu'` functions; on the CPU, convert colours with `srgbToLinear` from `@typegpu/color`.
- Shift `u32` right with `>>>`; wrap a left-hand literal in `d.u32`: `d.u32(0x32) >>> corner`.
- Take `fwidth`, `dpdx`, `dpdy` before a per-pixel branch or `return`.
- Far from the origin, interpolate eye-relative positions for derivatives and noise.
- Write GPU methods as properties, `alive: (i) => { 'use gpu'; ... }`, not shorthand methods.
- `tsc` accepts three mistakes that resolving a shader reports. Copy a named vector, matrix or struct when you assign or return it: `agent.velocity = d.vec3f(velocity)`. Write a struct literal's fields in its schema's order. Hold constants a shader indexes in `tgpu.const(d.arrayOf(d.f32, n), values)`, as [solar-system/planets.ts](../../docs/examples/solar-system/planets.ts) does.

## Platforms

Before writing code for Safari, Firefox or React Native, read [platforms](../../docs/platforms.md). On React Native, import subpaths such as `@typegpu/kit/frames`, and flat helpers from theirs, such as `@typegpu/kit/inject`.

## Testing

- On `recordingRoot()` with `offscreenSurface()`, pin passes and pipelines, check that steady frames create nothing and compare `log.alive({ uniforms: false })` before creation and after `destroy()`.
- Run the frames a host or `animate` requests one at a time with `steppedSchedule()`.
- Compile WGSL on dawn.node with `gpuRoot()` from `@typegpu/kit/testing/dawn`; fail on TypeGPU warnings with the `@typegpu/kit/testing/warnings` setup file.
