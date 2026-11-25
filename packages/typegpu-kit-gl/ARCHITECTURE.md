# How the WebGL fallback works

`@typegpu/kit-gl` runs an unchanged TypeGPU Kit program on WebGL 2. It does not reimplement the kit or TypeGPU. It gives TypeGPU a `GPUDevice` that is implemented on a `WebGL2RenderingContext`, and a shader generator that emits GLSL ES 3.00 and JavaScript instead of WGSL:

```ts
const device = createDevice(gl); // src/device
const root = tgpu.initFromDevice({ device, unstable_shaderGeneratorClass: KitGenerator });
```

Everything above the device is the real thing: TypeGPU's root, buffers, bind groups, layouts, pipelines and every kit module. So a module works on the fallback when the device supports the WebGPU calls it makes and the generators can express its shaders.

## Parts

| Path               | Owns                                                                                                                                                                                                                                                    |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `../typegpu-gl`    | The vendored `@typegpu/gl`: the WGSL-shaped AST to GLSL generator. Patches for upstream only; see `PATCHES.md`.                                                                                                                                         |
| `src/generator.ts` | `KitGenerator`, the class the root instantiates per pipeline. It hands a render pipeline to the GLSL generator and a compute pipeline to the CPU generator, which also attaches the pipeline's GPU form.                                                |
| `src/glsl/`        | `KitGlslGenerator extends GlslGenerator`: what is specific to the device's conventions, and the render module format, `renderModule.ts`. `KitGlslComputeGenerator` in `compute.ts` emits GPU forms of kernels, in the format `computeModule.ts` states. |
| `src/cpu/`         | The CPU compute backend: `CpuGenerator` emits JavaScript for a compute pipeline, `compile` turns it into a `CpuModule` the device dispatches.                                                                                                           |
| `src/device/`      | The WebGPU device on WebGL 2: buffers, textures, samplers, bind groups, pipelines, encoders and passes, the queue, canvas contexts. `gpuCompute.ts` runs a kernel's GPU form; `compute.ts` chooses it or the CPU module per dispatch.                   |
| `src/webgl.ts`     | The public `webgl` namespace: `init`, and the fallback `web.host` takes.                                                                                                                                                                                |

## A shader module is a string

TypeGPU resolves a pipeline with the generator and passes the result to `device.createShaderModule({ code })`. That string is all the device learns about a shader, so it carries its own metadata as directive comments, lines that start with `//#`. `parse` in `src/glsl/renderModule.ts` and `compile` in `src/cpu` read them; the device never reads directives itself. A module without a `//#kit-gl` directive is WGSL the generator never saw, such as TypeGPU's internal mipmap shaders, which the device replaces with GL calls or rejects.

TypeGPU replaces bind group placeholders in the whole string after generation, so a directive may name a group by its placeholder.

## Conventions the device and the GLSL agree on

- **Buffers are textures.** Every buffer binding, uniform or storage, reads from an `RGBA32UI` texture that holds the buffer's bytes as 32-bit words, four to a texel, rows of `textureSize(tex, 0).x` texels. Word `w` is component `w % 4` of texel `w / 4`. Generated loaders rebuild values with `uintBitsToFloat`, so the WGSL memory layout is the only layout. Each binding also has a `uvec2` uniform: the first word of the bound range, and its size in bytes for `arrayLength`.
- **Rows run top to bottom.** Texture row 0 is the top row, as in WebGPU, for uploads, `textureLoad`, render targets and readback. Every vertex shader ends by negating `gl_Position.y`, the device swaps the front face to compensate, and a canvas is presented with a flipped blit. `gl_FragCoord` and derivatives then match WebGPU with no further work.
- **Depth is 0 to 1.** The vertex epilogue remaps `z` to GL's clip range unless the device enabled `EXT_clip_control`, which it signals with `#define KIT_CLIP_ZERO_TO_ONE`.
- **Samplers are combined.** Each texture and sampler pair a shader uses is one GLSL sampler uniform; the device binds the texture and the sampler object to its unit.
- **Instance and vertex offsets are uniforms.** `kit_firstInstance` and `kit_baseVertex` carry what WebGL 2 draw calls cannot.

## Compute

WebGL 2 has no compute stage. A compute pipeline's module is JavaScript: `CpuGenerator` emits one function per shader function over typed-array views of the bound buffers, and the device can run it on dispatch, invocation by invocation. Its `f32` arithmetic rounds to 32 bits at each operation, as a GPU's does, so a seeded kernel gives the CPU and the GPU the same result. Buffers and textures then have two homes. The device tracks which copy is current, reads back from GL when a kernel needs data a draw produced, and uploads before a draw reads what a kernel wrote.

### GPU forms

Most kernels also have a GPU form, a set of passes GL draws in place of the CPU module. `KitGenerator` resolves the pipeline twice more with `KitGlslComputeGenerator` (`src/glsl/compute.ts`). The first resolution walks the entry and every function it reaches and builds a plan: what the kernel writes, how, and how many records each helper adds. It decides refusals from the whole walk at the entry, not function by function, since TypeGPU resolves a helper before the code that calls it: an `atomicLoad` of a buffer the kernel adds to, or a taken value deciding a branch in another function, refuses the form whichever function TypeGPU resolved first. The second resolution emits the passes from the plan. The CPU module carries the form on a `//#kit-gl gpu` line, or why it has none on a `//#kit-gl cpu-only` line.

`parseCompute` in `src/glsl/computeModule.ts` reads the form and states the contract: what each pass draws, and the conditions on a dispatch under which the device may use the form. `src/device/gpuCompute.ts` checks those conditions per dispatch and draws the passes. Every pass reads the contents from before the dispatch and renders the new contents into a fresh texture, and the device replaces each target with its texture once every pass ran. A form's programs start linking when the pipeline is made and are waited for at its first dispatch, or by `initAsync`. The device's `stats` report, per pipeline, the time spent and why a dispatch ran on the CPU.

A form has one pass per written resource, of one of four kinds:

- **Fragment passes**, for a resource whose cells each belong to one invocation. Every fragment runs the whole kernel for the invocation that owns its cell. Writes go to a copy of the owned cell, and a read of a written resource checks at run time whether it reads that cell. An invocation owns element `gid.x` of an array, or `gid.x + gid.y * width + gid.z * width * height` for constant sizes, times a constant stride, plus a phase all invocations share (a uniform's offset, or a masked value below the stride); texel `gid.xy` (`.xyz`) of a storage texture, or a texel whose coordinates are strided the same way, such as a red-black sweep's `(gid.x * 2 + ((gid.y + gid.z + parity) & 1), gid.y, gid.z)`; every texel of the 2d-array or 3d column at `gid.xy` when the dispatch is one invocation deep; and whole small bindings when the dispatch is one invocation or the binding is written only under conditions such as `gid.x == view.cursor.x` that choose one invocation by values all invocations share. Ownership passes into helpers through their arguments, and atomics on owned cells are plain arithmetic on the copy. A pass of a 2d-array or 3d target writes four layers a draw, and the device renders a whole single-level texture into new storage that replaces the old.
- **Passes of blocks**, for a grid each invocation owns a constant block of, up to 8 cells, written through a helper. The helper writes `row * width + col` of a vector parameter after an early return on a function that keeps the column in `[0, width)`. Each call from the entry, under a condition or not, names a cell of the block through its argument: a step per axis times the invocation, plus a constant offset within the step and a phase every invocation shares. Each invocation keeps one shadow per cell of its block, loaded from the old contents in `kit_begin`, and every read and write of the grid goes to the shadow whose cell it names or to the old contents. A fragment finds the site and invocation whose block holds its element and keeps that shadow.
- **Passes of points**, for a resource the kernel writes at cells it computes. A vertex shader runs the kernel once per invocation, and each write records the cell and the value in the next of the resource's slots: as many as the writes the entry and the functions it calls can make, none of them in a runtime loop. Transform feedback captures the slots. The device then draws those records as points over a copy of the old contents: once per slot, and for a buffer once per word and data-texel channel, or once per element where an element fits in a data texel. Each write must write a whole element or texel, and the kernel must not read the resource.
- **Passes of sums**, for a buffer the kernel only adds to with atomics whose results it drops. Each invocation records its additions to that buffer, at most the count found by adding up the additions of the entry and every function it calls, none in a loop, and for a buffer of a fixed size no more than its words: an addition to a word it already recorded adds to that record. Transform feedback captures the records. The device draws them as points with additive blending into eight float textures, one nibble of every amount each: at once with eight draw buffers, or in two rounds of four, which a uniform tells the shader, with fewer. Each channel stays exact for 2^24 / 15 records. A last draw renders the old words plus the summed nibbles in `uint` arithmetic, which wraps as the atomics do. A sum does not depend on the order of its additions.

  A counter the kernel takes values of, with `atomicAdd` of 1 whose result it uses, is such a buffer too, and the device ranks its takes before any pass: it runs the counter's capture once, counts each invocation's records, and renders each invocation's base, the takes of the invocations before it, into a texture every program of the form samples. A take then returns the counter before the dispatch plus the base plus the invocation's earlier takes, in every pass alike, as long as no taken value changes how many the invocation takes, which the generator checks over the whole kernel.

The device runs a dispatch on the CPU module where the form's contract does not hold for it: a dispatch wider than a pass's target, one whose records could put more than 2^24 / 15 additions on one cell, one that takes from more than 65 536 invocations or more than 15 times an invocation, and records past what transform feedback captures or past 255 bytes an invocation, the stride WebGL reads attributes at.

### Placement

A GPU form is not always the cheaper place. When the CPU reads a form's outputs back, each dispatch costs a wait for the GPU, and a small kernel whose inputs are already on the CPU can run there instead and leave GL an upload. `src/device/placement.ts` watches each pipeline over a window of dispatches, moves such a kernel of at most 65 536 invocations to its CPU module, and moves it back when the CPU run costs more than the waiting it saved or its inputs stop being on the CPU. `stats.placement` gives the reason per pipeline.

### Kernels that branch

Other atomics on shared cells, barriers and workgroup variables keep a kernel on the CPU, as do writes these rules exclude. A module can still choose other kernels on the fallback with `backends.of(root)`, usually ones these forms express:

- In the Kit, the culling, `exposure`, `environment`, `grid`, `liveLists`, `particles` and `poses`. The grid sorts with a bitonic network of passes that each write their own elements and bounds its cells with scattered writes. Particles spawn bursts by having each ring slot gather the particle the ring places in it. Poses find each joint's chain in a thread of its own instead of sharing a workgroup's. The culling and the live lists mark each row in a thread of its own, which the GPU runs, and list the marked rows in order on the CPU.
- In the examples, `ocean`, `slime-mould`, `censer`, `sph-sculpt` and `jump-flood`.

Each such branch leaves the WebGPU path as it is, and a test runs both on dawn by branding a real device as CPU-compute.

## What modules see

`backends.of(root)` in `@typegpu/kit` reports `'webgpu'` or `'webgl'` from a brand on the device, so a module can pick another kernel, capacity or technique where the automatic path is too slow or cannot work. `LIMITATIONS.md` lists what the fallback cannot do.
