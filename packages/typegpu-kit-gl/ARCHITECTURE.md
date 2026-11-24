# How the WebGL fallback works

`@typegpu/kit-gl` runs an unchanged TypeGPU Kit program on WebGL 2. It does not reimplement the kit or TypeGPU. It gives TypeGPU a `GPUDevice` that is implemented on a `WebGL2RenderingContext`, and a shader generator that emits GLSL ES 3.00 and JavaScript instead of WGSL:

```ts
const device = createDevice(gl); // src/device
const root = tgpu.initFromDevice({ device, unstable_shaderGeneratorClass: KitGenerator });
```

Everything above the device is the real thing: TypeGPU's root, buffers, bind groups, layouts, pipelines and every kit module. So a module works on the fallback when the device supports the WebGPU calls it makes and the generators can express its shaders.

## Parts

| Path               | Owns                                                                                                                                                        |
| ------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `../typegpu-gl`    | The vendored `@typegpu/gl`: the WGSL-shaped AST to GLSL generator. Patches for upstream only; see `PATCHES.md`.                                             |
| `src/generator.ts` | `KitGenerator`, the class the root instantiates per pipeline. It hands a render pipeline to the GLSL generator and a compute pipeline to the CPU generator. |
| `src/glsl/`        | `KitGlslGenerator extends GlslGenerator`: what is specific to the device's conventions, and the render module format, `renderModule.ts`.                    |
| `src/cpu/`         | The CPU compute backend: `CpuGenerator` emits JavaScript for a compute pipeline, `compile` turns it into a `CpuModule` the device dispatches.               |
| `src/device/`      | The WebGPU device on WebGL 2: buffers, textures, samplers, bind groups, pipelines, encoders and passes, the queue, canvas contexts.                         |
| `src/webgl.ts`     | The public `webgl` namespace: `init`, and the fallback `web.host` takes.                                                                                    |

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

WebGL 2 has no compute stage. A compute pipeline's module is JavaScript: `CpuGenerator` emits one function per shader function over typed-array views of the bound buffers, and the device runs it on dispatch, invocation by invocation. Buffers and textures then have two homes. The device tracks which copy is current, reads back from GL when a kernel needs data a draw produced, and uploads before a draw reads what a kernel wrote.

A kernel in which each invocation writes only cells it provably owns also runs on the GPU. `KitGenerator` resolves such a pipeline twice more with `KitGlslComputeGenerator`, which emits the kernel as fragment passes, one per written buffer array or storage texture, and the CPU module carries them on a `//#kit-gl gpu` line; `parseCompute` in `src/glsl/computeModule.ts` reads it and states when the device may use it. A fragment runs the whole kernel for the invocation that owns its cell. Writes go to copies of the owned cells, reads of them check at run time whether they are the invocation's own, and every other read sees the contents from before the dispatch, so each pass renders into a new texture and the device swaps it in afterwards.

An invocation owns element `gid.x` of an array, or `gid.x + gid.y * width + gid.z * width * height` for constant sizes, texel `gid.xy` (`.xyz`) of a storage texture or every texel of the 2d-array or 3d column at `gid.xy` when the dispatch is one invocation deep, and whole small bindings when the dispatch is one invocation or the binding is written only under conditions such as `gid.x == view.cursor.x` that choose one invocation by values all invocations share; the binding's pass computes that invocation. Ownership passes into helpers through their arguments. Atomics on cells the invocation owns run as plain arithmetic on its copy; atomics on shared cells, barriers, workgroup variables and writes the generator cannot prove own keep a kernel on the CPU; the CPU module then says why on a `//#kit-gl cpu-only` line.

## What modules see

`backends.of(root)` in `@typegpu/kit` reports `'webgpu'` or `'webgl'` from a brand on the device, so a module can pick another kernel, capacity or technique where the automatic path is too slow or cannot work. `LIMITATIONS.md` lists what the fallback cannot do.
