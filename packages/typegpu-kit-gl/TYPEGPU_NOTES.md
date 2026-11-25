# Notes for TypeGPU

Each entry is a limit of TypeGPU 0.12.11 that the WebGL fallback works around. It says what happens, gives a reproduction against `typegpu` alone, names the file in this package that works around it, and names the place in TypeGPU's repository where a change would go. `PATCHES.md` holds the changes this repository already applies as diffs; an entry that a patch covers points at it.

Paths under `src/` and `tests/` are relative to this package. Paths under `packages/typegpu/src/` are relative to TypeGPU's repository. The reproductions were run against 0.12.11 with this repository's patch applied; none of them depends on what the patch changes.

## Summary

Tier 1 entries are small, additive changes. Tier 2 entries are hooks that delete the most workaround code. Tier 3 entries are larger designs or remove little.

| #   | Entry                                                     | Kind         | Patch               | Tier |
| --- | --------------------------------------------------------- | ------------ | ------------------- | ---- |
| 1   | Root's generator class must claim WGSL                    | API gap      |                     | 1    |
| 2   | Generator is not told what it generates                   | missing hook |                     | 2    |
| 3   | Render pipeline is one module for both stages             | missing hook |                     | 3    |
| 4   | Shader text is the only channel to the device             | missing hook |                     | 2    |
| 5   | Generator cannot resolve a pipeline a second way          | missing hook |                     | 2    |
| 6   | Code written as WGSL text bypasses the generator          | API gap      | core 6, partly      | 2    |
| 7   | `WgslGenerator` has few seams for another target          | missing hook | core 3 to 5, partly | 3    |
| 8   | Shader-function calls bypass `emitCall`                   | missing hook |                     | 2    |
| 9   | Atomic targets arrive as text                             | missing hook |                     | 2    |
| 10  | Barriers carry no per-function effect                     | missing hook |                     | 3    |
| 11  | Sibling blocks reuse names without stable identities      | missing hook |                     | 3    |
| 12  | One function variant per argument type only               | missing hook |                     | 2    |
| 13  | Buffer access is text before the generator sees it        | API gap      |                     | 2    |
| 14  | Ternary is emitted as `select` past `emitCall`            | missing hook | gl F, partly        | 1    |
| 15  | `determineReturnType` must be called, undocumented        | API gap      |                     | 1    |
| 16  | Generators replace context methods, read untyped members  | API gap      |                     | 3    |
| 17  | Second resolution renumbers bindings                      | missing hook |                     | 2    |
| 18  | `@invariant` position rejected by `vertexFn` types        | bug          |                     | 1    |
| 19  | Swizzle validation tied to the `Swizzle` class            | missing hook |                     | 3    |
| 20  | Local mutability decided after its declaration            | API gap      |                     | 1    |
| 21  | Generator cannot ask what a function reads                | missing hook |                     | 3    |
| 22  | Local liveness and callee effects not exposed             | missing hook |                     | 3    |
| 23  | `configureContext` assumes a WebGPU canvas context        | API gap      |                     | 3    |
| 24  | Constant namespaces read from the global scope            | bug          |                     | 1    |
| 25  | Running a kernel on the CPU needs a second generator      | API gap      |                     | 3    |
| 26  | Cube or array viewing unknown at texture creation         | API gap      |                     | 3    |
| 27  | Internal blits allocate and submit their own encoder      | performance  |                     | 3    |
| 28  | Stored local copy is still treated as a reference         | API gap      |                     | 3    |
| 29  | Bitwise and shift operators are not folded                | bug          |                     | 1    |
| 30  | Language features come from `navigator.gpu`               | missing hook |                     | 1    |
| 31  | Multi-declarator statements fail at transform             | API gap      |                     | 3    |
| 32  | Matrix `columns` typed readonly                           | API gap      |                     | 1    |
| 33  | Captured names hit `Object.prototype` methods             | bug          |                     | 1    |
| 34  | Helper generated inside its caller's statement            | bug          |                     | 1    |
| 35  | No whole-pipeline view before emitting                    | missing hook |                     | 3    |
| 36  | Calling `'use gpu'` code in JavaScript is slow on vectors | performance  |                     | 3    |
| 37  | Unrolled `for…of` loops are not visible to generators     | missing hook |                     | 1    |

Several entries want the same change, and upstreaming that change once settles each of them:

- Passing the pipeline kind and item to `initGenerator` settles notes 2 and 5.
- Routing every call, operator, atomic and barrier through `emitCall` or a typed hook settles notes 7, 8, 9, 10 and 14.
- A structured place for buffer, member and index access settles notes 9, 13 and 19.
- A per-function summary of reads, writes and barriers, built before text is emitted, settles notes 10, 21, 22 and 35.
- Options on `initFromDevice` for presenting and language features settle notes 23 and 30.
- A CPU executor with dispatch settles notes 25 and 36.

## Generators

### 1. A root's generator class must claim to generate WGSL

`tgpu.init` and `tgpu.initFromDevice` type `unstable_shaderGeneratorClass` as `WgslGeneratorClass`, whose instances must have `languageKey: 'wgsl'`. `WgslGenerator` types `languageKey` as `string`, so the option rejects TypeGPU's own generator and every subclass of it, as well as a generator for another language.

```ts
class Glsl extends WgslGenerator {
  override readonly languageKey = 'glsl';
}
// Type 'typeof Glsl' is not assignable to type
// 'ShaderGeneratorClass<ShaderGenerator & { languageKey: "wgsl"; }>'
tgpu.initFromDevice({ device, unstable_shaderGeneratorClass: Glsl });
// The same error: Type 'string' is not assignable to type '"wgsl"'
tgpu.initFromDevice({ device, unstable_shaderGeneratorClass: WgslGenerator });
// Both should compile, as `tgpu.resolve` accepts any `ShaderGenerator`
```

**Fallback.** `KitGenerator` in `src/generator.ts` declares `languageKey = 'wgsl' as const`. So `std.getTargetShaderLanguage()`, which returns `ctx.gen.languageKey`, says `'wgsl'` in shaders the fallback writes as GLSL or JavaScript.

**Upstream.** In `packages/typegpu/src/core/root/init.ts`, `InitOptions`, `InitFromDeviceOptions` and the `TgpuRootImpl` constructor: type the generator class as `ShaderGeneratorClass`, as `ExperimentalTgpuRoot['shaderGeneratorClass']` in `core/root/rootTypes.ts` already is. Additive, since the type only widens.

### 2. A generator is not told what it is generating

`ComputePipelineCore.resolveAndCreateShaderModule` and `RenderPipelineCore.resolveAndCreateShaderModule` create the generator with `new root.shaderGeneratorClass()`, and the `ResolutionCtxImpl` constructor calls `initGenerator(ctx)` with nothing about the pipeline. A generator can read the kind from `ctx.topFunctionScope?.functionType` once a function is being generated. An entry written as a WGSL template never reaches `functionDefinition`, and the calls the generator does get, `declareGlobalVar` and `emitTypeAnnotation`, come with no function scope.

```ts
class Spy extends WgslGenerator {
  #ctx: ResolutionCtx | undefined;
  override initGenerator(ctx: ResolutionCtx) {
    this.#ctx = ctx; // the only argument
    super.initGenerator(ctx);
  }
  override declareGlobalVar(options: VariableDefinitionOptions) {
    console.log(this.#ctx?.topFunctionScope?.functionType); // undefined
    return super.declareGlobalVar(options);
  }
  override functionDefinition(options: FunctionDefinitionOptions) {
    console.log('functionDefinition'); // never logged
    return super.functionDefinition(options);
  }
}
const root = tgpu.initFromDevice({ device, unstable_shaderGeneratorClass: Spy as never }); // note 1
const data = root.createMutable(d.f32);
const main = tgpu.computeFn({ workgroupSize: [1] })`{ data = 1; }`.$uses({ data });
root.unwrap(root.createComputePipeline({ compute: main }));
// Logs `undefined` once. The generator should know it builds a compute pipeline from the start.
```

**Fallback.** `KitGenerator.#use` in `src/generator.ts` picks the CPU backend when `ctx.topFunctionScope?.functionType` is `'compute'` and GLSL otherwise, and `KitGenerator.functionDefinition` corrects the choice when a compute entry arrives first.

**Upstream.** In `packages/typegpu/src/resolutionCtx.ts`, `resolve` and the `ResolutionCtxImpl` constructor: take an optional `target: { kind: 'render' | 'compute'; item: object }` in `ResolutionCtxImplOptions` and call `initGenerator(ctx, target)`. The two `resolveAndCreateShaderModule` methods in `core/pipeline/computePipeline.ts` and `core/pipeline/renderPipeline.ts` pass it. Additive, since existing generators ignore the second argument. Note 5 needs the same `item`.

### 3. A render pipeline is one module, so one generator run must emit both stages

`RenderPipelineCore.resolveAndCreateShaderModule` resolves once, calls `device.createShaderModule` once and gives the same `GPUShaderModule` to `vertex` and `fragment`. GLSL needs a source per stage, and a vertex source fails to compile when it contains a function that calls `dFdx` or uses `discard`, even if nothing calls it. `@typegpu/gl`'s `GlslGenerator` takes the stage as a constructor argument, and its own root in `tgpuRootWebGL.ts` resolves each pipeline twice; a root from `tgpu.initFromDevice` cannot.

```ts
const pipeline = root.createRenderPipeline({
  vertex: common.fullScreenTriangle,
  fragment,
  targets: { format: 'rgba8unorm' },
});
root.unwrap(pipeline);
// One createShaderModule call; the pipeline descriptor's vertex.module === fragment.module
```

**Fallback.** `KitGlslGenerator` in `src/glsl/generator.ts` tags each declaration as a `shared`, `vertex` or `fragment` chunk, and `parse` in `src/glsl/renderModule.ts` keeps per stage what that stage's `main` reaches.

**Upstream.** Either let `GlslGenerator` in `packages/typegpu-gl/src/glslGenerator.ts` emit both stages from one resolution with the chunks marked, as `KitGlslGenerator` does, which needs no core change. Or, in `packages/typegpu/src/core/pipeline/renderPipeline.ts`, `RenderPipelineCore.resolveAndCreateShaderModule`: let a generator return one module per stage through the hook of note 4. Additive either way.

### 4. The only channel from generator to device is the shader text

A device that is not WebGPU needs more than code: which GLSL sampler pairs which texture and sampler, which uniform reads which binding, the workgroup size, which bindings a kernel writes. Both pipeline cores call `device.createShaderModule({ label, code })` and nothing else, so the fallback encodes all of it as `//#` comment lines and parses them back. Group indices in those lines are right only because `resolve` in `resolutionCtx.ts` replaces the bind group placeholders in the whole string, comments included.

```ts
class Tagged extends WgslGenerator {
  override declareGlobalVar(options: VariableDefinitionOptions) {
    const result = super.declareGlobalVar(options);
    return { ...result, value: `// binding ${options.group} ${options.binding}\n${result.value}` };
  }
}
// On a root over `Tagged`, the createShaderModule descriptor has only `label` and `code`,
// and `code` contains `// binding 0 0`: the comment's `#CATCHALL#` was replaced too.
```

**Fallback.** The directive formats: `KitGlslGenerator` writes `//#chunk`, `//#buffer`, `//#block`, `//#texture` and `//#levels`, which `parse` in `src/glsl/renderModule.ts` reads; `CpuGenerator` in `src/cpu/generator.ts` writes `//#entry`, `//#binding` and `//#scope`, which `compile` in `src/cpu/index.ts` reads; and `attachComputeForm` in `src/glsl/compute.ts` appends the compute form as a `//#kit-gl gpu` line, which `gpuFormOf` and `parseCompute` in `src/glsl/computeModule.ts` read.

**Upstream.** In `packages/typegpu/src/tgsl/shaderGenerator.ts`, add an optional `ShaderGenerator.moduleData?(replace: (text: string) => string): unknown`. `resolve` in `resolutionCtx.ts` calls it with a function that applies the placeholder replacements it already records for `declarations`, and returns the result in `ResolutionResult`. Both `resolveAndCreateShaderModule` methods pass it in the `GPUShaderModuleDescriptor` under a documented key a custom device reads. Additive.

### 5. A generator cannot resolve the same pipeline a second way

The fallback wants two forms of a compute pipeline: JavaScript for the CPU, and GLSL fragment passes where the kernel allows. A root resolves each pipeline once, and the generator never receives the pipeline it is for (note 2). It captures it by replacing `ctx.resolve` on the context instance, whose first call gets the pipeline's internal `ComputePipelineCore`, then resolves that again with `tgpu.resolveWithContext`.

```ts
let item: unknown;
class Capture extends WgslGenerator {
  override initGenerator(ctx: ResolutionCtx) {
    super.initGenerator(ctx);
    const resolve = ctx.resolve.bind(ctx);
    ctx.resolve = (what, schema) => {
      item ??= what; // the first call gets the pipeline's ComputePipelineCore
      return resolve(what, schema);
    };
  }
}
// After root.unwrap(pipeline) on a root over `Capture`, a second form:
tgpu.resolveWithContext([item as never], { unstable_shaderGenerator: new WgslGenerator() });
// The generator should receive the item, not take it from a replaced method
```

The second resolution numbers the catch-all group in its own visiting order, which note 17 covers.

**Fallback.** `attachComputeForm` in `src/glsl/compute.ts` replaces `ctx.resolve` and appends the form that `computeForm` returns. `prepareGpu` in `src/device/gpuCompute.ts` checks the form's bindings against the CPU module's; on a mismatch the compute pipeline in `src/device/pipeline.ts` records the reason in `ctx.stats.fallbacks` and runs the kernel on the CPU.

**Upstream.** Same hook as note 2. With the `item` passed to `initGenerator`, the second resolution needs only the public `tgpu.resolveWithContext`. Note 17 covers the binding numbers.

### 6. Code written as WGSL text is invisible to other generators

A function implemented with a WGSL template goes into the output as written, without `functionDefinition`. Three places in TypeGPU 0.12.11 do this:

- `createGuardedComputePipeline`'s entry, in `core/root/init.ts`. Its source comment says it is WGSL because TypeGPU does not run `unplugin-typegpu` on its own package.
- `common.fullScreenTriangle`, in `common/fullScreenTriangle.ts`.
- The mipmap and resampling shaders in `core/texture/textureUtils.ts`, which `texture.generateMipmaps()` and a resizing `texture.write()` compile with `device.createShaderModule` directly, bypassing the root's generator.

```ts
class Marked extends WgslGenerator {
  override functionDefinition(options: FunctionDefinitionOptions) {
    return `/* generated */ ${super.functionDefinition(options)}`;
  }
}
// On a root over `Marked`, a pipeline with `vertex: common.fullScreenTriangle` gets a module
// with `/* generated */ @fragment fn fragment(...)`, but the vertex function as written:
// `@vertex fn item(@builtin(vertex_index) vertexIndex: u32) -> _Output { ... }`.
// Every function should reach the root's generator.
```

**Fallback.** `KitGlslGenerator.#foreign` in `src/glsl/generator.ts` recognises `fullScreenTriangle`'s WGSL through a replaced `ctx.addDeclaration` and substitutes GLSL. `builtin` in `src/device/shader.ts` recognises the mipmap shaders by fragments of their WGSL text (`isBlitVertex`, `SAMPLE_FRAGMENT`, `GATHER_FRAGMENT`) and substitutes GLSL programs of its own, so an edit to those strings upstream breaks mipmaps on the fallback without an error.

**Patched.** The guarded entry is covered by PATCHES.md, core patch 6.

**Upstream.** In `packages/typegpu/src/common/fullScreenTriangle.ts`, write `fullScreenTriangle` as a `'use gpu'` function. In `packages/typegpu/src/core/texture/textureUtils.ts`, `generateTextureMipmaps` and `writeImages`: take the texture's root instead of its device and build the blit as a TypeGPU render pipeline, so it goes through the root's generator. Building TypeGPU with `unplugin-typegpu` turns all three into `'use gpu'` source, as patch 6 does by hand. Additive, since the WGSL output stays equivalent.

### 7. `WgslGenerator` has few seams for another target

A subclass retargets well where a hook exists: `emitCall`, `emitBinaryOp`, `typeInstantiation`, `numericLiteral`, `functionDefinition`, `declareGlobalVar`, `call`, and the protected `_emitVarDecl`, `_return`, `_block`, `_callShellless`, `_aliasConstStatement` and `_emitSwitchStatement`. Elsewhere it overrides `_expression` or `_statement` and matches tinyest nodes itself. In 0.12.11 these bypass every hook:

- unary `-`, through `neg` in `std/operators.ts`, which writes `-(x)` with `stitch`, and `!` and `~`, written inline in `_expressionImpl`;
- `i++` and `i--`, written as `${argStr}${op}` in `_statement`;
- member and index access, and so the left side of an assignment, which reaches `emitBinaryOp` as text (note 13);
- ternaries (note 14), calls to user functions (note 8), atomics (note 9) and barriers (note 10).

```ts
// `probe`, here and in notes 8 to 13, is a `WgslGenerator` subclass that records its hook calls
const f = tgpu.fn(
  [d.f32],
  d.f32,
)((x) => {
  'use gpu';
  let i = d.i32(0);
  i++;
  return -x + d.f32(i);
});
tgpu.resolve([f], { unstable_shaderGenerator: probe });
// `i++;` and `-(x)` are in the output; only `+` reached `emitBinaryOp`, nothing reached `emitCall`
```

**Fallback.** `src/cpu/generator.ts` overrides `CpuGenerator._expression` for unary minus on vectors and `i32`, member access and swizzles, and `CpuGenerator._statement` to rewrite `i++` as `+=`. `packages/typegpu-gl/src/glslGenerator.ts` overrides `_expression` for ternaries and `_return` for entry outputs.

**Patched.** Texture builtins, plain `name(args)` builtins and `for…of` declarations: PATCHES.md, core patches 3 to 5.

**Upstream.** In `packages/typegpu/src/std/operators.ts`, `neg`: emit through a hook, such as `ctx.gen.emitUnaryOp('-', arg)`, and use the same hook for `!` and `~` in `_expressionImpl` and for update statements in `_statement`, both in `packages/typegpu/src/tgsl/wgslGenerator.ts`. Additive: WGSL output stays the same. The other gaps have their own entries: notes 8, 9, 10, 13 and 14.

### 8. Shader-function calls bypass `emitCall`

Calls to a `tgpu.fn` are written by its `codegenImpl` in `core/function/tgpuFn.ts` as ``stitch`${name}(${args})` ``, and calls to shellless functions the same way in `WgslGenerator._callShellless`. Overriding `emitCall` changes builtins but no user call. A backend that passes a vector as separate numeric parameters must change both the definition and every call.

```ts
const twice = tgpu.fn([d.vec2f], d.vec2f)((v) => std.mul(v, 2));
const half = (v: d.v2f) => {
  'use gpu';
  return std.mul(v, 0.5);
};
const main = tgpu.fn(
  [],
  d.vec2f,
)(() => {
  'use gpu';
  return half(twice(d.vec2f(1, 2)));
});
tgpu.resolve([main], { unstable_shaderGenerator: probe });
// Output: `half(twice(vec2f(1, 2)))`; `emitCall` saw neither `twice` nor `half`
```

**Fallback.** `CpuGenerator.call` in `src/cpu/generator.ts` lets the base generator write the call, reads the callee's name from the text, splits the arguments again with `callArguments` from `src/cpu/references.ts` and re-emits the call through `emitCall`.

**Upstream.** In `packages/typegpu/src/core/function/tgpuFn.ts`, both `codegenImpl`s, and in `packages/typegpu/src/tgsl/wgslGenerator.ts`, `_callShellless`: return `ctx.gen.emitCall(name, [], convertedArgs)` with the converted argument snippets still typed. Additive: `WgslGenerator.emitCall` writes the same text.

### 9. Atomic references arrive as text

The `call` hook receives an atomic's target as a typed snippet whose value is already text: `rows[i].count`, or `(*row).count` when the row was bound to a `const` first, since that `const` becomes an implicit pointer. The atomics in `std/atomic.ts` then write `` stitch`atomicAdd(&${target}, …)` `` and never reach `emitCall`; core patch 4 leaves them alone. A backend that stores buffers as typed arrays needs the array and the index separately to update one field without touching its neighbours.

```ts
const Row = d.struct({ count: d.atomic(d.u32), other: d.u32 });
const layout = tgpu.bindGroupLayout({ rows: { storage: d.arrayOf(Row, 16), access: 'mutable' } });
const f = tgpu.fn([d.u32])((i) => {
  'use gpu';
  const row = layout.$.rows[i];
  std.atomicAdd(row.count, 1);
});
tgpu.resolve([f], { unstable_shaderGenerator: probe });
// `call` receives the target as the string '(*row).count'; `emitCall` is never called
```

**Fallback.** `CpuGenerator.call` in `src/cpu/generator.ts` handles every `atomic*` before the base generator: `lowerReferences` rewrites `(*row)` and `(&x)` text, and `addressParts` in `src/cpu/references.ts` splits the last `[i]` or `.field` off the target.

**Upstream.** In `packages/typegpu/src/std/atomic.ts`, each atomic's `codegenImpl`: emit through `ctx.gen.emitCall(name, [], [target, …])`, with the target as a typed pointer snippet. Additive. Getting the array and index back out of that pointer needs the structured place of note 13: same hook as note 13.

### 10. Barrier suspension and its call graph are not generator metadata

`workgroupBarrier`, `storageBarrier` and `textureBarrier` in `std/atomic.ts` return the text `workgroupBarrier()` without `emitCall`. The `call` hook sees `std.workgroupBarrier` itself, but nothing tells a generator that a function contains a barrier, or that a call reaches one through a helper. A CPU backend must suspend every lane at each barrier, including those in helpers.

```ts
const sync = () => {
  'use gpu';
  std.workgroupBarrier();
};
const kernel = tgpu.computeFn({ workgroupSize: [64] })(() => {
  'use gpu';
  sync();
});
tgpu.resolve([kernel], { unstable_shaderGenerator: probe });
// `emitCall` never sees the barrier; the entry's `sync()` call carries no mark that it suspends
```

**Fallback.** `CpuGenerator.functionDefinition` in `src/cpu/generator.ts` replaces barrier text with `(yield)` by regular expression, records the function in `#generators`, and rewrites calls to those functions as `(yield* name(…))`. `expandBarrierHelpers` in `src/cpu/phase-helpers.ts` inlines straight-line scalar helpers, `splitPhases` in `src/cpu/phases.ts` splits unconditional entry barriers into phase functions, and `src/cpu/index.ts` runs the phases across lanes. Other barriers keep JavaScript generator functions.

**Upstream.** In `packages/typegpu/src/std/atomic.ts`, the three barriers: emit through `ctx.gen.emitCall`. Additive. Then record per generated function whether it reaches a barrier, directly or through its callees, and pass it to `functionDefinition` and with each call's snippet, in `packages/typegpu/src/core/function/fnCore.ts` and the function scope in `packages/typegpu/src/resolutionCtx.ts`. Additive. Same per-function summary as notes 21 and 22.

### 11. Local names are reused across sibling blocks, without stable identities

`makeUniqueIdentifier(name, 'block')` returns the JavaScript name unchanged when no visible scope has it, so sibling blocks declare the same identifier. That is valid WGSL, but a backend that keeps facts per emitted name, such as a local's buffer address or its vector components, carries them into the sibling block. `_block` is overridable and wraps every block body, but loop headers push their own scope in `_statement`, and a variable has no identity beyond its printed name.

```ts
const layout = tgpu.bindGroupLayout({
  rows: { storage: d.arrayOf(d.vec4f, 16), access: 'readonly' },
});
const f = tgpu.fn(
  [d.u32, d.u32, d.bool],
  d.f32,
)((i, j, first) => {
  'use gpu';
  let sum = d.f32(0);
  if (first) {
    const item = layout.$.rows[i];
    sum += item.x;
  } else {
    const item = layout.$.rows[j];
    sum += item.y;
  }
  return sum;
});
tgpu.resolve([f]);
// Both blocks declare `let item = (&rows[…]);`: two bindings, one identifier
```

**Fallback.** `src/cpu/generator.ts` names its own temporaries with `ctx.makeUniqueIdentifier(…, 'global')` and resets `#addresses` and `#vectors` at each `functionDefinition`, but still keys user locals by name. `KitGlslComputeGenerator._emitVarDecl` in `src/glsl/compute.ts` forgets what it knew about a name whenever a declaration reuses it. `ctx.getById` does not help: it takes the JavaScript identifier, not the emitted one.

**Upstream.** In `packages/typegpu/src/tgsl/wgslGenerator.ts`, `_statement`: open the loop-header scopes of `for`, `while` and `for…of` through one protected method, so a subclass sees every scope enter and exit. Additive. Then give each local a stable identity that reaches `_emitVarDecl`, `_aliasConstStatement` and the snippets that read it, in `packages/typegpu/src/resolutionCtx.ts` `defineVariable`. Same identities as note 22.

### 12. A function is generated once per argument type, never per argument value

`ShelllessRepository.get` in `tgsl/shellless.ts` keys variants by argument types, and a `tgpu.fn` with a shell is generated once per resolution for all callers. GLSL needs a body per call site in two cases. A texture or sampler argument needs one, because GLSL samples through one combined uniform, so `blurred(a, s, uv)` and `blurred(b, s, uv)` need two bodies. An argument that is the invocation's own index needs one, because a helper writing `buf[i]` runs as a fragment pass only for calls that pass the own index. Tagging the argument's schema does not make a variant: `schemasEqualNominally` compares textures by sample type and samplers by kind alone.

```ts
const layout = tgpu.bindGroupLayout({
  a: { texture: d.texture2d() },
  b: { texture: d.texture2d() },
  s: { sampler: 'filtering' },
});
const blurred = (map: d.texture2d<d.F32>, s: d.sampler, uv: d.v2f) => {
  'use gpu';
  return std.textureSample(map, s, uv);
};
const main = tgpu.fn(
  [d.vec2f],
  d.vec4f,
)((uv) => {
  'use gpu';
  return std.add(blurred(layout.$.a, layout.$.s, uv), blurred(layout.$.b, layout.$.s, uv));
});
tgpu.resolve([main]);
// One `fn blurred`, called with `a` and with `b`
```

**Fallback.** For texture and sampler arguments of shellless functions, `KitGlslGenerator._callShellless` in `src/glsl/generator.ts` swaps `ctx.shelllessRepo.cache` for a list keyed by the handles while the lookup runs. For own-index arguments, of shellless and shelled functions alike, `KitGlslComputeGenerator.#calling` in `src/glsl/compute.ts` keeps the body generated for the first call's own arguments and rejects the kernel when a later call passes others.

**Upstream.** In `packages/typegpu/src/tgsl/shellless.ts`, `ShelllessRepository.get`: ask the generator for an extra key per argument, such as `variantKey(arg: Snippet): string | undefined`, and match it with the types. For shelled functions, in `packages/typegpu/src/core/function/tgpuFn.ts`, `codegenImpl`: resolve a separate body per distinct key. Additive: `WgslGenerator` returns `undefined` and keeps one variant.

### 13. Buffer access becomes text before a generator sees what it reads

`accessProp` and `accessIndex` in `tgsl/accessProp.ts` and `tgsl/accessIndex.ts` write member and index access with `stitch`, so `layout.$.items[i].weight` reaches the generator as the string `items[i].weight`. A backend that reads buffers another way, such as words from a data texture, cannot load one field without first parsing the text. Overriding `_expression` catches access in the AST, but access on a value proxy goes through `valueProxyHandler` in `core/valueProxyUtils.ts`, which calls the same functions. A generator cannot keep its own object as a snippet's value either: `asComptime` in `types.ts` treats an object without `[$comptimeValueOf]` as known at compile time, so a comparison with it throws "requires numeric operands", and neither `$comptimeValueOf` nor `runtimeOnly` is exported.

```ts
const Item = d.struct({ position: d.vec3f, weight: d.f32 });
const layout = tgpu.bindGroupLayout({
  items: { storage: d.arrayOf(Item, 16), access: 'readonly' },
});
const f = tgpu.fn(
  [d.u32],
  d.f32,
)((i) => {
  'use gpu';
  return layout.$.items[i].weight + 1;
});
tgpu.resolve([f], { unstable_shaderGenerator: probe });
// `emitBinaryOp` receives the left side as the string 'items[i].weight'
```

**Fallback.** `KitGlslGenerator._expression` in `src/glsl/generator.ts` narrows places itself and keeps each place's load expression as the snippet's value, with `#places` mapping that text back to the place. `CpuGenerator._expression` in `src/cpu/generator.ts` recovers buffer offsets from the text with regular expressions and keeps them in `#addresses`.

**Upstream.** In `packages/typegpu/src/tgsl/accessProp.ts` and `packages/typegpu/src/tgsl/accessIndex.ts`: represent buffer, member and index access as a place the generator turns into code, such as `emitLoad(place)`, with the WGSL generator writing today's text. Additive. Export `runtimeOnly` and `$comptimeValueOf` from `typegpu/~internal`, in `packages/typegpu/src/internal.ts`. Additive. Atomics (note 9), assignment targets (note 7) and swizzle provenance (note 19) need the same place.

### 14. A ternary is written as WGSL's `select`

`WgslGenerator._expressionImpl` writes a runtime ternary as `select(f, t, c)` with `stitch`, so it never reaches `emitCall`. `std.select` does go through `emitCall('select', …)`. GLSL and JavaScript have no `select`.

```ts
class Recording extends WgslGenerator {
  calls: string[] = [];
  override emitCall(name: string, t: readonly Snippet[], a: readonly Snippet[]): string {
    this.calls.push(name);
    return super.emitCall(name, t, a);
  }
}
const f = tgpu.fn(
  [d.f32],
  d.f32,
)((x) => {
  'use gpu';
  return x < 0.5 ? 1 : 2;
});
const gen = new Recording();
tgpu.resolveWithContext([f], { unstable_shaderGenerator: gen });
// Code has `select(2i, 1i, (x < 0.5f))`, but `gen.calls` has no 'select'.
// It should, as it does for `std.select(d.f32(2), d.f32(1), x < 0.5)`.
```

**Fallback.** `GlslGenerator._expression` in `packages/typegpu-gl/src/glslGenerator.ts` declares `SELECT_OVERLOADS` when a ternary's text starts with `select(`. `CpuGenerator._expression` in `src/cpu/generator.ts` parses the `select(…)` text back into three arguments and passes them to `emitCall('select', …)`.

**Patched.** The GLSL half is `@typegpu/gl` fork patch F in PATCHES.md. The CPU half is not patched.

**Upstream.** In `packages/typegpu/src/tgsl/wgslGenerator.ts`, `_expressionImpl`, the `NODE.conditionalExpr` branch: return `this.emitCall('select', [], [alt, con, convertedTest])` in place of the `stitch`. The WGSL output stays the same. Additive for WGSL; a generator that overrides `emitCall('select')` now also sees ternaries.

### 15. `functionDefinition` must call `determineReturnType`, which nothing says

`ResolutionCtxImpl.resolveFunction` sets the return type only inside the `determineReturnType` callback it passes to `gen.functionDefinition`. A generator that never calls it fails afterwards with `Failed to determine return type`, which does not name the cause. `FunctionDefinitionOptions.determineReturnType` has no doc comment.

```ts
class OwnEntry extends WgslGenerator {
  override functionDefinition(options: FunctionDefinitionOptions): string {
    return `fn ${options.name}() {}`;
  }
}
const main = tgpu.computeFn({ workgroupSize: [1] })(() => {
  'use gpu';
});
tgpu.resolveWithContext([main], { unstable_shaderGenerator: new OwnEntry() });
// Throws 'Failed to determine return type'; should work or say to call determineReturnType()
```

**Fallback.** Every generator that writes its own function calls it: `KitGlslComputeGenerator.computeEntry` in `src/glsl/compute.ts`, `KitGlslGenerator.#entryDefinition` in `src/glsl/generator.ts` and `CpuGenerator.functionDefinition` in `src/cpu/generator.ts`.

**Upstream.** In `packages/typegpu/src/resolutionCtx.ts`, `resolveFunction`: when `returnType` is still unset after `functionDefinition` returns, call the same callback instead of throwing. Document on `FunctionDefinitionOptions.determineReturnType` in `packages/typegpu/src/tgsl/shaderGenerator.ts` that it must run after the body is generated. Additive.

### 16. Generators replace `ResolutionCtx` methods and read members its type lacks

The exported `ResolutionCtx` type in `typegpu/~internal` already has `topFunctionScope`, `shelllessRepo`, `setBlockExternals`, `addDeclaration` and `resolve`. It lacks `fixedBindings` and `bindGroupLayoutsToPlaceholderMap`, which only `ResolutionCtxImpl` has. The generators also replace `ctx.resolve`, `ctx.addDeclaration` and `ctx.shelllessRepo.cache` on the instance, which nothing promises TypeGPU will keep calling. And they bind a generated snippet to a temporary external with `setBlockExternals` so that `_expression` can continue a member or index access from it, because `accessProp` and `accessIndex` are not exported.

```ts
import type { ResolutionCtx } from 'typegpu/~internal';
declare const ctx: ResolutionCtx;
ctx.topFunctionScope?.functionType; // typed
ctx.shelllessRepo.cache; // typed
ctx.fixedBindings; // TS2339: not on ResolutionCtx
ctx.bindGroupLayoutsToPlaceholderMap; // TS2339: not on ResolutionCtx
```

**Fallback.** `#numbered` in `src/glsl/compute.ts` casts the context to read the two binding members (note 17). `attachComputeForm` in `src/glsl/compute.ts` replaces `ctx.resolve` to capture the pipeline (note 5). `KitGlslGenerator.initGenerator` in `src/glsl/generator.ts` replaces `ctx.addDeclaration` to translate WGSL-text declarations (note 6). `KitGlslGenerator._callShellless` swaps `shelllessRepo.cache` (note 12). `KitGlslGenerator.#bind` in `src/glsl/generator.ts` calls `setBlockExternals` and re-enters `_expression` with a synthetic AST node.

**Upstream.** In `packages/typegpu/src/internal.ts`, export `accessProp` (`tgsl/accessProp.ts`) and `accessIndex` (`tgsl/accessIndex.ts`), so a generator continues an access from a snippet without `#bind`. Additive. Same hooks as notes 5, 6, 12 and 17 for the rest.

### 17. A second resolution of a pipeline numbers its bindings its own way

`ResolutionCtxImpl.allocateFixedEntry` numbers the catch-all group's bindings in the order a resolution first reaches each resource. `allocateLayoutEntry` gives layouts without an explicit index placeholders in the order they are reached, and `resolve` turns those into group indices in that order. Resolving the same pipeline again in another order gives other numbers.

```ts
const a = root.createReadonly(d.arrayOf(d.f32, 4));
const b = root.createMutable(d.arrayOf(d.f32, 4));
const ab = tgpu.computeFn({ workgroupSize: [1] })(() => {
  'use gpu';
  const x = a.$[0];
  b.$[0] = x;
});
const ba = tgpu.computeFn({ workgroupSize: [1] })(() => {
  'use gpu';
  const y = b.$[1];
  b.$[0] = a.$[0] + y;
});
tgpu.resolve([ab]); // `a` at @binding(0)
tgpu.resolve([ba]); // `a` at @binding(1): numbers follow first use, not the resources
```

**Fallback.** The GPU form of a compute pipeline resolves it again with `KitGlslComputeGenerator`. `KitGlslComputeGenerator.#numbered` in `src/glsl/compute.ts` maps each binding to the outer resolution's catch-all index or layout placeholder through `fixedBindings` and `bindGroupLayoutsToPlaceholderMap`, and writes `#KIT_OUTER_` placeholders that the outer resolution replaces.

**Upstream.** In `packages/typegpu/src/core/resolve/tgpuResolve.ts`, `resolveWithContext`: accept the parent `ResolutionCtx` as an option, and in `packages/typegpu/src/resolutionCtx.ts`, `allocateFixedEntry` and `allocateLayoutEntry`, return the parent's binding and placeholder for a resource or layout it already has. Additive. The smaller step is adding `readonly fixedBindings` and `readonly bindGroupLayoutsToPlaceholderMap` to `ResolutionCtx` in `packages/typegpu/src/types.ts`, which makes `#numbered` typed but keeps it. Note 5 asks for the same.

### 18. A vertex output position cannot be typed `@invariant`

`d.invariant(d.builtin.position)` returns `Decorated<Vec4f, [Builtin<'position'>, Invariant]>`, and resolution emits `@invariant @builtin(position)` for it. `tgpu.vertexFn`'s `out` type does not accept it, because `AnyVertexOutputBuiltin` has only `BuiltinPosition` and `BuiltinClipDistances`.

```ts
tgpu.vertexFn({ out: { position: d.builtin.position } }); // accepted
tgpu.vertexFn({ out: { position: d.invariant(d.builtin.position) } });
// TS2769: not assignable to VertexOutConstrained; should be accepted
```

**Fallback.** `packages/typegpu-kit/src/passes.ts` casts `tgpu.vertexFn` to its own `Shell` type for `vertexShell`, which `vertexOf` calls with the invariant position. `tests/glsl/stages.test.ts` casts the schema through `unknown`.

**Upstream.** In `packages/typegpu/src/builtin.ts`, add `Decorated<Vec4f, [Builtin<'position'>, Invariant]>` to `AnyVertexOutputBuiltin`, which `VertexOutConstrained` in `packages/typegpu/src/core/function/tgpuVertexFn.ts` reads. Additive.

### 19. Swizzle mutation validation depends on the emitted value's class

`WgslGenerator.validateSnippetMutation` is private and checks `mutated.value instanceof Swizzle`. For a swizzle it validates the base's origin; for anything else, the snippet's own. `Swizzle[$resolve]` in `tgsl/swizzle.ts` writes `base.components` itself. A generator that emits swizzles in another form must replace the snippet's value, and then either valid writes or valid copies fail. `accessIndex` also makes a `Swizzle` for a constant vector index, so `v[0]` takes the same path.

```ts
// A generator whose `_expression` returns a Swizzle's text as a string snippet
const write = tgpu.fn([])(() => {
  'use gpu';
  const value = d.vec4f(1, 2, 3, 4);
  value.yx = d.vec2f(value.xy); // with the swizzle's origin: "not a reference to an existing value"
});
const copy = tgpu.fn([])(() => {
  'use gpu';
  let value = d.vec4f(1, 2, 3, 4);
  value = value.yxwz; // with the base's origin: "references cannot be assigned"
});
// Both resolve with the plain WgslGenerator; both should with any emitted form
```

**Fallback.** `CpuGenerator.#lowerSwizzle` in `src/cpu/generator.ts` clones the `Swizzle` instance and uses `makeResolvable` to change only its emitted code, for member and index access alike.

**Upstream.** In `packages/typegpu/src/tgsl/swizzle.ts`, `Swizzle[$resolve]`: emit through a new optional `ShaderGenerator.emitSwizzle(base: Snippet, components: string): string` in `packages/typegpu/src/tgsl/shaderGenerator.ts`, defaulting to `${base}.${components}`. The value stays a `Swizzle`, so validation is unchanged. Additive.

### 20. Whether a local is reassigned is decided after its declaration is generated

`WgslGenerator._letStatement`, and `_constStatement` for a new struct or array value such as `View(layout.$.view)`, pass `_emitVarDecl` a placeholder keyword `#VAR_n#`. `WgslGenerator.functionDefinition` replaces it with `var` or `let` after the body, from `FunctionScopeLayer.modifiedVariables`. That set is readable through `ctx.topFunctionScope` once the body is generated, but not at the declaration, and its contract is undocumented.

```ts
const f = tgpu.fn(
  [d.u32],
  d.u32,
)((x) => {
  'use gpu';
  let a = x;
  let b = x;
  if (x > 2) {
    b = 1;
  }
  return a + b;
});
// `_emitVarDecl` gets '#VAR_0#' and '#VAR_1#'. After the body,
// `ctx.topFunctionScope.modifiedVariables` holds `b` only, and the code has `let a` and `var b`.
```

**Fallback.** `KitGlslComputeGenerator.#checkWrite` in `src/glsl/compute.ts` records the root name of every assignment and post-update target in `frame.assigned`, and `KitGlslComputeGenerator.functionDefinition` rejects the kernel when a name in `frame.choosing` was assigned. It decides at the function's end, so it could read `ctx.topFunctionScope.modifiedVariables` instead.

**Upstream.** In `packages/typegpu/src/types.ts`, document `FunctionScopeLayer.modifiedVariables` and `placeholderForVariable` as what a generator reads after `_block` returns to learn each local's final mutability. Additive. A generator that needs it at the declaration cannot have it without a second pass.

### 21. A generator cannot ask what a function reads

An accessor filled with a function reaches a kernel as a call to a generated function, and `FunctionDefinitionOptions` gives a generator only the function's type, name, workgroup size, arguments and body. `ResolutionCtxImpl._getOrInstantiate` keeps a `ProbeFrame` per resolved item, but it is private, and its only probe, `bindingUseProbe`, records fixed bindings from `allocateFixedEntry`; `allocateLayoutEntry`, private variables and builtins report nothing.

```ts
const Pass = d.struct({ source: d.u32 });
const layout = tgpu.bindGroupLayout({
  pass: { uniform: Pass },
  cells: { storage: d.arrayOf(d.f32, 8), access: 'mutable' },
});
const current = tgpu.accessor(Pass, () => {
  'use gpu';
  return Pass(layout.$.pass);
});
const main = tgpu.computeFn({ workgroupSize: [1], in: { gid: d.builtin.globalInvocationId } })((
  input,
) => {
  'use gpu';
  layout.$.cells[input.gid.x + (1 - current.$.source) * 4] = 1;
});
tgpu.resolve([main], { unstable_shaderGenerator: new Spy() }); // Spy extends WgslGenerator
// Emits `cells[(gid.x + ((1u - current().source) * 4u))]`. `Spy.functionDefinition` gets the same
// six fields for `current` as for `main`: nothing says `current` reads only a uniform.
```

**Fallback.** `KitGlslComputeGenerator.functionDefinition` in `src/glsl/compute.ts` adds a function to `#pure` when it has no parameters and `#varies` finds in its generated GLSL no invocation id, private variable, written buffer or texture, or function outside `#pure`.

**Upstream.** In `packages/typegpu/src/resolutionCtx.ts`, report a probe from `allocateLayoutEntry`, from private and workgroup variable access in `packages/typegpu/src/core/variable/tgpuVariable.ts`, and from builtin access, and let a generator read the probe set of a resolved callee, for example `ctx.probesOf(item)`. Additive. One per-function summary serves notes 10, 21 and 22.

### 22. Local liveness and callee effects are not exposed together

`FunctionArgument.used` tells a generator whether a body read an argument, but locals have no equivalent: `_emitVarDecl` gets strings at the declaration, before any use. Every call to a `'use gpu'` function has `possibleSideEffects: true` (`sideEffects: true` in `core/function/tgpuFn.ts`, the default of `snip` for shellless calls in `tgsl/wgslGenerator.ts`), so a generator cannot tell that dropping an unused initializer is safe. Record arguments and returns carry no field use either.

```ts
const double = (x: number) => {
  'use gpu';
  return x * 2;
};
const f = (x: number) => {
  'use gpu';
  const unused = double(x);
  return x;
};
tgpu.resolve([tgpu.fn([d.f32], d.f32)((x) => f(x))]);
// Emits `let unused = double(x);`. Nothing tells a generator that `unused` is never read and
// that `double` has no effects, so it cannot drop the call.
```

**Fallback.** `CpuGenerator` in `src/cpu/generator.ts` keeps `#functionEffects` per generated function. `prunePureLocals` in `src/cpu/dead-locals.ts` drops unused declarations whose initializer `isPureExpression` accepts, and `usedRecordFields` and `projectRecordBody` in `src/cpu/record-fields.ts` project pure helpers onto the record fields their callers read.

**Upstream.** In `packages/typegpu/src/resolutionCtx.ts`, count reads per variable that `defineVariable` registers and expose the counts when a function body is done. In `packages/typegpu/src/tgsl/wgslGenerator.ts`, set a user function call's `possibleSideEffects` from its callee's effect summary instead of `true`. Additive: the WGSL output does not change. Same per-function summary as note 21; note 10 adds reachable barriers to it.

## Roots and devices

### 23. `root.configureContext` assumes a WebGPU canvas context

`configureContext` in `core/root/init.ts` calls `canvas.getContext('webgpu')`, then `navigator.gpu.getPreferredCanvasFormat()` when no `format` is given. Its options take a canvas only, never a context, so a root over a device that presents another way cannot use it. The default target format in `core/pipeline/connectTargetsToShader.ts` reads `navigator.gpu` too.

```ts
const root = tgpu.initFromDevice({ device: webglDevice });
root.configureContext({ canvas, format: 'rgba8unorm' });
// Throws "Unable to initialize 'webgpu' context on the provided canvas."
// It should ask the device's presenter for the context.
```

**Fallback.** `web.surface` in `packages/typegpu-kit/src/web.ts` calls `backends.of(root).configure` from `packages/typegpu-kit/src/backends.ts` instead of `root.configureContext` when the backend is not WebGPU. The fallback device supplies `configure` and `preferredCanvasFormat` under the `brand` symbol in `src/device/index.ts`, through `configure` in `src/device/canvas.ts`.

**Upstream.** In `packages/typegpu/src/core/root/init.ts`, accept an optional presenter in `initFromDevice`'s options, `{ getContext(canvas): GPUCanvasContext | null; preferredFormat: GPUTextureFormat }`, and use it in `configureContext` and in `connectTargetsToShader` in place of `canvas.getContext('webgpu')` and `navigator.gpu.getPreferredCanvasFormat()`. Additive. It extends the same `initFromDevice` options as note 30.

### 24. TypeGPU reads WebGPU's constant namespaces from the global scope

`GPUBufferUsage`, `GPUTextureUsage`, `GPUShaderStage` and `GPUMapMode` are read as globals when a buffer, texture, query set or layout is created. A browser without WebGPU does not define them, so creating a uniform throws before any device is involved.

```ts
tgpu.initFromDevice({ device: webglDevice }).createUniform(d.f32);
// ReferenceError: GPUBufferUsage is not defined. It should need no global.
```

**Fallback.** `defineConstants` in `src/globals.ts`, called by `init` in `src/webgl.ts`, defines the missing namespaces, and `GPUColorWrite` for user code.

**Upstream.** Keep the constants in a TypeGPU module, since the WebGPU specification fixes their values, and import them in `packages/typegpu/src/core/buffer/buffer.ts`, `packages/typegpu/src/core/texture/texture.ts`, `packages/typegpu/src/core/texture/textureUtils.ts`, `packages/typegpu/src/core/querySet/querySet.ts` and `packages/typegpu/src/tgpuBindGroupLayout.ts`. Additive.

### 25. Running a kernel on the CPU needs a second code generator

Calling a `'use gpu'` function in JavaScript runs one invocation without its kernel. `tgpu['~unstable'].simulate` fixes the dispatch at one workgroup of one invocation. `workgroupBarrier`, `storageBarrier` and the atomics throw `MissingCpuImplError`, and so do the sampling, load and store texture functions. A slot or accessor read outside resolution throws in 0.12.11; [TypeGPU #3220](https://github.com/software-mansion/TypeGPU/pull/3220), unmerged, adds them to CPU execution.

```ts
const counter = tgpu.workgroupVar(d.atomic(d.u32));
const barrier = () => {
  'use gpu';
  std.workgroupBarrier();
};
barrier(); // MissingCpuImplError: workgroupBarrier is a no-op outside of CODEGEN mode.
tgpu['~unstable'].simulate(() => std.atomicAdd(counter.$, 1)); // MissingCpuImplError
tgpu['~unstable'].simulate(() => 1).privateVars.flat(2).length; // 1: one invocation, no dispatch
// A kernel should run on the CPU over a dispatch, with its barriers and atomics.
```

**Fallback.** `CpuGenerator` in `src/cpu/generator.ts` generates JavaScript from the typed AST, and `compile` in `src/cpu/index.ts` runs it over the dispatch, with atomics in the generator and texture access in `src/cpu/textures.ts`.

**Upstream.** In `packages/typegpu/src/core/simulate/tgpuSimulate.ts`, `simulate`: a CPU executor that takes an entry, bound resources and dispatch dimensions. Additive. Same executor as note 36, which gives its design; barriers need note 10.

### 26. A texture's cube or array viewing is unknown when it is created

WebGPU lets a 6-layer texture be viewed as `cube` or `2d-array`, but WebGL 2 picks `TEXTURE_CUBE_MAP` or `TEXTURE_2D_ARRAY` when it allocates storage. `GPUTextureDescriptor.textureBindingViewDimension` carries that choice, but TypeGPU's texture props have no such field. `materialize` in `TgpuTextureImpl` creates the texture from `props` alone, and a view reaches `GPUTexture.createView` only when a bind group first unwraps it.

```ts
const sky = root
  .createTexture({ size: [256, 256, 6], format: 'rgba16float' })
  .$usage('sampled', 'render');
sky.write(faces); // device.createTexture now, without textureBindingViewDimension
const cube = sky.createView(d.textureCube(d.f32)); // reaches the device only at the first bind
// createTexture should receive textureBindingViewDimension: 'cube'.
```

**Fallback.** `src/device/texture.ts` stores the texture as the views created before allocation ask, through `hint` in `makeView` and `chooseTarget`. `sampled` reallocates it if the first sampled view disagrees and nothing was written yet, and otherwise `derivedCopy` keeps a copy per view that it refreshes after each write. `createTexture` there already reads `textureBindingViewDimension`.

**Upstream.** In `packages/typegpu/src/core/texture/textureProps.ts`, add `textureBindingViewDimension` to `TextureProps`, and pass it to `device.createTexture` in `materialize` in `packages/typegpu/src/core/texture/texture.ts`. Additive.

### 27. TypeGPU's internal blits allocate per call and submit their own encoder

`blit` in `core/texture/textureUtils.ts` caches its pipeline but creates a bind group and two texture views on every call. `generateTextureMipmaps`, `writeImages` and `clearTextureWithColor` each create a command encoder and submit it at once, as does `copyFrom` in `core/texture/texture.ts`. A module that regenerates mips every frame therefore allocates each frame and submits outside the frame's encoder, so its work cannot be ordered with the frame's passes.

```ts
const target = root
  .createTexture({ size: [8, 8], format: 'rgba8unorm', mipLevelCount: 4 })
  .$usage('sampled', 'render');
target.generateMipmaps();
// 3 bind groups, 6 texture views, 1 command encoder and 1 submit, again on every call.
// It should reuse them and record into an encoder it is given.
```

**Fallback.** The fallback device runs each submit as it comes and replaces the blit shaders, as note 6 says. The Kit's `mipmaps` module in `packages/typegpu-kit/src/mipmaps.ts` records mip levels into the frame's encoder instead, and `images` in `packages/typegpu-kit/src/images.ts` calls `generateMipmaps` only at setup.

**Upstream.** In `packages/typegpu/src/core/texture/textureUtils.ts`, `blit`: cache the bind group and views per source view. In `packages/typegpu/src/core/texture/texture.ts`, `generateMipmaps`: take an optional command encoder to record into, and submit only without one. Additive.

### 28. A copy bound to a local is still a reference when stored into an array

`_binaryOrAssignmentExpression` rejects `x = b` whenever `b` names a local vector, matrix, struct or array, because in JavaScript the store would alias the two. Binding a fresh copy, `const b = d.vec2u(group[1])`, does not change this. A compare-and-swap over a local array therefore wraps each store in a second constructor, which emits the same WGSL.

```ts
const Pairs = d.arrayOf(d.vec2u, 8);
const swap = tgpu.fn(
  [],
  d.u32,
)(() => {
  'use gpu';
  const group = Pairs();
  const b = d.vec2u(group[1]);
  group[0] = b; // throws: "references cannot be assigned. Try 'group[0] = vec2u(b)'"
  return group[0].x;
});
tgpu.resolve([swap]);
// With `group[0] = d.vec2u(b)` it resolves to `group[0i] = b;`: the copy changes no WGSL.
```

**Fallback.** Not in the fallback. The `merge` kernel of `bitonic` in `packages/typegpu-kit/src/grid.ts` writes `group[e] = d.vec2u(b)` and `group[e | apart] = d.vec2u(a)`.

**Upstream.** In `packages/typegpu/src/tgsl/wgslGenerator.ts`, `_binaryOrAssignmentExpression`: accept `=` from a local that is neither read nor written after the store, since no alias can then be observed. Additive. It needs the per-local liveness that note 22 asks to expose.

### 29. Bitwise and shift operators on known numbers are not folded

`d.arrayOf(T, n)` inside a `'use gpu'` function needs `n` known at resolution. `_binaryOrAssignmentExpression` folds known operands only for `+`, `-`, `*`, `/`, `**` (through `binaryOpCodeToCodegen`) and the comparisons. `%`, `&`, `|`, `^`, `<<`, `>>` and `>>>` become runtime expressions, so `1 << N` with a module constant `N` cannot be a count.

```ts
const N = 3;
const f = tgpu.fn(
  [],
  d.u32,
)(() => {
  'use gpu';
  const group = d.arrayOf(d.vec2u, 1 << N)(); // `2 ** N` resolves, `1 << N` and `N & 1` do not
  return group[0].x;
});
tgpu.resolve([f]); // "Called comptime function with runtime-known values: '1 << N'"
```

**Fallback.** Not in the fallback. `packages/typegpu-kit/src/grid.ts` declares `Gathered = d.arrayOf(d.vec2u, 2 ** STEPS)` at module scope for the `merge` kernel of `bitonic`.

**Upstream.** In `packages/typegpu/src/tgsl/wgslGenerator.ts`, `_binaryOrAssignmentExpression`: add `'%': mod` to `binaryOpCodeToCodegen`, and fold the bitwise and shift operators when `asComptime` knows both operands, as the comparisons already do. Additive.

### 30. WGSL language features come from the global GPU, not the root's device

`enabledWgslLanguageFeatures` on `TgpuRootImpl` returns `navigator.gpu.wgslLanguageFeatures` whatever device the root holds, and `initFromDevice` has no option to set it. A root over a WebGL or CPU device advertises `immediate_address_space` when the browser's WebGPU has it. A check like the `jump-flood` example's then picks `tgpu.immediateVar` on a device whose `maxImmediateSize` is 0.

```ts
// A host whose WebGPU has immediates, and a device that has none
Object.defineProperty(navigator, 'gpu', {
  value: { wgslLanguageFeatures: new Set(['immediate_address_space']) },
});
const device = { limits: { maxImmediateSize: 0 } } as unknown as GPUDevice;
const root = tgpu.initFromDevice({ device });
root.enabledWgslLanguageFeatures.has('immediate_address_space'); // true, should be false
```

**Fallback.** `init` in `src/webgl.ts` redefines `enabledWgslLanguageFeatures` on the root as an empty set. `cpuRoot` in `tests/cpu/device.ts` does the same.

**Upstream.** In `packages/typegpu/src/core/root/init.ts`: add `unstable_wgslLanguageFeatures?: ReadonlySet<string>` to `InitFromDeviceOptions`, keep it in `TgpuRootImpl`, and return it from `enabledWgslLanguageFeatures` when given. Additive. Note 23 is another place where the root reads `navigator.gpu` instead of asking its device.

## Source transforms

### 31. Multiple declarations in one shader statement fail before resolution

`VariableDeclaration` in tinyest-for-wgsl throws for a declaration with more than one declarator. unplugin-typegpu runs it at build time, so `const a = ..., b = ...` in a `'use gpu'` body fails the build and no generator sees it.

```ts
const kernel = tgpu.fn(
  [],
  d.f32,
)(() => {
  'use gpu';
  const a = d.f32(1),
    b = a + 1;
  return a + b;
});
// unplugin-typegpu fails the build: "Currently only one declaration in a statement is supported."
```

**Fallback.** No workaround code. Shader bodies declare one name per statement.

**Upstream.** In `packages/tinyest-for-wgsl/src/transpilers.ts`: have `VariableDeclaration` return one `const` or `let` node per declarator in source order, and have `BlockStatement` splice them into its body. Order is kept, so a declarator that reads an earlier one still works. In a `for` init, wrap the declarations and the loop in a block. Additive.

## Value types

### 32. Matrix column references are writable in WGSL but readonly in TypeScript

The `mat2x2`, `mat3x3` and `mat4x4` interfaces type `columns` as a readonly tuple, so assigning a column of a local matrix fails type checking: TS2540 with a literal index, TS2542 with a computed one. TypeGPU resolves the assignment and runs it on the CPU. Component assignments such as `matrix.columns[1].y = 9` already type-check.

```ts
const kernel = tgpu.fn(
  [],
  d.mat2x2f,
)(() => {
  'use gpu';
  const matrix = d.mat2x2f();
  matrix.columns[0] = d.vec2f(7, 8); // TS2540: '0' is a read-only property
  return matrix;
});
tgpu.resolve([kernel]); // `matrix[0i] = vec2f(7, 8);`
kernel(); // mat2x2f(7, 8, 0, 0) on the CPU
```

**Fallback.** The column-assignment case in `tests/cpu/indexed.ts` casts `matrix.columns` to `[d.v2f, d.v2f]`.

**Upstream.** In `packages/typegpu/src/data/wgslTypes.ts`, `mat2x2`, `mat3x3` and `mat4x4`, and in `data/matrix.ts`, `MatBase` and the `mat*Impl` classes: drop the inner `readonly` from the `columns` tuple and keep the property readonly. Additive for code that only reads `columns`. A store of a local vector is already rejected by the check in note 28, which keeps the CPU alias out of shaders.

## Name resolution

### 33. Captured names can resolve to inherited Object prototype methods

`ItemStateStackImpl.getSnippetById` reads `layer.argAccess[id]` without an own-property check, and `resolveFunction` creates that table as `{}`. A captured function named `valueOf`, `toString` or `hasOwnProperty` finds the `Object.prototype` method instead, which is called as an argument accessor and throws.

```ts
const valueOf = () => {
  'use gpu';
  return d.u32(23);
};
const kernel = tgpu.fn(
  [],
  d.u32,
)(() => {
  'use gpu';
  return valueOf(); // finds Object.prototype.valueOf, not the captured function
});
tgpu.resolve([kernel]); // "Cannot convert undefined or null to object"
```

**Fallback.** The effectful helper in `tests/cpu/indexed.ts` is named `nextValue` instead of `valueOf`.

**Upstream.** In `packages/typegpu/src/resolutionCtx.ts`, `ItemStateStackImpl.getSnippetById`: check `Object.hasOwn(layer.argAccess, id)`, as the same function already does for `externalMap`, or create the table with `Object.create(null)` in `resolveFunction`. Additive.

## Resolution order

### 34. A helper is generated inside the statement that first calls it

TypeGPU generates a function when a call first reaches it, so `ResolutionCtxImpl.resolveFunction` calls `ShaderGenerator.functionDefinition` in the middle of the caller's statement. State a generator keeps per statement still describes the caller while the helper's body is generated. `WgslGenerator` itself leaks its private `#unrollingChain` this way: a helper first called inside a `tgpu.unroll` loop fails on a `break` in its own `switch`, and its own unrolled loops get the caller's labels.

```ts
const pick = (k: number) => {
  'use gpu';
  let r = d.f32();
  switch (k) {
    case 0:
      r = 1;
      break;
    default:
      r = 2;
  }
  return r;
};
const entry = () => {
  'use gpu';
  let t = d.f32();
  for (const k of tgpu.unroll([0, 1])) t += pick(d.i32(k));
  return t;
};
tgpu.resolve([entry]); // throws 'Cannot unroll loop containing `break`'
// Resolves when `entry` first calls `pick(d.i32(0))` outside the loop: it should resolve either way
```

**Fallback.** The GPU compute form only accepts an early `return` as a function's guard at the function's own top level. `KitGlslComputeGenerator.functionDefinition` in `src/glsl/compute.ts` records the `if` depth at entry as `Frame.depth`, and `_statement` compares `#branches` against it.

**Upstream.** In `packages/typegpu/src/tgsl/wgslGenerator.ts`, `WgslGenerator.functionDefinition`: save `#unrollingChain`, set it to `[]` for the body, and restore it after. Document on `ShaderGenerator.functionDefinition` in `tgsl/shaderGenerator.ts` that it runs nested inside the caller's statement, so a generator saves its per-statement state there. Additive.

### 35. A generator cannot see the whole pipeline before it emits

TypeGPU emits each function as the first call reaches it: `fnCore` passes the text `ctx.resolveFunction` returns straight to `ctx.addDeclaration`. A helper's text is final before the generator has seen the rest of its caller. Facts spread over the kernel, such as every atomic on a buffer or the summed additions of a helper over all its calls, are known only after the code that depends on them is written.

```ts
const counts = tgpu.workgroupVar(d.arrayOf(d.atomic(d.u32), 8));
const count = (cell: number) => {
  'use gpu';
  std.atomicAdd(counts.$[cell % 8], 1);
};
const kernel = tgpu.computeFn({ workgroupSize: [64], in: { gid: d.builtin.globalInvocationId } })(({
  gid,
}) => {
  'use gpu';
  count(gid.x);
  count(gid.x + 1);
  const seen = std.atomicLoad(counts.$[0]);
});
// A generator logging `functionDefinition` and `_statement` sees:
// ['count emitted', 'const seen', 'kernel emitted']
// `count` is final before the load of `counts` is reached; nothing lets a generator revise it
```

**Fallback.** `computeForm` in `src/glsl/compute.ts` resolves the pipeline at least twice with `KitGlslComputeGenerator`. The first run has no plan: it walks the kernel and throws `Ineligible` or builds the plan, and repeats when a `NotBlocks` error names new buffers. The second emits from the plan and throws if it reaches a write the plan did not list. This adds to the CPU module's own resolution (note 5).

**Upstream.** In `packages/typegpu/src/resolutionCtx.ts`, `ResolutionCtxImpl.resolve` at item depth 0, before it joins `_declarations`: call an optional `ShaderGenerator.finalizeDeclarations(declarations)` that may rewrite each declaration's code. The module function `resolve` in the same file already replaces bind group placeholders after the whole item resolves. Additive. A pre-pass over the call graph would also answer notes 21 and 22.

## CPU execution

### 36. Calling `'use gpu'` code in JavaScript is slow on vectors

Calling `'use gpu'` code in JavaScript is slow on vectors. Each `std.add` or `std.mul` runs `cpuAdd` or `cpuMul` in `std/operators.ts`, which checks operand kinds, up-casts and allocates a new vector. The particle body below runs 316 times slower called directly than as the fallback's scalarized code (`tests/cpu/NOTES.md`).

```ts
const step = (position: d.v3f, velocity: d.v3f) => {
  'use gpu';
  let p = d.vec3f(position);
  for (let i = 0; i < 10; i++) p = std.add(std.mul(p, 0.999), std.mul(velocity, 0.01));
  return p;
};
// About 0.04 million calls/s in Node: 30 vector operations allocate 30 vectors per call
for (let i = 0; i < 20000; i++) step(d.vec3f(1, 2, 3), d.vec3f(0.5, 0.25, 0.125));
```

**Fallback.** `src/cpu/generator.ts` generates scalarized JavaScript over typed-array views, run by `compile` in `src/cpu/index.ts`. From the repository root, `CPU_SOURCE_BENCHMARK=1 npx vitest run --project 'typegpu-kit-gl (unit)' tests/cpu/source-performance.test.ts` compares it with direct calls, `tgpu.fn` and `simulate`, and rewrites `tests/cpu/results.json`.

**Upstream.** In `packages/typegpu/src/std/operators.ts`, `cpuAdd`, `cpuSub`, `cpuMul` and `cpuDiv`: a fast path for two vectors of the same type, or a vector and a number, before the generic kind checks. Additive. A compiled executor that scalarizes vector temporaries is the executor note 25 asks for.

## Loops

### 37. Nothing tells a generator which `for…of` loops TypeGPU unrolled

`WgslGenerator._statement` unrolls a `for…of` when its iterable's value is an `UnrollableIterable`, the value `tgpu.unroll` gives a shader. Any other `for…of` stays a runtime loop. `UnrollableIterable` is not exported from `typegpu` or `typegpu/~internal`, so an overriding generator cannot tell the two apart. The call's spelling is no evidence, since `tgpu` may be rebound and `unroll` aliased.

```ts
import { d, std, tgpu as real } from 'typegpu';

const tgpu = { unroll: std.range }; // spelled `tgpu.unroll`, but a runtime range
const unroll = real.unroll; // the real one under another name
const cells = real.workgroupVar(d.arrayOf(d.atomic(d.u32), 4));
const kernel = real.computeFn({ workgroupSize: [4] })(() => {
  'use gpu';
  for (const v of tgpu.unroll(4)) std.atomicOr(cells.$[v], 1); // a `for` loop in WGSL
  for (const v of unroll([0, 1, 2, 3])) std.atomicOr(cells.$[v], 1); // four unrolled copies
});
// Both reach a generator as `forOf`; only `instanceof UnrollableIterable` tells them apart
```

**Fallback.** The GPU compute form bounds additions and writes per invocation, so it counts a runtime loop as unbounded. `UNROLLED` in `src/glsl/compute.ts` recovers the class by calling `tgpu.unroll`'s gpu-callable symbol once and reading the result's constructor. `KitGlslComputeGenerator._expression` stops counting a `for…of` as a loop once its iterable resolves to an instance of it. If the class cannot be recovered, every loop counts and `tests/glsl/unroll-hidden.test.ts` checks that kernels then run on the CPU.

**Upstream.** In `packages/typegpu/src/internal.ts`: export `UnrollableIterable` from `core/unroll/tgpuUnroll.ts`, whose own doc comment already tells shader generators to check for it. Additive.
