# Patches for upstream

The WebGL fallback changes two things it does not own: TypeGPU's core, through `patches/typegpu@0.12.11.patch` at the repository root, and the vendored `@typegpu/gl` in `../typegpu-gl`. Every change is listed here with the kit code that needs it, a reproduction and its diff, so each can go upstream on its own. `git diff d75490bc -- packages/typegpu-gl/src` prints the fork's changes against upstream's file.

Conventions of this fallback's device stay out of both: render modules and their directives, buffers read through data textures and uniform blocks, the vertex epilogue, `kit_firstInstance` and `kit_baseVertex`, and both stages in one string live in `src/glsl/generator.ts`, which extends the fork through the hooks below.

The core diffs are against the published `dist` JavaScript, which is what the patch changes. Each names the TypeScript source file upstream would change.

## TypeGPU core

### 1. Laid-out samplers and texture views declare themselves through `declareGlobalVar`

Applies [TypeGPU #3387](https://github.com/software-mansion/TypeGPU/pull/3387) ("fix(@typegpu/gl): Resolve laid-out samplers and texture views through declareGlobalVar").

**Why.** Every lit kit scene reads its shadow map through a bind group layout (`afterimage`, `cloister`, `forest`, …), and post-processing (`bloom`, `taa`) samples laid-out views. The declarations were written as WGSL text, so a GLSL generator could not declare them.

**Reproduction.**

```ts
const layout = tgpu.bindGroupLayout({
  map: { texture: d.texture2d() },
  s: { sampler: 'filtering' },
});
const pipeline = root.createRenderPipeline({
  vertex: common.fullScreenTriangle,
  fragment: tgpu.fragmentFn({ in: { uv: d.vec2f }, out: d.vec4f })(({ uv }) => {
    'use gpu';
    return std.textureSample(layout.$.map, layout.$.s, uv);
  }),
  targets: { format: 'rgba8unorm' },
});
tgpu.resolve([pipeline], glOptions()); // contains `@group(0) @binding(0) var map: texture_2d<f32>;`
```

**Diff** (`core/sampler/sampler.ts`, `core/texture/texture.ts`; the fixed sampler as well):

```diff
-                ctx.addDeclaration(`@group(${group}) @binding(${this.#membership.idx}) var ${id}: ${ctx.resolve(this.schema).value};`, id);
-                return snip(id, this.schema, /* origin */ 'handle');
+                return ctx.gen.declareGlobalVar({
+                    group,
+                    binding: this.#membership.idx,
+                    scope: 'handle',
+                    id,
+                    dataType: this.schema,
+                    init: undefined,
+                });
```

`WgslGenerator.declareGlobalVar` writes the same `@group(g) @binding(b) var id: type;` for the `handle` scope, so WGSL output is unchanged.

### 2. External textures declare themselves through `declareGlobalVar`

New.

**Why.** `ember-mirror` draws the camera feed through `video`, a `texture_external`. Same cause as patch 1, in a file #3387 does not touch.

**Reproduction.** As above with `{ texture: d.textureExternal() }` and `std.textureSampleBaseClampToEdge`: the GLSL output contains `var frame: texture_external;`.

**Diff** (`core/texture/externalTexture.ts`):

```diff
     [$resolve](ctx) {
         const id = ctx.makeUniqueIdentifier(getName(this), 'global');
         const group = ctx.allocateLayoutEntry(this.#membership.layout);
-        ctx.addDeclaration(`@group(${group}) @binding(${this.#membership.idx}) var ${id}: ${ctx.resolve(this.schema).value};`, id);
-        return snip(id, textureExternal(), 'handle');
+        return ctx.gen.declareGlobalVar({
+            group,
+            binding: this.#membership.idx,
+            id,
+            dataType: this.schema,
+            scope: 'handle',
+            init: undefined,
+        });
     }
```

`grep -rn "@group(" node_modules/typegpu --include='*.js'` finds no other declaration outside the generator, apart from the WGSL of TypeGPU's internal mipmap shaders (`core/texture/textureUtils.js`), which the device replaces with GL calls.

### 3. Every texture builtin emits through `ShaderGenerator.emitCall`

New.

**Why.** `textureSample`, `textureSampleLevel`, `textureLoad` and `textureDimensions` already called `ctx.gen.emitCall`. The rest wrote WGSL with `stitch`, so another language could not translate them. The kit's shadow filter samples with `textureSampleCompareLevel` (in every lit example), terrain and decals use `textureSampleGrad` (`ocean`, `quiver-trees`), reflections ask `textureNumLevels`, and `ember-mirror` calls `textureSampleBaseClampToEdge`.

**Reproduction.** Resolve any fragment that calls `std.textureSampleCompareLevel(depthArray, comparisonSampler, uv, layer, depth)` with a GLSL generator: the output keeps the WGSL call.

**Diff** (`std/texture.ts`, the same change for `textureStore`, `textureNumLayers`, `textureNumLevels`, `textureNumSamples`, `textureGather`, `textureGatherCompare`, `textureSampleCompare`, `textureSampleCompareLevel`, `textureSampleGrad` and `textureSampleBaseClampToEdge`):

```diff
-    codegenImpl: (_ctx, args) => stitch `textureSampleCompareLevel(${args})`,
+    codegenImpl: (ctx, args) => ctx.gen.emitCall('textureSampleCompareLevel', [], compactSnippetArgs(args)),
```

`WgslGenerator.emitCall` joins arguments as `stitch` did, so WGSL output is unchanged.

### 4. The other `name(args)` builtins emit through `ShaderGenerator.emitCall`

New.

**Why.** WGSL and GLSL name several builtins differently, and the GLSL generator renames them in `emitCall` (fork patch E). The std functions wrote their names with `stitch`, so the renames never ran: `forest`, `stampede` and every glTF mesh decode normals with `unpack2x16snorm` and `unpack2x16float`, `blockfall`, `censer` and `jellies` call `atan2`, and `jelly-world` counts bits with `countTrailingZeros`.

**Reproduction.** `const f = (v: number) => { 'use gpu'; return std.unpack2x16snorm(v); };` resolved with a GLSL generator gives `unpack2x16snorm(v)`, which GLSL does not have.

**Diff.** In `std/numeric.ts`, `std/packing.ts`, `std/derivative.ts`, `std/boolean.ts` and `std/subgroup.ts`, each `codegenImpl` whose template is exactly the builtin's name over its arguments in order, 95 of them:

```diff
-    codegenImpl: (_ctx, [y, x]) => stitch `atan2(${y}, ${x})`,
+    codegenImpl: (ctx, [y, x]) => ctx.gen.emitCall('atan2', [], [y, x]),
```

Templates that are not a plain call (`!(x)`, `(a | b)`, variadic `min` and `max`, the atomics with `&`) are left alone. WGSL output is unchanged.

### 5. `for…of` loops declare their index and element through `_emitVarDecl`

New.

**Why.** `_letStatement` and `_constStatement` go through the overridable `_emitVarDecl`, but a `for…of` header wrote `var i = …` and its element `let x = …` directly. GLSL needs typed declarations. `hex-isle` and `night-market` scan tiles with `for (const y of std.range(8))`.

**Reproduction.** `for (const i of std.range(8)) { sum += d.f32(i); }` in a function resolved with the GLSL generator gives `for (var i = 0u; i < 8u; i += 1u)`.

**Diff** (`tgsl/wgslGenerator.ts`):

```diff
                 const index = this.ctx.makeUniqueIdentifier('i', 'block');
-                const forHeaderStr = stitch `${this.ctx.pre}for (var ${index} = ${range.start}; ${index} ${range.comparison} ${range.end}; ${index} += ${range.step})`;
+                const indexDeclStr = this.ctx.withResetIndentLevel(() => this._emitVarDecl('var', index, concretize(range.start.dataType), this.ctx.resolveSnippet(range.start).value)).slice(0, -1);
+                const forHeaderStr = stitch `${this.ctx.pre}for (${indexDeclStr}; ${index} ${range.comparison} ${range.end}; ${index} += ${range.step})`;
 …
-                    const loopVarDeclStr = stitch `${this.ctx.pre}${loopVarKind} ${loopVarName} = ${tryConvertSnippet(this.ctx, elementSnippet, elementType, false)};`;
+                    const loopVarDeclStr = this._emitVarDecl(loopVarKind, loopVarName, elementType, this.ctx.resolveSnippet(tryConvertSnippet(this.ctx, elementSnippet, elementType, false)).value);
```

The WGSL generator's `_emitVarDecl` writes `${pre}${keyword} ${name} = ${rhs};`, so WGSL output is unchanged.

### 6. Guarded compute pipelines build their entry from a TypeGPU function

New.

**Why.** `createGuardedComputePipeline` built `mainCompute` from a WGSL string, which never reaches `ShaderGenerator.functionDefinition`, so no other generator can translate it. `jelly-world` and every `@typegpu/noise` static cache dispatch a guarded pipeline, which blocks the fallback's CPU compute backend.

**Reproduction.** `root.createGuardedComputePipeline((x) => { 'use gpu'; out.$[x] = x; })` on a root with `unstable_shaderGeneratorClass` set to any non-WGSL generator still yields the WGSL entry `@compute … fn …(…) { if (any(in.id >= sizeUniform)) …`.

**Diff** (`core/root/init.ts`). TypeGPU is not built with unplugin-typegpu, so the patch attaches by hand the metadata unplugin emits for the same `'use gpu'` source (checked against unplugin's output). Upstream could build this file with the plugin instead.

```diff
-        // WGSL instead of JS because we do not run unplugin
-        // before shipping the typegpu package
-        const mainCompute = computeFn({
-            workgroupSize,
-            in: { id: builtin.globalInvocationId },
-        }) `{
-  if (any(in.id >= sizeUniform)) {
-    return;
-  }
-  wrappedCallback(in.id.x, in.id.y, in.id.z);
-}`.$uses({ sizeUniform, wrappedCallback });
+        const guarded = (input) => {
+            if (any(ge(input.id, sizeUniform.$))) {
+                return;
+            }
+            wrappedCallback(input.id.x, input.id.y, input.id.z);
+        };
+        (globalThis.__TYPEGPU_META__ ??= new WeakMap()).set(guarded, {
+            v: 2,
+            name: 'mainCompute',
+            ast: { params: [{ type: 'i', name: 'input' }], body: [0, [
+                [11, [6, 'std.any', [[6, 'std.ge', [[7, 'input', 'id'], [7, 'sizeUniform', '$']]]]], [0, [[10]]]],
+                [6, 'wrappedCallback', [[7, [7, 'input', 'id'], 'x'], [7, [7, 'input', 'id'], 'y'], [7, [7, 'input', 'id'], 'z']]],
+            ]] },
+            externals: { 'std.any': () => any, 'std.ge': () => ge, sizeUniform: () => sizeUniform, wrappedCallback: () => wrappedCallback },
+        });
+        const mainCompute = computeFn({ workgroupSize, in: { id: builtin.globalInvocationId } })(guarded);
```

The WGSL entry becomes `fn mainCompute(@builtin(global_invocation_id) id: vec3u) { if (any((id >= sizeUniform))) { return; } wrappedCallback(id.x, id.y, id.z); }`, the same program. The kit's guarded-pipeline tests pass on dawn.node.

## `@typegpu/gl` (`src/glslGenerator.ts`)

### A. Built-in function names are reserved identifiers

New.

**Why.** GLSL forbids redeclaring a built-in function, and TypeGPU names user functions after their JS names. `solar-system` exports a quaternion `inverse`.

**Reproduction.** `const inverse = (q: d.v4f) => { 'use gpu'; return d.vec4f(-q.xyz, q.w); };` called from a fragment gives `vec4 inverse(vec4 q)`: "Name of a built-in function cannot be redeclared".

**Diff.**

```diff
   'namespace',
   'using',
+  //
+  // Built-in functions, which user functions cannot redeclare
+  //
+  ...[
+    'radians', 'degrees', 'sin', 'cos', 'tan', /* … every GLSL ES 3.00 built-in … */ 'fwidth', 'main',
+  ],
 ];
```

### B. Array types resolve their element type through the context

New.

**Why.** `emitTypeAnnotation` called itself for an array's element type, which bypasses the context's memoization, so a struct element was declared twice under two names. `fireworks` keeps a `const` table of `Traits` structs.

**Reproduction.** `const table = tgpu.const(d.arrayOf(Traits, 6), …)` read from a fragment gives `const Traits traitsOf[6] = Traits_1[6](…)`: "cannot convert from 'Traits_1[6]' to 'Traits[6]'".

**Diff.**

```diff
     if (d.isWgslArray(data)) {
-      // The array size suffix is handled elsewhere
-      return this.emitTypeAnnotation(data.elementType);
+      // The array size suffix is handled elsewhere. Resolving through the context declares a
+      // struct element once.
+      return this.ctx.resolve(data.elementType).value;
     }
```

### C. Functions with empty bodies, array parameters and array results

New.

**Why.** An empty body generated `void f(Corner corner)` with no braces; `classroom` passes a `handOver` hook that does nothing. Array parameters and results lost their size: `kinetic` returns `array<f32, 9>` from `basisOf`.

**Reproduction.** `const nothing = (x: number) => { 'use gpu'; };` called from a fragment gives `void nothing(float x)`, a syntax error. `const basis = () => { 'use gpu'; return [1, 2, 3] as number[]; }` gives `float basis()`: "cannot convert from 'float[3]' to 'float'".

**Diff.**

```diff
       const argList = options.args
-        .map((arg) => `${this.ctx.resolve(arg.decoratedType).value} ${arg.name}`)
+        .map(
+          (arg) =>
+            `${this.ctx.resolve(arg.decoratedType).value} ${arg.name}${resolveArraySizeSuffix(this.ctx, arg.decoratedType)}`,
+        )
         .join(', ');

-      return `${this.ctx.resolve(returnType).value} ${options.name}(${argList}) ${body}`;
+      return `${this.ctx.resolve(returnType).value}${resolveArraySizeSuffix(this.ctx, returnType)} ${options.name}(${argList}) ${body || '{}'}`;
```

### D. Zero-argument matrix and struct constructors

New.

**Why.** WGSL's `mat4x4f()` and `S()` are zero values; GLSL needs every component. `forest` and every instanced table builds rows from `TreeRow()`.

**Reproduction.** `const r = Row();` for a struct `Row` gives `Row r = Row();`: "constructor does not have any arguments".

**Diff.**

```diff
+    // GLSL matrix and struct constructors take every component; WGSL's zero-argument ones zero them.
+    if (schema.type.startsWith('mat') && args.length === 0) {
+      return super.typeInstantiation(schema, [snip(0, d.f32, 'constant')]);
+    }
+
+    if (d.isWgslStruct(schema) && args.length === 0) {
+      return super.typeInstantiation(
+        schema,
+        Object.values(schema.propTypes).map((prop) =>
+          this.typeInstantiation(undecorateDataType(prop), []),
+        ),
+      );
+    }
```

### E. Builtins GLSL names differently or lacks

New. Needs core patch 4 to see the `std` builtins.

**Why.** `atan2` is `atan`, the `pack2x16*` and `unpack2x16*` functions are `packSnorm2x16` and so on, `dpdx` and friends are `dFdx`, `faceForward` is `faceforward`, and WGSL's `round` rounds half to even, which is `roundEven`. GLSL ES 3.00 has no `fma`, no bit counting (that is ES 3.1), no `dot` on integer vectors and no `abs` on `uint`. The kit decodes normals with `unpack2x16snorm` (`forest`, every glTF mesh), `jelly-world` iterates set bits with `countTrailingZeros`, and spatial grids count with `countOneBits`.

**Reproduction.** `std.countOneBits(x)` in a fragment: "'countOneBits' : no matching overloaded function found". `tests/glsl/execution.test.ts` runs each polyfill against JS references on real WebGL.

**Diff.** A rename table and polyfills written as `'use gpu'` functions, called through `_callShellless` like the existing `HELPERS`, componentwise for vectors:

```diff
+const RENAMED_BUILTINS: Record<string, string> = {
+  atan2: 'atan',
+  inverseSqrt: 'inversesqrt',
+  dpdx: 'dFdx', dpdxCoarse: 'dFdx', dpdxFine: 'dFdx',
+  dpdy: 'dFdy', dpdyCoarse: 'dFdy', dpdyFine: 'dFdy',
+  fwidthCoarse: 'fwidth', fwidthFine: 'fwidth',
+  faceForward: 'faceforward',
+  round: 'roundEven',
+  pack2x16snorm: 'packSnorm2x16', unpack2x16snorm: 'unpackSnorm2x16',
+  pack2x16unorm: 'packUnorm2x16', unpack2x16unorm: 'unpackUnorm2x16',
+  pack2x16float: 'packHalf2x16', unpack2x16float: 'unpackHalf2x16',
+};
+const BIT_POLYFILLS = { countOneBits, countLeadingZeros, countTrailingZeros, firstTrailingBit, reverseBits };
 …
   override emitCall(name, templateParams, args) {
-    if (name === 'inverseSqrt') { … }
-    if (name === 'dpdx') { … }
-    if (name === 'dpdy') { … }
+    const renamed = RENAMED_BUILTINS[name];
+    if (renamed) return super.emitCall(renamed, templateParams, args);
+    if (name === 'fma') return `((${a} * ${b}) + ${c})`;
+    const polyfill = BIT_POLYFILLS[name];
+    if (polyfill) return this.#componentwise(args[0], (c) => this._callShellless(polyfill, [c]).value);
+    if (name === 'firstLeadingBit') { … the highest set bit, of the complement for a negative i32 … }
+    if (name === 'extractBits' || name === 'insertBits') throw new Error(`'${name}' is not supported by the GLSL generator yet.`);
+    if (name === 'dot' && isIntegerVector(args[0].dataType)) return `(${a}.x * ${b}.x + …)`;
+    if (name === 'abs' && isUnsigned(args[0].dataType)) return a;
```

### F. `select` and ternaries for every type WGSL allows

New.

**Why.** The WGSL generator writes a ternary as WGSL's `select(f, t, c)`, which GLSL lacks, and `std.select` with a vector condition became `mix`, which GLSL ES 3.00 defines for floats only. `hex-isle`'s tone mapping writes `x < 0.08 ? … : …`.

**Reproduction.** `const y = x < 0.5 ? 1 : 2;` in a fragment: "'select' : no matching overloaded function found". `std.select(d.vec2i(1), d.vec2i(2), d.vec2b(a, b))` gives `mix(ivec2, ivec2, bvec2)`, which ES 3.00 does not define.

**Diff.** A set of `select` overloads, declared on first use through a new hook (patch G), for every scalar and vector type with a `bool` or matching `bvecN` condition; a scalar condition becomes a GLSL ternary for any type:

```diff
+  protected override _expression(expression: Expression): Snippet {
+    const result = super._expression(expression);
+    if (Array.isArray(expression) && expression[0] === NODE.conditionalExpr &&
+        typeof result.value === 'string' && result.value.startsWith('select(')) {
+      // The WGSL generator writes ternaries as WGSL's `select`
+      this._declareHelper('select', SELECT_OVERLOADS);
+    }
+    return result;
+  }
 …
-      if (falsy.dataType !== UnknownData && falsy.dataType.type.startsWith('vec')) {
-        if (cond.dataType !== UnknownData && cond.dataType.type.startsWith('vec')) {
-          return super.emitCall('mix', templateParams, args);
-        }
-        return super.emitCall('mix', templateParams, [falsy, truthy, this.typeInstantiation(correspondingBooleanVectorSchema(falsy.dataType), [cond])]);
-      }
-      if (cond.dataType !== UnknownData && cond.dataType.type.startsWith('vec')) {
-        throw new Error(`GLSL select() with scalar branches requires a scalar boolean condition`);
-      }
+      if (cond.dataType !== UnknownData && cond.dataType.type.startsWith('vec')) {
+        if (isF32VecfSchema(falsy.dataType)) {
+          return super.emitCall('mix', templateParams, args);
+        }
+        // GLSL ES 3.00 mixes floats only
+        this._declareHelper('select', SELECT_OVERLOADS);
+        return super.emitCall('select', templateParams, args);
+      }
+      // Generating a ternary expression, which GLSL has for any type with a scalar condition
```

### G. `_declareHelper`, a hook for code the output relies on

New.

**Why.** Patches E and F need GLSL helpers declared once per resolution. A subclass that lays declarations out differently, as the kit's generator does with its chunks, overrides the hook.

**Diff.**

```diff
+  protected _declareHelper(name: string, code: string): void {
+    if (this.#declaredHelpers.has(name)) {
+      return;
+    }
+    this.#declaredHelpers.add(name);
+    this.ctx.addDeclaration(code);
+  }
```

### H. Abstract scalars next to vectors take the vector's component type

New.

**Why.** WGSL converts an abstract literal to the component type of the vector it meets; TypeGPU leaves `vec2u % 3` with an abstract `3`, which GLSL reads as `int`. `blockfall` takes `vec2u(position.xy) % 3`.

**Reproduction.** `const cell = d.vec2u(p) % 3;` gives `(uvec2(p) % 3)`: "'%' : wrong operand types … 'uvec2' and 'const int'".

**Diff.**

```diff
+  #matchVectorComponent(operand: Snippet, other: Snippet): Snippet {
+    … when `operand` is abstractInt or abstractFloat and `other` a vector, write it as
+    `other`'s component type: a literal through numericLiteral, anything else in a constructor …
+  }
+
   override emitBinaryOp(lhs: Snippet, op: BinaryOperator, rhs: Snippet): string {
+    lhs = this.#matchVectorComponent(lhs, rhs);
+    rhs = this.#matchVectorComponent(rhs, lhs);
```

The full fork diff, including the unchanged upstream code around each hunk, is `git diff d75490bc -- packages/typegpu-gl/src`.
