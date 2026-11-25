# @typegpu/kit-gl

A WebGL 2 fallback for TypeGPU Kit. It runs an unchanged Kit program in a browser without WebGPU: the root it creates is TypeGPU's own, on a `GPUDevice` implemented over a WebGL 2 context, and shaders compile to GLSL ES 3.00. Render pipelines draw with WebGL. Compute kernels run on the GPU as fragment passes or passes of points where their writes allow it, and on the CPU otherwise.

The package is private and `@unstable` while it depends on TypeGPU patches that are not upstream yet.

## Using it

Pick one of three ways in.

**Hosting a scene.** Give `web.host` the fallback. The host creates the root on WebGPU where the browser gives a device, and on WebGL 2 where it does not:

```ts
import * as web from '@typegpu/kit/web';
import { webgl } from '@typegpu/kit-gl';

const hosted = web.host(createScene, { canvas, fallback: webgl.fallback() });
```

`fallback` takes any object with `init(canvas)`, so one that imports this package inside `init` keeps it out of what a WebGPU browser downloads.

**Your own root.** `await webgl.initWithFallback({ canvas })` returns a WebGPU root where WebGPU works and a WebGL 2 root otherwise. `optionalFeatures` requests WebGPU features where the adapter has them.

**Always WebGL 2.** `await webgl.init({ canvas })` creates a WebGL 2 root, and throws where WebGL 2 is missing. `webgl.supported()` reports whether it is there. Pass the canvas you draw to, so the device presents to it without a copy; `web.surface` configures it for either kind of root.

A root's `destroy` also frees the WebGL device.

## Telling the backends apart

`backends.of(root)` from `@typegpu/kit` reports `kind`, `'webgpu'` or `'webgl'`, and `compute`, `'gpu'` or `'cpu'`. The fallback reports `'webgl'` and `'cpu'`: kernels that share workgroup memory, barriers or atomics on shared cells run on the CPU there. A module reads it where it creates its pipelines to choose a smaller capacity or kernels without workgroup memory, as the Kit's `exposure`, `environment` and culling do. Such a branch leaves the WebGPU path as it is, and its tests run both paths on dawn by branding a real device as CPU-compute.

## Testing on the fallback

- `webgl.fallback({ force: true })` hosts on WebGL 2 in a browser that has WebGPU too. The docs' example pages do the same with `?backend=webgl`.
- `webgl.init({ gpuCompute: false })` runs every kernel on the CPU, to compare a GPU form with the CPU module.
- The device's `stats` report the time spent per kernel, readback and uploads, and why each pipeline ran on the CPU. `stats.reset()` starts over.
- `pnpm test:gl` runs `*.test.gl.ts` in headless Chromium, on WebGL 2 beside WebGPU. `tests/device/diff.ts` runs a scene on both and compares every texture and buffer.
- `node packages/typegpu-kit-gl/scripts/examples.ts --base <docs server> --backend both --reference <dir>` opens every example page on both backends, reports which ones draw and their frame rates, and scores screenshots against references.

## Using it outside this repository

Until TypeGPU takes the patches this package relies on, a program that depends on it needs both of these:

- TypeGPU 0.12.11 with `patches/typegpu@0.12.11.patch` from this repository applied. With pnpm, list it under `patchedDependencies` in `pnpm-workspace.yaml`: `typegpu@0.12.11: patches/typegpu@0.12.11.patch`.
- The vendored `@typegpu/gl` from `packages/typegpu-gl`, which carries the GLSL generator's patches, in place of the published one.

## More

- `ARCHITECTURE.md` explains how the device, the generators and the CPU backend fit together.
- `LIMITATIONS.md` lists everything the fallback cannot do or does differently.
- `PATCHES.md` lists each change to TypeGPU's core and to `@typegpu/gl`, with a reproduction, for upstream.
- `TYPEGPU_NOTES.md` lists what TypeGPU could offer so that the fallback needs fewer patches and workarounds.
