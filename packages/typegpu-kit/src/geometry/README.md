# @typegpu/geometry (vendored)

Vendored from TypeGPU `packages/typegpu-geometry` at `177de9920`.

Local differences:

- The default vertex schema is `meshes.Vertex`, named `Surface` upstream, so it does not clash with the kit's `Surface`.

Edit this copy only to compile under TypeGPU Kit's compiler flags. Record each local change here so you can account for it when syncing upstream. `render/lines.ts` imports `miterJoinLimitSlot` from `lines/joins/miter.ts`, which the package root does not export; keep that path when syncing.
