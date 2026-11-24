# @typegpu/gl (vendored)

Vendored from TypeGPU `packages/typegpu-gl` at `f631b3539` (tag `v0.12.11`, package version 0.12.10). `@typegpu/kit-gl` generates its GLSL with this package.

The first commit of this directory is the upstream source, unchanged. Every later change is a patch meant for upstream, and `../typegpu-kit-gl/PATCHES.md` records each one with its reason, a reproduction and its diff. `git diff <vendoring commit> -- packages/typegpu-gl/src` prints them all against upstream's own file layout.

Upstream's tests are not vendored: they need `typegpu-testing-utility`, which TypeGPU does not publish. This copy's tests live in `tests/`.

Once upstream ships the patches, delete this directory and depend on the published `@typegpu/gl`.
