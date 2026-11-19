# Inter Display SemiBold

`inter.json` and `inter.rgba` contain baked Inter Display SemiBold for `@typegpu/kit`. They cover printable ASCII and `…`, at 40 texels per em with a distance range of 8, the defaults of `typegpu-kit-bake-font`. Bake them again from this directory with:

    typegpu-kit-bake-font --input InterDisplay-SemiBold.ttf --output inter --characters "…"

The source font is `InterDisplay-SemiBold.ttf` from `extras/ttf/` in the Inter 4.1 release (`Inter-4.1.zip`, SHA-256 `9883fdd4a49d4fb66bd8177ba6625ef9a64aa45899767dde3d36aa425756b11e`), subset with fontTools to printable ASCII and `…` with its kerning. Its SHA-256 is `80b037d0e5689cadb8dead1ed7d7b945e8b7ba72edc2d3e5362aeecd694d2a9a`.

Copyright 2016 The Inter Project Authors. The SIL Open Font License 1.1 covers the font and baked files, with no Reserved Font Name. See `OFL.txt` for the licence.
