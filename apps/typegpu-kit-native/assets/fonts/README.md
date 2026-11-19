# JetBrains Mono

`mono.json` and `mono.rgba` contain baked JetBrains Mono Regular, the docs site's mono face, for the chest's card. They cover only the characters the card prints in it, at the defaults of `typegpu-kit-bake-font`: the examples' lines, kickers, rarities and line counts. `tests/chest.test.ts` fails when an example's title or line needs a character the card's fonts lack.

The source is `jetbrains-mono-latin-wght-normal.woff2` from `@fontsource-variable/jetbrains-mono` 5.3.0, instanced at weight 400 with fontTools. The baker packs every glyph of the font it reads, so subset the font to the card's characters first, adding any the test names, then bake it:

    fonttools varLib.instancer jetbrains-mono-latin-wght-normal.woff2 wght=400 -o instance.woff2
    pyftsubset instance.woff2 --text=" ',-./0123456789:;ABCDEFGHILMNOPQRSTUWXYabcdefghijklmnopqrstuvwxy…" --layout-features='' --flavor= --output-file=JetBrainsMono-Card.ttf
    typegpu-kit-bake-font --input JetBrainsMono-Card.ttf --output mono --characters "…"

The baker reports the rest of printable ASCII as absent, which leaves it blank.

Copyright 2020 The JetBrains Mono Project Authors. The SIL Open Font License 1.1 covers the font and baked files. See `OFL.txt` for the licence.
