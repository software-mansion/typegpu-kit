import { gpuRoot } from '@typegpu/kit/testing/dawn';
import { it } from 'vitest';
import { createBox } from '../src/box.scene.ts';
import { COLORS } from '../src/dither.ts';
import { createFoil } from '../src/foil.scene.ts';

it('compiles the box and the card face on a real device', async () => {
  const root = await gpuRoot();
  const box = createBox(root, 'bgra8unorm', COLORS);
  const foil = createFoil(root, 'bgra8unorm');
  await Promise.all([box.pipeline.initAsync(), foil.pipeline.initAsync()]);
  root.destroy();
});
