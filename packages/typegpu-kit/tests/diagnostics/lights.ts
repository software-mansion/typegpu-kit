import type { TgpuRoot } from 'typegpu';
import { lighting, shading } from '../../src/index.ts';

declare const root: TgpuRoot;

// An option the lighting lacks is an error at its key, and a wrong value at the value. Neither
// changes what `create` returns: the shadow is still there, and so are its members.
export const misspelt = lighting.create(root, { shadow: { size: 1024, extend: 20 } });
export const mistyped = lighting.create(root, { shadow: { size: 'large' } });
misspelt.shadow.resize(512);
mistyped.shadow.resize(512);

// A hook types its parameters from the options.
export const hooked: lighting.Shadowed = lighting.create(root, {
  shadow: { size: 512 },
  shading: { ambient: (material, point) => shading.hemisphere(material, point) },
});
