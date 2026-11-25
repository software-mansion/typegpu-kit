import { expect, it } from 'vitest';
import { compile, type CpuResources, type CpuTexture } from '../../src/cpu/index.ts';

it('rebinds ranges and handles and drops uniform caches after each reusable dispatch', () => {
  const module = compile(`//#kit-gl compute
//#kit-gl reusable
//#entry {"name":"kernel","size":[1,1,1],"inputs":[]}
//#binding {"group":0,"binding":0,"kind":"buffer","reads":true,"writes":true}
const data = $buffer(0,0,Uint32Array);
const texture = $resources.texture(0,1);
const sampler = $resources.sampler(0,2);
//#dispatch-cache ["cached"]
let cached;
function kernel(){
  if(cached===undefined) cached=data[0];
  if(data[0]===99) throw new Error('kernel failure');
  data[1]=cached+texture.levels[0].data[0]+sampler.lodMinClamp;
}`);
  const bytes = Uint32Array.of(3, 0, 7, 0);
  let offset = 0;
  let texel = 10;
  let lod = 1;
  const resources: CpuResources = {
    buffer: () => ({ bytes: bytes.buffer, offset, size: 8 }),
    texture: () =>
      ({
        format: 'r32uint',
        dimension: '2d',
        levels: [{ width: 1, height: 1, depthOrArrayLayers: 1, data: Uint32Array.of(texel) }],
      }) as CpuTexture,
    sampler: () => ({ lodMinClamp: lod }),
  };
  module.dispatch(1, 1, 1, resources);
  expect(bytes[1]).toBe(14);
  bytes[0] = 5;
  module.dispatch(1, 1, 1, resources);
  expect(bytes[1]).toBe(16);
  offset = 8;
  texel = 20;
  lod = 2;
  module.dispatch(1, 1, 1, resources);
  expect(bytes[3]).toBe(29);
  const other = Uint32Array.of(11, 0);
  const replacement = { ...resources, buffer: () => ({ bytes: other.buffer, offset: 0, size: 8 }) };
  module.dispatch(1, 1, 1, replacement);
  expect(other[1]).toBe(33);
  other[0] = 99;
  expect(() => module.dispatch(1, 1, 1, replacement)).toThrow('kernel failure');
  other[0] = 13;
  module.dispatch(1, 1, 1, replacement);
  expect(other[1]).toBe(35);
  expect(() =>
    module.dispatch(1, 1, 1, {
      ...replacement,
      texture: () => {
        throw new Error('binding failure');
      },
    }),
  ).toThrow('binding failure');
  other[0] = 17;
  module.dispatch(1, 1, 1, replacement);
  expect(other[1]).toBe(39);
});

it('keeps the original initialization semantics for modules without reuse metadata', () => {
  const module = compile(`//#kit-gl compute
//#entry {"name":"kernel","size":[1,1,1],"inputs":[]}
const data = $buffer(0,0,Uint32Array);
let calls=0;
function kernel(){data[0]=++calls;}`);
  const bytes = new ArrayBuffer(4);
  const resources: CpuResources = {
    buffer: () => ({ bytes, offset: 0, size: 4 }),
    texture: () => {
      throw new Error('unused');
    },
    sampler: () => ({}),
  };
  module.dispatch(1, 1, 1, resources);
  module.dispatch(1, 1, 1, resources);
  expect(new Uint32Array(bytes)[0]).toBe(1);
});
