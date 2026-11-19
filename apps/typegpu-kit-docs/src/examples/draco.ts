import { gltf } from '@typegpu/kit';
// draco3d's entry also loads its encoder, so take the decoder alone
import createDecoderModule from 'draco3d/draco_decoder_nodejs.js';
import wasm from 'draco3d/draco_decoder.wasm?url';

/** Makes a Draco decoder from `draco3d`'s WebAssembly build, fetched from the site. */
export async function decoder(): Promise<gltf.Draco> {
  const response = await fetch(wasm);
  if (!response.ok) throw new Error(`${wasm}: ${response.status} ${response.statusText}`);
  return gltf.draco(await createDecoderModule({ wasmBinary: await response.arrayBuffer() }));
}
