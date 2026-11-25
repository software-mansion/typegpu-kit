import { compile } from '../../src/cpu/index.ts';
import { textureOps, encodeHalf, decodeHalf } from '../../src/cpu/textures.ts';
import type { CpuBuffer, CpuResources, CpuTexture } from '../../src/cpu/types.ts';
import type { ReplaySnapshot } from './replay.ts';
import { readBuffer } from './example-comparison.ts';

const ROWS = 4096;
interface Sample {
  texture: CpuTexture;
  sampler: GPUSamplerDescriptor;
  coords: number[];
  lod: number;
  value: number[];
}
function expand(v: number, sign: number) {
  const bits = encodeHalf(v);
  return (bits & 0x7fff) === 0 ? sign * decodeHalf(1) : decodeHalf(bits + (v < 0 ? -sign : sign));
}

function envelope(sample: Sample, coords: number[], lod: number) {
  if (lod !== 0) throw new Error('Sampling trace expects level zero');
  const level = sample.texture.levels[0]!,
    dimensions = [level.width, level.height, level.depthOrArrayLayers];
  const axes = coords.map((value, axis) => {
    const size = dimensions[axis]!,
      lo = value - 1 / (256 * size),
      hi = value + 1 / (256 * size);
    const cuts = [lo, hi];
    const at = Math.floor(value * size - 0.5);
    for (const index of [at, at + 1]) {
      const boundary = (index + 0.5) / size;
      if (boundary > lo && boundary < hi) cuts.push(boundary);
    }
    return cuts;
  });
  let points: number[][] = [[]];
  for (const axis of axes)
    points = points.flatMap((point) => axis.map((value) => [...point, value]));
  const values = points.map(
    (point) =>
      textureOps.textureSampleLevel(sample.texture, sample.sampler, point, lod) as number[],
  );
  return [0, 1, 2, 3].map((c) => {
    const lo = Math.min(...values.map((v) => v[c]!)),
      hi = Math.max(...values.map((v) => v[c]!));
    if (sample.texture.format.includes('16float')) {
      return [expand(lo, -1), expand(hi, 1)];
    }
    const conversion = sample.texture.format === 'rgba8unorm' ? 1 / 4080 : 0;
    return [lo - conversion - 2 ** -22, hi + conversion + 2 ** -22];
  });
}

export async function traceSampling(
  device: GPUDevice,
  wgsl: string,
  groups: Map<number, GPUBindGroupEntry[]>,
  snapshot: ReplaySnapshot,
  resources: CpuResources,
  texels: number[],
  size: number[],
  output: CpuTexture | CpuBuffer,
  expected: Uint16Array | Uint32Array,
  geometry = false,
) {
  const samples: Sample[][] = texels.map(() => []);
  let controlledSamples: Sample[][] = texels.map(() => []);
  const geometryRows: { tag: number; value: number[] }[][] = texels.map(() => []);
  let injectedGeometry: typeof geometryRows | undefined;
  const cursors = new Map<number, number>();
  const geometrical = (tag: number, value: number[]) => {
    if (!injected) geometryRows[selected]!.push({ tag, value: [...value] });
    const cursor = cursors.get(tag) ?? 0;
    cursors.set(tag, cursor + 1);
    return injectedGeometry?.[selected]?.filter((row) => row.tag === tag)[cursor]?.value ?? value;
  };
  let selected = -1,
    index = 0,
    injected: number[][][] | undefined;
  const select = (id: number[]) => {
    selected = texels.indexOf(id[0]! + size[0]! * (id[1]! + size[1]! * id[2]!));
    index = 0;
    cursors.clear();
    return selected >= 0;
  };
  const sample = (
    texture: CpuTexture,
    sampler: GPUSamplerDescriptor,
    coords: number[],
    lod: number,
    value: number[],
  ) => {
    if (!injected)
      samples[selected]!.push({ texture, sampler, coords: [...coords], lod, value: [...value] });
    if (injected)
      controlledSamples[selected]!.push({
        texture,
        sampler,
        coords: [...coords],
        lod,
        value: [...value],
      });
    return injected?.[selected]?.[index++] ?? value;
  };
  const scope = globalThis as typeof globalThis & {
    cpuSamplingTrace?: {
      select: typeof select;
      sample: typeof sample;
      geometrical: typeof geometrical;
    };
  };
  const previous = scope.cpuSamplingTrace;
  const entry = JSON.parse(/^\/\/#entry (.*)$/m.exec(snapshot.code)![1]!) as {
    name: string;
    inputs: string[];
  };
  if (entry.inputs.length && !entry.inputs.includes('global_invocation_id'))
    throw new Error('Sampling trace needs a global invocation input');
  if (/\b(?:workgroupBarrier|storageBarrier)\(/.test(wgsl))
    throw new Error('Sampling trace cannot isolate invocations across barriers');
  const definition = new RegExp(`function ${entry.name}\\(([^)]*)\\) \\{`);
  const parameters = definition.exec(snapshot.code)?.[1];
  if (parameters === undefined) throw new Error('Sampling trace entry shape changed');
  const argumentsList = parameters.split(',').map((value) => value.trim());
  const invocation = entry.inputs.length
    ? argumentsList[entry.inputs.indexOf('global_invocation_id')]!
    : '[0,0,0]';
  let cpu = `const traceSample=$textures.textureSampleLevel;$textures={...$textures,textureSampleLevel:(t,s,c,l)=>globalThis.cpuSamplingTrace.sample(t,s,c,l,traceSample(t,s,c,l))};\n${snapshot.code.replace(definition, `function ${entry.name}(${parameters}) {if(!globalThis.cpuSamplingTrace.select(${invocation}))return;`)}`;
  if (geometry) {
    const sphere = /function intersectSphere\(([^)]+)\)/.exec(cpu)?.[1]?.split(',');
    if (!sphere || sphere.length !== 7) throw new Error('Atmosphere sphere helper changed');
    cpu = cpu.replace('function intersectSphere(', 'function geometrySphere(');
    cpu += `\nfunction intersectSphere(${sphere.join(',')}){const result=geometrySphere(${sphere.join(',')});globalThis.cpuSamplingTrace.geometrical(1,[${sphere.slice(0, 3).join(',')},${sphere[6]}]);return globalThis.cpuSamplingTrace.geometrical(2,[${sphere.slice(3, 6).join(',')},result])[3];}`;
    cpu = cpu.replace(
      /function (mediumAt|cpuRecordFields(?:_\d+)?)\(r,haze\) \{/g,
      '$&r=globalThis.cpuSamplingTrace.geometrical(3,[r,haze,Math.max(r-6360,0),0])[0];',
    );
  }
  const run = () => {
    if (injected) controlledSamples = texels.map(() => []);
    scope.cpuSamplingTrace = { select, sample, geometrical };
    try {
      compile(cpu).dispatch(...snapshot.workgroups, resources);
    } finally {
      if (previous) scope.cpuSamplingTrace = previous;
      else delete scope.cpuSamplingTrace;
    }
  };
  run();
  const readOutput = () =>
    'levels' in output
      ? new Uint16Array(
          output.levels[0]!.data.buffer,
          output.levels[0]!.data.byteOffset,
          output.levels[0]!.data.byteLength / 2,
        )
      : new Uint32Array(output.bytes, output.offset, output.size / 4);
  const original = readOutput().slice();
  const group = Math.max(...groups.keys()) + 1;
  const wrappers = new Map<string, string>();
  let shader = wgsl.replace(/textureSampleLevel\((\w+)\s*,/g, (_call, name: string) => {
    const type = new RegExp(`\\b${name}\\s*:\\s*(texture_(?:2d|3d)<f32>)`).exec(wgsl)?.[1];
    if (!type) throw new Error(`Sampling trace cannot infer ${name}`);
    const dimension = type.includes('3d') ? 3 : 2;
    const wrapper = `sampling_${name}`;
    wrappers.set(
      wrapper,
      `fn ${wrapper}(t:${type},s:sampler,c:vec${dimension}f,l:f32)->vec4f{let v=textureSampleLevel(t,s,c,l);sampling_record(0u,vec4f(c,${dimension === 2 ? '0,' : ''}l),v);return v;}`,
    );
    return `${wrapper}(${name},`;
  });
  const start = shader.indexOf('@compute'),
    open = shader.indexOf('{', start);
  const gid = /@builtin\(global_invocation_id\)\s+(\w+)/.exec(shader.slice(start, open))?.[1];
  if (!gid && entry.inputs.length) throw new Error('Sampling trace WGSL has no global input');
  shader =
    shader.slice(0, open + 1) +
    (gid
      ? `let linear=${gid}.x+${size[0]}u*(${gid}.y+${size[1]}u*${gid}.z);${texels.map((texel, i) => `if(linear==${texel}u){sampling_slot=${i}u;}`).join('')}if(sampling_slot==0xffffffffu){return;}`
      : 'sampling_slot=0u;') +
    shader.slice(open + 1);
  shader += `\nvar<private>sampling_slot:u32=0xffffffffu;var<private>sampling_row:u32=0u;@group(${group})@binding(0)var<storage,read_write>sampling_output:array<vec4f>;
fn sampling_record(tag:u32,c:vec4f,v:vec4f){if(sampling_row>=${ROWS}u){return;}let at=(sampling_slot*${ROWS}u+sampling_row)*3u;sampling_output[at]=vec4f(f32(tag+1u),0,0,0);sampling_output[at+1u]=c;sampling_output[at+2u]=v;sampling_row++;}\n${[...wrappers.values()].join('\n')}`;
  if (geometry) {
    shader = shader
      .replace('fn intersectSphere(', 'fn geometrySphere(')
      .replace('fn mediumAt(', 'fn geometryMedium(');
    shader += `\nfn intersectSphere(origin:vec3f,dir:vec3f,radius:i32)->f32{let result=geometrySphere(origin,dir,radius);sampling_record(1u,vec4f(origin,f32(radius)),vec4f());sampling_record(2u,vec4f(dir,result),vec4f());return result;}fn mediumAt(r:f32,haze:f32)->Medium{sampling_record(3u,vec4f(r,haze,max(r-6360,0),0),vec4f());return geometryMedium(r,haze);}`;
  }
  const buffer = device.createBuffer({
    size: texels.length * ROWS * 48,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  try {
    const pipeline = await device.createComputePipelineAsync({
      layout: 'auto',
      compute: { module: device.createShaderModule({ code: shader }) },
    });
    const encoder = device.createCommandEncoder(),
      pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    for (const [g, entries] of groups)
      pass.setBindGroup(
        g,
        device.createBindGroup({ layout: pipeline.getBindGroupLayout(g), entries }),
      );
    pass.setBindGroup(
      group,
      device.createBindGroup({
        layout: pipeline.getBindGroupLayout(group),
        entries: [{ binding: 0, resource: { buffer } }],
      }),
    );
    pass.dispatchWorkgroups(...snapshot.workgroups);
    pass.end();
    device.queue.submit([encoder.finish()]);
    const words = new Float32Array(await readBuffer(device, buffer));
    const logged = texels.map((_texel, i) =>
      Array.from({ length: ROWS }, (_, row) => {
        const at = (i * ROWS + row) * 12;
        return words[at]! > 0
          ? {
              tag: words[at]! - 1,
              coords: Array.from(words.subarray(at + 4, at + 8)),
              value: Array.from(words.subarray(at + 8, at + 12)),
            }
          : undefined;
      }).filter((row) => row !== undefined),
    );
    const dawn = logged.map((rows) => rows.filter((row) => row.tag === 0));
    injected = dawn.map((rows) => rows.map((row) => row.value));
    run();
    const actual = readOutput();
    const sampleOnly = actual.slice();
    if (geometry) {
      injectedGeometry = logged.map((rows) =>
        rows.filter((row) => row.tag !== 0).map((row) => ({ tag: row.tag, value: row.coords })),
      );
      run();
    }
    return texels.map((texel, i) => {
      const comparisons = dawn[i]!.map((row, j) => {
        const source = controlledSamples[i]![j];
        if (!source) return { unmatched: true, dawn: row };
        const coordinates = row.coords.slice(0, source.coords.length),
          bounds = envelope(source, coordinates, row.coords[3]!);
        return {
          texture: {
            format: source.texture.format,
            size: [
              source.texture.levels[0]!.width,
              source.texture.levels[0]!.height,
              source.texture.levels[0]!.depthOrArrayLayers,
            ],
          },
          cpu: samples[i]![j]
            ? {
                coords: samples[i]![j]!.coords,
                lod: samples[i]![j]!.lod,
                value: samples[i]![j]!.value,
              }
            : undefined,
          controlledCPU: { coords: source.coords, lod: source.lod, value: source.value },
          dawn: row,
          bounds,
          outside: row.value.some((v, c) => v < bounds[c]![0]! || v > bounds[c]![1]!),
        };
      });
      return {
        bits: 'levels' in output ? 16 : 32,
        texel,
        cell: [
          texel % size[0]!,
          Math.floor(texel / size[0]!) % size[1]!,
          Math.floor(texel / (size[0]! * size[1]!)),
        ],
        counts: [samples[i]!.length, dawn[i]!.length],
        controlledCounts: [controlledSamples[i]!.length, dawn[i]!.length],
        original: Array.from(original.subarray(texel * 4, texel * 4 + 4)),
        injected: Array.from(sampleOnly.subarray(texel * 4, texel * 4 + 4)),
        ...(geometry
          ? {
              geometryInjected: Array.from(actual.subarray(texel * 4, texel * 4 + 4)),
              geometryCPU: geometryRows[i],
              geometryDawn: injectedGeometry![i],
            }
          : {}),
        expected: Array.from(expected.subarray(texel * 4, texel * 4 + 4)),
        comparisons,
      };
    });
  } finally {
    buffer.destroy();
  }
}

const halfOrder = (v: number) => (v & 0x8000 ? -(v & 0x7fff) : v);

export function summarizeSamplingTrace(rows: Awaited<ReturnType<typeof traceSampling>>) {
  const bits = new Uint32Array(1),
    value = new Float32Array(bits.buffer);
  const encoded = (v: number) => {
    value[0] = v;
    return bits[0]!;
  };
  let radiusMaxUlps = 0,
    storeMaxUlps = 0,
    boundsMisses = 0,
    matched = true;
  const branchFlips = [];
  for (const row of rows) {
    matched &&= row.controlledCounts[0] === row.controlledCounts[1];
    boundsMisses += row.comparisons.filter((sample) => sample.unmatched || sample.outside).length;
    const actual = row.geometryInjected ?? row.injected;
    for (let c = 0; c < actual.length; c++)
      storeMaxUlps = Math.max(
        storeMaxUlps,
        Math.abs(
          (row.bits === 16 ? halfOrder(actual[c]!) : actual[c]!) -
            (row.bits === 16 ? halfOrder(row.expected[c]!) : row.expected[c]!),
        ),
      );
    if (!row.geometryCPU || !row.geometryDawn) continue;
    const radii = row.geometryCPU.filter((item) => item.tag === 3),
      expectedRadii = row.geometryDawn.filter((item) => item.tag === 3);
    for (let i = 0; i < Math.min(radii.length, expectedRadii.length); i++)
      radiusMaxUlps = Math.max(
        radiusMaxUlps,
        Math.abs(encoded(radii[i]!.value[0]!) - encoded(expectedRadii[i]!.value[0]!)),
      );
    const origins = row.geometryCPU.filter((item) => item.tag === 1),
      expectedOrigins = row.geometryDawn.filter((item) => item.tag === 1);
    const rays = row.geometryCPU.filter((item) => item.tag === 2),
      expectedRays = row.geometryDawn.filter((item) => item.tag === 2);
    for (let i = 0; i < Math.min(rays.length, expectedRays.length); i++) {
      const a = rays[i]!.value,
        b = expectedRays[i]!.value;
      if (a[3]! < 0 === b[3]! < 0) continue;
      const dot = origins[i]!.value.slice(0, 3).reduce((sum, x, axis) => sum + x * a[axis]!, 0);
      const ulp = 2 ** (Math.floor(Math.log2(Math.abs(dot))) - 23);
      branchFlips.push({
        cell: row.cell,
        call: i,
        cpuOrigin: origins[i]!.value,
        dawnOrigin: expectedOrigins[i]!.value,
        cpuDirectionResult: a,
        dawnDirectionResult: b,
        resultGapInBUlps: Math.abs(a[3]! - b[3]!) / ulp,
      });
    }
  }
  return { matched, boundsMisses, radiusMaxUlps, storeMaxUlps, branchFlips };
}
