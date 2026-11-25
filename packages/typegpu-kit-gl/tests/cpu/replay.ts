import { traceSampling } from './sampling-trace.ts';
import { tracePrefilter } from './prefilter-trace.ts';
import { traceSky } from './sky-trace.ts';
import { traceMaterial } from './material-trace.ts';
import { writeFileSync } from 'node:fs';
import { traceAdvection, summarizeAdvection } from './advection-trace.ts';
import type { CpuBindingUse, CpuBuffer, CpuResources, CpuTexture } from '../../src/cpu/types.ts';
import { compile } from '../../src/cpu/index.ts';
import type { DispatchSnapshot } from './device.ts';
import type { observeScene, ObservedDispatch, Resource } from './scene-observer.ts';
import {
  compareCaptures,
  readBuffer,
  readTexture,
  type CapturedResource,
} from './example-comparison.ts';

const ordered = (bits: number) => (bits & 0x8000 ? -(bits & 0x7fff) : bits);

const unorm = (value: number) => Math.round(Math.max(0, Math.min(1, value)) * 255);

interface HeldBinding {
  use: CpuBindingUse;
  key: string;
  resource?: Resource;
  buffer?: CpuBuffer;
  texture?: CpuTexture;
  sampler?: GPUSamplerDescriptor;
}
export interface ReplaySnapshot {
  code: string;
  label: string;
  workgroups: [number, number, number];
  bindings: HeldBinding[];
}
export function snapshotDispatch(
  code: string,
  label: string,
  dispatch: DispatchSnapshot,
  observer: ReturnType<typeof observeScene>,
): ReplaySnapshot {
  const copied = new Map<ArrayBuffer, ArrayBuffer>();
  const clone = (bytes: ArrayBuffer) => {
    let result = copied.get(bytes);
    if (!result) {
      result = bytes.slice(0);
      copied.set(bytes, result);
    }
    return result;
  };
  return {
    code,
    label,
    workgroups: dispatch.workgroups,
    bindings: dispatch.module.bindings.map((use) => {
      const group = dispatch.groups.get(use.group)!;
      const entry = Array.from(group.entries).find((item) => item.binding === use.binding)!;
      const native = entry.resource as GPUBufferBinding;
      const resource = observer.resourceOf(native.buffer ?? entry.resource);
      const base = {
        use,
        key: resource?.key ?? `sampler:${use.group}:${use.binding}`,
        ...(resource ? { resource: { ...resource } } : {}),
      };
      if (use.kind === 'buffer') {
        const value = dispatch.resources.buffer(use.group, use.binding);
        return { ...base, buffer: { ...value, bytes: clone(value.bytes) } };
      }
      if (use.kind === 'sampler')
        return { ...base, sampler: { ...dispatch.resources.sampler(use.group, use.binding) } };
      const texture = dispatch.resources.texture(use.group, use.binding);
      return {
        ...base,
        texture: {
          ...texture,
          levels: texture.levels.map((level) => {
            const bytes = level.data.buffer as ArrayBuffer;
            const constructor = level.data.constructor as new (
              buffer: ArrayBuffer,
              offset: number,
              count: number,
            ) => Uint8Array;
            return {
              ...level,
              data: new constructor(
                clone(bytes),
                level.data.byteOffset,
                (level.data as Uint8Array).length,
              ),
            };
          }),
        },
      };
    }),
  };
}

function captured(binding: HeldBinding, levels: CapturedResource['levels']): CapturedResource {
  const { handle: _handle, ...metadata } = binding.resource!;
  return { ...metadata, reasons: [], order: [], levels };
}

/** Runs one compiled CPU kernel and its WGSL counterpart from the same captured bytes. */
export async function replay(
  device: GPUDevice,
  events: ObservedDispatch[],
  snapshot: ReplaySnapshot,
  filterDiagnostic = false,
  options: { sampling?: boolean; geometry?: boolean; prefilter?: boolean } = {},
) {
  const keys = snapshot.bindings
    .filter((binding) => binding.resource)
    .map((binding) => `${binding.key}:${binding.use.writes ? 'write' : 'read'}`)
    .toSorted();
  const candidates = events.filter(
    (event) =>
      event.label === snapshot.label &&
      JSON.stringify(
        event.bindings
          .map((binding) => `${binding.resource.key}:${binding.writes ? 'write' : 'read'}`)
          .toSorted(),
      ) === JSON.stringify(keys),
  );
  const event = candidates[0];
  if (!event)
    throw new Error(
      `No WGSL dispatch matches ${snapshot.label}: ${keys.join(', ')}; available ${[...new Set(events.map((item) => item.label))].join(', ')}`,
    );

  if (process.env.CPU_REPLAY_SOURCE) {
    writeFileSync(`${process.env.CPU_REPLAY_SOURCE}.wgsl`, event.code);
    writeFileSync(`${process.env.CPU_REPLAY_SOURCE}.js`, snapshot.code);
  }
  const groups = new Map<number, GPUBindGroupEntry[]>();
  const append = (group: number, binding: number, resource: GPUBindingResource) => {
    const entries = groups.get(group) ?? [];
    entries.push({ binding, resource });
    groups.set(group, entries);
  };
  const buffers = new Map<ArrayBuffer, GPUBuffer>(),
    textures = new Map<HeldBinding, GPUTexture>();
  const find = (group: number, binding: number) =>
    snapshot.bindings.find((item) => item.use.group === group && item.use.binding === binding)!;
  const resources: CpuResources = {
    buffer: (group, binding) => find(group, binding).buffer!,
    texture: (group, binding) => find(group, binding).texture!,
    sampler: (group, binding) => find(group, binding).sampler!,
  };
  try {
    for (const binding of snapshot.bindings) {
      if (binding.sampler) continue;
      const matches = event.bindings.filter(
        (item) => item.resource.key === binding.key && item.writes === binding.use.writes,
      );
      const gpuBinding =
        matches.find(
          (item) => item.group === binding.use.group && item.binding === binding.use.binding,
        ) ?? matches[0];
      if (!gpuBinding) throw new Error(`Missing WGSL binding ${binding.key}`);
      if (binding.buffer) {
        const source = binding.buffer;
        let buffer = buffers.get(source.bytes);
        if (!buffer) {
          buffer = device.createBuffer({
            size: source.bytes.byteLength,
            usage:
              GPUBufferUsage.COPY_SRC |
              GPUBufferUsage.COPY_DST |
              GPUBufferUsage.UNIFORM |
              GPUBufferUsage.STORAGE,
          });
          device.queue.writeBuffer(buffer, 0, source.bytes);
          buffers.set(source.bytes, buffer);
        }
        append(gpuBinding.group, gpuBinding.binding, {
          buffer,
          offset: source.offset,
          size: source.size,
        });
      } else {
        const source = binding.texture!,
          first = source.levels[0]!;
        const texture = device.createTexture({
          size: [first.width, first.height, first.depthOrArrayLayers],
          mipLevelCount: source.levels.length,
          format: source.format,
          dimension: source.dimension === '3d' ? '3d' : '2d',
          usage:
            GPUTextureUsage.COPY_DST |
            GPUTextureUsage.COPY_SRC |
            GPUTextureUsage.TEXTURE_BINDING |
            (binding.use.kind === 'storageTexture' ? GPUTextureUsage.STORAGE_BINDING : 0),
        });
        textures.set(binding, texture);
        for (const [level, data] of source.levels.entries()) {
          const row = data.data.byteLength / data.height / data.depthOrArrayLayers;
          device.queue.writeTexture(
            { texture, mipLevel: level },
            data.data as Uint8Array<ArrayBuffer>,
            { bytesPerRow: row, rowsPerImage: data.height },
            [data.width, data.height, data.depthOrArrayLayers],
          );
        }
        append(
          gpuBinding.group,
          gpuBinding.binding,
          texture.createView({ dimension: source.dimension }),
        );
      }
    }
    for (const sampler of event.samplers)
      append(sampler.group, sampler.binding, device.createSampler(sampler.descriptor));
    // Captures contain statically used bindings, not unused entries of explicit TypeGPU layouts.
    const pipeline = await device.createComputePipelineAsync({
      layout: 'auto',
      compute: { module: device.createShaderModule({ code: event.code }) },
    });
    const encoder = device.createCommandEncoder(),
      pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    for (const [group, entries] of groups)
      pass.setBindGroup(
        group,
        device.createBindGroup({ layout: pipeline.getBindGroupLayout(group), entries }),
      );
    pass.dispatchWorkgroups(...snapshot.workgroups);
    pass.end();
    device.queue.submit([encoder.finish()]);
    const savedInputs = filterDiagnostic
      ? snapshot.bindings.flatMap((binding) => {
          const arrays = binding.buffer
            ? [new Uint8Array(binding.buffer.bytes)]
            : (binding.texture?.levels.map(
                (level) =>
                  new Uint8Array(level.data.buffer, level.data.byteOffset, level.data.byteLength),
              ) ?? []);
          return arrays.map((target) => ({ target, bytes: target.slice() }));
        })
      : [];
    compile(snapshot.code).dispatch(...snapshot.workgroups, resources);
    await device.queue.onSubmittedWorkDone();
    const cpu: CapturedResource[] = [],
      gpu: CapturedResource[] = [];
    for (const binding of snapshot.bindings.filter((item) => item.use.writes && item.resource)) {
      if (binding.buffer) {
        const bytes = binding.buffer.bytes;
        cpu.push(captured(binding, [{ width: 0, height: 0, depth: 0, bytes }]));
        gpu.push(
          captured(binding, [
            { width: 0, height: 0, depth: 0, bytes: await readBuffer(device, buffers.get(bytes)!) },
          ]),
        );
      } else {
        cpu.push(
          captured(
            binding,
            binding.texture!.levels.map((level) => ({
              width: level.width,
              height: level.height,
              depth: level.depthOrArrayLayers,
              bytes: new Uint8Array(
                level.data.buffer,
                level.data.byteOffset,
                level.data.byteLength,
              ).slice().buffer,
            })),
          ),
        );
        gpu.push(captured(binding, await readTexture(device, textures.get(binding)!)));
      }
      const encoded = event.bindings.find((item) => item.resource.key === binding.key)?.encoded;
      if (encoded) gpu.at(-1)!.encoded = encoded;
    }
    const filterModels = [];
    if (filterDiagnostic) {
      // This test-only model measures hardware filtering precision without changing the runtime.
      for (const halfResult of [false, true]) {
        for (const { target, bytes } of savedInputs) target.set(bytes);
        const code = `
const cpuDiagnosticSample = $textures.textureSampleLevel;
$textures = {...$textures, textureSampleLevel: (texture, sampler, coords, lod) => {
  if (texture.dimension !== '3d' || lod !== 0) throw new Error('Filter probe requires a 3d level-zero sample');
  const level = texture.levels[0];
  const size = [level.width, level.height, level.depthOrArrayLayers];
  const snapped = coords.map((value, axis) => (Math.round((value * size[axis] - 0.5) * 256) / 256 + 0.5) / size[axis]);
  const sampled = cpuDiagnosticSample(texture, sampler, snapped, lod);
  return ${halfResult ? "sampled.map(value => $unpack($pack([value, 0], 16, 'float'), 16, 'float')[0])" : 'sampled'};
}};
${snapshot.code}`;
        compile(code).dispatch(...snapshot.workgroups, resources);
        const modeled = snapshot.bindings
          .filter((binding) => binding.use.writes && binding.resource)
          .map((binding) =>
            captured(
              binding,
              binding.buffer
                ? [{ width: 0, height: 0, depth: 0, bytes: binding.buffer.bytes.slice(0) }]
                : binding.texture!.levels.map((level) => ({
                    width: level.width,
                    height: level.height,
                    depth: level.depthOrArrayLayers,
                    bytes: new Uint8Array(
                      level.data.buffer,
                      level.data.byteOffset,
                      level.data.byteLength,
                    ).slice().buffer,
                  })),
            ),
          );
        filterModels.push({
          halfResult,
          ...compareCaptures(
            { resources: gpu, skipped: [], frames: 0 },
            { resources: modeled, skipped: [], frames: 0 },
          ),
        });
      }
    }
    let advectionTrace;
    if (process.env.CPU_ADVECTION_TRACE === '1') {
      const binding = snapshot.bindings.find((item) =>
        item.key.startsWith('texture:smoke advected:'),
      )!;
      const level = binding.texture!.levels[0]!;
      const actual = level.data as Uint16Array;
      const expected = new Uint16Array(
        gpu.find((item) => item.key === binding.key)!.levels[0]!.bytes,
      );

      const gap = (word: number) => Math.abs(ordered(actual[word]!) - ordered(expected[word]!));
      const worst = Array.from({ length: actual.length / 4 }, (_, texel) => {
        let channel = 0;
        for (let k = 1; k < 4; k++) if (gap(texel * 4 + k) > gap(texel * 4 + channel)) channel = k;
        const word = texel * 4 + channel;
        return {
          texel,
          channel,
          cpu: actual[word]!,
          dawn: expected[word]!,
          storedCpu: Array.from(actual.subarray(texel * 4, texel * 4 + 4)),
          storedDawn: Array.from(expected.subarray(texel * 4, texel * 4 + 4)),
          ulps: gap(word),
        };
      })
        .toSorted((a, b) => b.ulps - a.ulps)
        .filter((item) => item.ulps > 1);
      for (const { target, bytes } of savedInputs) target.set(bytes);
      const traces = await traceAdvection(
        device,
        event.code,
        groups,
        snapshot,
        resources,
        worst.map((item) => item.texel),
        [level.width, level.height, level.depthOrArrayLayers],
      );
      advectionTrace = {
        summary: summarizeAdvection(traces, worst),
        worst: worst.slice(0, 10),
        traces: traces.slice(0, 10),
      };
    }
    let materialTrace;
    if (process.env.CPU_MATERIAL_TRACE === '1') {
      const binding = snapshot.bindings.find((item) => item.texture && item.use.writes)!;
      const level = binding.texture!.levels[0]!;
      const actual = level.data as Uint8Array;
      const expected = new Uint8Array(
        gpu.find((item) => item.key === binding.key)!.levels[0]!.bytes,
      );
      const worst = Array.from({ length: actual.length / 4 }, (_, texel) => ({
        texel,
        gap: Math.abs(actual[texel * 4]! - expected[texel * 4]!),
        cpu: actual[texel * 4],
        dawn: expected[texel * 4],
      }))
        .toSorted((a, b) => b.gap - a.gap)
        .filter((item) => item.gap > 1);
      const traces = await traceMaterial(
        device,
        event.code,
        groups,
        snapshot,
        resources,
        worst.map((item) => item.texel),
        [level.width, level.height, level.depthOrArrayLayers],
      );
      const rawBits = new Uint32Array(1),
        float = new Float32Array(rawBits.buffer);
      const bits = (value: number) => {
        float[0] = value;
        return rawBits[0]!;
      };

      const summary = traces.map((trace) => {
        const c = trace.cpu.find((row) => row.tag === 3)!.value,
          d = trace.dawn.find((row) => row.tag === 3)!.value;
        const actualStore = trace.sameCenter.find((row) => row.tag === 112)!.value,
          expectedStore = trace.dawn.find((row) => row.tag === 112)!.value;
        return {
          cell: trace.cell,
          coordinateUlps: Math.max(
            ...c.slice(0, 2).map((value, i) => Math.abs(bits(value) - bits(d[i]!))),
          ),
          centerChanged: c[2] !== d[2] || c[3] !== d[3],
          sameCenterMaxBytes: Math.max(
            ...actualStore.map((value, i) => Math.abs(unorm(value) - unorm(expectedStore[i]!))),
          ),
        };
      });
      materialTrace = { summary, worst: worst.slice(0, 10), traces: traces.slice(0, 10) };
    }
    let skyTrace;
    let prefilterTrace;
    if (
      process.env.CPU_SKY_TRACE === '1' ||
      process.env.CPU_PREFILTER_TRACE === '1' ||
      options.prefilter
    ) {
      const binding = snapshot.bindings.find((item) => item.texture && item.use.writes)!;
      const level = binding.texture!.levels[0]!;
      const actual = level.data as Uint16Array,
        expected = new Uint16Array(gpu.find((item) => item.key === binding.key)!.levels[0]!.bytes);
      const worst = Array.from({ length: actual.length / 4 }, (_, texel) => ({
        texel,
        gap: Math.max(
          ...[0, 1, 2].map((c) => Math.abs(actual[texel * 4 + c]! - expected[texel * 4 + c]!)),
        ),
      }))
        .toSorted((a, b) => b.gap - a.gap)
        .slice(0, 10);
      skyTrace =
        process.env.CPU_PREFILTER_TRACE === '1' || options.prefilter
          ? undefined
          : await traceSky(
              device,
              event.code,
              groups,
              snapshot,
              resources,
              worst.map((item) => item.texel),
              level.width,
            );
      if (process.env.CPU_PREFILTER_TRACE === '1' || options.prefilter)
        prefilterTrace = await tracePrefilter(
          device,
          event.code,
          groups,
          snapshot,
          resources,
          worst.map((item) => item.texel),
          level.width,
        );
    }
    let samplingTrace;
    if (options.sampling || options.geometry) {
      const binding = snapshot.bindings.find(
        (item) => (item.texture || item.buffer) && item.use.writes,
      )!;
      const level = binding.texture?.levels[0];
      if (binding.texture && binding.texture.format !== 'rgba16float')
        throw new Error('Sampling trace output must be rgba16float');
      const actual = level?.data as Uint16Array | undefined;
      const expectedBytes = gpu.find((item) => item.key === binding.key)!.levels[0]!.bytes;
      const expected = level ? new Uint16Array(expectedBytes) : new Uint32Array(expectedBytes);
      const worst = actual
        ? Array.from({ length: actual.length / 4 }, (_, texel) => ({
            texel,
            gap: Math.max(
              ...[0, 1, 2, 3].map((c) =>
                Math.abs(ordered(actual[texel * 4 + c]!) - ordered(expected[texel * 4 + c]!)),
              ),
            ),
          }))
            .toSorted((a, b) => b.gap - a.gap)
            .slice(0, 10)
        : [{ texel: 0, gap: 0 }];
      samplingTrace = await traceSampling(
        device,
        event.code,
        groups,
        snapshot,
        resources,
        worst.map((item) => item.texel),
        level ? [level.width, level.height, level.depthOrArrayLayers] : [1, 1, 1],
        binding.texture ?? binding.buffer!,
        expected,
        options.geometry,
      );
    }
    return {
      samplingTrace,
      prefilterTrace,
      skyTrace,
      materialTrace,
      advectionTrace,
      filterModels,
      pipeline: snapshot.label,
      workgroups: snapshot.workgroups,
      ...compareCaptures(
        { resources: gpu, skipped: [], frames: 0 },
        { resources: cpu, skipped: [], frames: 0 },
      ),
    };
  } finally {
    for (const buffer of buffers.values()) buffer.destroy();
    for (const texture of textures.values()) texture.destroy();
  }
}
