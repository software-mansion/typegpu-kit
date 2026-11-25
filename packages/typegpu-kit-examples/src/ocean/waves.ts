import { backends, pipelines } from '@typegpu/kit';
import { randf } from '@typegpu/noise';
import { d, std, tgpu, type TgpuComputePass, type TgpuRoot } from 'typegpu';
import { createWaveMips } from './wave-mips.ts';
import {
  CASCADES,
  Cascade,
  cascadeValues,
  dispersion,
  Foaming,
  LOG2N,
  LOOP_PERIOD,
  MIP_LEVELS,
  mirrorCell,
  N,
  REACH,
  SIGNALS,
  SpectrumParams,
  surfaceJacobian,
  waveVariance,
  worldToPatch,
} from './spectrum.ts';

const TILE = 16;
const EVOLVE_GROUP = 256;
const HALF_N = N / 2;
const OMEGA_0 = (2 * Math.PI) / LOOP_PERIOD;
/** How far the waves' crests lean and sharpen, as a share of the linear waves' motion. */
const CHOPPINESS = 1.3;
const STRIDES = Array.from({ length: LOG2N }, (_, stage) => 1 << stage);
const SIGNAL_INDICES = Array.from({ length: SIGNALS }, (_, i) => i);

/** Kernels visit only the cells within `REACH`, listed across cascades from these starts. */
const SIDES = REACH.map((reach) => 2 * reach + 1);
const startsOf = (counts: number[]) =>
  counts.map((_, c) => counts.slice(0, c).reduce((a, b) => a + b, 0));
const ROW_STARTS = startsOf(SIDES);
const ROWS = ROW_STARTS[CASCADES - 1] + SIDES[CASCADES - 1];
const CELL_STARTS = startsOf(SIDES.map((side) => side * side));
const CELLS = CELL_STARTS[CASCADES - 1] + SIDES[CASCADES - 1] ** 2;

const reachOf = tgpu.const(d.arrayOf(d.u32, CASCADES), REACH);
const rowStarts = tgpu.const(d.arrayOf(d.u32, CASCADES), ROW_STARTS);
const cellStarts = tgpu.const(d.arrayOf(d.u32, CASCADES), CELL_STARTS);

const cascadeAt = (starts: number[]) => {
  const later = starts.slice(1);
  return (index: number) => {
    'use gpu';
    let c = d.u32(0);
    for (const start of tgpu.unroll(later)) {
      if (index >= start) {
        c += 1;
      }
    }
    return c;
  };
};
const cascadeOfRow = cascadeAt(ROW_STARTS);
const cascadeOfCell = cascadeAt(CELL_STARTS);

const Step = d.struct({ time: d.f32, dt: d.f32 });

const rowsLayout = tgpu.bindGroupLayout({
  data: { storage: d.arrayOf(d.vec2f), access: 'mutable' },
});

const field = d.textureStorage2dArray('rgba16float', 'write-only');
const columnsLayout = tgpu.bindGroupLayout({
  rows: { storage: d.arrayOf(d.vec2f), access: 'readonly' },
  displacement: { storageTexture: field },
  derivatives: { storageTexture: field },
  moments: { storageTexture: field },
});

const cmul = (a: d.v2f, b: d.v2f) => {
  'use gpu';
  return d.vec2f(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
};

/** a + i·b for two spectra whose inverse transforms are real. */
const pack = (a: d.v2f, b: d.v2f) => {
  'use gpu';
  return d.vec2f(a.x - b.y, a.y + b.x);
};

/** Where thread `t` loads its two samples, `t` and `t` + N/2, in bit-reversed order. */
const reversedOf = (t: number) => {
  'use gpu';
  return std.reverseBits(t) >>> (32 - LOG2N);
};

/** Inverse-transforms `count` signals of `lines` in place: bit-reversed in, natural order out. */
const butterfliesOf = (lines: { $: d.v2f[] }, count: number) => {
  const signals = SIGNAL_INDICES.slice(0, count);
  return (t: number) => {
    'use gpu';
    std.workgroupBarrier();
    for (const stride of tgpu.unroll(STRIDES)) {
      const j = t & (stride - 1);
      const first = (t - j) * 2 + j;
      const angle = (Math.PI / stride) * d.f32(j);
      const twiddle = d.vec2f(std.cos(angle), std.sin(angle));
      for (const signal of tgpu.unroll(signals)) {
        const at = signal * N + first;
        const a = d.vec2f(lines.$[at]);
        const b = cmul(lines.$[at + stride], twiddle);
        lines.$[at] = a + b;
        lines.$[at + stride] = a - b;
      }
      std.workgroupBarrier();
    }
  };
};

const row = tgpu.workgroupVar(d.arrayOf(d.vec2f, N));
const rowButterflies = butterfliesOf(row, 1);

/** One workgroup inverse-transforms one row within reach; samples beyond the reach are zero. */
const transformRows = tgpu.computeFn({
  workgroupSize: [HALF_N],
  in: { lid: d.builtin.localInvocationId, wid: d.builtin.workgroupId },
})(({ lid, wid }) => {
  'use gpu';
  const t = lid.x;
  const cascade = cascadeOfRow(wid.x);
  const reach = reachOf.$[cascade];
  const base =
    ((cascade * SIGNALS + wid.y) * N + HALF_N - reach + wid.x - rowStarts.$[cascade]) * N;
  const reversed = reversedOf(t);
  row.$[reversed] = d.vec2f();
  row.$[reversed + 1] = d.vec2f();
  if (t + reach >= HALF_N) {
    row.$[reversed] = d.vec2f(rowsLayout.$.data[base + t]);
  }
  if (t <= reach) {
    row.$[reversed + 1] = d.vec2f(rowsLayout.$.data[base + t + HALF_N]);
  }
  rowButterflies(t);
  rowsLayout.$.data[base + t] = d.vec2f(row.$[t]);
  rowsLayout.$.data[base + t + HALF_N] = d.vec2f(row.$[t + HALF_N]);
});

const columns = tgpu.workgroupVar(d.arrayOf(d.vec2f, SIGNALS * N));

/**
 * Radix-2 Stockham steps of an inverse transform, from `from` into `into`, for backends where
 * workgroups share nothing. Each invocation gathers its own output element, so the steps read and
 * write natural order; a pass takes three steps.
 */
const stageLayout = tgpu.bindGroupLayout({
  from: { storage: d.arrayOf(d.vec2f), access: 'readonly' },
  into: { storage: d.arrayOf(d.vec2f), access: 'mutable' },
  span: { uniform: d.u32 },
});

/** The butterfly output at `o` of a step whose pairs lie `span` apart, from inputs `a` and `b`. */
const butterfly = (o: number, span: number, a: d.v2f, b: d.v2f) => {
  'use gpu';
  const angle = (Math.PI * d.f32(o % span)) / d.f32(span);
  const w = cmul(b, d.vec2f(std.cos(angle), std.sin(angle)));
  return std.select(a - w, a + w, o % (span * 2) < span);
};

/** The input pair's first element for output `o` of a step whose pairs lie `span` apart. */
const pairOf = (o: number, span: number) => {
  'use gpu';
  return std.intdiv(o, span * 2) * span + (o % span);
};

/**
 * Two radix-2 steps in one, spans `span` and `2 * span`: output `o` of the second gathers the two
 * outputs of the first it pairs, each from its two inputs, so it does the same arithmetic as two
 * passes with one pass's reads and writes fewer. Input `i` of the line is at `base + i * stride`.
 */
const twice = (o: number, span: number, base: number, stride: number) => {
  'use gpu';
  const wide = span * 2;
  const k = pairOf(o, wide);
  const first = pairOf(k, span);
  const second = pairOf(k + HALF_N, span);
  const from = stageLayout.$.from;
  const x = butterfly(
    k,
    span,
    d.vec2f(from[base + first * stride]),
    d.vec2f(from[base + (first + HALF_N) * stride]),
  );
  const y = butterfly(
    k + HALF_N,
    span,
    d.vec2f(from[base + second * stride]),
    d.vec2f(from[base + (second + HALF_N) * stride]),
  );
  return butterfly(o, wide, x, y);
};

/** Three radix-2 steps in one, spans `span`, `2 * span` and `4 * span`, as `twice` gathers two. */
const thrice = (o: number, span: number, base: number, stride: number) => {
  'use gpu';
  const wide = span * 4;
  const k = pairOf(o, wide);
  const x = twice(k, span, base, stride);
  const y = twice(k + HALF_N, span, base, stride);
  return butterfly(o, wide, x, y);
};

/** Steps every row of every signal three times, as `thrice` does. */
const rowStages = tgpu.computeFn({
  workgroupSize: [64],
  in: { gid: d.builtin.globalInvocationId },
})(({ gid }) => {
  'use gpu';
  const line = (gid.z * N + gid.y) * N;
  stageLayout.$.into[line + gid.x] = thrice(gid.x, stageLayout.$.span, line, d.u32(1));
});

/** Steps every column of every signal three times, as `thrice` does. */
const columnStages = tgpu.computeFn({
  workgroupSize: [64],
  in: { gid: d.builtin.globalInvocationId },
})(({ gid }) => {
  'use gpu';
  const plane = gid.z * N * N + gid.x;
  stageLayout.$.into[plane + gid.y * N] = thrice(gid.y, stageLayout.$.span, plane, d.u32(N));
});

const columnButterflies = butterfliesOf(columns, SIGNALS);

export type Waves = ReturnType<typeof createWaves>;

/** Four FFT cascades of an ocean spectrum, mipmapped as displacement, slope and moment maps. */
export function createWaves(root: TgpuRoot) {
  const spectrum = root.createUniform(SpectrumParams);
  const cascades = root.createUniform(d.arrayOf(Cascade, CASCADES), cascadeValues);
  const step = root.createUniform(Step, { time: 0, dt: 0 });
  const foaming = root.createUniform(d.arrayOf(Foaming, CASCADES));
  const foam = root.createMutable(d.arrayOf(d.f32, CASCADES * N * N));
  const amplitudes = root.createMutable(d.arrayOf(d.vec4f, CASCADES * N * N));
  const omegas = root.createMutable(d.arrayOf(d.f32, CASCADES * N * N));

  const lattice = root
    .createBuffer(d.arrayOf(d.vec2f, CASCADES * SIGNALS * N * N))
    .$usage('storage');
  const signals = lattice.as('mutable');
  const sampler = root.createSampler({
    magFilter: 'linear',
    minFilter: 'linear',
    mipmapFilter: 'linear',
    addressModeU: 'repeat',
    addressModeV: 'repeat',
    maxAnisotropy: 8,
  });

  const createMap = () =>
    root
      .createTexture({ size: [N, N, CASCADES], format: 'rgba16float', mipLevelCount: MIP_LEVELS })
      .$usage('sampled', 'storage');
  const maps = [createMap(), createMap(), createMap()] as const;
  const displacement = maps[0].createView(d.texture2dArray(d.f32));
  const derivatives = maps[1].createView(d.texture2dArray(d.f32));
  const moments = maps[2].createView(d.texture2dArray(d.f32));
  const mips = createWaveMips(root, maps);
  const top = { baseMipLevel: 0, mipLevelCount: 1 };

  /** h0(k), conj(h0(-k)) and the looped frequency of every lattice cell of every cascade. */
  const initialSpectrum = tgpu.computeFn({
    workgroupSize: [TILE, TILE],
    in: { gid: d.builtin.globalInvocationId },
  })(({ gid }) => {
    'use gpu';
    if (gid.x >= N || gid.y >= N) return;
    const cascade = Cascade(cascades.$[gid.z]);
    const params = SpectrumParams(spectrum.$);
    const dk = (2 * Math.PI) / cascade.patchSize;
    const kVec = (d.vec2f(gid.xy) - N / 2) * dk;
    const seed = params.seed + d.f32(gid.z) * 17;
    const valid = std.select(d.f32(1), d.f32(0), gid.x === 0 || gid.y === 0);
    randf.seed3(d.vec3f(d.vec2f(gid.xy), seed));
    const h0 =
      d.vec2f(randf.normal(0, 1), randf.normal(0, 1)) *
      std.sqrt(0.5 * waveVariance(kVec, cascade, params) * valid);
    randf.seed3(d.vec3f(d.vec2f(mirrorCell(gid.xy)), seed));
    const h0Mirror =
      d.vec2f(randf.normal(0, 1), randf.normal(0, 1)) *
      std.sqrt(0.5 * waveVariance(kVec * -1, cascade, params) * valid);
    const idx = (gid.z * N + gid.y) * N + gid.x;
    amplitudes.$[idx] = d.vec4f(h0, h0Mirror.x, -h0Mirror.y);
    const omega = dispersion(std.max(std.length(kVec), 1e-6)).x;
    const looped = std.floor(omega / OMEGA_0) * OMEGA_0;
    omegas.$[idx] = looped;
  });

  const evolve = tgpu.computeFn({
    workgroupSize: [EVOLVE_GROUP],
    in: { gid: d.builtin.globalInvocationId },
  })(({ gid }) => {
    'use gpu';
    if (gid.x >= CELLS) return;
    const c = cascadeOfCell(gid.x);
    const reach = reachOf.$[c];
    const side = reach * 2 + 1;
    const local = gid.x - cellStarts.$[c];
    const cell = d.vec2u(local % side, std.intdiv(local, side)) + (HALF_N - reach);
    const idx = (c * N + cell.y) * N + cell.x;
    const h0 = amplitudes.$[idx];
    const phase = omegas.$[idx] * step.$.time;
    const rotation = d.vec2f(std.cos(phase), std.sin(phase));
    const h = cmul(h0.xy, rotation) + cmul(h0.zw, d.vec2f(rotation.x, -rotation.y));
    const dk = (2 * Math.PI) / cascades.$[c].patchSize;
    const k = (d.vec2f(cell) - N / 2) * dk;
    const kLen = std.max(std.length(k), 1e-6);
    const ih = d.vec2f(-h.y, h.x);
    const base = c * SIGNALS * N * N + cell.y * N + cell.x;
    // Displacement x, height; displacement z, slope x; slope z, ∂Dx/∂x; ∂Dz/∂z, ∂Dx/∂z
    signals.$[base] = pack(ih * (-k.x / kLen), h);
    signals.$[base + N * N] = pack(ih * (-k.y / kLen), ih * k.x);
    signals.$[base + 2 * N * N] = pack(ih * k.y, h * ((k.x * k.x) / kLen));
    signals.$[base + 3 * N * N] = pack(h * ((k.y * k.y) / kLen), h * ((k.x * k.y) / kLen));
  });

  /**
   * Packs a transformed texel's displacement, slopes and moments from its four signals, and
   * carries its foam in `cell`.
   */
  const packSignals = (
    texel: d.v2u,
    c: number,
    cell: number,
    signal0: d.v2f,
    signal1: d.v2f,
    signal2: d.v2f,
    signal3: d.v2f,
  ) => {
    'use gpu';
    const sign = std.select(d.f32(1), d.f32(-1), ((texel.x + texel.y) & 1) === 1);
    const s0 = signal0 * sign;
    const s1 = signal1 * sign;
    const s2 = signal2 * sign;
    const s3 = signal3 * sign;
    // With -i·k/|k|, a negative gain compresses the crests; a positive one sharpens the troughs
    const lambda = -CHOPPINESS * cascades.$[c].chop;
    const gradient = d.vec2f(s1.y, s2.x);
    const xx = lambda * s2.y;
    const zz = lambda * s3.x;
    const xz = lambda * s3.y;
    const jacobian = surfaceJacobian(d.vec3f(xx, zz, xz));
    const rule = Foaming(foaming.$[c]);
    const foamed = std.saturate(
      foam.$[cell] * std.exp(-rule.decay * step.$.dt) +
        std.max(rule.whitecap - jacobian, 0) * rule.grow * step.$.dt,
    );
    foam.$[cell] = foamed;
    std.textureStore(
      columnsLayout.$.displacement,
      texel,
      c,
      d.vec4f(lambda * s0.x, s0.y, lambda * s1.x, xz),
    );
    std.textureStore(columnsLayout.$.derivatives, texel, c, d.vec4f(gradient, xx, zz));
    std.textureStore(
      columnsLayout.$.moments,
      texel,
      c,
      d.vec4f(gradient.x * gradient.x, gradient.y * gradient.y, gradient.x * gradient.y, foamed),
    );
  };

  /** Packs a texel of the column the workgroup transformed. */
  const packTexel = (texel: d.v2u, c: number) => {
    'use gpu';
    packSignals(
      texel,
      c,
      (c * N + texel.x) * N + texel.y,
      d.vec2f(columns.$[texel.y]),
      d.vec2f(columns.$[N + texel.y]),
      d.vec2f(columns.$[2 * N + texel.y]),
      d.vec2f(columns.$[3 * N + texel.y]),
    );
  };

  /** One workgroup inverse-transforms one column of a cascade's signals and packs its texels. */
  const transformColumns = tgpu.computeFn({
    workgroupSize: [HALF_N],
    in: { lid: d.builtin.localInvocationId, wid: d.builtin.workgroupId },
  })(({ lid, wid }) => {
    'use gpu';
    const t = lid.x;
    const x = wid.x;
    const c = wid.y;
    const reach = reachOf.$[c];
    const reversed = reversedOf(t);
    for (const signal of tgpu.unroll(SIGNAL_INDICES)) {
      const column = (c * SIGNALS + signal) * N * N + x;
      let low = d.vec2f();
      let high = d.vec2f();
      if (t + reach >= HALF_N) {
        low = d.vec2f(columnsLayout.$.rows[column + t * N]);
      }
      if (t <= reach) {
        high = d.vec2f(columnsLayout.$.rows[column + (t + HALF_N) * N]);
      }
      columns.$[signal * N + reversed] = d.vec2f(low);
      columns.$[signal * N + reversed + 1] = d.vec2f(high);
    }
    columnButterflies(t);
    for (const half of tgpu.unroll([0, HALF_N])) {
      packTexel(d.vec2u(x, t + half), c);
    }
  });

  // Where compute runs on the CPU, the WebGL fallback can run kernels that each write their own
  // element on its GPU, but not workgroups that share memory: there the transforms step through
  // Stockham stages over a second lattice instead, with the evolve and the packing per element
  const cpu = backends.of(root).compute === 'cpu';

  /** Evolves one signal of one cell, `gid` being (x, y, signal of a cascade), zero beyond reach. */
  const evolveSignal = tgpu.computeFn({
    workgroupSize: [64],
    in: { gid: d.builtin.globalInvocationId },
  })(({ gid }) => {
    'use gpu';
    const at = (gid.z * N + gid.y) * N + gid.x;
    const c = std.intdiv(gid.z, d.u32(SIGNALS));
    const signal = gid.z % SIGNALS;
    const reach = reachOf.$[c];
    if (
      gid.x + reach < HALF_N ||
      gid.x > HALF_N + reach ||
      gid.y + reach < HALF_N ||
      gid.y > HALF_N + reach
    ) {
      signals.$[at] = d.vec2f();
      return;
    }
    // As `evolve` does for the cell, keeping one of its four signals
    const idx = (c * N + gid.y) * N + gid.x;
    const h0 = amplitudes.$[idx];
    const phase = omegas.$[idx] * step.$.time;
    const rotation = d.vec2f(std.cos(phase), std.sin(phase));
    const h = cmul(h0.xy, rotation) + cmul(h0.zw, d.vec2f(rotation.x, -rotation.y));
    const dk = (2 * Math.PI) / cascades.$[c].patchSize;
    const k = (d.vec2f(gid.xy) - N / 2) * dk;
    const kLen = std.max(std.length(k), 1e-6);
    const ih = d.vec2f(-h.y, h.x);
    let value = pack(h * ((k.y * k.y) / kLen), h * ((k.x * k.y) / kLen));
    if (signal === 0) value = pack(ih * (-k.x / kLen), h);
    if (signal === 1) value = pack(ih * (-k.y / kLen), ih * k.x);
    if (signal === 2) value = pack(ih * k.y, h * ((k.x * k.x) / kLen));
    signals.$[at] = d.vec2f(value);
  });

  /** Packs one texel of one cascade, `gid` being (x, y, cascade), from the transformed lattice. */
  const packDense = tgpu.computeFn({
    workgroupSize: [8, 8],
    in: { gid: d.builtin.globalInvocationId },
  })(({ gid }) => {
    'use gpu';
    const at = (gid.z * SIGNALS * N + gid.y) * N + gid.x;
    packSignals(
      gid.xy,
      gid.z,
      (gid.z * N + gid.y) * N + gid.x,
      d.vec2f(columnsLayout.$.rows[at]),
      d.vec2f(columnsLayout.$.rows[at + N * N]),
      d.vec2f(columnsLayout.$.rows[at + 2 * N * N]),
      d.vec2f(columnsLayout.$.rows[at + 3 * N * N]),
    );
  });

  const spare = cpu
    ? root.createBuffer(d.arrayOf(d.vec2f, CASCADES * SIGNALS * N * N)).$usage('storage')
    : undefined;
  // Steps go three spans a pass, LOG2N being a multiple of three: an odd number of passes, as of
  // steps, so rows step from the lattice and end in the spare one, columns step back
  const spans = cpu ? STRIDES.filter((_, s) => s % 3 === 0) : [];
  const stagesOf = (start: 'lattice' | 'spare') =>
    spans.map((first, s) => {
      const span = root.createUniform(d.u32, first);
      const fromLattice = (s % 2 === 0) === (start === 'lattice');
      const group = root.createBindGroup(stageLayout, {
        from: fromLattice ? lattice : (spare as typeof lattice),
        into: fromLattice ? (spare as typeof lattice) : lattice,
        span,
      });
      return group;
    });
  const rowPasses = stagesOf('lattice');
  const columnPasses = stagesOf('spare');
  const evolveSignalPipeline = cpu
    ? root.createComputePipeline({ compute: evolveSignal })
    : undefined;
  const rowStagesPipeline = cpu ? root.createComputePipeline({ compute: rowStages }) : undefined;
  const columnStagesPipeline = cpu
    ? root.createComputePipeline({ compute: columnStages })
    : undefined;
  const packPipeline = cpu ? root.createComputePipeline({ compute: packDense }) : undefined;

  const initialPipeline = root.createComputePipeline({ compute: initialSpectrum });
  const evolvePipeline = cpu ? undefined : root.createComputePipeline({ compute: evolve });
  const rowsPipeline = cpu ? undefined : root.createComputePipeline({ compute: transformRows });
  const columnsPipeline = cpu
    ? undefined
    : root.createComputePipeline({ compute: transformColumns });

  const rowsGroup = root.createBindGroup(rowsLayout, { data: lattice });
  const columnsGroup = root.createBindGroup(columnsLayout, {
    rows: lattice,
    displacement: maps[0].createView(field, top),
    derivatives: maps[1].createView(field, top),
    moments: maps[2].createView(field, top),
  });

  const patchUv = (xz: d.v2f, c: number) => {
    'use gpu';
    return worldToPatch(xz, cascades.$[c].rotation) / cascades.$[c].patchSize;
  };

  /** Samples a wave field filtered over a pixel's world footprint, `dx` by `dy`. */
  const filtered = (map: d.texture2dArray<d.F32>, xz: d.v2f, c: number, dx: d.v2f, dy: d.v2f) => {
    'use gpu';
    const uv = patchUv(xz, c);
    const cascade = cascades.$[c];
    const uvDx = worldToPatch(dx, cascade.rotation) / cascade.patchSize;
    const uvDy = worldToPatch(dy, cascade.rotation) / cascade.patchSize;
    return std.textureSampleGrad(map, sampler.$, uv, c, uvDx, uvDy);
  };

  /** Cubic B-spline through four bilinear taps, for fields magnified past their texels. */
  const smoothField = (map: d.texture2dArray<d.F32>, xz: d.v2f, c: number) => {
    'use gpu';
    const texel = patchUv(xz, c) * N - 0.5;
    const base = std.floor(texel);
    const f = texel - base;
    const f2 = f * f;
    const f3 = f2 * f;
    const w0 = (f2 * 3 - f3 - f * 3 + 1) / 6;
    const w1 = (f3 * 3 - f2 * 6 + 4) / 6;
    const w2 = (f2 * 3 - f3 * 3 + f * 3 + 1) / 6;
    const w3 = f3 / 6;
    const g0 = w0 + w1;
    const g1 = w2 + w3;
    const at0 = (base + 0.5 + w1 / g0 - 1) / N;
    const at1 = (base + 0.5 + w3 / g1 + 1) / N;
    const a = std.textureSampleLevel(map, sampler.$, d.vec2f(at0.x, at0.y), c, 0);
    const b = std.textureSampleLevel(map, sampler.$, d.vec2f(at1.x, at0.y), c, 0);
    const e = std.textureSampleLevel(map, sampler.$, d.vec2f(at0.x, at1.y), c, 0);
    const h = std.textureSampleLevel(map, sampler.$, d.vec2f(at1.x, at1.y), c, 0);
    return (a * g0.x + b * g1.x) * g0.y + (e * g0.x + h * g1.x) * g1.y;
  };

  const smoothed = (map: d.texture2dArray<d.F32>, xz: d.v2f, c: number, dx: d.v2f, dy: d.v2f) => {
    'use gpu';
    if (std.max(std.length(dx), std.length(dy)) < cascades.$[c].patchSize / N) {
      return smoothField(map, xz, c);
    }
    return filtered(map, xz, c, dx, dy);
  };

  return {
    initAsync: () =>
      pipelines.compile(
        cpu
          ? [
              initialPipeline,
              evolveSignalPipeline!,
              rowStagesPipeline!,
              columnStagesPipeline!,
              packPipeline!,
              mips.pipeline,
            ]
          : [initialPipeline, evolvePipeline!, rowsPipeline!, columnsPipeline!, mips.pipeline],
      ),
    spectrum,
    cascades,
    step,
    foaming,

    /** The displacement at mip level `lod`, for kernels and vertices. */
    displacementAt: (xz: d.v2f, c: number, lod: number) => {
      'use gpu';
      return std.textureSampleLevel(displacement.$, sampler.$, patchUv(xz, c), c, lod);
    },
    /** The fields filtered over a pixel's world footprint, `dx` by `dy`, for fragments. */
    displacement: (xz: d.v2f, c: number, dx: d.v2f, dy: d.v2f) => {
      'use gpu';
      return filtered(displacement.$, xz, c, dx, dy);
    },
    derivatives: (xz: d.v2f, c: number, dx: d.v2f, dy: d.v2f) => {
      'use gpu';
      return filtered(derivatives.$, xz, c, dx, dy);
    },
    moments: (xz: d.v2f, c: number, dx: d.v2f, dy: d.v2f) => {
      'use gpu';
      return filtered(moments.$, xz, c, dx, dy);
    },
    /**
     * Like `derivatives` and `moments`, but smooth where a pixel is smaller than the cascade's
     * texel. Use them together: the same positive weights keep the slopes' variance non-negative.
     */
    smoothDerivatives: (xz: d.v2f, c: number, dx: d.v2f, dy: d.v2f) => {
      'use gpu';
      return smoothed(derivatives.$, xz, c, dx, dy);
    },
    smoothMoments: (xz: d.v2f, c: number, dx: d.v2f, dy: d.v2f) => {
      'use gpu';
      return smoothed(moments.$, xz, c, dx, dy);
    },

    /** Records the initial amplitudes of the spectrum `spectrum` holds; write it first. */
    rebuild(pass: TgpuComputePass) {
      initialPipeline.with(pass).dispatchWorkgroups(N / TILE, N / TILE, CASCADES);
    },
    record(pass: TgpuComputePass) {
      if (cpu) {
        const lines = [N / 64, N, CASCADES * SIGNALS] as const;
        evolveSignalPipeline!.with(pass).dispatchWorkgroups(...lines);
        for (const group of rowPasses) {
          rowStagesPipeline!
            .with(pass)
            .with(group)
            .dispatchWorkgroups(...lines);
        }
        for (const group of columnPasses) {
          columnStagesPipeline!
            .with(pass)
            .with(group)
            .dispatchWorkgroups(...lines);
        }
        packPipeline!
          .with(pass)
          .with(columnsGroup)
          .dispatchWorkgroups(N / 8, N / 8, CASCADES);
      } else {
        evolvePipeline!.with(pass).dispatchWorkgroups(Math.ceil(CELLS / EVOLVE_GROUP));
        rowsPipeline!.with(pass).with(rowsGroup).dispatchWorkgroups(ROWS, SIGNALS);
        columnsPipeline!.with(pass).with(columnsGroup).dispatchWorkgroups(N, CASCADES);
      }
      mips.record(pass);
    },
    destroy() {
      for (const resource of [foam.buffer, amplitudes.buffer, omegas.buffer, lattice, ...maps]) {
        resource.destroy();
      }
      spare?.destroy();
    },
  };
}
