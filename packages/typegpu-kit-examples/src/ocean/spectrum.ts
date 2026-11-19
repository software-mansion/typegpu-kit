import { d, std } from 'typegpu';

export const N = 512;
export const LOG2N = 9;
export const CASCADES = 4;
/** The wind waves' cascade, whose crests break and pass light. */
export const WIND = 1;
/** The ripples' cascade only shades: its waves are shorter than the mesh resolves. */
export const RIPPLES = 3;
/** Four packed complex signals carry eight real fields per cascade. */
export const SIGNALS = 4;
export const MIP_LEVELS = LOG2N + 1;
/** Rotated, differently sized domains keep the shorter cascades from showing repeated tiles. */
const PATCH_SIZES = [2048, 613, 101, 12];
const PATCH_ROTATIONS = [13, 47, 101, 71];
/** Wavelength boundaries in metres between the swell, wind wave, chop and ripple cascades. */
const BAND_EDGES = [640, 35, 5, 0.9, 0.06];
const BAND_TAPER = 0.15;
const CASCADE_GAINS = [
  { height: 1.2, chop: 1 },
  { height: 1.15, chop: 1 },
  { height: 1, chop: 1 },
  { height: 1, chop: 0.3 },
];
const GRAVITY = 9.81;
/** Frequencies snap to multiples of 2π over this many seconds, so the sea repeats exactly. */
export const LOOP_PERIOD = 200;
const SWELL_PERIOD_RATIO = 1.7;
const SWELL_PEAKEDNESS = 7;
const SWELL_NARROWING = 1.6;
const PEAK_BROADENING = 0.45;
const PEAK_BROADENING_WIDTH = 0.3;
const ISOTROPIC_FLOOR = 0.35;

export const SpectrumParams = d.struct({
  windDirection: d.f32,
  peakFrequency: d.f32,
  energyScale: d.f32,
  peakedness: d.f32,
  directionality: d.f32,
  swellDirection: d.f32,
  swellFrequency: d.f32,
  swellScale: d.f32,
  waveHeight: d.f32,
  seed: d.f32,
});

/** `rotation` holds the cosine and sine of the angle that takes patch space to world space. */
export const Cascade = d.struct({
  patchSize: d.f32,
  kLow: d.f32,
  kHigh: d.f32,
  maxWavelength: d.f32,
  rotation: d.vec2f,
  heightGain: d.f32,
  chop: d.f32,
});

/**
 * A cascade's foam rule, after GodotOceanWaves: foam grows at `grow` per unit its Jacobian falls
 * below `whitecap`, and decays by `decay`, each second. Padded to a uniform array's 16-byte stride.
 */
export const Foaming = d.struct({ whitecap: d.f32, grow: d.f32, decay: d.size(8, d.f32) });

export const cascadeValues: d.Infer<typeof Cascade>[] = PATCH_SIZES.map((patchSize, c) => {
  const angle = (PATCH_ROTATIONS[c] * Math.PI) / 180;
  return {
    patchSize,
    kLow: (2 * Math.PI) / BAND_EDGES[c],
    kHigh: (2 * Math.PI) / BAND_EDGES[c + 1],
    maxWavelength: BAND_EDGES[c],
    rotation: d.vec2f(Math.cos(angle), Math.sin(angle)),
    heightGain: CASCADE_GAINS[c].height,
    chop: CASCADE_GAINS[c].chop,
  };
});

/** Each cascade's waves lie within this many cells of its lattice's centre; FFTs skip the rest. */
export const REACH = cascadeValues.map(({ kHigh, patchSize }) =>
  Math.min(N / 2 - 1, Math.floor(((1 + BAND_TAPER) * kHigh * patchSize) / (2 * Math.PI)) + 1),
);

export const patchToWorld = (v: d.v2f, rotation: d.v2f) => {
  'use gpu';
  return d.vec2f(rotation.x * v.x - rotation.y * v.y, rotation.y * v.x + rotation.x * v.y);
};

export const worldToPatch = (v: d.v2f, rotation: d.v2f) => {
  'use gpu';
  return d.vec2f(rotation.x * v.x + rotation.y * v.y, rotation.x * v.y - rotation.y * v.x);
};

/** Rotates a symmetric tensor packed as (xx, yy, xy) from patch space to world space. */
export const tensorToWorld = (t: d.v3f, rotation: d.v2f) => {
  'use gpu';
  const c = rotation.x;
  const s = rotation.y;
  return d.vec3f(
    c * c * t.x - 2 * c * s * t.z + s * s * t.y,
    s * s * t.x + 2 * c * s * t.z + c * c * t.y,
    c * s * t.x + (c * c - s * s) * t.z - c * s * t.y,
  );
};

/** Determinant of the horizontal deformation, packed as (xx, zz, xz). */
export const surfaceJacobian = (deformation: d.v3f) => {
  'use gpu';
  return (1 + deformation.x) * (1 + deformation.y) - deformation.z * deformation.z;
};

/** The height gradient through the horizontal deformation, bounded where cells fold. */
export const surfaceSlope = (gradient: d.v2f, deformation: d.v3f) => {
  'use gpu';
  const determinant = std.max(surfaceJacobian(deformation), 0.05);
  const x = ((1 + deformation.y) * gradient.x - deformation.z * gradient.y) / determinant;
  const z = ((1 + deformation.x) * gradient.y - deformation.z * gradient.x) / determinant;
  const scale = std.max(std.sqrt(x * x + z * z) / 8, 1);
  return d.vec2f(x, z) / scale;
};

/** First-order slope covariance after horizontal deformation: M⁻¹ C M⁻ᵀ. */
export const surfaceCovariance = (cov: d.v3f, deformation: d.v3f) => {
  'use gpu';
  const determinant = std.max(surfaceJacobian(deformation), 0.05);
  const a = (1 + deformation.y) / determinant;
  const b = -deformation.z / determinant;
  const c = (1 + deformation.x) / determinant;
  return d.vec3f(
    a * a * cov.x + 2 * a * b * cov.z + b * b * cov.y,
    b * b * cov.x + 2 * b * c * cov.z + c * c * cov.y,
    a * b * cov.x + (a * c + b * b) * cov.z + b * c * cov.y,
  );
};

/** Deep water dispersion: omega and d omega / dk. */
export const dispersion = (k: number) => {
  'use gpu';
  const omega = std.sqrt(GRAVITY * k);
  return d.vec2f(omega, (0.5 * GRAVITY) / omega);
};

function jonswapEnergyScale(waveHeight: number, peakPeriod: number, gamma: number) {
  const omegaP = (2 * Math.PI) / peakPeriod;
  const steps = 4000;
  const from = 0.3 * omegaP;
  const dw = (12 * omegaP - from) / steps;
  let m0 = 0;
  for (let i = 0; i < steps; i++) m0 += jonswap(from + (i + 0.5) * dw, omegaP, gamma) * dw;
  return (waveHeight * waveHeight) / 16 / m0;
}

/** A young sea, shorter and steeper than a developed one, spread wide so its crests cross. */
const SEA = {
  height: [0.25, 0.012],
  period: [2, 0.27],
  peakedness: 3.3,
  directionality: 0.6,
  swell: 0.25,
  angle: 0.35,
};

/** The sea's spectrum under a wind of `speed` metres per second, blowing towards `direction`. */
export function spectrumAt(speed: number, direction: number): d.Infer<typeof SpectrumParams> {
  const waveHeight = SEA.height[0] + SEA.height[1] * speed * speed * Math.sqrt(speed / 18);
  const peakPeriod = SEA.period[0] + SEA.period[1] * speed;
  const swellPeriod = peakPeriod * SWELL_PERIOD_RATIO;
  return {
    windDirection: direction,
    peakFrequency: (2 * Math.PI) / peakPeriod,
    energyScale: jonswapEnergyScale(
      Math.sqrt(1 - SEA.swell * SEA.swell) * waveHeight,
      peakPeriod,
      SEA.peakedness,
    ),
    peakedness: SEA.peakedness,
    directionality: SEA.directionality,
    swellDirection: direction + SEA.angle,
    swellFrequency: (2 * Math.PI) / swellPeriod,
    swellScale: jonswapEnergyScale(SEA.swell * waveHeight, swellPeriod, SWELL_PEAKEDNESS),
    waveHeight,
    seed: 3.7,
  };
}

/** Only the wind waves foam: their whitecap rises from `calm` to `gale` with the wind. */
const FOAMING = { calm: 0.45, gale: 0.9, grow: 67.5, decay: 1.15 };

/** Each cascade's foam rule under a wind of `speed` metres per second. */
export function foamingAt(speed: number): d.Infer<typeof Foaming>[] {
  const gale = Math.min(Math.max((speed - 6) / 16, 0), 1);
  const none = { whitecap: 0, grow: 0, decay: 0 };
  const whitecap = FOAMING.calm + (FOAMING.gale - FOAMING.calm) * gale;
  const foaming = { whitecap, grow: FOAMING.grow, decay: FOAMING.decay };
  return Array.from({ length: CASCADES }, (_, c) => (c === WIND ? foaming : none));
}

const jonswap = (omega: number, omegaP: number, gamma: number) => {
  'use gpu';
  const sigma = std.select(0.09, 0.07, omega <= omegaP);
  const dw = omega - omegaP;
  const r = std.exp(-(dw * dw) / (2 * sigma * sigma * omegaP * omegaP));
  const ratio = omegaP / omega;
  const ratio4 = ratio * ratio * ratio * ratio;
  const omega5 = omega * omega * omega * omega * omega;
  return ((GRAVITY * GRAVITY) / omega5) * std.exp(-1.25 * ratio4) * std.pow(gamma, r);
};

const wrapAngle = (theta: number) => {
  'use gpu';
  return std.atan2(std.sin(theta), std.cos(theta));
};

/** Donelan-Banner spreading, broadened round the peak so crests stay short. */
const donelanBanner = (r: number, theta: number, narrowing: number) => {
  'use gpu';
  let beta = 2.61 * std.pow(r, 1.3);
  if (r >= 1.6) {
    beta = std.pow(10, -0.4 + 0.8393 * std.exp(-0.567 * std.log(r * r)));
  } else if (r >= 0.95) {
    beta = 2.28 * std.pow(r, -1.3);
  }
  const lnr = std.log(r);
  const width = PEAK_BROADENING_WIDTH * PEAK_BROADENING_WIDTH;
  beta = beta * narrowing * (1 - PEAK_BROADENING * std.exp(-(lnr * lnr) / (2 * width)));
  const sech = 1 / std.cosh(beta * wrapAngle(theta));
  return (beta / (2 * std.tanh(beta * Math.PI))) * sech * sech;
};

const spreading = (r: number, theta: number, params: d.Infer<typeof SpectrumParams>) => {
  'use gpu';
  const lobe = donelanBanner(r, theta, std.mix(0.35, 1.1, params.directionality));
  return std.mix(lobe, 0.5 / Math.PI, (1 - params.directionality) * ISOTROPIC_FLOOR);
};

const bandEdge = (k: number, edge: number) => {
  'use gpu';
  return std.smoothstep(edge * (1 - BAND_TAPER), edge * (1 + BAND_TAPER), k);
};

/** Expected |h0(k)|² of a lattice cell: a wind sea plus a swell from another direction. */
export const waveVariance = (
  kVec: d.v2f,
  cascade: d.Infer<typeof Cascade>,
  params: d.Infer<typeof SpectrumParams>,
) => {
  'use gpu';
  const k = std.length(kVec);
  if (k < 1e-4) {
    return d.f32(0);
  }
  const disp = dispersion(k);
  const kWorld = patchToWorld(kVec, cascade.rotation);
  const angle = std.atan2(kWorld.y, kWorld.x);
  const windSea =
    jonswap(disp.x, params.peakFrequency, params.peakedness) *
    params.energyScale *
    spreading(disp.x / params.peakFrequency, angle - params.windDirection, params);
  const swell =
    jonswap(disp.x, params.swellFrequency, d.f32(SWELL_PEAKEDNESS)) *
    params.swellScale *
    donelanBanner(disp.x / params.swellFrequency, angle - params.swellDirection, SWELL_NARROWING);
  const dk = (2 * Math.PI) / cascade.patchSize;
  const band = bandEdge(k, cascade.kLow) * (1 - bandEdge(k, cascade.kHigh));
  const gain = cascade.heightGain * cascade.heightGain;
  return ((0.5 * (windSea + swell) * disp.y * dk * dk) / k) * band * gain;
};

/** Conjugate cell in a centred power-of-two FFT lattice. Integer keys avoid -0 on the axes. */
export const mirrorCell = (cell: d.v2u) => {
  'use gpu';
  return d.vec2u((N - cell.x) & (N - 1), (N - cell.y) & (N - 1));
};
