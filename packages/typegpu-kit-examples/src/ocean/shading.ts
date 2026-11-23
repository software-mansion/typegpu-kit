import { d, std } from 'typegpu';

const WATER_IOR = 1.333;
/** Below this cosine the lobe is subpixel; keeps intermediate maths in f32 range. */
const MIN_COSINE = 1e-6;

/** Unpolarized air-to-water Fresnel reflectance. */
export const fresnelWater = (cosI: number) => {
  'use gpu';
  const sinT2 = (1 - cosI * cosI) / (WATER_IOR * WATER_IOR);
  const cosT = std.sqrt(std.max(1 - sinT2, 0));
  const rs = (cosI - WATER_IOR * cosT) / (cosI + WATER_IOR * cosT);
  const rp = (WATER_IOR * cosI - cosT) / (WATER_IOR * cosI + cosT);
  return std.saturate(0.5 * (rs * rs + rp * rp));
};

/** Smith masking for a Beckmann slope distribution, Walter's rational fit. */
const beckmannLambda = (a: number) => {
  'use gpu';
  if (a >= 1.6) {
    return d.f32(0);
  }
  return (1 - 1.259 * a + 0.396 * a * a) / (3.535 * a + 2.181 * a * a);
};

/** Slope variance of a covariance packed as (xx, yy, xy) along the azimuth of v, times sin². */
const projectedVariance = (v: d.v3f, cov: d.v3f) => {
  'use gpu';
  return cov.x * v.x * v.x + 2 * cov.z * v.x * v.z + cov.y * v.z * v.z;
};

const smithParam = (v: d.v3f, cov: d.v3f) => {
  'use gpu';
  return std.max(v.y, MIN_COSINE) / std.sqrt(2 * std.max(projectedVariance(v, cov), 0) + 1e-8);
};

const slopeDensity = (zeta: d.v2f, cov: d.v3f) => {
  'use gpu';
  const det = std.max(cov.x * cov.y - cov.z * cov.z, 1e-8);
  const quad =
    (cov.y * zeta.x * zeta.x - 2 * cov.z * zeta.x * zeta.y + cov.x * zeta.y * zeta.y) / det;
  // Rounded moments can make this slightly negative; negligible tails skip subnormal maths
  if (quad > 160) {
    return d.f32(0);
  }
  return std.exp(-0.5 * std.max(quad, 0)) / (2 * Math.PI * std.sqrt(det));
};

/** Sun glint over Gaussian slopes, with vectors in the resolved normal's frame: x, up, z. */
export const sunGlint = (v: d.v3f, l: d.v3f, cov: d.v3f, sun: d.v3f) => {
  'use gpu';
  if (v.y <= 0 || l.y <= 0) {
    return d.vec3f();
  }
  const halfway = v + l;
  const h = halfway / std.max(std.length(halfway), MIN_COSINE);
  const cosH = std.max(h.y, MIN_COSINE);
  const p = slopeDensity(d.vec2f(-h.x / cosH, -h.z / cosH), cov);
  const h4 = cosH * cosH * cosH * cosH;
  const masking = 1 + beckmannLambda(smithParam(v, cov)) + beckmannLambda(smithParam(l, cov));
  return sun * ((p * fresnelWater(std.dot(v, h))) / (4 * h4 * std.max(v.y, MIN_COSINE) * masking));
};

/** Fresnel over slopes of deviation `sigma` at `cosine`, Bruneton's fit, as GodotOceanWaves. */
export const roughFresnel = (cosine: number, sigma: number) => {
  'use gpu';
  const oneMinus = 1 - std.saturate(cosine);
  return (
    0.02 + 0.98 * (std.pow(oneMinus, 5 * std.exp(-2.69 * sigma)) / (1 + 22.7 * std.pow(sigma, 1.5)))
  );
};

/** Smith's Λ for slopes of width `alpha` seen at `cosine` from the mean normal. */
export const smithLambda = (cosine: number, alpha: number) => {
  'use gpu';
  return beckmannLambda(cosine / (alpha * std.sqrt(std.max(1 - cosine * cosine, 1e-6))));
};

/** Fresnel averaged over the slope distribution seen from v, Bruneton's fit. */
export const meanFresnel = (v: d.v3f, cov: d.v3f) => {
  'use gpu';
  const horizontal = v.x * v.x + v.z * v.z + 1e-8;
  const sigmaV = std.sqrt(std.max(projectedVariance(v, cov), 0) / horizontal);
  const oneMinus = 1 - v.y;
  const fifth = oneMinus * oneMinus * oneMinus * oneMinus * oneMinus;
  return 0.02 + 0.98 * ((fifth * std.exp(-2.69 * sigmaV)) / (1 + 22.7 * std.pow(sigmaV, 1.5)));
};
