import { hexToRgb, srgbToLinear } from '@typegpu/color';
import { d } from 'typegpu';

export const [PLAIN, BANDED, EARTHLIKE, BURNING] = [0, 1, 2, 3];

// name | kind | day | year | diameter in km | gravity in g | distance in AU
const FACTS = `
Mercury | Rocky planet | 58.6 days  | 88 days    | 4,879   | 0.38 | 0.39
Venus   | Rocky planet | 243 days   | 225 days   | 12,104  | 0.91 | 0.72
Earth   | Rocky planet | 23.9 hours | 365 days   | 12,742  | 1.00 | 1.00
Mars    | Rocky planet | 24.6 hours | 687 days   | 6,779   | 0.38 | 1.52
Jupiter | Gas giant    | 9.9 hours  | 11.9 years | 139,820 | 2.53 | 5.20
Saturn  | Gas giant    | 10.7 hours | 29.5 years | 116,460 | 1.07 | 9.54
Uranus  | Ice giant    | 17.2 hours | 84 years   | 50,724  | 0.89 | 19.2
Neptune | Ice giant    | 16.1 hours | 165 years  | 49,244  | 1.14 | 30.1`;

// orbit, radius in world units | phase, tilt in radians | spin in radians a second | colours | pattern
const LOOKS = `
2.6  | 0.3  | 4.92 | 0.02 | 0.3   | 9e9a95 | 67635f | 0
3.9  | 0.42 | 3.0  | 0.05 | -0.15 | f5dc9c | e0b871 | 1
5.2  | 0.45 | 4.44 | 0.41 | 0.8   | 3a77e6 | 5fb352 | 2
6.5  | 0.36 | 2.17 | 0.44 | 0.75  | d66b38 | 9a4526 | 0
8.3  | 1.0  | 3.7  | 0.05 | 1.2   | e0b385 | faedd1 | 1
10.1 | 0.82 | 5.6  | 0.47 | 1.1   | f2d99e | dbb780 | 1
11.8 | 0.6  | 0.9  | 1.71 | 0.9   | a8e8ed | 8ad1de | 1
13.4 | 0.58 | 2.3  | 0.49 | 0.95  | 3861e6 | 6b94ff | 1`;

const rows = (table: string) =>
  table
    .trim()
    .split('\n')
    .map((row) => row.split('|').map((cell) => cell.trim()));

/** Converts an sRGB hex colour to linear light. */
export const hex = (rgb: string) => srgbToLinear(hexToRgb(rgb));

export const EARTH = 2;
export const SATURN = 5;
/** Saturn's two ring bands: radius in planet radii, and colour. */
export const SATURN_RINGS = [
  { radius: 1.55, color: hex('efe2c0') },
  { radius: 2.05, color: hex('b8a27c') },
];

const looks = rows(LOOKS);
/** Each planet's facts, and how it looks and moves, stylised and not to scale. */
export const PLANETS = rows(FACTS).map((fact, i) => {
  const [name, kind, day, year, diameter, gravity, au] = fact as [string, ...string[]];
  const [orbit, radius, phase, tilt, spin, color, accent, pattern] = looks[i];
  const extent = Number(radius) * (i === SATURN ? SATURN_RINGS[1].radius + 0.25 : 1);
  return {
    name,
    band: `orbit ${year}  /  day ${day}  /  `.toUpperCase(),
    details:
      `${kind}\n${diameter} km across\nGravity ${gravity} g\n${au} AU from the sun`.toUpperCase(),
    orbit: Number(orbit),
    radius: Number(radius),
    /** World units from its centre to its outermost edge, Saturn's rings included. */
    extent,
    /** World units from its centre to the band of facts round it, outside Saturn's ring. */
    bandRadius: Math.max(Number(radius) * 1.5, extent + Number(radius) * 0.15) + 0.35,
    phase: Number(phase),
    tilt: Number(tilt),
    spin: Number(spin),
    color: hex(color!),
    accent: hex(accent!),
    pattern: Number(pattern),
  };
});

export const SUN = { radius: 1.3, spin: 0.08, emission: d.vec3f(3.6, 1.25, 0.22) };
export const MOON = {
  orbit: 0.85,
  radius: 0.12,
  rate: 2,
  color: hex('bebec2'),
  accent: hex('76767a'),
};

/** Radians a second round the sun for an orbit `radius` world units out, as in Kepler's third law. */
export const rateOf = (radius: number) => 2.2 / radius ** 1.5;
