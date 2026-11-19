import { cameras, type CameraBase } from '@typegpu/kit';
import { common, d, std, tgpu, type TgpuRenderCommands, type TgpuRoot } from 'typegpu';

/** Signed distance from the nearest surface, negative inside. Runs on the GPU. */
export type Field = (point: d.v3f) => number;

/** Unlit surface colour at a point. Runs on the GPU. */
export type Paint = (point: d.v3f) => d.v3f;

/** Maximum ray steps per pixel and maximum travel distance in world units. */
const STEPS = 128;
const FAR = 40;
/** Surface-hit tolerance per world unit travelled. */
const HIT = 0.0005;
/** Normal-sampling offset in world units. */
const NORMAL_STEP = 0.0005;
/** Fog density. A surface `t` world units away retains `exp(-t × FOG)` of its colour. */
const FOG = 0.035;
/** Sun direction and linear sun, zenith and horizon colours. */
const SUN = std.normalize(d.vec3f(0.5, 0.75, 0.45));
const SUNLIGHT = d.vec3f(1.7, 1.55, 1.35);
const ZENITH = d.vec3f(0.05, 0.12, 0.32);
const HORIZON = d.vec3f(0.32, 0.38, 0.48);
/** Sky-light weights for sideways and upward-facing surfaces. */
const SKYLIGHT = { sideways: 0.35, up: 0.25 };

/** Maximum shadow-ray steps and minimum and maximum step lengths in world units. */
const SHADOW = { steps: 48, least: 0.02, most: 0.3 };
/** World units off the surface where the shadow's ray starts, so it doesn't meet it at once. */
const LIFT = 0.002;
/**
 * Shadow sharpness. A ray passing distance `d` from the field at travel distance `t` transmits
 * `PENUMBRA × d / t` of sunlight.
 */
const PENUMBRA = 8;
/** Shadow-ray reach toward the sun in world units. */
const SHADOW_REACH = 10;
/**
 * Distance beyond which the field cannot darken a shadow. Allows cheaper distance bounds
 * outside this range.
 */
export const UNSHADED = SHADOW_REACH / PENUMBRA;

/**
 * Raymarches a distance field with one ray per pixel, sunlight, soft shadows, sky light and
 * fog. Draws one triangle into the caller's pass.
 */
export function createRaymarch(
  root: TgpuRoot,
  options: { camera: CameraBase; format: GPUTextureFormat; field: Field; paint: Paint },
) {
  const { camera, format, field, paint } = options;
  /** Estimates the normal from field differences at tetrahedron corners. */
  const normalAt = (point: d.v3f) => {
    'use gpu';
    const e = d.vec2f(1, -1) * NORMAL_STEP;
    return std.normalize(
      e.xyy * field(point + e.xyy) +
        e.yyx * field(point + e.yyx) +
        e.yxy * field(point + e.yxy) +
        e.xxx * field(point + e.xxx),
    );
  };

  /** The share of sunlight reaching `point`, less where the shadow ray grazes the field. */
  const sunlight = (point: d.v3f) => {
    'use gpu';
    let light = d.f32(1);
    let t = d.f32(SHADOW.least);
    for (let i = d.u32(0); i < SHADOW.steps; i++) {
      const near = field(point + SUN * t);
      light = std.min(light, (PENUMBRA * near) / t);
      t += std.clamp(near, SHADOW.least, SHADOW.most);
      if (light < 0.001 || t > SHADOW_REACH) break;
    }
    return std.saturate(light);
  };

  const fragment = tgpu.fragmentFn({ in: { uv: d.vec2f }, out: d.vec4f })(({ uv }) => {
    'use gpu';
    const ray = cameras.rayOf(cameras.current.$, d.vec2f(uv.x, 1 - uv.y) * 2 - 1);
    const sky = std.mix(HORIZON, ZENITH, std.saturate(ray.direction.y));

    let t = d.f32(0);
    let hit = false;
    for (let i = d.u32(0); i < STEPS; i++) {
      const distance = field(ray.origin + ray.direction * t);
      if (distance < HIT * t) {
        hit = true;
        break;
      }
      t += distance;
      if (t > FAR) break;
    }
    if (!hit) return d.vec4f(sky, 1);

    const point = ray.origin + ray.direction * t;
    const normal = normalAt(point);
    const sun = std.max(std.dot(normal, SUN), 0) * sunlight(point + normal * LIFT);
    const lit = SUNLIGHT * sun + sky * (SKYLIGHT.sideways + normal.y * SKYLIGHT.up);
    const color = paint(point) * lit;
    return d.vec4f(std.mix(sky, color, std.exp(-t * FOG)), 1);
  });

  const pipeline = root.pipe(camera.inject()).createRenderPipeline({
    vertex: common.fullScreenTriangle,
    fragment,
    targets: { format },
  });

  return {
    initAsync: () => pipeline.initAsync(),

    draw(pass: TgpuRenderCommands) {
      pass.setPipeline(pipeline);
      pass.draw(3);
    },
  };
}
