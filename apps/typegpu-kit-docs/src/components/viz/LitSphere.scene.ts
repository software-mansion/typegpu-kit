import {
  cameras,
  Camera,
  type Frame,
  type Surface,
  lighting,
  materials,
  shading,
} from '@typegpu/kit';
import { linearToSrgb } from '@typegpu/color';
import { d, std, tgpu, type TgpuRoot } from 'typegpu';

export const PRESETS = ['matte', 'glossy', 'metal', 'glow'] as const;

/** The reflect functions, a pipeline each; the first at first. */
export const REFLECTS = ['pbr', 'lambert'] as const;
export type Reflect = (typeof REFLECTS)[number];

/** Where a sphere is drawn: x, y, width and height, as fractions of the surface, y down. */
export type Rect = readonly [number, number, number, number];

const stone = d.vec3f(0.48, 0.48, 0.48);
const silver = d.vec3f(0.85, 0.85, 0.85);
const amber = d.vec3f(0.97, 0.71, 0.31);

const presets = (roughness: number) => [
  materials.matte(stone),
  materials.glossy(stone, roughness),
  materials.metal(silver, roughness),
  materials.glow(amber, 1),
];

const Materials = d.arrayOf(shading.Material, PRESETS.length);
const surfaces = tgpu.accessor(Materials);

const Rects = d.arrayOf(d.vec4f, PRESETS.length);
const squares = tgpu.accessor(Rects);

/** Four squares in a row, filling the surface's height, until the page lays them out. */
const ROW: Rect[] = PRESETS.map((_, i) => [i / PRESETS.length, 0, 1 / PRESETS.length, 1]);

/** A quad over the instance's square, as a strip, that passes on which sphere it draws. */
const quad = tgpu.vertexFn({
  in: { vertexIndex: d.builtin.vertexIndex, instanceIndex: d.builtin.instanceIndex },
  out: { position: d.builtin.position, uv: d.vec2f, which: d.interpolate('flat', d.u32) },
})(({ vertexIndex, instanceIndex }) => {
  'use gpu';
  const corner = d.vec2f(d.f32(vertexIndex & 1), d.f32((vertexIndex >>> 1) & 1));
  const square = squares.$[instanceIndex];
  const at = square.xy + corner * square.zw;

  return {
    position: d.vec4f(at.x * 2 - 1, 1 - at.y * 2, 0, 1),
    uv: corner,
    which: instanceIndex,
  };
});

/** A unit sphere seen from the front, filling the target, with premultiplied edges. */
const sphere = tgpu.fragmentFn({
  in: { uv: d.vec2f, which: d.interpolate('flat', d.u32) },
  out: d.vec4f,
})(({ uv, which }) => {
  'use gpu';
  const xy = d.vec2f(uv.x * 2 - 1, 1 - uv.y * 2);
  const radius = std.length(xy);
  const coverage = std.saturate((1 - radius) / std.fwidth(radius) + 0.5);

  const edge = std.min(radius, 1);
  const normal = d.vec3f(xy, std.sqrt(1 - edge * edge));
  const material = shading.Material(surfaces.$[which]);
  const color = shading.shade(material, shading.pointAt(normal, normal));

  return d.vec4f(std.saturate(linearToSrgb(color)) * coverage, coverage);
});

export function createScene(root: TgpuRoot, surface: Surface) {
  const eye = root.createUniform(Camera);
  eye.patch({ position: d.vec3f(0, 0, 50), right: d.vec3f(1, 0, 0), up: d.vec3f(0, 1, 0) });
  const values = root.createUniform(Materials, presets(0.35));
  const rects = root.createUniform(
    Rects,
    ROW.map((rect) => d.vec4f(...rect)),
  );

  const light = lighting.create(root);
  const lit = root
    .with(cameras.current, eye)
    .with(surfaces, values)
    .with(squares, rects)
    .pipe(light.inject());

  const make = (cfg: typeof lit) =>
    cfg.createRenderPipeline({
      vertex: quad,
      fragment: sphere,
      targets: { format: surface.format },
      primitive: { topology: 'triangle-strip' },
    });
  const pipelines = { pbr: make(lit), lambert: make(lit.with(shading.reflect, shading.lambert)) };
  let reflect: Reflect = REFLECTS[0];

  return {
    /** Writes the presets at `roughness`, and the sun from `angle` radians around the spheres. */
    set(roughness: number, angle: number) {
      values.write(presets(roughness));
      light.settings.patch({ sunDirection: [Math.sin(angle), 0.7, Math.cos(angle)] });
    },

    /** Chooses the function each sphere reflects the sun with: a pipeline each. */
    reflect(chosen: Reflect) {
      reflect = chosen;
    },

    /** Where each preset's sphere goes, in the order of PRESETS. */
    layout(placed: readonly Rect[]) {
      rects.write(placed.map((rect) => d.vec4f(...rect)));
    },

    /** Draws every preset's sphere in its square, in one draw. */
    frame(frame: Frame) {
      const pass = frame.encoder.beginRenderPass({
        colorAttachments: { view: surface.view(frame), clearValue: [0, 0, 0, 0] },
      });
      pass.setPipeline(pipelines[reflect]);
      pass.draw(4, PRESETS.length);
      pass.end();
    },

    destroy() {
      light.destroy();
      eye.buffer.destroy();
      values.buffer.destroy();
      rects.buffer.destroy();
    },
  };
}
