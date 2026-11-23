import {
  bloom,
  cameras,
  finish,
  instances,
  lighting,
  lines,
  materials,
  meshPool,
  passes,
  pipelines,
  shading,
  taa,
  targets,
  tone,
  type Frame,
  type Surface,
} from '@typegpu/kit';
import { caps, meshes } from '@typegpu/kit/geometry';
import { srgbToLinear } from '@typegpu/color';
import { d, std, type TgpuRoot } from 'typegpu';
import type { Pointing } from '../contract.ts';
import type { Settings } from '../settings.ts';
import { createContours } from './contours.ts';

const BACKDROP = [0.022, 0.024, 0.028] as const;
/** Seconds the view takes to tip from above the model to its working angle. */
const TIP = 4;
/** Seconds the rings take to rise, from `RISE[0]` to `RISE[1]`. */
const RISE = [0.4, 3] as const;
/** The view's pitch above the model, and the angle it settles at. */
const PITCH = [1.42, 0.74] as const;
/** The plinth's half width and thickness, its top at 0. */
const PLINTH = { half: 5, thickness: 0.3 };

const paint = (r: number, g: number, b: number) => srgbToLinear(d.vec3f(r, g, b));
const linear = (rgb: readonly [number, number, number]) => {
  const { x, y, z } = paint(...rgb);
  return [x, y, z, 1];
};
const SLATE = materials.glossy(paint(0.065, 0.072, 0.085), 0.55);
const BASE = materials.glossy(paint(0.08, 0.085, 0.09), 0.5);
const BRASS = [0.78, 0.6, 0.32] as const;
const BONE = [0.86, 0.83, 0.76] as const;
/** The graticule's spacing and half width in world units, and its ink's opacity. */
const GRID = { spacing: 1, half: 4.75, ink: 0.07 };
/** World units the graticule's ink lies over the plinth's top. */
const PRINT = 0.004;
/** Share of the sun's light that reaches into a shadow. */
const SHADE = 0.42;
/**
 * Where the scale bar lies on the plinth, its `east` and its `south` end, with the north arrow
 * west of it, and the bar's length in world units and its parts.
 */
const LEGEND = { east: 4.3, south: -2.2, bar: 2, parts: 4 };
/** World units the legend's inlay stands over the plinth's top, above the graticule's ink. */
const INLAY = 0.008;
/** The north arrow's head: from its centre to its tip, and the turn pointing its tip north, to -z. */
const HEAD = { radius: 0.24, rotation: [0, Math.SQRT1_2, 0, Math.SQRT1_2] as const };

/** Shades the graticule as faint ink printed flat on the plinth. */
const ink = (fragment: lines.Fragment) => {
  'use gpu';
  const lit = shading.shade(
    materials.matte(fragment.color.xyz),
    shading.pointAt(fragment.position, d.vec3f(0, 1, 0)),
  );
  const alpha = fragment.coverage * GRID.ink;
  return d.vec4f(lit * alpha, alpha);
};

/** Lets part of the sun's light into shadows, so they stay lighter than the rings. */
const softened = (point: shading.Point) => {
  'use gpu';
  return std.mix(SHADE, 1, shading.sunShadowed(point));
};

/** A unit vector towards the sun from the south-west, `height` its rise. */
const sunAt = (height: number): [number, number, number] => [-0.82, height, 0.5];

/** The graticule's lines across the plinth at every `GRID.spacing`, and its border. */
const graticule = () => {
  const { spacing, half } = GRID;
  const count = Math.floor(half / spacing);
  const rows: number[][] = [];
  for (let k = -count; k <= count; k++) {
    rows.push([k * spacing, PRINT, -half, k * spacing, PRINT, half]);
    rows.push([-half, PRINT, k * spacing, half, PRINT, k * spacing]);
  }
  for (const side of [-half, half]) {
    rows.push([side, PRINT, -half, side, PRINT, half]);
    rows.push([-half, PRINT, side, half, PRINT, side]);
  }
  return rows;
};

/**
 * Draws a noise island as contour rings that rise off a slate plinth. One set of lines draws
 * every ring, its points written by a kernel. Another prints a graticule a point wide on the
 * plinth, a north arrow and a scale bar lie inlaid in it, and the rings cast the sun's softened
 * shadow.
 */
export async function createScene(
  root: TgpuRoot,
  surface: Surface,
  { taa: temporal = true, dashed = true, width = 0.03, sun = 0.86 } = {},
) {
  const values = { taa: temporal, dashed, width, sun };
  const screen = targets.create(root, surface, {
    hdr: { format: 'rgba16float', sampled: true },
    depth: { format: 'depth24plus', sampled: true },
  });
  const view = cameras.orbit(root, {
    size: surface,
    target: [0.5, 0.15, -0.45],
    distance: 8.1,
    yaw: 0.75,
    pitch: PITCH[0],
    pitchRange: [0.12, 1.45],
    distanceRange: [4, 22],
  });

  // TAA reprojects the last frame by the camera's motion and the scene's depth
  const aa = taa.create(root, {
    source: () => screen.textures.hdr,
    depth: () => screen.textures.depth,
    camera: view,
  });
  const light = lighting.create(root, {
    settings: {
      sunDirection: sunAt(values.sun),
      sunColor: [1, 0.78, 0.55],
      sunIntensity: 2.8,
      skyTop: [0.16, 0.2, 0.28],
      skyBottom: [0.04, 0.035, 0.03],
      skyIntensity: 0.8,
    },
    shadow: { eye: view, cascades: 2, far: 26, filter: 9 },
    shading: { sunShadow: softened },
  });
  // The lines' shading reads the lighting, which fills beneath their own fills
  const scene = passes
    .create(
      root,
      screen,
      { hdr: passes.color, depth: passes.depth },
      { jitter: aa.jitter, clear: [...BACKDROP, 1] },
    )
    .pipe(view.inject())
    .beneath(light.inject());

  const pool = meshPool.create(root, { capacity: 64 });
  const slab = pool.add(
    meshes.roundedBox({
      width: PLINTH.half * 2,
      height: PLINTH.thickness,
      depth: PLINTH.half * 2,
      radius: 0.06,
    }),
  );
  const strip = pool.add(meshes.box({ width: 1, height: INLAY * 2, depth: 1 }));
  // A cylinder with three sides is a triangular prism, with one corner along +x
  const head = pool.add(meshes.cylinder({ radius: 1, height: INLAY * 2, radialSegments: 3 }));
  const { east, south, bar, parts } = LEGEND;
  const things = instances.create(root, pool, { capacity: 4 + parts });
  things.add(slab, { position: [0, -PLINTH.thickness / 2, 0], ...SLATE });
  things.add(slab, {
    position: [0, -PLINTH.thickness * 1.5, 0],
    scale: [1.03, 1, 1.03],
    ...BASE,
  });
  // A brass north arrow and a scale bar of brass and bone, inlaid in the plinth's near corner
  const brass = materials.glossy(paint(...BRASS), 0.3);
  const bone = materials.glossy(paint(...BONE), 0.3);
  const tip = south - bar - 0.2;
  const tail = south + 0.2;
  const neck = tip + HEAD.radius;
  things.add(strip, {
    position: [east - 0.5, 0, (tail + neck) / 2],
    scale: [0.12, 1, tail - neck],
    ...brass,
  });
  things.add(head, {
    position: [east - 0.5, 0, neck],
    rotation: HEAD.rotation,
    scale: [HEAD.radius, 1, HEAD.radius],
    ...brass,
  });
  for (let k = 0; k < parts; k++) {
    things.add(strip, {
      position: [east, 0, south - ((k + 0.5) * bar) / parts],
      scale: [0.07, 1, bar / parts],
      ...(k % 2 ? bone : brass),
    });
  }

  const rows = graticule();
  const grid = lines.create(root, {
    capacity: rows.length,
    length: 2,
    units: 'points',
    startCap: caps.butt,
    endCap: caps.butt,
    shade: ink,
    label: 'graticule',
  });
  for (const [k, row] of rows.entries()) grid.write(k, row, { color: linear(BONE) });

  const contours = createContours(root, { width: values.width, dashed: values.dashed });

  const plinth = things.renderer();
  const drawn = [scene.add(plinth), scene.add(grid.drawing), scene.add(contours.drawing)];
  const casters = [light.shadow.add(plinth), light.shadow.add(contours.drawing)];

  const glow = bloom.create(root, { source: aa, settings: { threshold: 1.6 } });
  const final = finish.create(root, { format: surface.format, source: aa, tone: tone.aces, glow });

  await pipelines.compile([aa, light, scene, contours, glow, final]);

  // Trace the rings once to read their radii for picking
  const tracing = root.createCommandEncoder();
  contours.trace(tracing);
  tracing.submit();
  await contours.readRadii();

  const settings: Settings<typeof values> = {
    values,
    knobs: {
      taa: { label: 'TAA' },
      dashed: { label: 'dashed soundings' },
      width: { label: 'ring width', min: 0.005, max: 0.08, step: 0.005 },
      sun: { label: 'sun height', min: 0.1, max: 1.5, step: 0.02 },
    },

    write() {
      aa.enable(values.taa);
      contours.dashed = values.dashed;
      contours.width = values.width;
      light.settings.patch({ sunDirection: sunAt(values.sun) });
    },
  };

  // The pointer picks a ring each frame, as the view and the rings move under it
  const pointer = { x: 0, y: 0, over: false };
  const ray = { origin: [0, 0, 0], direction: [0, 0, 0] };
  const pointing: Pointing = {
    move(x, y) {
      pointer.x = x;
      pointer.y = y;
      pointer.over = true;
    },
    leave() {
      pointer.over = false;
      contours.hover();
    },
  };

  return {
    view,
    settings,
    pointing,

    frame(frame: Frame) {
      // Tip the view down to the model, and raise the rings, as the scene opens
      if (frame.time < TIP) {
        view.goal.pitch = std.mix(PITCH[0], PITCH[1], std.smoothstep(0, TIP, frame.time));
      }
      contours.lift(std.smoothstep(RISE[0], RISE[1], frame.time));
      view.update(frame.dt);
      if (pointer.over) {
        view.rayAt(pointer.x, pointer.y, ray);
        contours.hover(ray);
      }

      contours.record(frame.encoder);

      screen.fit();
      things.flush();
      light.shadow.record(frame, casters);
      scene.record(frame, drawn);
      aa.record(frame);
      glow.record(frame);
      final.record(frame, surface.view(frame));
    },

    destroy() {
      for (const made of [screen, aa, light, pool, things, grid, contours, glow]) made.destroy();
    },
  };
}
