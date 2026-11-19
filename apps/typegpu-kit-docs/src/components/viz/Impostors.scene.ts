import {
  cameras,
  passes,
  targets,
  type Drawn,
  type Frame,
  type Surface,
  finish,
  impostors,
  instances,
  lighting,
  materials,
  meshPool,
  shading,
} from '@typegpu/kit';
import { srgbToLinear } from '@typegpu/color';
import { meshes } from '@typegpu/kit/geometry';
import { d, type TgpuRoot } from 'typegpu';
import { mat4, quat, vec3, type Vec3 } from 'wgpu-matrix';

export const SAMPLINGS = ['nearest', 'blended', 'parallax'] as const;
/** The views a side the slider offers, as `capture` takes them. */
export const VIEWS = { min: 2, max: 16 } as const;
export const START = { views: 12, sampling: 'parallax' } as const;
/** Texels a side of each view, the capture's default. */
const SIZE = 128;

type Triple = [number, number, number];
type Quaternion = [number, number, number, number];

/** MB an atlas of `views` a side takes, mipmaps included. */
export const megabytesAt = (views: number) => (views * views * SIZE * SIZE * 8 * 4) / 3 / 1e6;

const HDR = 'rgba16float';
/** What shows behind the trees: a hazy sky, in linear light. */
const SKY = [0.16, 0.22, 0.32, 1] as const;
const SAMPLES = 4;

const linear = (r: number, g: number, b: number): Triple => {
  const { x, y, z } = srgbToLinear(d.vec3f(r, g, b));
  return [x, y, z];
};

const turn = (axis: Triple, angle: number): Quaternion => {
  const [x, y, z, w] = quat.fromAxisAngle(axis, angle);
  return [x!, y!, z!, w!];
};

const BARK: Triple = [0.36, 0.24, 0.15];
/** An apple's radius, in the tree's units. */
const APPLE = 0.14;

/** The tree's pieces in its own space, each with half its shape's box and its colour, sRGB. */
const PIECES = [
  {
    shape: () => meshes.cylinder({ radius: 0.12, height: 1.6, radialSegments: 12 }),
    half: [0.12, 0.8, 0.12],
    position: [0, 0.8, 0],
    color: BARK,
  },
  {
    shape: () => meshes.cylinder({ radius: 0.07, height: 0.9, radialSegments: 8 }),
    half: [0.07, 0.45, 0.07],
    position: [0.32, 1.35, 0],
    rotation: turn([0, 0, 1], -0.9),
    color: BARK,
  },
  {
    shape: () => meshes.icosphere({ radius: 0.85, segments: 2 }),
    half: [0.85, 0.85, 0.85],
    position: [0, 2.2, 0],
    color: [0.3, 0.5, 0.2],
  },
  {
    shape: () => meshes.icosphere({ radius: 0.5, segments: 2 }),
    half: [0.5, 0.5, 0.5],
    position: [0.78, 1.85, 0.1],
    color: [0.47, 0.64, 0.25],
  },
  {
    shape: () => meshes.icosphere({ radius: 0.42, segments: 2 }),
    half: [0.42, 0.42, 0.42],
    position: [-0.38, 2.85, -0.25],
    color: [0.2, 0.38, 0.16],
  },
  ...(
    [
      [0.5, 1.95, 0.62],
      [-0.55, 2.35, 0.6],
      [1.05, 1.62, 0.42],
    ] satisfies Triple[]
  ).map((position) => ({
    shape: () => meshes.icosphere({ radius: APPLE, segments: 1 }),
    half: [APPLE, APPLE, APPLE] satisfies Triple,
    position,
    color: [0.78, 0.12, 0.08] satisfies Triple,
  })),
] satisfies {
  shape: () => meshPool.Shape;
  half: Triple;
  position: Triple;
  rotation?: Quaternion;
  color: Triple;
}[];

/** Where the tree stands as meshes, and as an impostor. */
const MESHES_AT: Triple = [-1.9, 0, 0];
export const IMPOSTOR_AT: Triple = [1.9, 0, 0];
const MIDDLE: Triple = [0, 1.6, 0];

/** The box around every piece's box as `capture` places them, and the sphere around it. */
function enclose() {
  const corners = PIECES.flatMap(({ half, position, rotation = [0, 0, 0, 1] }) =>
    [0, 1, 2, 3, 4, 5, 6, 7].map((corner) => {
      const offset = half.map((h, axis) => (((corner >> axis) & 1) * 2 - 1) * h);
      return vec3.add(vec3.transformQuat(offset, rotation), position);
    }),
  );
  const low = corners.reduce((a, b) => vec3.min(a, b));
  const high = corners.reduce((a, b) => vec3.max(a, b));
  const center = vec3.lerp(low, high, 0.5);
  const radius = Math.max(...corners.map((corner) => vec3.distance(corner, center)));
  return { center, radius };
}

// What follows lays the views out as `impostors.ts` does for a hemisphere, so the figure can turn a
// card to each view and say which views the impostor reads

/** A direction above the horizon as a point of the unit square. */
const encode = ([x, y, z]: Vec3 | number[]) => {
  const sum = Math.max(Math.abs(x!) + Math.max(y!, 0) + Math.abs(z!), 1e-6);
  return [((x! + z!) / sum) * 0.5 + 0.5, ((z! - x!) / sum) * 0.5 + 0.5] as const;
};

/** The direction a view looks back along, by its column and row. */
const directionOf = (column: number, row: number, views: number) => {
  const fx = (column / (views - 1)) * 2 - 1;
  const fy = (row / (views - 1)) * 2 - 1;
  const x = (fx - fy) * 0.5;
  const z = (fx + fy) * 0.5;
  return vec3.normalize([x, 1 - Math.abs(x) - Math.abs(z), z]);
};

/** A view's screen axes: right and up, around the pole, or around +z from straight above. */
const basisOf = (direction: Vec3) => {
  const pole = Math.abs(direction[1]) > 0.999 ? [0, 0, 1] : [0, 1, 0];
  const right = vec3.normalize(vec3.cross(pole, direction));
  return { right, up: vec3.cross(direction, right) };
};

/** The views a pixel at the subject's centre reads from `toward`, by layer, and their weights. */
export function viewsRead(toward: Vec3, views: number, sampling: impostors.Sampling) {
  const last = views - 1;
  const [gx, gy] = encode(toward).map((c) => c * last) as [number, number];
  if (sampling === 'nearest')
    return [{ layer: Math.round(gy) * views + Math.round(gx), weight: 1 }];

  const cx = Math.min(Math.floor(gx), last - 1);
  const cy = Math.min(Math.floor(gy), last - 1);
  const fx = gx - cx;
  const fy = gy - cy;
  const lower = fx + fy < 1;
  const corner = lower ? 0 : 1;
  const weights = lower ? [1 - fx - fy, fx, fy] : [fx + fy - 1, 1 - fy, 1 - fx];
  const cells = [
    [cx + corner, cy + corner],
    [cx + 1, cy],
    [cx, cy + 1],
  ] as const;
  return cells.map(([x, y], k) => ({ layer: y * views + x, weight: weights[k] }));
}

/**
 * A card for each view in a square one unit across, row by row: turned so the direction it looks
 * back along points at a camera looking down -z, with its right and up along x and y, and scaled
 * so the subject fits its cell.
 */
export function layOut(views: number, subject: ReturnType<typeof enclose>) {
  const cell = 1 / views;
  const scale = (cell / (2 * subject.radius)) * 0.94;

  return Array.from({ length: views * views }, (_, layer) => {
    const column = layer % views;
    const row = Math.floor(layer / views);
    const direction = directionOf(column, row, views);
    const { right, up } = basisOf(direction);
    // Its rows are the view's axes, so it takes them to x, y and z
    const turned = mat4.create(
      right[0],
      up[0],
      direction[0],
      0,
      right[1],
      up[1],
      direction[1],
      0,
      right[2],
      up[2],
      direction[2],
      0,
      0,
      0,
      0,
      1,
    );
    const centre = vec3.transformMat4Upper3x3(subject.center, turned);
    const [x, y, z, w] = quat.fromMat(turned);
    const at: Triple = [
      (column + 0.5) * cell - 0.5 - centre[0] * scale,
      0.5 - (row + 0.5) * cell - centre[1] * scale,
      -centre[2] * scale,
    ];
    return { position: at, rotation: [x!, y!, z!, w!] as Quaternion, scale };
  });
}

/** Shows a view's colours as they were captured, lit by nothing. */
const unlit = (material: shading.Material, _point: shading.Point) => {
  'use gpu';
  return d.vec3f(material.color);
};

/** How bright a view the impostor doesn't read shows. */
const DIM = 0.22;

export async function createScene(root: TgpuRoot, surface: Surface) {
  const shown = { atlas: false };
  const side = () => Math.max(Math.min(surface.height, Math.floor(surface.width / 2)), 1);
  const region = {
    get width() {
      return shown.atlas ? surface.width - side() : surface.width;
    },
    get height() {
      return surface.height;
    },
  };
  const square = {
    get width() {
      return side();
    },
    get height() {
      return side();
    },
  };

  const pool = meshPool.create(root, { vertices: 4096, indices: 16384, capacity: 256 });
  const ground = pool.add(meshes.plane({ width: 160, depth: 160 }));
  const shapes = PIECES.map(({ shape }) => pool.add(shape()));
  const card = pool.add(impostors.CARD).meshes[0];
  const pieces = PIECES.map(({ position, rotation, color }, i): impostors.Piece => ({
    mesh: shapes[i],
    position,
    ...(rotation && { rotation }),
    color: linear(...color),
  }));
  const subject = enclose();

  const solid = instances.create(root, pool, { capacity: 1 + PIECES.length });
  solid.add(ground, materials.matte(d.vec3f(...linear(0.5, 0.45, 0.36))));
  for (const [i, { position, rotation, color }] of PIECES.entries()) {
    const at = vec3.add(MESHES_AT, position);
    solid.add(shapes[i], {
      position: [at[0], at[1], at[2]],
      ...(rotation && { rotation }),
      color: linear(...color),
      roughness: 0.9,
    });
  }

  const light = lighting.create(root, {
    settings: {
      sunDirection: [-0.7, 0.62, 0.36],
      sunColor: [1, 0.9, 0.76],
      sunIntensity: 2.6,
      skyTop: [0.45, 0.56, 0.78],
      skyBottom: [0.28, 0.22, 0.16],
      skyIntensity: 0.5,
    },
    shadow: { size: 1024, extent: 14 },
  });
  const view = cameras.orbit(root, {
    size: region,
    target: MIDDLE,
    distance: 9,
    yaw: 0.5,
    pitch: 0.2,
    pitchRange: [0.02, 1.45],
  });
  const screen = targets.create(root, region, {
    scene: { format: HDR, samples: SAMPLES },
    depth: { format: 'depth24plus', samples: SAMPLES },
    hdr: { format: HDR, sampled: true },
  });
  const scene = passes
    .create(
      root,
      screen,
      { scene: passes.color, depth: passes.depth },
      { resolve: { scene: 'hdr' }, clear: SKY },
    )
    .pipe(view.inject())
    .beneath(light.inject());
  const asMeshes = solid.renderer();
  const meshesShown = scene.add(asMeshes);
  const meshesCast = light.shadow.add(asMeshes);
  const final = finish.create(root, { format: surface.format, source: () => screen.textures.hdr });

  const flat = lighting.create(root, {
    settings: { sunIntensity: 0 },
    shading: { ambient: unlit },
  });
  const panelView = cameras.flat(root, { size: square, height: 1 });
  const panelScreen = targets.create(root, square, {
    scene: { format: HDR, samples: SAMPLES },
    depth: { format: 'depth24plus', samples: SAMPLES },
    hdr: { format: HDR, sampled: true },
  });
  const panel = passes
    .create(
      root,
      panelScreen,
      { scene: passes.color, depth: passes.depth },
      { resolve: { scene: 'hdr' }, clear: [0.02, 0.025, 0.03, 1] },
    )
    .pipe(panelView.inject())
    .beneath(flat.inject());
  const panelFinal = finish.create(root, {
    format: surface.format,
    source: () => panelScreen.textures.hdr,
    tone: null,
    settings: { vignette: 0 },
  });

  interface Reading {
    readonly shown: Drawn;
    readonly cast: Drawn;
    readonly ready: Promise<unknown>;
  }

  /** An atlas of `views` a side, its impostor, its views laid out, and each way it is read. */
  const captured = (views: number) => {
    const atlas = impostors.create(root, pool, pieces, { card, views, size: SIZE });
    const capturing = root.createCommandEncoder();
    atlas.record(capturing);
    capturing.submit();
    const cards = instances.create(root, pool, { kind: impostors.Impostor, capacity: 1 });
    cards.add(card, { position: IMPOSTOR_AT, roughness: 0.9 });

    const tiles = instances.create(root, pool, {
      kind: impostors.Impostor,
      capacity: views * views,
    });
    const ids = layOut(views, subject).map((placed) =>
      tiles.add(card, { ...placed, color: [DIM, DIM, DIM] }),
    );
    const brightness = ids.map(() => DIM);
    const tilesShown = panel.add(tiles.renderer().pipe(atlas.inject('nearest')));

    const readings = new Map<impostors.Sampling, Reading>();
    return {
      views,
      atlas,
      cards,
      tiles,
      tilesShown,
      ready: tilesShown.initAsync(),

      read(sampling: impostors.Sampling) {
        let reading = readings.get(sampling);
        if (!reading) {
          const renderer = cards.renderer().pipe(atlas.inject(sampling));
          const cardsShown = scene.add(renderer);
          const cast = light.shadow.add(renderer);
          const ready = Promise.all([cardsShown.initAsync(), cast.initAsync()]);
          reading = { shown: cardsShown, cast, ready };
          readings.set(sampling, reading);
        }
        return reading;
      },

      /** Lights each view by its weight among those the impostor reads from `toward`. */
      highlight(toward: Vec3, sampling: impostors.Sampling) {
        const next = ids.map(() => DIM);
        for (const { layer, weight } of viewsRead(toward, views, sampling)) {
          if (layer >= 0 && layer < next.length) next[layer] = next[layer] + (1 - DIM) * weight;
        }
        for (const [i, id] of ids.entries()) {
          if (next[i] === brightness[i]) continue;
          brightness[i] = next[i];
          tiles.write(id, { color: [next[i], next[i], next[i]] });
        }
      },

      destroy() {
        for (const { shown: cardsShown, cast } of readings.values()) {
          cardsShown.destroy();
          cast.destroy();
        }
        tilesShown.destroy();
        cards.destroy();
        tiles.destroy();
        atlas.destroy();
      },
    };
  };
  type Captured = ReturnType<typeof captured>;

  const wanted = { sampling: START.sampling as impostors.Sampling, views: START.views as number };
  let current: { captured: Captured; reading: Reading };
  {
    const first = captured(wanted.views);
    const reading = first.read(wanted.sampling);
    await Promise.all([
      first.ready,
      reading.ready,
      scene.initAsync(),
      light.shadow.initAsync(),
      final.initAsync(),
      panelFinal.initAsync(),
    ]);
    current = { captured: first, reading };
  }

  // One change at a time, each compiled before it shows
  let queue: Promise<unknown> = Promise.resolve();
  const queued = (work: () => Promise<void>) => {
    const done = queue.then(work);
    queue = done.catch(() => {});
    return done;
  };

  const toward = vec3.create();
  const eye = vec3.create();
  const subjectAt = vec3.add(IMPOSTOR_AT, subject.center);

  return {
    view,
    subject,

    /** Reads the atlas `sampling`'s way, once its pipelines have compiled. */
    read: (sampling: impostors.Sampling) =>
      queued(async () => {
        wanted.sampling = sampling;
        const reading = current.captured.read(sampling);
        await reading.ready;
        current = { captured: current.captured, reading };
      }),

    /** Captures the tree again at `views` a side, and shows it once it has compiled. */
    capture: (views: number) =>
      queued(async () => {
        if (views === current.captured.views) return;
        wanted.views = views;
        const next = captured(views);
        const reading = next.read(wanted.sampling);
        await Promise.all([next.ready, reading.ready]);
        const last = current.captured;
        current = { captured: next, reading };
        last.destroy();
      }),

    /** Shows the atlas beside the impostor, which the view turns to, or the whole scene. */
    showAtlas(on: boolean) {
      shown.atlas = on;
      view.goal.target = on ? [IMPOSTOR_AT[0], MIDDLE[1], IMPOSTOR_AT[2]] : [...MIDDLE];
    },

    frame(frame: Frame) {
      // The figure draws only when asked, so the view goes straight to where it was turned
      view.jump();
      screen.fit();

      const { captured: now, reading } = current;
      solid.flush();
      now.cards.flush();
      light.shadow.record(frame, [meshesCast, reading.cast]);

      const pass = scene.begin(frame);
      meshesShown.draw(pass);
      reading.shown.draw(pass);
      pass.end();

      const width = region.width;
      if (shown.atlas) {
        const { yaw, pitch, distance, target } = view.pose;
        eye[0] = target[0] + distance * Math.cos(pitch) * Math.sin(yaw);
        eye[1] = target[1] + distance * Math.sin(pitch);
        eye[2] = target[2] + distance * Math.cos(pitch) * Math.cos(yaw);
        now.highlight(
          vec3.normalize(vec3.subtract(eye, subjectAt, toward), toward),
          wanted.sampling,
        );
        now.tiles.flush();

        panelView.update(0);
        panelScreen.fit();
        panel.record(frame, now.tilesShown);
      }

      const last = frame.encoder.beginRenderPass({
        colorAttachments: { view: surface.view(frame), clearValue: [0, 0, 0, 1] },
      });
      if (shown.atlas) last.setViewport(0, 0, width, surface.height, 0, 1);
      final.draw(last);
      if (shown.atlas) {
        const across = side();
        last.setViewport(width, (surface.height - across) / 2, across, across, 0, 1);
        panelFinal.draw(last);
      }
      last.end();
    },

    destroy() {
      current.captured.destroy();
      meshesShown.destroy();
      meshesCast.destroy();
      solid.destroy();
      pool.destroy();
      light.destroy();
      flat.destroy();
      screen.destroy();
      panelScreen.destroy();
    },
  };
}
