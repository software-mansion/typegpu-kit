import { targets, tone, type Frame, type Surface } from '@typegpu/kit';
import {
  common,
  d,
  std,
  tgpu,
  type TgpuRenderPass,
  type TgpuRoot,
  type TgpuTextureView,
} from 'typegpu';
import { mat4, vec3 } from 'wgpu-matrix';
import { bayerCell, type Colors } from '../site/dither.ts';
import { AIR, Bird, CAPACITY, COLUMNS, isLamp, LOW } from './birds.ts';
import { clamp } from './geometry.ts';
import { RULE_SIZE, type Subject } from './subject.ts';

/**
 * What a sheet shows: a texture a module wrote, the bird table, the grid's cells, the forces the
 * birds' rule gave, or nothing, a socket where a module will go.
 */
export const KIND = {
  final: 0,
  bloom: 1,
  lit: 2,
  normals: 3,
  shadow: 4,
  birds: 5,
  grid: 6,
  rule: 7,
  socket: 8,
} as const;
export type Kind = keyof typeof KIND;

/** Sheets a view draws at most. */
export const MAX = 8;

/** Birds a side of the table: 32 × 32 rows, each a 3 × 3 block of words. */
const TABLE = Math.sqrt(CAPACITY);

/** The sheets' light, towards it, for the shadow each casts on the one below. */
const LIGHT = (() => {
  const [x, y, z] = vec3.normalize([0.45, 1, 0.3]);
  return d.vec3f(x!, y!, z!);
})();

const Palette = d.struct({ grey: d.arrayOf(d.vec4f, 7), accent: d.arrayOf(d.vec4f, 2) });

/** `place` stores (x, height, z, half-side). `cover` stores the sheet above for shadows. Remaining fields control content, highlight and brightness. */
const Sheet = d.struct({
  place: d.vec4f,
  cover: d.vec4f,
  kind: d.u32,
  hot: d.f32,
  shade: d.f32,
  shown: d.f32,
  keeps: d.f32,
  rim: d.f32,
});

const View = d.struct({
  viewProjection: d.mat4x4f,
  /** How much light an empty part of a sheet has, so each reads as a slab. */
  base: d.f32,
  /** The hairlines' opacity. */
  line: d.f32,
  /** The slab's thickness, as a fraction of half its side. */
  thick: d.f32,
  /** Device pixels a cell spans. */
  cell: d.f32,
});

const viewLayout = tgpu.bindGroupLayout({
  view: { uniform: View },
  sheets: { uniform: d.arrayOf(Sheet, MAX) },
});
const cellsLayout = tgpu.bindGroupLayout({ cells: { texture: d.texture2d(d.f32) } });

export interface SheetState {
  kind: Kind;
  /** Centre, in world units. */
  x: number;
  y: number;
  z: number;
  half: number;
  /** How much its edges take the accent, from 0 to 1, as while it is pointed at. */
  hot: number;
  /** How much of it has dithered in, from 0, a socket's dashed outline, to 1. */
  shown: number;
  /** How much of its light it keeps: 1 in the frame, less once its module is taken out. */
  keeps: number;
  /**
   * How bright its seat's dashed outline is while it is not all there, from 0 to 1. The seat is
   * its place on the stack's axis, at x and z 0, which it slides into as it dithers in. Default 1.
   */
  rim?: number;
}

export interface Camera {
  yaw: number;
  pitch: number;
  /** World units from the middle of the view to its top. */
  scale: number;
  /** What the view looks at. */
  target: [number, number, number];
  /** Where on the canvas the target lands, as a fraction from its top left. */
  anchor: [number, number];
}

const toSrgb = (c: readonly number[]) => d.vec4f(c[0], c[1], c[2], 1);

/** Linear light to a value on the ramp: a soft shoulder, then roughly perceptual. */
const display = (amount: number) => {
  'use gpu';
  return std.sqrt(amount / (1 + amount));
};

/** How blue a colour is, 0 for greys and warm colours, 1 for the lamps'. */
const blueOf = (color: d.v3f) => {
  'use gpu';
  const lean = (color.z - std.max(color.x, color.y)) / std.max(color.z, 1e-4);
  return std.saturate((lean - 0.1) * 4);
};

/** A step of `count` along a ramp, from `t` in 0 to 1, dithered by `threshold`. */
const step = (t: number, count: number, threshold: number) => {
  'use gpu';
  const last = d.f32(count - 1);
  const at = std.saturate(t) * last;
  const low = std.floor(at);
  return d.u32(std.min(std.select(low, low + 1, at - low > threshold), last));
};

/** The pipelines every view of the stack shares, reading the subject's resources. */
export function createStacks(
  root: TgpuRoot,
  subject: Subject,
  colors: Colors,
  format: GPUTextureFormat,
) {
  const palette = root.createUniform(Palette, {
    grey: colors.grey.map(toSrgb),
    accent: colors.accent.map(toSrgb),
  });
  const { screen, glow, light, birds } = subject;
  const finalView = screen.textures.final.createView(d.texture2d(d.f32));
  const litView = screen.textures.hdr.createView(d.texture2d(d.f32));
  const normalView = screen.textures.normals.createView(d.texture2d(d.f32));
  const ruleView = subject.ruleMap.textures.strokes.createView(d.texture2d(d.f32));
  const glowView = glow.output.createView(d.texture2d(d.f32));
  const shadowView = light.shadow.part.resources.map as TgpuTextureView<d.WgslTextureDepth2dArray>;
  const columns = subject.columns;
  const walk = subject.walk;
  const smooth = root.createSampler({ magFilter: 'linear', minFilter: 'linear' });

  /** Whether bird `slot` is in the nine columns a walk from the pointer reads. */
  const walked = (slot: number) => {
    'use gpu';
    const at = walk.$.at;
    const column = std.floor((Bird.position(slot).xz - LOW.xz) / (AIR.x / COLUMNS));
    const reach = std.abs(column - at.xy);
    return at.z > 0.5 && reach.x <= 1 && reach.y <= 1;
  };

  /** Map one of a bird's nine words to ramp brightness and accent. Words hold position, velocity, speed, lamp status and force. */
  const wordOf = (slot: number, word: number) => {
    'use gpu';
    if (!Bird.alive(slot)) return d.vec2f(0.03, 0);
    const position = Bird.position(slot);
    const velocity = Bird.velocity(slot);
    let value = d.f32(0);
    let accent = d.f32(0);
    if (word < 3) {
      value = ((position - LOW) / AIR)[word];
      if (walked(slot)) accent = 1;
    } else if (word < 6) value = 0.5 + velocity[word - 3] * 0.12;
    else if (word === 6) value = std.length(velocity) / 5;
    else if (word === 7) {
      value = std.select(d.f32(0.1), d.f32(0.9), isLamp(slot));
      accent = std.select(d.f32(0), d.f32(1), isLamp(slot));
    } else value = std.length(Bird.force(slot)) / 30;
    return d.vec2f(0.15 + std.saturate(value) * 0.85, accent);
  };

  /** The bird table at `uv`: a block a slot, three words by three, with a gap between blocks. */
  const tableAt = (uv: d.v2f) => {
    'use gpu';
    const at = std.clamp(uv, d.vec2f(0), d.vec2f(0.9999)) * TABLE;
    const block = d.vec2u(std.floor(at));
    const local = d.vec2u(std.floor(std.fract(at) * 3.5));
    if (local.x > 2 || local.y > 2) return d.vec2f(0, 0);
    return wordOf(block.y * TABLE + block.x, local.y * 3 + local.x);
  };

  /** The grid's columns at `uv`: how many birds each holds, its hairlines, and the walk in the accent. */
  const gridAt = (uv: d.v2f) => {
    'use gpu';
    const at = std.clamp(uv, d.vec2f(0), d.vec2f(0.9999)) * COLUMNS;
    const cell = std.floor(at);
    const count = d.f32(columns.$[d.u32(cell.y) * COLUMNS + d.u32(cell.x)]);
    const inside = std.min(std.fract(at), 1 - std.fract(at));
    const line = std.min(inside.x, inside.y) < 0.035;
    let value = std.select(0.06 + std.sqrt(std.saturate(count / 40)) * 0.9, d.f32(0.3), line);
    const w = walk.$.at;
    const reach = std.abs(cell - w.xy);
    const walking = w.z > 0.5 && reach.x <= 1 && reach.y <= 1;
    if (walking && line) value = 0.9;
    return d.vec2f(value, std.select(d.f32(0), d.f32(1), walking));
  };

  /**
   * Return brightness and accent amounts in [0, 1] at `uv`. Only the graded frame and bloom keep
   * the lamps' blue.
   */
  const valueAt = (kind: number, uv: d.v2f) => {
    'use gpu';
    if (kind === KIND.final) {
      const color = std.textureSampleLevel(finalView.$, smooth.$, uv, 0).xyz;
      return d.vec2f(tone.luminance(color), blueOf(color));
    }
    if (kind === KIND.bloom) {
      const color = std.textureSampleLevel(glowView.$, smooth.$, uv, 0).xyz;
      return d.vec2f(display(tone.luminance(color) * 8), blueOf(color));
    }
    if (kind === KIND.lit) {
      return d.vec2f(
        display(tone.luminance(std.textureSampleLevel(litView.$, smooth.$, uv, 0).xyz) * 2.5),
        0,
      );
    }
    if (kind === KIND.rule) {
      // The brightest of the four texels nearest, so a stroke a texel wide never drops out
      const at = d.vec2i(std.clamp(uv * RULE_SIZE - 0.5, d.vec2f(0), d.vec2f(RULE_SIZE - 2)));
      const a = std.textureLoad(ruleView.$, at, 0).xy;
      const b = std.textureLoad(ruleView.$, at + d.vec2i(1, 0), 0).xy;
      const c = std.textureLoad(ruleView.$, at + d.vec2i(0, 1), 0).xy;
      const e = std.textureLoad(ruleView.$, at + d.vec2i(1, 1), 0).xy;
      const most = std.max(std.max(a, b), std.max(c, e));
      return d.vec2f(most.x * 0.9, most.y);
    }
    if (kind === KIND.normals) {
      const read = std.textureSampleLevel(normalView.$, smooth.$, uv, 0);
      const normal = read.xyz * 2 - 1;
      return d.vec2f(
        read.w * (0.12 + 0.88 * std.saturate(std.dot(normal, d.vec3f(-0.5, 0.75, 0.45)))),
        0,
      );
    }
    if (kind === KIND.grid) return gridAt(uv);
    if (kind === KIND.shadow) {
      const size = d.vec2f(std.textureDimensions(shadowView.$));
      const texel = d.vec2i(std.min(uv * size, size - 1));
      const depth = std.textureLoad(shadowView.$, texel, 0, 0);
      const contour = std.select(d.f32(0), d.f32(0.18), std.fract(depth * 40) < 0.1);
      return d.vec2f(
        std.select(std.saturate((0.92 - depth) * 1.2) + contour, d.f32(0), depth >= 1),
        0,
      );
    }
    return tableAt(uv);
  };

  /**
   * A cell's colour from its value: the greys, and the accent where `emitted` passes the
   * threshold.
   */
  const ramp = (value: number, emitted: number, threshold: number) => {
    'use gpu';
    const p = palette.$;
    if (emitted > threshold) {
      const i = step(value, 4, threshold);
      let color = p.grey[0].xyz;
      if (i === 1) color = p.grey[2].xyz;
      if (i === 2) color = p.accent[0].xyz;
      if (i === 3) color = p.accent[1].xyz;
      return color;
    }
    return p.grey[step(value, 7, threshold)].xyz;
  };

  const Corner = {
    position: d.builtin.position,
    world: d.vec3f,
    uv: d.vec2f,
    which: d.interpolate('flat', d.u32),
    /** 0 for the top face, then the four sides. */
    facet: d.interpolate('flat', d.u32),
  };

  /** A slab: its top face, then its four sides, two triangles each. */
  const face = tgpu.vertexFn({
    in: { vertexIndex: d.builtin.vertexIndex, instanceIndex: d.builtin.instanceIndex },
    out: Corner,
  })(({ vertexIndex, instanceIndex }) => {
    'use gpu';
    const view = viewLayout.$.view;
    const sheet = viewLayout.$.sheets[instanceIndex];
    const facet = d.u32(vertexIndex / 6);
    const k = vertexIndex % 6;
    const a = d.f32(k === 1 || k === 4 || k === 5);
    const b = d.f32(k === 2 || k === 3 || k === 5);

    let xz = d.vec2f(a, b) * 2 - 1;
    let down = d.f32(0);
    if (facet > 0) {
      const along = a * 2 - 1;
      down = b;
      if (facet === 1) xz = d.vec2f(along, -1);
      if (facet === 2) xz = d.vec2f(1, along);
      if (facet === 3) xz = d.vec2f(-along, 1);
      if (facet === 4) xz = d.vec2f(-1, -along);
    }
    const flat = sheet.place.xz + xz * sheet.place.w;
    const world = d.vec3f(flat.x, sheet.place.y - down * view.thick * sheet.place.w, flat.y);
    const socket = sheet.kind === KIND.socket;
    return {
      position: std.select(view.viewProjection * d.vec4f(world, 1), d.vec4f(), socket),
      world,
      uv: d.vec2f(a, b),
      which: instanceIndex,
      facet,
    };
  });

  /** Sample at the cell centre before dithering. */
  const cell = tgpu.fragmentFn({
    in: {
      position: d.builtin.position,
      world: d.vec3f,
      uv: d.vec2f,
      which: d.interpolate('flat', d.u32),
      facet: d.interpolate('flat', d.u32),
    },
    out: d.vec4f,
  })(({ position, world, uv, which, facet }) => {
    'use gpu';
    const view = viewLayout.$.view;
    const sheet = viewLayout.$.sheets[which];
    const threshold = bayerCell(d.vec2u(position.xy));
    if (threshold >= sheet.shown) std.discard();

    if (facet > 0) {
      let normal = d.vec3f(0, 0, -1);
      if (facet === 2) normal = d.vec3f(1, 0, 0);
      if (facet === 3) normal = d.vec3f(0, 0, 1);
      if (facet === 4) normal = d.vec3f(-1, 0, 0);
      const lit = 0.16 + 0.3 * std.max(std.dot(normal, LIGHT), 0);
      return d.vec4f(ramp(lit * sheet.keeps, d.f32(0), threshold), 1);
    }

    const read = valueAt(sheet.kind, uv);
    let value = (view.base + (1 - view.base) * read.x) * sheet.keeps;

    const cover = sheet.cover;
    const lift = (cover.y - world.y) / LIGHT.y;
    const there = world.xz + LIGHT.xz * lift;
    const inside = std.abs(there - cover.xz);
    if (sheet.shade > 0 && inside.x < cover.w && inside.y < cover.w) value *= 1 - 0.5 * sheet.shade;

    return d.vec4f(ramp(value, read.y * sheet.keeps, threshold), 1);
  });

  const cellsPipeline = root.pipe(birds.inject()).createRenderPipeline({
    vertex: face,
    fragment: cell,
    targets: { format: 'rgba8unorm' },
    depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
  });

  /** The cells, each a block of whole pixels: what the canvas shows under the hairlines. */
  const upscale = tgpu.fragmentFn({ in: { position: d.builtin.position }, out: d.vec4f })(({
    position,
  }) => {
    'use gpu';
    const at = d.vec2i(std.floor(position.xy / viewLayout.$.view.cell));
    return std.textureLoad(cellsLayout.$.cells, at, 0);
  });

  const upscalePipeline = root.createRenderPipeline({
    vertex: common.fullScreenTriangle,
    fragment: upscale,
    targets: { format },
    depthStencil: { format: 'depth24plus', depthWriteEnabled: false, depthCompare: 'always' },
  });

  /** Depth alone, with the holes of a sheet dithering in, cell for cell. */
  const nothing = tgpu.fragmentFn({
    in: { position: d.builtin.position, which: d.interpolate('flat', d.u32) },
    out: d.vec4f,
  })(({ position, which }) => {
    'use gpu';
    const view = viewLayout.$.view;
    const threshold = bayerCell(d.vec2u(std.floor(position.xy / view.cell)));
    if (threshold >= viewLayout.$.sheets[which].shown) std.discard();
    return d.vec4f();
  });

  const depthPipeline = root.createRenderPipeline({
    vertex: face,
    fragment: nothing,
    targets: { format, writeMask: 0 },
    depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
  });

  /** Twelve slab edges, then the four of its seat, with two endpoints each. */
  const edge = tgpu.vertexFn({
    in: { vertexIndex: d.builtin.vertexIndex, instanceIndex: d.builtin.instanceIndex },
    out: {
      position: d.builtin.position,
      color: d.vec4f,
      along: d.f32,
      solid: d.interpolate('flat', d.f32),
    },
  })(({ vertexIndex, instanceIndex }) => {
    'use gpu';
    const view = viewLayout.$.view;
    const sheet = viewLayout.$.sheets[instanceIndex];
    const p = palette.$;
    const e = vertexIndex >>> 1;
    const end = vertexIndex & 1;
    const socket = sheet.kind === KIND.socket;
    const seat = e >= 12;
    const missing = 1 - sheet.shown;

    // A socket has only its seat, and a sheet all there none
    if ((socket && !seat) || (seat && missing * sheet.rim < 0.001)) {
      return { position: d.vec4f(), color: d.vec4f(), along: d.f32(-1), solid: d.f32(1) };
    }

    let corner = (e + end) & 3;
    let level = d.u32(0);
    if (e >= 4 && e < 8) level = 1;
    if (e >= 8 && e < 12) {
      corner = e - 8;
      level = end;
    }
    const x = std.select(d.f32(-1), d.f32(1), corner === 1 || corner === 2);
    const z = std.select(d.f32(-1), d.f32(1), corner >= 2);
    const centre = std.select(sheet.place.xz, d.vec2f(), seat);
    const xz = centre + d.vec2f(x, z) * sheet.place.w;
    const world = d.vec3f(xz.x, sheet.place.y - d.f32(level) * view.thick * sheet.place.w, xz.y);
    let position = view.viewProjection * d.vec4f(world, 1);
    position = d.vec4f(position.xy, position.z - 0.0008 * position.w, position.w);

    // The slab's lines dither in with it, dashed until it is all there; its seat's fade as it fills
    const empty = std.min(view.line * 2.4, d.f32(1)) * 0.5;
    const line = std.select(view.line * sheet.shown, empty * sheet.rim * missing, seat);
    const hot = std.select(sheet.hot, sheet.hot * missing, seat);
    const color = std.mix(p.accent[0].xyz * line, p.accent[1].xyz, hot);
    const alpha = std.mix(line, d.f32(1), hot);
    const dashed = e < 4 || seat;
    const along = std.select(d.f32(-1), d.f32(end) * 10, dashed);
    const solid = std.select(sheet.shown, d.f32(0), seat);
    return { position, color: d.vec4f(color, alpha), along, solid };
  });

  const ink = tgpu.fragmentFn({
    in: { color: d.vec4f, along: d.f32, solid: d.interpolate('flat', d.f32) },
    out: d.vec4f,
  })(({ color, along, solid }) => {
    'use gpu';
    if (along >= 0 && std.fract(along) > 0.55 + solid * 0.5) std.discard();
    return d.vec4f(color);
  });

  const linesPipeline = root.createRenderPipeline({
    vertex: edge,
    fragment: ink,
    targets: {
      format,
      blend: {
        color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
        alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
      },
    },
    primitive: { topology: 'line-list' },
    depthStencil: { format: 'depth24plus', depthWriteEnabled: false, depthCompare: 'less-equal' },
  });

  return {
    async initAsync() {
      await Promise.all([
        cellsPipeline.initAsync(),
        upscalePipeline.initAsync(),
        depthPipeline.initAsync(),
        linesPipeline.initAsync(),
      ]);
    },

    /** A view of the stack on `surface`, with cells `cellCss` CSS pixels across. */
    view(surface: Surface, cellCss: () => number) {
      const uniform = root.createBuffer(View).$usage('uniform');
      const sheetsBuffer = root.createBuffer(d.arrayOf(Sheet, MAX)).$usage('uniform');
      const group = root.createBindGroup(viewLayout, { view: uniform, sheets: sheetsBuffer });
      const cellPx = () => Math.max(1, Math.round(cellCss() * surface.pixelRatio));
      const cellsSize = {
        get width() {
          return Math.ceil(surface.width / cellPx());
        },
        get height() {
          return Math.ceil(surface.height / cellPx());
        },
      };
      const cells = targets.create(root, cellsSize, {
        color: { format: 'rgba8unorm', sampled: true },
        depth: { format: 'depth24plus', transient: true },
      });
      const full = targets.create(root, surface, {
        depth: { format: 'depth24plus', transient: true },
      });
      let cellsGroup = root.createBindGroup(cellsLayout, { cells: cells.textures.color });

      const matrix = mat4.identity();
      const flat = d.mat4x4f();
      const records = Array.from({ length: MAX }, () => ({
        place: d.vec4f(),
        cover: d.vec4f(),
        kind: 0,
        hot: 0,
        shade: 0,
        shown: 0,
        keeps: 1,
        rim: 1,
      }));

      /** The view-projection for `camera` on this surface, and its inverse. */
      const project = (camera: Camera) => {
        const aspect = surface.width / Math.max(surface.height, 1);
        const { yaw, pitch, scale, target, anchor } = camera;
        const forward = [
          Math.cos(pitch) * Math.sin(yaw),
          Math.sin(pitch),
          Math.cos(pitch) * Math.cos(yaw),
        ];
        const eye = [
          target[0] + forward[0] * 30,
          target[1] + forward[1] * 30,
          target[2] + forward[2] * 30,
        ];
        const up =
          Math.abs(Math.sin(pitch)) > 0.999 ? [-Math.sin(yaw), 0, -Math.cos(yaw)] : [0, 1, 0];
        const dx = (anchor[0] - 0.5) * 2 * scale * aspect;
        const dy = (0.5 - anchor[1]) * 2 * scale;
        const view = mat4.lookAt(eye, target, up);
        const ortho = mat4.ortho(
          -scale * aspect - dx,
          scale * aspect - dx,
          -scale - dy,
          scale - dy,
          1,
          60,
        );
        mat4.multiply(ortho, view, matrix);
        return matrix;
      };

      return {
        surface,
        cells,

        /** CSS pixels on the canvas of a point in the world, under the last camera drawn. */
        toCanvas(point: readonly number[]) {
          const clip = vec3.transformMat4(point, matrix);
          const width = surface.width / surface.pixelRatio;
          const height = surface.height / surface.pixelRatio;
          return [(clip[0] * 0.5 + 0.5) * width, (0.5 - clip[1] * 0.5) * height] as const;
        },

        /** Points the view through `camera` for `toCanvas`, without writing it for `record`. */
        aim(camera: Camera) {
          project(camera);
        },

        /** Writes the camera and the sheets, top first, for `record`. */
        write(
          camera: Camera,
          sheets: readonly SheetState[],
          look: {
            base: number;
            line: number;
            thick: number;
          },
        ) {
          project(camera);
          for (let i = 0; i < 16; i++) flat.columns[i >> 2][i & 3] = matrix[i];
          const count = Math.min(sheets.length, MAX);
          for (let i = 0; i < count; i++) {
            const sheet = sheets[i];
            const above = sheets[i - 1];
            const record = records[i];
            record.place = d.vec4f(sheet.x, sheet.y, sheet.z, sheet.half);
            record.cover = above
              ? d.vec4f(above.x, above.y, above.z, above.half)
              : d.vec4f(0, 0, 0, 0);
            record.kind = KIND[sheet.kind];
            record.hot = sheet.hot;
            record.shade =
              above && above.kind !== 'socket'
                ? clamp((above.y - sheet.y) / 0.25) * above.shown
                : 0;
            record.shown = sheet.shown;
            record.keeps = sheet.keeps;
            record.rim = sheet.rim ?? 1;
          }
          sheetsBuffer.write(records);
          uniform.write({
            viewProjection: flat,
            base: look.base,
            line: look.line,
            thick: look.thick,
            cell: cellPx(),
          });
          return count;
        },

        /**
         * Draws `count` sheets, as `write` left them, into the surface, then `after` into the same
         * pass.
         */
        record(frame: Frame, count: number, after?: (pass: TgpuRenderPass) => void) {
          if (cells.fit())
            cellsGroup = root.createBindGroup(cellsLayout, { cells: cells.textures.color });
          full.fit();

          const pass = frame.encoder.beginRenderPass({
            label: 'stack cells',
            colorAttachments: { view: cells.views.color, clearValue: [0, 0, 0, 0] },
            depthStencilAttachment: {
              view: cells.views.depth,
              depthClearValue: 1,
              depthLoadOp: 'clear',
              depthStoreOp: 'discard',
            },
          });
          pass.setBindGroup(group);
          pass.setPipeline(cellsPipeline);
          pass.draw(30, count);
          pass.end();

          const last = frame.encoder.beginRenderPass({
            label: 'stack',
            colorAttachments: { view: surface.view(frame), clearValue: [0, 0, 0, 0] },
            depthStencilAttachment: {
              view: full.views.depth,
              depthClearValue: 1,
              depthLoadOp: 'clear',
              depthStoreOp: 'discard',
            },
          });
          last.setBindGroup(group);
          last.setBindGroup(cellsGroup);
          last.setPipeline(upscalePipeline);
          last.draw(3);
          last.setPipeline(depthPipeline);
          last.draw(30, count);
          last.setPipeline(linesPipeline);
          last.draw(32, count);
          after?.(last);
          last.end();
        },

        cellPx,

        destroy() {
          uniform.destroy();
          sheetsBuffer.destroy();
          cells.destroy();
          full.destroy();
        },
      };
    },

    destroy() {
      palette.buffer.destroy();
    },
  };
}

export type Stacks = ReturnType<typeof createStacks>;
export type StackView = ReturnType<Stacks['view']>;
