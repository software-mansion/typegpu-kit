import { cameras, clock, targets, type Frame } from '@typegpu/kit';
import { d, std, tgpu, type TgpuRoot } from 'typegpu';
import { mat4, vec4 } from 'wgpu-matrix';
import { AIR, Bird, bornAt, CAPACITY, CELL, COLUMNS, degreesUp, LOW, STEP } from './birds.ts';
import { createFrame, LIT, SIZE } from './frame.code.ts';
import { smooth } from './geometry.ts';
import { KNOBS, type Knobs } from './knobs.ts';
import type { RuleName } from './rules.ts';

/** Texels a side of the rule's plate. */
export const RULE_SIZE = 128;

/** The frame's modules in the order its code builds them. */
export const MODULES = ['table', 'grid', 'shadow', 'lit', 'passes', 'bloom', 'finish'] as const;
export type Module = (typeof MODULES)[number];

/** What the reader can take out of the whole frame: the modules whose plates slide out. */
export type Removable = 'bloom' | 'lit' | 'shadow' | 'grid';

const EYE = [0, 13.5, 8.2] as const;
const AIM = [0, 0, -0.9] as const;
const FOV = 0.86;
/** The height the pointer's ray is read at, metres. */
const POINTER_HEIGHT = 1.2;

/** The shadow without the lighting's sun and sky: a plain white sun over an even grey. */
const PLAIN = {
  sunIntensity: 1.3,
  sunColor: d.vec3f(1, 1, 1),
  skyIntensity: 0.7,
  skyTop: d.vec3f(0.3, 0.3, 0.3),
  skyBottom: d.vec3f(0.3, 0.3, 0.3),
};

/** Neither: an even grey light. */
const FLAT = {
  sunIntensity: 0,
  sunColor: d.vec3f(1, 1, 1),
  skyIntensity: 1,
  skyTop: d.vec3f(0.45, 0.45, 0.45),
  skyBottom: d.vec3f(0.45, 0.45, 0.45),
};

/** Where the walk under the pointer starts, in columns of the grid, and 1 in `z` while it shows. */
const Walk = d.struct({ at: d.vec4f });

/** A stroke of the rule's plate, from a bird along the force its rule gave it. */
const Stroke = { position: d.builtin.position, value: d.vec2f };

/** The frame the page shows, with what the stack's plates read besides its textures. */
export function createSubject(root: TgpuRoot) {
  const size = { width: SIZE, height: SIZE };
  const knobs: Knobs = Object.fromEntries(
    Object.entries(KNOBS).map(([key, knob]) => [key, knob.value]),
  ) as unknown as Knobs;
  const out = new Set<Removable>();
  let built: number = MODULES.length;
  let ruleName: RuleName = 'flock';
  /** Whether `module` is in the frame: built, and not taken out. */
  const has = (module: Module) => MODULES.indexOf(module) < built && !out.has(module as Removable);

  const view = cameras.custom(root);
  const viewMatrix = mat4.lookAt(EYE, AIM, [0, 1, 0]);
  const projection = mat4.perspective(FOV, 1, 0.1, 100);
  const inverse = mat4.invert(mat4.multiply(projection, viewMatrix));
  view.write(viewMatrix, projection);

  const screen = targets.create(root, size, {
    hdr: { format: 'rgba16float', sampled: true },
    normals: { format: 'rgba8unorm', sampled: true },
    depth: { format: 'depth24plus' },
    final: { format: 'rgba8unorm', sampled: true },
  });

  const parts = createFrame(root, screen, view, has);
  const { birds, rows, cells, steering, lamps, light, scene, chevrons, glow, print, final } = parts;
  const [ground] = parts.shown;

  const populate = (count: number) => {
    while (rows.length < count) rows.push(birds.add(bornAt()));
    while (rows.length > count) birds.remove(rows.pop()!);
  };

  const columns = root.createBuffer(d.arrayOf(d.u32, COLUMNS * COLUMNS)).$usage('storage');
  const counting = columns.as('mutable');
  const countKernel = tgpu.computeFn({
    workgroupSize: [64],
    in: { gid: d.builtin.globalInvocationId },
  })(({ gid }) => {
    'use gpu';
    const c = gid.x;
    if (c >= COLUMNS * COLUMNS) return;
    const column = d.vec2f(d.f32(c % COLUMNS), d.f32(std.intdiv(c, d.u32(COLUMNS))));
    let count = d.u32(0);
    for (let i = d.u32(0); i < CAPACITY; i++) {
      if (!Bird.alive(i)) continue;
      const at = std.floor((Bird.position(i).xz - LOW.xz) / CELL);
      if (at.x === column.x && at.y === column.y) count += 1;
    }
    counting.$[c] = count;
  });
  const countPipeline = root.pipe(birds.inject()).createComputePipeline({ compute: countKernel });
  const walk = root.createUniform(Walk);

  const ruleMap = targets.create(
    root,
    { width: RULE_SIZE, height: RULE_SIZE },
    { strokes: { format: 'rgba8unorm', sampled: true } },
  );
  const strokeAt = tgpu.vertexFn({
    in: { vertexIndex: d.builtin.vertexIndex, instanceIndex: d.builtin.instanceIndex },
    out: Stroke,
  })(({ vertexIndex, instanceIndex }) => {
    'use gpu';
    if (!Bird.alive(instanceIndex)) return { position: d.vec4f(), value: d.vec2f() };
    const force = Bird.force(instanceIndex).xz;
    const length = std.length(force);
    const along = (force / std.max(length, 1e-4)) * std.min(length * 0.12, 1.4);
    const tip = d.f32(vertexIndex);
    const xz = Bird.position(instanceIndex).xz + along * tip;
    const uv = (xz - LOW.xz) / AIR.xz;
    return {
      position: d.vec4f(uv.x * 2 - 1, 1 - uv.y * 2, 0, 1),
      value: d.vec2f(0.35 + tip * 0.65, tip),
    };
  });
  const strokeColor = tgpu.fragmentFn({ in: { value: d.vec2f }, out: d.vec4f })(({ value }) => {
    'use gpu';
    return d.vec4f(value, 0, 1);
  });
  const strokes = root.pipe(birds.inject()).createRenderPipeline({
    vertex: strokeAt,
    fragment: strokeColor,
    targets: {
      format: 'rgba8unorm',
      blend: {
        color: { operation: 'max', srcFactor: 'one', dstFactor: 'one' },
        alpha: { operation: 'max', srcFactor: 'one', dstFactor: 'one' },
      },
    },
    primitive: { topology: 'line-list' },
  });

  const sim = clock.fixedStep({ step: STEP, maxSteps: 3 });
  let time = 0;
  let startle = 0;
  let shape = 1;
  const pointer = { x: 0, z: 0, over: 0 };
  const walkAt = [0, 0, 0];

  /** Writes what depends on which modules are in. */
  const apply = () => {
    final.settings.patch({
      glowIntensity: has('bloom') ? knobs.strength : 0,
      vignette: has('finish') ? 0.25 : 0,
    });
    print.settings.patch({ on: has('finish') ? 1 : 0 });
    light.settings.patch(has('lit') ? LIT : has('shadow') ? PLAIN : FLAT);
    lamps.settings.patch({ intensity: has('lit') && has('grid') ? knobs.intensity : 0 });
  };

  /** The point on the birds' height under (u, v) of the frame, from its top left. */
  const groundAt = (u: number, v: number, height = POINTER_HEIGHT) => {
    const ndc = [u * 2 - 1, 1 - v * 2];
    const near = vec4.transformMat4([ndc[0], ndc[1], 0, 1], inverse);
    const far = vec4.transformMat4([ndc[0], ndc[1], 1, 1], inverse);
    const a = [near[0] / near[3], near[1] / near[3], near[2] / near[3]];
    const b = [far[0] / far[3], far[1] / far[3], far[2] / far[3]];
    const t = (height - a[1]) / (b[1] - a[1] || -1e-6);
    return [a[0] + (b[0] - a[0]) * t, height, a[2] + (b[2] - a[2]) * t] as const;
  };

  return {
    size,
    screen,
    glow,
    light,
    birds,
    ruleMap,
    columns: columns.as('readonly'),
    walk,
    knobs: knobs as Readonly<Knobs>,

    /** Whether `module` is in the frame. */
    has,

    /** Whether the reader has taken `module` out. */
    isOut: (module: Removable) => out.has(module),

    /** Takes `module` out of the frame, or puts it back. */
    remove(module: Removable, removed: boolean) {
      if (removed === out.has(module)) return;
      if (removed) out.add(module);
      else out.delete(module);
      apply();
    },

    /** How many of `MODULES` are built, from the first: the frame does without the rest. */
    get built() {
      return built;
    },
    set built(count: number) {
      if (count === built) return;
      built = count;
      apply();
    },

    /** The birds' rule. */
    get rule() {
      return ruleName;
    },
    set rule(name: RuleName) {
      ruleName = name;
    },

    /** Sets one of the numbers the reader drags. Each is a uniform, so the next frame shows it. */
    knob(key: keyof Knobs, value: number) {
      knobs[key] = value;
      if (key === 'count') populate(Math.round(value));
      if (key === 'elevation') light.settings.patch({ sunDirection: degreesUp(value) });
      if (key === 'range') lamps.settings.patch({ range: value });
      if (key === 'length') chevrons.settings.patch({ length: value });
      if (key === 'threshold') glow.settings.patch({ threshold: value });
      if (key === 'cell') print.settings.patch({ cell: value });
      if (key === 'levels') print.settings.patch({ levels: value });
      if (key === 'toward') steering.tuning.patch({ toward: value });
      if (key === 'radius') steering.tuning.patch({ radius: value });
      if (key === 'crowd') steering.tuning.patch({ crowd: value });
      if (key === 'intensity' || key === 'strength') apply();
    },

    /**
     * The pointer over the frame, at (u, v) from its top left, or off it: the birds flee it and the
     * grid's plate shows the walk from it.
     */
    aim(at: readonly [number, number] | undefined) {
      if (!at) {
        pointer.over = 0;
        return;
      }
      const [x, , z] = groundAt(at[0], at[1]);
      pointer.x = x;
      pointer.z = z;
      pointer.over = 1;
    },

    /** The pointer over a plate laid out as the air, at (u, v) from its far left corner. */
    aimFlat(at: readonly [number, number] | undefined) {
      if (!at) {
        pointer.over = 0;
        return;
      }
      pointer.x = LOW.x + at[0] * AIR.x;
      pointer.z = LOW.z + at[1] * AIR.z;
      pointer.over = 1;
    },

    /** Scatters the flock, as a press on the lockup does. */
    startle() {
      startle = 1;
    },

    async initAsync() {
      await Promise.all([
        scene.initAsync(),
        glow.initAsync(),
        final.initAsync(),
        cells.initAsync(),
        steering.initAsync(),
        countPipeline.initAsync(),
        strokes.initAsync(),
        light.shadow.initAsync(),
      ]);
    },

    /** Records the whole frame into the frame's encoder: every texture the stack shows. */
    record(frame: Frame, dt: number) {
      time += dt;
      startle *= Math.exp(-dt * 3);
      const shaped = has('passes') ? Math.min(shape + dt * 2.5, 1) : Math.max(shape - dt * 2.5, 0);
      if (shaped !== shape) {
        shape = shaped;
        chevrons.settings.patch({ shape: smooth(shape) });
      }
      const onGrid = has('grid');
      steering.settings.write({
        pointer: d.vec4f(pointer.x, POINTER_HEIGHT, pointer.z, pointer.over),
        time,
        sight: knobs.sight,
        startle: startle > 0.01 ? startle : 0,
      });
      const column = [
        Math.floor((pointer.x - LOW.x) / CELL),
        Math.floor((pointer.z - LOW.z) / CELL),
        pointer.over,
      ];
      if (column.some((value, i) => value !== walkAt[i])) {
        walkAt.splice(0, 3, ...column);
        walk.write({ at: d.vec4f(column[0], column[1], onGrid ? column[2] : 0, 0) });
      }
      birds.flush();

      const compute = frame.encoder.beginComputePass({ label: 'flock' });
      const steps = sim.update(dt);
      for (let left = steps; left > 0; left--) {
        if (onGrid) cells.sort(compute);
        steering.step(compute, onGrid, ruleName);
      }
      // The lamps find the ground through the grid, sorted where the birds now are
      if (onGrid) cells.sort(compute);
      compute.setPipeline(countPipeline);
      compute.dispatchWorkgroups(Math.ceil((COLUMNS * COLUMNS) / 64));
      compute.end();

      const strokePass = frame.encoder.beginRenderPass({
        label: 'rule',
        colorAttachments: { view: ruleMap.views.strokes, clearValue: [0, 0, 0, 0] },
      });
      if (birds.span > 0 && has('table')) {
        strokePass.setPipeline(strokes);
        strokePass.draw(2, birds.span);
      }
      strokePass.end();

      // Recorded even while out: its `enabled` turns false then, and the frame goes sunlit
      const drawn = has('table') ? parts.shown : [ground!];
      light.shadow.record(frame, has('table') ? parts.cast : []);
      scene.record(frame, drawn);
      if (has('bloom')) glow.record(frame);

      const last = frame.encoder.beginRenderPass({
        label: 'finish',
        colorAttachments: { view: screen.views.final },
      });
      final.draw(last);
      last.end();
    },

    destroy() {
      glow.destroy();
      scene.destroy();
      light.destroy();
      cells.destroy();
      birds.destroy();
      columns.destroy();
      ruleMap.destroy();
      for (const each of [steering.settings, steering.tuning, walk, lamps.settings]) {
        each.buffer.destroy();
      }
      for (const each of [chevrons.settings, print.settings]) each.buffer.destroy();
      screen.destroy();
    },
  };
}

export type Subject = ReturnType<typeof createSubject>;
