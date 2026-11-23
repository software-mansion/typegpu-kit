import { pipelines, type SizeSource } from '@typegpu/kit';
import { perlin2d, randf } from '@typegpu/noise';
import {
  d,
  std,
  tgpu,
  type TgpuComputePass,
  type TgpuRenderCommands,
  type TgpuRoot,
  type TgpuUniform,
} from 'typegpu';

const WORKGROUP = 64;
const TAU = Math.PI * 2;

/** Chance per step of respawning a particle elsewhere. Prevents particles collecting in sinks. */
const RESPAWN = 1 / 240;

/** A random position across the canvas, in clip space. */
const spawn = () => {
  'use gpu';
  return d.vec2f(randf.sample(), randf.sample()) * 2 - 1;
};

const Particle = d.struct({ position: d.vec2f, direction: d.vec2f });

export const FlowSettings = d.struct({
  /** Clip-space units per second. */
  speed: d.f32,
  /** Noise cells across the height. */
  scale: d.f32,
  /** Particle diameter in pixels. */
  pointSize: d.f32,
});

export interface FlowFieldOptions {
  readonly count: number;
  /** Seconds per `step`. */
  readonly step: number;
  /**
   * Seconds since the last fixed step, written from `sim.ahead` each frame. Particles are drawn
   * that far along their direction, so they move at the display rate. Default 0.
   */
  readonly ahead?: TgpuUniform<d.F32>;
  readonly format: GPUTextureFormat;
  /** What the particles are drawn into, such as the canvas surface. */
  readonly size: SizeSource;
  /** Initial speed, scale and point size. */
  readonly settings?: Partial<d.InferInput<typeof FlowSettings>>;
}

export const FLOW = { speed: 0.2, scale: 1.5, pointSize: 2 };

/** Advects particles through Perlin noise. Owns the buffers and records into caller passes. */
export function createFlowField(root: TgpuRoot, options: FlowFieldOptions) {
  const { count, step, format, size } = options;
  const ahead = tgpu.accessor(d.f32);

  const particles = root.createBuffer(d.arrayOf(Particle, count)).$usage('storage');
  const writable = particles.as('mutable');
  const readable = particles.as('readonly');

  const settings = root.createUniform(FlowSettings, { ...FLOW, ...options.settings });
  const viewport = root.createUniform(d.vec2f, d.vec2f(size.width, size.height));
  const seedValue = root.createUniform(d.f32);

  const workgroups = Math.ceil(count / WORKGROUP);

  const seedKernel = tgpu.computeFn({
    workgroupSize: [WORKGROUP],
    in: { gid: d.builtin.globalInvocationId },
  })(({ gid }) => {
    'use gpu';
    if (gid.x >= count) return;

    randf.seed2(d.vec2f(d.f32(gid.x), seedValue.$));
    writable.$[gid.x] = Particle({ position: spawn(), direction: d.vec2f() });
  });

  const stepKernel = tgpu.computeFn({
    workgroupSize: [WORKGROUP],
    in: { gid: d.builtin.globalInvocationId },
  })(({ gid }) => {
    'use gpu';
    if (gid.x >= count) return;
    const particle = writable.$[gid.x];

    const aspect = viewport.$.x / viewport.$.y;
    const sample = particle.position * d.vec2f(aspect, 1) * settings.$.scale;
    const angle = perlin2d.sample(sample) * TAU;
    const direction = d.vec2f(std.cos(angle), std.sin(angle));

    let position = particle.position + direction * (settings.$.speed * step);

    randf.seed3(d.vec3f(position, d.f32(gid.x)));
    if (randf.sample() < RESPAWN || std.any(std.gt(std.abs(position), d.vec2f(1)))) {
      position = spawn();
    }

    writable.$[gid.x] = Particle({ position, direction });
  });

  const vertex = tgpu.vertexFn({
    in: { vertexIndex: d.builtin.vertexIndex, instanceIndex: d.builtin.instanceIndex },
    out: { position: d.builtin.position, color: d.vec3f, corner: d.vec2f },
  })(({ vertexIndex, instanceIndex }) => {
    'use gpu';
    const particle = readable.$[instanceIndex];

    const corner = d.vec2f(d.f32(vertexIndex & 1), d.f32(vertexIndex >>> 1)) * 2 - 1;
    const extent = (corner * settings.$.pointSize) / viewport.$;
    const center = particle.position + particle.direction * (settings.$.speed * ahead.$);

    return {
      position: d.vec4f(center + extent, 0, 1),
      color: d.vec3f(particle.direction * 0.5 + 0.5, 0.9),
      corner,
    };
  });

  const fragment = tgpu.fragmentFn({
    in: { color: d.vec3f, corner: d.vec2f },
    out: d.vec4f,
  })(({ color, corner }) => {
    'use gpu';
    const falloff = 1 - std.smoothstep(0.4, 1, std.length(corner));

    return d.vec4f(color * (falloff * 0.5), falloff);
  });

  const seedPipeline = root.createComputePipeline({ compute: seedKernel });
  const stepPipeline = root.createComputePipeline({ compute: stepKernel });
  const drawPipeline = root.with(ahead, options.ahead ?? 0).createRenderPipeline({
    vertex,
    fragment,
    targets: { format, blend: pipelines.blend.add },
    primitive: { topology: 'triangle-strip' },
  });

  let width = size.width;
  let height = size.height;

  return {
    settings,
    initAsync: () => pipelines.compile([seedPipeline, stepPipeline, drawPipeline]),

    seed(pass: TgpuComputePass) {
      seedValue.write(Math.random());
      pass.setPipeline(seedPipeline);
      pass.dispatchWorkgroups(workgroups);
    },

    step(pass: TgpuComputePass) {
      pass.setPipeline(stepPipeline);
      pass.dispatchWorkgroups(workgroups);
    },

    draw(pass: TgpuRenderCommands) {
      if (size.width !== width || size.height !== height) {
        width = size.width;
        height = size.height;
        viewport.write(d.vec2f(width, height));
      }

      pass.setPipeline(drawPipeline);
      pass.draw(4, count);
    },

    destroy() {
      particles.destroy();
    },
  };
}
