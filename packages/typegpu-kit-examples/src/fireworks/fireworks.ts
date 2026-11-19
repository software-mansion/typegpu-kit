import {
  lighting,
  Particle,
  particles,
  pipelines,
  shading,
  trails,
  type CameraBase,
  type Lighting,
} from '@typegpu/kit';
import type { Inject } from '@typegpu/kit/inject';
import { perlin2d, randf } from '@typegpu/noise';
import { d, std, tgpu, type TgpuComputePass, type TgpuRoot } from 'typegpu';

const WORKGROUP = 64;
const workgroups = (span: number) => Math.ceil(span / WORKGROUP);
const GRAVITY = 9.81;
/** Hertz of the glitter on crackle stars and flashes. */
const GLITTER_HZ = 17;
/** Brightness of stars, shells, their streaks and tails, and embers. */
const BRIGHTNESS = { star: 9, shell: 14, streak: 3.2, tail: 2.2, ember: 6 };

/** Fraction of the shell's velocity its stars carry. */
const CARRIED = 0.25;
/** How far a star's speed and life stray from their pattern's, as fractions. */
const SPREAD = { speed: 0.12, life: 0.3 };
/** Seconds a star burns white-hot before it shows its colour. */
const HOT_SECONDS = 0.12;
const EMBER_RED = d.vec3f(1, 0.18, 0.04);
const EMBER_GOLD = d.vec3f(1, 0.6, 0.22);
const SHELL_COLOR = d.vec3f(1, 0.55, 0.2);
/** Per-step probability that a rising shell sheds an ember. */
const SHELL_SHEDS = 0.6;

/**
 * Each flash slot adds a point light that every lit pixel visits. A finale bursts 26 shells in
 * 6 s, and flashes last 1.6 s. The slot ring therefore reuses slots after their flashes expire.
 */
const FLASHES = 32;
/** Metres a flash's light reaches, and its intensity per squared flash size. */
const FLASH = { range: 320, power: 3 };

const SMOKE_PUFFS = 14;
const SMOKE_COLOR = d.vec3f(0.5, 0.52, 0.58);
/** Smoke brightness inside a burst. Falls with distance outside the burst. */
const SMOKE_GLOW = 0.3;
/** Uniformly lit smoke-core radius as a fraction of flash size. */
const SMOKE_CORE = 0.4;
/** The share of a flash's light on a surface that lights smoke at its core's edge. */
const SMOKE_SHARE = (SMOKE_GLOW * SMOKE_CORE * SMOKE_CORE) / FLASH.power;

/** Burst patterns of shells, also carried by their stars and flashes. */
export const Pattern = {
  peony: 0,
  chrysanthemum: 1,
  willow: 2,
  palm: 3,
  ring: 4,
  crackle: 5,
} as const;
export type Pattern = (typeof Pattern)[keyof typeof Pattern];

const Traits = d.struct({
  /** Stars a shell bursts into. */
  count: d.u32,
  /** Base star lifetime in seconds before per-star variation. */
  life: d.f32,
  /** Speed lost per second, as a fraction. */
  drag: d.f32,
  /** Gravity multiplier relative to Earth. */
  fall: d.f32,
  /** Metres across a star's glow. */
  size: d.f32,
  /** Streak brightness relative to its star. */
  streak: d.f32,
  /** Fixed steps between the points of its streak, so slow stars leave long ones. */
  every: d.u32,
  /** Per-step probability of shedding an ember. */
  sheds: d.f32,
});

/** Each pattern's traits, which its shells' stars share. */
const TRAITS: Record<keyof typeof Pattern, d.InferInput<typeof Traits>> = {
  peony: { count: 120, life: 1.9, drag: 1.2, fall: 0.5, size: 2, streak: 0.35, every: 1, sheds: 0 },
  chrysanthemum: {
    count: 140,
    life: 2.2,
    drag: 1.1,
    fall: 0.6,
    size: 1.6,
    streak: 1,
    every: 2,
    sheds: 0.05,
  },
  willow: {
    count: 110,
    life: 4.2,
    drag: 1.9,
    fall: 0.35,
    size: 1.4,
    streak: 0.8,
    every: 5,
    sheds: 0.35,
  },
  palm: {
    count: 14,
    life: 2.6,
    drag: 0.9,
    fall: 0.8,
    size: 3.2,
    streak: 1.4,
    every: 3,
    sheds: 0.9,
  },
  ring: { count: 64, life: 1.8, drag: 1.2, fall: 0.5, size: 2, streak: 0.4, every: 1, sheds: 0 },
  crackle: {
    count: 180,
    life: 1.6,
    drag: 1.6,
    fall: 0.6,
    size: 0.9,
    streak: 0.15,
    every: 1,
    sheds: 0,
  },
};

const names = Object.keys(Pattern) as (keyof typeof Pattern)[];
/** `TRAITS` by `Pattern`. */
const traitsOf = tgpu.const(
  d.arrayOf(Traits, names.length),
  names.map((name) => TRAITS[name]),
);

/** A shell for `launch`. */
export interface Launch {
  readonly position: readonly [number, number, number];
  readonly velocity: readonly [number, number, number];
  /** Seconds until it bursts. */
  readonly fuse: number;
  readonly color: readonly [number, number, number];
  readonly pattern: Pattern;
  /** Metres a second the burst throws its stars. */
  readonly burstSpeed: number;
}

/** Shell pattern and burst speed in metres per second. */
const Shell = particles.schema('Shell', { pattern: d.u32, reach: d.f32 });
type Shell = d.InferGPU<typeof Shell>;

/** Star pattern and elapsed steps for spacing streak points. */
const Star = particles.schema('Star', { pattern: d.u32, steps: d.u32 });
type Star = d.InferGPU<typeof Star>;

const Flash = particles.schema('Flash', { pattern: d.u32 });
type Flash = d.InferGPU<typeof Flash>;

/** Flash brightness by age. A sharp spike fades, with flicker for glitter shells. */
const flashLevel = (flash: Flash) => {
  'use gpu';
  const glitter = std.select(
    d.f32(1),
    1 - 0.5 * std.step(0.5, std.fract(flash.age * GLITTER_HZ + flash.seed)),
    flash.pattern === Pattern.crackle,
  );
  return std.exp(-flash.age * 3.2) * glitter;
};

/** Star colour by age, progressing from white-hot to its colour and then dim red. */
const burning = (star: Star) => {
  'use gpu';
  const pattern = star.pattern;
  const t = star.age / star.life;
  const hot = std.mix(d.vec3f(1), star.color.xyz, std.saturate(star.age / HOT_SECONDS));
  const fading = std.mix(
    hot,
    EMBER_RED * std.max(star.color.xyz.x, 0.4),
    std.smoothstep(0.55, 1, t),
  );
  const level =
    std.saturate((1 - t) * 3) * std.select(d.f32(1), 1 - t * 0.5, pattern === Pattern.willow);
  const strobe = std.select(
    d.f32(1),
    std.step(0.45, std.fract(star.age * GLITTER_HZ + star.seed * 7)) * 1.6,
    pattern === Pattern.crackle && star.age > star.life * 0.35,
  );
  return fading * level * strobe;
};

/** Smoke shows only in the bursts' light, so moonlight leaves it dark. */
const moonless = (_point: shading.Point) => {
  'use gpu';
  return d.f32(0);
};

/** Radial glow with a hot core. */
const spark = (color: d.v3f, local: d.v2f) => {
  'use gpu';
  const r2 = std.dot(local, local);
  return d.vec4f(color * (std.exp(-r2 * 14) + std.exp(-r2 * 4) * 0.2), 0);
};

/** Flashes, each a point light for `lighting.create`'s `points.light`. */
export type Flashes = ReturnType<typeof createFlashes>;

export const FIREWORKS = { wind: 1.4, light: 1, smoke: 1 };

/**
 * Creates the flashes that bursts leave, each a scene point light. Create them before the
 * lighting, which reads their lights, then pass both to `createFireworks`, whose shells burst
 * into the flashes and whose smoke the lighting lights.
 */
export function createFlashes(root: TgpuRoot, camera: CameraBase, options: { step: number }) {
  const { step } = options;
  /** Multiplies the flashes' light. */
  const brightness = root.createUniform(d.f32, FIREWORKS.light);
  const flashes = particles.create(root, camera, {
    label: 'flashes',
    capacity: FLASHES,
    particle: Flash,
    step,
    settings: { gravity: [0, 0, 0], drag: 2 },
    shade: (flash: Flash, local: d.v2f) => {
      'use gpu';
      const r2 = std.dot(local, local);
      const glow = std.exp(-flash.age * 14) * 1.5;
      return d.vec4f(flash.color.xyz * glow * std.exp(-r2 * 8), 0);
    },
  });

  const lightOf = (slot: number) => {
    'use gpu';
    const flash = flashes.particle(slot);
    return lighting.Light({
      position: flash.position,
      range: std.select(d.f32(0), FLASH.range, flashes.alive(slot)),
      color: flash.color.xyz,
      intensity: flashLevel(flash) * brightness.$ * FLASH.power * flash.size * flash.size,
    });
  };

  return {
    /** The system, which `createFireworks` emits into, steps and draws. */
    particles: flashes,
    capacity: flashes.capacity,
    initAsync: () => flashes.initAsync(),
    destroy() {
      flashes.destroy();
    },
    brightness,
    /** Flash lights for `lighting.create`'s `points.light`. */
    light: lightOf,
  };
}

interface FireworksOptions {
  readonly camera: CameraBase;
  /** The length of a step in seconds. */
  readonly step: number;
  /** Cached Perlin gradients for smoke. */
  readonly noise: { inject(): Inject };
  /** How many of each the systems hold at once. */
  readonly capacity: Readonly<Record<'shells' | 'stars' | 'embers' | 'smoke', number>>;
}

/**
 * Spawns shells that burst into stars, `flashes` and smoke, which `light` lights, and embers.
 * Records the flashes' steps with its own.
 */
export function createFireworks(
  root: TgpuRoot,
  flashes: Flashes,
  light: Lighting,
  { camera, step, noise, capacity }: FireworksOptions,
) {
  /** Metres a second the wind blows smoke, across the lake along x. */
  const wind = root.createUniform(d.f32, FIREWORKS.wind);
  /** Multiplies the smoke's opacity. */
  const opacity = root.createUniform(d.f32, FIREWORKS.smoke);
  const flashing = flashes.particles;
  const still = { gravity: [0, 0, 0] as [number, number, number], drag: 0 };

  /** A flash lights smoke evenly across the burst's core. */
  const throughSmoke = (
    _point: shading.Point,
    _toLight: d.v3f,
    distance: number,
    range: number,
    slot: number,
  ) => {
    'use gpu';
    const core = flashing.particle(slot).size * SMOKE_CORE;
    return SMOKE_SHARE * shading.falloff(std.max(distance, core), range);
  };

  const smoke = particles.create(root, camera, {
    label: 'smoke',
    capacity: capacity.smoke,
    step,
    blend: 'over',
    settings: still,
    update: tgpu
      .fn((puff: Particle, dt: number) => {
        'use gpu';
        let next = Particle(puff);
        const drift = d.vec3f(wind.$, 0.5, 0);
        next.velocity = std.mix(puff.velocity, drift, dt / 2.5);
        next.size = puff.size + 0.9 * dt;
        next.color = d.vec4f(shading.incoming(puff.position) * SMOKE_COLOR, puff.color.w);
        return next;
      })
      .pipe(light.inject())
      .with(shading.attenuation, throughSmoke)
      .with(shading.sunShadow, moonless),
    shade: tgpu
      .fn((puff: Particle, local: d.v2f) => {
        'use gpu';
        const r2 = std.dot(local, local);
        const billow = perlin2d.sample(local * 2.2 + puff.seed * 31) * 0.5 + 0.5;
        const body =
          std.saturate(1 - r2) * std.saturate(1 - r2) * std.smoothstep(0.25, 0.75, billow) * 1.6;
        const life =
          std.saturate(puff.age / 1.5) * std.saturate((puff.life - puff.age) / (puff.life * 0.5));
        const alpha = std.saturate(body * life * puff.color.w * opacity.$);
        return d.vec4f(puff.color.xyz * alpha, alpha);
      })
      .pipe(noise.inject()),
  });

  const embers = particles.create(root, camera, {
    label: 'embers',
    capacity: capacity.embers,
    step,
    settings: { gravity: [0, -2.4, 0], drag: 1.6 },
    shade: (ember: Particle, local: d.v2f) => {
      'use gpu';
      const t = ember.age / ember.life;
      const twinkle = 0.55 + 0.45 * std.sin(ember.age * 40 + ember.seed * 60);
      return spark(ember.color.xyz * (BRIGHTNESS.ember * (1 - t) * (1 - t) * twinkle), local);
    },
  });

  /** Emits an ember behind a burning particle. */
  const shed = (position: d.v3f, velocity: d.v3f, color: d.v3f) => {
    'use gpu';
    embers.emit(
      Particle({
        position,
        age: 0,
        velocity: velocity * 0.15 + randf.inUnitSphere() * 1.5,
        life: 1.6 * (0.5 + randf.sample()),
        color: d.vec4f(color, 1),
        size: 0.45,
        seed: randf.sample(),
      }),
    );
  };

  const stars = particles.create(root, camera, {
    label: 'stars',
    capacity: capacity.stars,
    particle: Star,
    step,
    settings: still,
    update: (star: Star, dt: number) => {
      'use gpu';
      let next = Star(star);
      const traits = traitsOf.$[star.pattern];
      next.velocity =
        star.velocity * std.exp(-traits.drag * dt) - d.vec3f(0, GRAVITY * traits.fall * dt, 0);
      next.steps = star.steps + 1;
      if (star.position.y < 0) next.life = 0;
      return next;
    },
    shade: (star: Star, local: d.v2f) => {
      'use gpu';
      return spark(burning(star) * BRIGHTNESS.star, local);
    },
  });

  const streaks = trails.create(root, {
    capacity: capacity.stars,
    length: 9,
    width: tgpu.fn((point: trails.Sample) => {
      'use gpu';
      const star = stars.particle(point.trail);
      const shown = std.select(d.f32(0), star.size * 0.3, stars.alive(point.trail));
      return shown * (1 - point.along * 0.7);
    }),
    shade: tgpu.fn((point: trails.Sample, across: number) => {
      'use gpu';
      const star = stars.particle(point.trail);
      const streak = traitsOf.$[star.pattern].streak;
      const edge = 1 - std.smoothstep(0.3, 1, std.abs(across));
      const fade = (1 - point.along) * (1 - point.along);
      return d.vec4f(burning(star) * (BRIGHTNESS.streak * streak * edge * fade), 0);
    }),
    // Keep each streak on its star, which moves on between the steps that push its points
    head: stars.drawnAt,
  });

  const tails = trails.create(root, {
    label: 'tails',
    capacity: capacity.shells,
    length: 28,
    width: (point: trails.Sample) => {
      'use gpu';
      return 0.5 * (1 - point.along);
    },
    shade: (point: trails.Sample, across: number) => {
      'use gpu';
      const edge = 1 - std.smoothstep(0.2, 1, std.abs(across));
      return d.vec4f(SHELL_COLOR * (BRIGHTNESS.tail * edge * (1 - point.along)), 0);
    },
  });

  const shells = particles.create(root, camera, {
    label: 'shells',
    capacity: capacity.shells,
    particle: Shell,
    step,
    settings: { gravity: [0, -GRAVITY, 0], drag: 0 },
    // Start the tail at shell birth and end it with the shell
    born: tgpu.fn((shell: Shell, slot: number) => {
      'use gpu';
      tails.reset(slot);
      return Shell(shell);
    }),
    shade: (_shell: Shell, local: d.v2f) => {
      'use gpu';
      return spark(SHELL_COLOR * BRIGHTNESS.shell, local);
    },
    died: tgpu.fn((shell: Shell, slot: number) => {
      'use gpu';
      tails.reset(slot);
      const pattern = shell.pattern;
      const traits = traitsOf.$[pattern];
      const reach = shell.reach;
      const color = d.vec4f(shell.color.xyz, 1);
      randf.seed3(shell.position + shell.seed);

      flashing.emit(
        Flash({
          position: shell.position,
          age: 0,
          velocity: d.vec3f(),
          life: 1.6,
          color,
          size: reach * 1.4,
          seed: randf.sample(),
          pattern,
        }),
      );

      // Choose a random orientation for each burst pattern, including the ring plane
      const axis = randf.onUnitSphere();
      const side = std.normalize(std.cross(axis, d.vec3f(0.3, 1, 0.1)));
      const across = std.cross(axis, side);
      const golden = Math.PI * (3 - Math.sqrt(5));
      const count = d.f32(traits.count);
      for (let i = d.u32(0); i < traits.count; i++) {
        const k = d.f32(i) + 0.5;
        let direction = d.vec3f();
        if (pattern === Pattern.ring) {
          const angle = (k / count) * Math.PI * 2;
          direction = side * std.cos(angle) + across * std.sin(angle);
        } else {
          const height = 1 - (2 * k) / count;
          const round = std.sqrt(std.max(1 - height * height, 0));
          const angle = k * golden;
          direction = axis * height + (side * std.cos(angle) + across * std.sin(angle)) * round;
        }
        if (pattern === Pattern.palm) {
          direction = std.normalize(direction + d.vec3f(0, 0.6, 0));
        }
        const speed = reach * (1 + (randf.sample() - 0.5) * SPREAD.speed * 2);
        stars.emit(
          Star({
            position: shell.position,
            age: 0,
            velocity: shell.velocity * CARRIED + direction * speed,
            life: traits.life * (1 + (randf.sample() - 0.5) * SPREAD.life),
            color,
            size: traits.size,
            seed: randf.sample(),
            pattern,
            steps: 0,
          }),
        );
      }

      for (let i = 0; i < SMOKE_PUFFS; i++) {
        smoke.emit(
          Particle({
            position: shell.position + randf.inUnitSphere() * reach * 0.9,
            age: 0,
            velocity: randf.inUnitSphere() * reach * 0.25,
            life: 22 * (0.7 + randf.sample() * 0.6),
            color: d.vec4f(0, 0, 0, 0.16 * (0.5 + randf.sample())),
            size: reach * 0.75,
            seed: randf.sample(),
          }),
        );
      }
    }),
  });

  const followStars = tgpu.computeFn({
    workgroupSize: [WORKGROUP],
    in: { id: d.builtin.globalInvocationId },
  })(({ id }) => {
    'use gpu';
    if (id.x >= stars.span) return;
    // Empty a dead star's streak so its points stop drawing
    if (!stars.alive(id.x)) {
      streaks.reset(id.x);
      return;
    }
    const star = stars.particle(id.x);
    const traits = traitsOf.$[star.pattern];
    // Reset the streak on birth because the slot may retain an earlier star's points
    if (star.steps === 1) streaks.reset(id.x);
    if (star.steps % traits.every === 0 || star.steps === 1) streaks.push(id.x, star.position);

    randf.seed3(d.vec3f(star.position.xy, star.age + star.seed));
    if (randf.sample() < traits.sheds) {
      const gold = std.mix(EMBER_GOLD, star.color.xyz, 0.3);
      shed(star.position, star.velocity, gold);
    }
  });

  const followShells = tgpu.computeFn({
    workgroupSize: [WORKGROUP],
    in: { id: d.builtin.globalInvocationId },
  })(({ id }) => {
    'use gpu';
    if (id.x >= shells.span) return;
    if (!shells.alive(id.x)) return;
    const shell = shells.particle(id.x);
    tails.push(id.x, shell.position);
    randf.seed3(d.vec3f(shell.position.xz, shell.age + shell.seed));
    if (randf.sample() < SHELL_SHEDS) shed(shell.position, shell.velocity * -1, EMBER_GOLD);
  });

  const starsFollowed = root.createComputePipeline({ compute: followStars });
  const shellsFollowed = root.createComputePipeline({ compute: followShells });

  const owned = [shells, stars, embers, smoke];

  return {
    wind,
    opacity,
    /** Smoke, which covers what is drawn before it. */
    smoke: smoke.drawing,
    /** Embers, stars, shells and flashes, which add their glow. Draw them after `smoke`. */
    sprites: [embers.drawing, stars.drawing, shells.drawing, flashing.drawing],
    /** The stars' streaks and the shells' tails, which add their glow. Draw them after `smoke`. */
    streaks: [streaks.drawing, tails.drawing],
    /** Queues a shell to launch in the next `record`. */
    launch(shell: Launch) {
      shells.burst({
        count: 1,
        position: shell.position,
        velocity: shell.velocity,
        life: shell.fuse,
        color: [...shell.color, 1],
        size: 1.2,
        variance: 0,
        fields: { pattern: shell.pattern, reach: shell.burstSpeed },
      });
    },
    /**
     * Records frame spawning, fixed steps and live lists in one compute pass, drawn `ahead`
     * seconds past the last step.
     */
    record(pass: TgpuComputePass, steps: number, ahead: number) {
      shells.spawn(pass);
      for (let i = 0; i < steps; i++) {
        shells.step(pass);
        stars.step(pass);
        flashing.step(pass);
        smoke.step(pass);
        embers.step(pass);
        pass.setPipeline(starsFollowed);
        pass.dispatchWorkgroups(workgroups(stars.span));
        pass.setPipeline(shellsFollowed);
        pass.dispatchWorkgroups(workgroups(shells.span));
      }
      for (const system of owned) system.compact(pass, ahead);
      flashing.compact(pass, ahead);
    },
    initAsync: () =>
      pipelines.compile([shells, stars, embers, smoke, starsFollowed, shellsFollowed]),
    destroy() {
      for (const system of owned) system.destroy();
      streaks.destroy();
      tails.destroy();
    },
  };
}
