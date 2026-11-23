import { type Checked, type Inject } from '../inject.ts';
import { type Candidates } from '../grid.ts';
import * as table from '../table.ts';
import { type Table } from '../table.ts';
import { d, std, type TgpuRoot, type TgpuUniform } from 'typegpu';
import {
  beams,
  LightSettings,
  lightSettings,
  Light,
  pointLightFields,
  pointLights,
  pointsArriving,
  pointShadowing,
  shadowSampler,
  shadowShape,
  Spot,
  spotLightFields,
} from './shared.ts';
import {
  createShadow,
  type Cascades,
  type Shadow,
  type ShadowOptions,
  type SunShadow,
} from './shadow.ts';
import type { PointShadow, PointShadowDefinition } from './pointShadow.ts';
import {
  ambient,
  attenuation,
  beam,
  fog as fogHook,
  Material,
  reflect,
  sunShadow as sunShadowHook,
  type AmbientHook,
  type AttenuationHook,
  type FogHook,
  type ReflectHook,
  type Point,
  type SunShadowHook,
} from './shading.ts';

export { Light, Spot } from './shared.ts';
export { fitSlices } from './shadow.ts';
export type {
  CascadeFit,
  CascadeFitInput,
  CascadePlace,
  Cascades,
  FittedShadowOptions,
  FixedShadowOptions,
  Shadow,
  ShadowMapOptions,
  ShadowOptions,
  ShadowPrepareOptions,
  ShadowRecordOptions,
  SunShadow,
} from './shadow.ts';

/** Defines the schema for the sun, sky and fog settings in `light.settings`. */
export const Settings = LightSettings;

/**
 * The default settings provide a white sun, a pale blue sky, dark brown ground and no fog.
 * `sunDirection` is `[0.4, 1, 0.3]`, `sunColor` is white and `sunIntensity` is 1. A white matte
 * surface facing the sun has a brightness of about 1.
 *
 * `skyTop` is `[0.2, 0.26, 0.36]`, `skyBottom` is `[0.06, 0.05, 0.045]`, and both `skyIntensity`
 * and `skyReflection` are 1. `fogColor` is the pale grey-blue `[0.55, 0.62, 0.7]`. `fogDensity`,
 * `fogFalloff` and `fogStart` are 0.
 */
export const defaults: d.Infer<typeof Settings> = {
  sunDirection: d.vec3f(0.4, 1, 0.3),
  sunIntensity: 1,
  sunColor: d.vec3f(1, 1, 1),
  skyIntensity: 1,
  skyTop: d.vec3f(0.2, 0.26, 0.36),
  fogDensity: 0,
  skyBottom: d.vec3f(0.06, 0.05, 0.045),
  fogFalloff: 0,
  fogColor: d.vec3f(0.55, 0.62, 0.7),
  skyReflection: 1,
  fogStart: 0,
};

/**
 * These functions control how `shading.shade` lights a surface. Pipelines use them through
 * `lighting.inject()`. The functions are fixed when you create the lighting.
 */
export interface Shading {
  /** Computes how a surface reflects the sun and each point light. Default `shading.pbr`. */
  readonly reflect?: ReflectHook;
  /** Computes ambient light. Default `shading.hemisphere`. */
  readonly ambient?: AmbientHook;
  /**
   * Gives the light the air adds between the eye and a point, and the share of the point's light
   * it lets through. Default `shading.heightFog`.
   */
  readonly fog?: FogHook;
  /**
   * Returns the share of sunlight that reaches a point. A value of 0 means full shadow and 1
   * means full light. Values above 1 can model focused light, such as light through moving
   * water. Default `shading.sunShadowed`, which reads the shadow map.
   */
  readonly sunShadow?: SunShadowHook;
  /**
   * Returns the share of a point or spot light's intensity that reaches a point. Default
   * `shading.reaching`, which multiplies `falloff`, `beamOf` and `shading.lampShadowed`.
   */
  readonly attenuation?: AttenuationHook;
}

/** Any of the four tables of lights a lighting reads. */
export type LightTable = PointLights | MovingPointLights | SpotLights | MovingSpotLights;

/**
 * Point lights from a table you own. Its definition decides whether lights have cones and whether
 * kernels write their positions. Every lit point visits each slot up to the table's highest live
 * one, unless `candidates` names the slots that may reach it.
 */
export interface TablePoints {
  /**
   * Supplies a table of `lighting.PointLight`, `MovingPointLight`, `SpotLight` or
   * `MovingSpotLight`. Its fields start at zero, so give it `defaults` such as `lighting.WHITE`.
   * Flush and destroy it yourself. Default none, lights from `light` or `spot`.
   */
  readonly table: LightTable;
  /**
   * Limits the loop to lights that may reach each shaded point, as `grid.Candidates` name them.
   * Default every live slot.
   */
  readonly candidates?: Candidates;
  readonly light?: never;
  readonly spot?: never;
  readonly count?: never;
}

/** What point lights from a `'use gpu'` function share. Attach the function's fills with `pipe`. */
interface FunctionPoints {
  /**
   * Sets how many slots, from 0, every lit point visits. Required unless `candidates` names them.
   * A light with range 0 lights nothing. Default none, the slots `candidates` names.
   */
  readonly count?: number;
  /**
   * Limits the loop to lights that may reach each shaded point, such as the slots a neighbour
   * grid finds. Each light still contributes only within its range. Default `count` slots.
   */
  readonly candidates?: Candidates;
  readonly table?: never;
}

/** Point lights that a `'use gpu'` function returns by slot, lighting every direction. */
export interface LightPoints extends FunctionPoints {
  /** Returns the light at a slot. Default none, lights from `table` or `spot`. @gpu */
  readonly light: (slot: number) => Light;
  readonly spot?: never;
}

/** Spot lights that a `'use gpu'` function returns by slot, each with an aim and a cone. */
export interface SpotPoints extends FunctionPoints {
  /**
   * Returns the spot light at a slot, whose `cone` comes from `lighting.cone`. Default none,
   * lights from `table` or `light`.
   * @gpu
   */
  readonly spot: (slot: number) => Spot;
  readonly light?: never;
}

/** Point lights: a table you own, or a `'use gpu'` function of the slot. */
export type PointsOptions = TablePoints | LightPoints | SpotPoints;

export interface Options {
  /**
   * Sets initial sun, sky and fog values over `defaults`. Default `defaults`, which has no fog.
   */
  readonly settings?: Partial<d.InferInput<typeof Settings>>;
  /**
   * Sets the functions that `shading.shade` calls. Default `pbr`, `hemisphere`, `heightFog`,
   * `sunShadowed` and `reaching`.
   */
  readonly shading?: Shading;
  /** Enables point lights from a table or a function. Default none. */
  readonly points?: PointsOptions;
  /** Enables and configures the shadow from the sun. Default none. */
  readonly shadow?: ShadowOptions;
  /**
   * Gives up to `count` point or spot lights a shadow each, as `pointShadow.define` describes
   * them, which `light.pointShadow.cast` assigns. Needs `points.table` of lights the CPU places,
   * not a moving one. Default none.
   */
  readonly pointShadow?: PointShadowDefinition;
}

/** This schema holds the first candidate index and the number of indices in a range. */
export const Span = d.struct({ first: d.u32, count: d.u32 });

export const PointLight = /* @__PURE__ */ table.define('PointLight', { host: pointLightFields });

const { position, ...unmoved } = pointLightFields;

/**
 * This point-light table stores positions in a device field. Kernels write positions through
 * `MovingPointLight.device.$[slot].position`.
 */
export const MovingPointLight = /* @__PURE__ */ table.define('MovingPointLight', {
  host: unmoved,
  device: { position },
});

export const SpotLight = /* @__PURE__ */ table.define('SpotLight', { host: spotLightFields });

const { position: _moved, ...unmovedSpot } = spotLightFields;

/**
 * This spot-light table stores positions in a device field. Kernels write positions through
 * `MovingSpotLight.device.$[slot].position`.
 */
export const MovingSpotLight = /* @__PURE__ */ table.define('MovingSpotLight', {
  host: unmovedSpot,
  device: { position },
});

/** A table of point lights whose positions the CPU writes. */
export type PointLights = Table<typeof PointLight>;
/** A table of point lights whose positions kernels write. */
export type MovingPointLights = Table<typeof MovingPointLight>;
/** A table of spot lights whose positions the CPU writes. */
export type SpotLights = Table<typeof SpotLight>;
/** A table of spot lights whose positions kernels write. */
export type MovingSpotLights = Table<typeof MovingSpotLight>;

/**
 * Holds white, intensity 1 and range 10, for the `defaults` of a table of point lights, since
 * table fields start at zero: `table.create(root, lighting.PointLight, { capacity, defaults: {
 * ...lighting.WHITE } })`.
 */
export const WHITE: {
  readonly intensity: number;
  readonly color: d.v3f;
  readonly range: number;
} = Object.freeze({
  intensity: 1,
  color: d.vec3f(1),
  range: 10,
});

/** Holds `WHITE` with an aim straight down and a full cone, for a table of spot lights. */
export const WHITE_SPOT: typeof WHITE & {
  readonly aim: d.v3f;
  readonly cone: d.v2f;
} = Object.freeze({
  ...WHITE,
  aim: d.vec3f(0, -1, 0),
  cone: d.vec2f(0, 1),
});

/**
 * Returns a cone's scale and offset for a spot light's `cone`, from its inner and outer
 * half-angles in radians. Light falls from full at `inner` to none at `outer`, as glTF's spot
 * lights do. It runs on the CPU and in shaders.
 */
export const cone = (inner: number, outer: number) => {
  'use gpu';
  const edge = std.cos(outer);
  const scale = 1 / std.max(std.cos(inner) - edge, 1e-3);
  return d.vec2f(scale, -edge * scale);
};

export interface Lighting extends Disposable {
  /**
   * Holds the sun, sky and fog values from `defaults` and the `settings` option. Use `patch` to
   * upload only the fields you supply. The shadow's next `record` follows a `sunDirection` written
   * here.
   */
  readonly settings: TgpuUniform<typeof Settings>;
  /** Provides the sun shadow when enabled in the options. */
  readonly shadow?: Shadow | Cascades;
  /** Provides the point and spot lights' shadows when enabled in the options. */
  readonly pointShadow?: PointShadow;
  /**
   * Fills the shading hooks and resources for pipelines created from the configuration you pipe
   * it into.
   */
  inject(): Inject;
  /** Compiles the pipelines of its shadows. Resolves at once without one. */
  initAsync(): Promise<void>;
  /** Destroys the shadows' maps and the casters added to them. */
  destroy(): void;
}

/** This lighting includes `shadow` and `pointShadow` when the options enable them. */
export type LightingOf<O> = Lighting &
  (O extends { shadow: { eye: object } }
    ? { readonly shadow: Cascades }
    : O extends { shadow: object }
      ? { readonly shadow: Shadow }
      : {}) &
  (O extends { pointShadow: object } ? { readonly pointShadow: PointShadow } : {});

/** A lighting whose sun casts a shadow, from one map or from cascades. */
export type Shadowed = Lighting & { readonly shadow: SunShadow };

/** A lighting whose point or spot lights cast shadows. */
export type PointShadowed = Lighting & { readonly pointShadow: PointShadow };

type PointTable = Pick<
  typeof PointLight,
  'span' | 'alive' | 'position' | 'intensity' | 'color' | 'range'
>;
type SpotTable = PointTable & Pick<typeof SpotLight, 'aim' | 'cone'>;

interface Source {
  readonly reach: (slot: number) => d.v4f;
  /** Tests a slot whose range reaches the point. Without it, every such slot is live. */
  readonly live?: (slot: number) => boolean;
  readonly glow: (slot: number) => d.v4f;
}

/** Candidates whose index is the slot when they have no `slot`. */
type Slots = Omit<Candidates, 'slot'> & Partial<Pick<Candidates, 'slot'>>;

// Empty slots retain light data, so a passing range test must also check the live bit
const tableSource = (rows: PointTable): Source => {
  const lightReach = (slot: number) => {
    'use gpu';
    return d.vec4f(rows.position(slot), rows.range(slot));
  };
  const lightGlow = (slot: number) => {
    'use gpu';
    return d.vec4f(rows.color(slot), rows.intensity(slot));
  };
  return { reach: lightReach, live: rows.alive, glow: lightGlow };
};

const tableCone = (rows: SpotTable) => {
  const spotCone = (toLight: d.v3f, slot: number) => {
    'use gpu';
    return beam(toLight, rows.aim(slot), rows.cone(slot));
  };
  return spotCone;
};

// The compiler merges these calls when both results are needed
const functionSource = (lightAt: (slot: number) => Light | Spot): Source => {
  const lightReach = (slot: number) => {
    'use gpu';
    const light = lightAt(slot);
    return d.vec4f(light.position, light.range);
  };
  const lightGlow = (slot: number) => {
    'use gpu';
    const light = lightAt(slot);
    return d.vec4f(light.color, light.intensity);
  };
  return { reach: lightReach, glow: lightGlow };
};

const functionCone = (spotAt: (slot: number) => Spot) => {
  const spotCone = (toLight: d.v3f, slot: number) => {
    'use gpu';
    const spot = spotAt(slot);
    return beam(toLight, spot.aim, spot.cone);
  };
  return spotCone;
};

/**
 * The range test must surround `reflect` so out-of-range lights skip reflection even if the
 * compiler flattens the loop. Colour and intensity load only for a light whose share is above 0.
 */
const candidateLights = (candidates: Slots, source: Source, reflects: boolean) => {
  const { ranges, range: rangeOf, slot: slotOf } = candidates;
  const { reach: reachOf, live, glow: glowOf } = source;

  const pointLightSum = (material: Material, point: Point) => {
    'use gpu';
    let color = d.vec3f();
    for (let r = d.u32(0); r < ranges; r++) {
      const span = rangeOf(point.position, r);
      const end = span.first + span.count;
      for (let i = d.u32(span.first); i < end; i++) {
        const slot = slotOf === undefined ? i : slotOf(i);
        const reach = reachOf(slot);
        const toLight = reach.xyz - point.position;
        const squared = std.dot(toLight, toLight);
        // At zero distance, the direction to the light is undefined
        if (squared < reach.w * reach.w && squared > 0 && (live === undefined || live(slot))) {
          const distance = std.sqrt(squared);
          const direction = toLight / distance;
          const share = attenuation.$(point, direction, distance, reach.w, slot);
          if (share > 0) {
            const glow = glowOf(slot);
            const radiance = glow.xyz * (glow.w * share);
            if (reflects) color += reflect.$(material, point, direction, radiance);
            else color += radiance;
          }
        }
      }
    }
    return color;
  };
  return pointLightSum;
};

/** The point-light loops: reflected by a material, and arriving from every direction. */
const loopsOf = (candidates: Slots, source: Source) => {
  const arriving = candidateLights(candidates, source, false);
  return {
    reflected: candidateLights(candidates, source, true),
    arriving: (point: Point) => {
      'use gpu';
      return arriving(Material(), point);
    },
  };
};

const everySlot = (span: () => number): Slots => {
  const allSlots = (_position: d.v3f, _r: number) => {
    'use gpu';
    return Span({ first: 0, count: span() });
  };
  return { ranges: 1, range: allSlots };
};

const tables = /* @__PURE__ */ [PointLight, MovingPointLight, SpotLight, MovingSpotLight].map(
  (rows) => {
    const source = tableSource(rows);
    const spread = 'aim' in rows ? tableCone(rows) : undefined;
    return { source, cone: spread, loops: loopsOf(everySlot(rows.span), source) };
  },
);

/**
 * Creates lighting with a sun, sky and fog. It can also read point lights from a table or a
 * function and cast shadows. Pipe its injection into each pipeline that uses it. Throws if
 * `points` names no source or more than one, a light function has neither `count` nor
 * `candidates`, a table comes with `count`, or `candidates.ranges` is anything other than a
 * positive integer.
 *
 * @injects hook the shading hooks its options name: `reflect`, `ambient`, `fog`, `sunShadow` and
 * `attenuation`
 * @injects definition the sun, sky and fog, the shadows and the point lights, which `shading`
 * reads
 * @binds 1 uniform: `shading.shade`, the sun, sky and fog
 * @binds 2 storage, 1 uniform: `shading.shade` with a table of point lights
 * @binds 3 storage, 1 uniform: `shading.shade` with a table of lights whose positions are on the
 * device, the table's two parts
 * @binds 1 uniform: `shading.shade` with `points.light` or `points.spot`, plus what the function
 * binds
 * @binds 0 storage: `shading.shade` with `points.candidates`, plus what their functions bind, 2
 * storage for a neighbour grid's `range` and `slot`
 * @binds 1 uniform: `shading.shade` with a shadow, the sun's light space and cascades, beside the
 * depth array and its comparison sampler
 * @binds 1 uniform: `shading.shade` with `pointShadow`, each shadow's light, layers and views,
 * beside their depth array and the comparison sampler
 * @owns 1 to 3 buffers, a sampler, targets, a cameras.layered, passes: the sun, sky and fog; with a
 * shadow, its light space, the views of its cascades, a comparison sampler, the depth array and
 * the passes that draw it; with `pointShadow`, the same for the lights' faces
 * @records `shadow.record` and `pointShadow.record` with what was added to them, before the passes
 * that shade
 * @param options.shadow the sun's shadow, with a depth pass per cascade
 * @param options.pointShadow shadows from `pointShadow.define` that `cast` gives to lights of
 * `points.table`, with a depth pass per face
 * @param options.points point lights from a table or a `'use gpu'` function. Every lit point visits
 * each slot up to the table's highest live one, or `count` with a function, unless `candidates`
 * names the slots that may reach it
 * @returns lighting you destroy
 */
export function create<const O = {}>(
  root: TgpuRoot,
  options?: Checked<O, Options> & Options,
): LightingOf<O>;
export function create(root: TgpuRoot, options: Options = {}) {
  const hooks = options.shading ?? {};
  const points = options.points;
  const given = points?.table;
  const lightAt = points?.light;
  const spotAt = points?.spot;
  const candidates = points?.candidates;
  const count = points?.count;
  if (points) {
    const sources = [given, lightAt, spotAt].filter((source) => source !== undefined).length;
    if (sources !== 1) {
      throw new Error(
        'lighting.create reads point lights from one of `points.table`, `points.light` and `points.spot`.',
      );
    }
    if (given && count !== undefined) {
      throw new Error(
        "lighting.create visits a table's slots up to its highest live one, so `points.count` goes with a light function.",
      );
    }
    if (!given && count === undefined && !candidates) {
      throw new Error(
        'lighting.create requires `points.count` or `points.candidates` with a light function.',
      );
    }
    if (count !== undefined && !(Number.isInteger(count) && count >= 0)) {
      throw new RangeError(
        `lighting.create requires \`points.count\` to be a whole number. Received ${count}.`,
      );
    }
  }
  if (candidates && !(Number.isInteger(candidates.ranges) && candidates.ranges >= 1)) {
    throw new RangeError(
      `lighting.create requires \`points.candidates.ranges\` to be an integer of at least 1. Received ${candidates.ranges}.`,
    );
  }

  const moving = given ? given.buffers.device !== undefined : false;
  const cones = given ? 'aim' in given.fields : spotAt !== undefined;
  if (options.pointShadow && !given) {
    throw new Error(
      'To use `pointShadow`, give `lighting.create` a `points.table`. Point shadows require lights placed by the CPU.',
    );
  }
  if (options.pointShadow && moving) {
    throw new Error(
      'Point shadows require lights placed by the CPU. The lights in `points.table` move on the device. For `pointShadow`, supply a table of `lighting.PointLight` or `lighting.SpotLight`.',
    );
  }
  // Allocate once every option checks out
  const initial = { ...defaults, ...options.settings };
  const settings = root.createUniform(LightSettings, initial);
  // The CPU copy that `patch` and `write` update; the shadow follows the sun direction in it
  const sunward = new Float32Array(settings.buffer.arrayBuffer, 0, 3);
  const shadow = options.shadow && createShadow(root, options.shadow, sunward);
  const sampler =
    (shadow || options.pointShadow) &&
    root
      .createComparisonSampler({ compare: 'less', magFilter: 'linear', minFilter: 'linear' })
      .$name('shadowSampler');
  // Create the sampler under the lighting before any shading pipeline uses it
  if (sampler) root.unwrap(sampler);
  const sun = initial.sunDirection;
  if (shadow && sun) shadow.shadow.aim([sun[0], sun[1], sun[2]]);

  const pointShadow =
    options.pointShadow && options.pointShadow.create(root, given as PointLights | SpotLights);

  const span = () => {
    'use gpu';
    return d.u32(count ?? 0);
  };
  const lightFrom = lightAt ?? spotAt;
  const tabled = given ? tables[(cones ? 2 : 0) + (moving ? 1 : 0)] : undefined;
  const source = lightFrom ? functionSource(lightFrom) : tabled?.source;
  const loops =
    source && (candidates || lightFrom)
      ? loopsOf(candidates ?? everySlot(span), source)
      : tabled?.loops;
  const spot = !cones ? undefined : spotAt ? functionCone(spotAt) : tabled?.cone;

  // Preserve hooks set on the pipeline unless the options override them, and drop the loops and
  // shadows of a lighting piped before this one
  const inject: Inject = (cfg) => {
    let lit = cfg
      .with(lightSettings, settings)
      .with(pointLights, loops?.reflected ?? null)
      .with(pointsArriving, loops?.arriving ?? null)
      .with(beams, spot ?? null)
      .with(shadowShape, null)
      .with(pointShadowing, null);
    if (hooks.reflect) lit = lit.with(reflect, hooks.reflect);
    if (hooks.ambient) lit = lit.with(ambient, hooks.ambient);
    if (hooks.fog) lit = lit.with(fogHook, hooks.fog);
    if (hooks.sunShadow) lit = lit.with(sunShadowHook, hooks.sunShadow);
    if (hooks.attenuation) lit = lit.with(attenuation, hooks.attenuation);

    if (shadow) lit = shadow.inject(lit);
    if (pointShadow) lit = pointShadow.inject(lit);
    if (sampler) lit = lit.with(shadowSampler, () => sampler.$);
    return given ? given.inject()(lit) : lit;
  };

  const destroy = () => {
    shadow?.destroy();
    pointShadow?.destroy();
  };

  return {
    settings,
    ...(shadow && { shadow: shadow.shadow }),
    ...(pointShadow && { pointShadow: pointShadow.pointShadow }),

    inject: () => inject,

    async initAsync() {
      await Promise.all([shadow?.shadow.initAsync(), pointShadow?.pointShadow.initAsync()]);
    },

    destroy,
    [Symbol.dispose]: destroy,
  };
}
