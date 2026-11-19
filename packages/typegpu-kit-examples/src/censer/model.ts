import { cameras, materials, passes, shading } from '@typegpu/kit';
import { perlin3d } from '@typegpu/noise';
import { meshes } from '@typegpu/kit/geometry';
import { d, std, tgpu, type TgpuRoot } from 'typegpu';
import type { Censer } from './censer.ts';
import { CROWN, LENGTH, PIVOT, PROFILE, SIZE } from './chain.ts';

const Corner = d.struct({
  position: d.vec4f,
  world: d.vec3f,
  normal: d.vec3f,
  local: d.vec3f,
  iron: d.u32,
});
const Seen = d.struct({ color: d.vec4f });

/** A chain link's length, width and wire radius, in metres, and the length each adds to a chain. */
const LINK = [0.042, 0.025, 0.0045] as const;
const PITCH = LINK[0] - LINK[2] * 2;
/** How far the chain runs on up from the pivot, past the top of the view. */
const ABOVE = 1.2;
/** Points around the body, and around each half of a link and across its wire. */
const SIDES = 72;
const BEND = 8;
const WIRE = 6;

/** The profile's points with their normals, radius and height in `xy` and the normal in `zw`. */
const OUTLINE = tgpu.const(
  d.arrayOf(d.vec4f, PROFILE.length),
  PROFILE.map(([r, y], i) => {
    const before = PROFILE[Math.max(0, i - 1)];
    const after = PROFILE[Math.min(PROFILE.length - 1, i + 1)];
    const dr = after[0] - before[0];
    const dy = after[1] - before[1];
    const length = Math.hypot(dr, dy) || 1;
    return d.vec4f(r, y, -dy / length, dr / length);
  }),
);
const outlineAt = (v: number) => {
  'use gpu';
  return d.vec4f(OUTLINE.$[d.u32(std.round(v * (PROFILE.length - 1)))]);
};
const aroundAt = (u: number) => {
  'use gpu';
  const a = u * 2 * Math.PI;
  return d.vec3f(std.cos(a), 1, std.sin(a));
};

/** The censer's body, the profile turned about the chain, its middle `LENGTH` below the pivot. */
const body = meshes.parametric(
  {
    at: (u, v) => {
      'use gpu';
      return aroundAt(u) * outlineAt(v).xyx * SIZE - d.vec3f(0, LENGTH, 0);
    },
    normalAt: (u, v) => {
      'use gpu';
      return aroundAt(u) * outlineAt(v).zwz;
    },
  },
  { cols: SIDES, rows: PROFILE.length - 1 },
);

/**
 * The way out from a link's centre line in its plane: each end bends through half a turn, and the
 * straight sides join them.
 */
const outAt = (u: number) => {
  'use gpu';
  const column = std.round(u * (BEND * 2 + 2));
  const a = std.min(std.select(column - 1, column, column <= BEND), BEND * 2) * (Math.PI / BEND);
  return d.vec3f(std.cos(a), std.sin(a), 0);
};
const centreAt = (u: number) => {
  'use gpu';
  const column = std.round(u * (BEND * 2 + 2));
  const half = (LINK[0] - LINK[1]) / 2;
  const end = std.select(d.f32(-half), half, column <= BEND || column > BEND * 2 + 1);
  return d.vec3f(0, end, 0) + outAt(u) * (LINK[1] / 2 - LINK[2]);
};
const acrossAt = (u: number, v: number) => {
  'use gpu';
  const b = v * 2 * Math.PI;
  return outAt(u) * std.cos(b) - d.vec3f(0, 0, std.sin(b));
};

/** One chain link about its own middle: a stadium of wire, long along y and wide along x. */
const link = meshes.parametric(
  {
    at: (u, v) => {
      'use gpu';
      return centreAt(u) + acrossAt(u, v) * LINK[2];
    },
    normalAt: acrossAt,
  },
  { cols: BEND * 2 + 2, rows: WIRE },
);

/**
 * The links along one piece of chain: the first in `links`, how many, the particles they run
 * through, -1 standing for the point above the pivot, and the foot they end at, if any.
 */
interface Run {
  first: number;
  count: number;
  particles: readonly number[];
  foot: number;
}

/**
 * The censer's bronze body and its chains, swinging with `censer`. Each chain link is an instance,
 * laid along a smooth curve through the chain's particles and turned a quarter from the last.
 */
export async function createModel(root: TgpuRoot, censer: Censer) {
  const mesh = await meshes.bakeAsync(root, meshes.concat(body, link));
  const { chain } = censer;

  // The curve's control points, gathered for each run in turn
  const points = new Float64Array(3 * 24);
  const footAt: [number, number, number] = [0, 0, 0];
  const put = (n: number, p: ArrayLike<number>) => {
    points[n * 3] = p[0];
    points[n * 3 + 1] = p[1];
    points[n * 3 + 2] = p[2];
  };
  const runs: Run[] = [];
  let total = 0;
  const run = (length: number, particles: readonly number[], foot = -1) => {
    const count = Math.max(1, Math.round(length / PITCH));
    runs.push({ first: total, count, particles, foot });
    total += count;
  };
  run(ABOVE, [-1, 0]);
  run(LENGTH - CROWN, chain.chain);
  chain.legs.forEach((leg, f) => {
    const [a, b] = [chain.positions[leg[0]], chain.positions[leg[1]]];
    run(Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]) * leg.length, leg, f);
  });
  const above = [PIVOT[0], PIVOT[1] + ABOVE, PIVOT[2]];
  /** Puts a run's control points, and returns how many. */
  const gather = ({ particles, foot }: Run) => {
    for (let n = 0; n < particles.length; n++) {
      put(n, particles[n] < 0 ? above : chain.positions[particles[n]]);
    }
    if (foot < 0) return particles.length;
    put(particles.length, chain.footOf(foot, footAt));
    return particles.length + 1;
  };
  // Per link: its middle, the way it runs and the way across it, each a vec4
  const links = root.createReadonly(d.arrayOf(d.vec4f, total * 3)).$name('censer links');
  const laid = new Float32Array(total * 12);

  const at = [0, 0, 0];
  const slope = [0, 0, 0];
  /**
   * Writes into `at` and `slope` the point and slope of a Catmull–Rom curve through the first `n`
   * points, at `t` from 0 to `n - 1`.
   */
  const curve = (n: number, t: number) => {
    const i = Math.min(n - 2, Math.floor(t));
    const f = t - i;
    for (let k = 0; k < 3; k++) {
      const p1 = points[i * 3 + k];
      const p2 = points[(i + 1) * 3 + k];
      const p0 = i > 0 ? points[(i - 1) * 3 + k] : 2 * p1 - p2;
      const p3 = i + 2 < n ? points[(i + 2) * 3 + k] : 2 * p2 - p1;
      const a = -p0 + p2;
      const b = 2 * p0 - 5 * p1 + 4 * p2 - p3;
      const c = -p0 + 3 * p1 - 3 * p2 + p3;
      at[k] = 0.5 * (2 * p1 + a * f + b * f * f + c * f * f * f);
      slope[k] = 0.5 * (a + 2 * b * f + 3 * c * f * f);
    }
  };
  /** Lays every link along its run, carrying the way across each from the last without twist. */
  const lay = () => {
    for (let r = 0; r < runs.length; r++) {
      const { first, count } = runs[r];
      const n = gather(runs[r]);
      let ux = 1;
      let uy = 0;
      let uz = 0;
      for (let k = 0; k < count; k++) {
        curve(n, ((k + 0.5) / count) * (n - 1));
        const length = Math.hypot(slope[0], slope[1], slope[2]) || 1;
        const wx = slope[0] / length;
        const wy = slope[1] / length;
        const wz = slope[2] / length;
        // The last way across, less its part along this link, or any way across if none is left
        const along = ux * wx + uy * wy + uz * wz;
        ux -= wx * along;
        uy -= wy * along;
        uz -= wz * along;
        let across = Math.hypot(ux, uy, uz);
        if (across < 1e-4) {
          const useX = Math.abs(wx) < 0.9;
          ux = useX ? 1 - wx * wx : -wy * wx;
          uy = useX ? -wx * wy : 1 - wy * wy;
          uz = useX ? -wx * wz : -wy * wz;
          across = Math.hypot(ux, uy, uz);
        }
        ux /= across;
        uy /= across;
        uz /= across;
        // Every other link turns a quarter about the chain
        const odd = (first + k) % 2 === 1;
        const o = (first + k) * 12;
        laid[o] = at[0];
        laid[o + 1] = at[1];
        laid[o + 2] = at[2];
        laid[o + 4] = wx;
        laid[o + 5] = wy;
        laid[o + 6] = wz;
        laid[o + 8] = odd ? wy * uz - wz * uy : ux;
        laid[o + 9] = odd ? wz * ux - wx * uz : uy;
        laid[o + 10] = odd ? wx * uy - wy * ux : uz;
      }
    }
    links.buffer.write(laid.buffer);
  };

  const drawing: passes.Drawing<typeof Corner, d.Infer<typeof Seen>> = {
    label: 'censer',
    Vertex: Corner,
    at: (vertexIndex: number, instanceIndex: number) => {
      'use gpu';
      const vertex = mesh.vertexAt(mesh.indexAt(vertexIndex));
      const iron = vertexIndex >= body.indexCount;
      let world = censer.place(vertex.position);
      let normal = censer.turn(vertex.normal);
      if (iron) {
        const middle = links.$[instanceIndex * 3].xyz;
        const along = links.$[instanceIndex * 3 + 1].xyz;
        const across = links.$[instanceIndex * 3 + 2].xyz;
        const turned = d.mat3x3f(across, along, std.cross(across, along));
        world = middle + turned * vertex.position;
        normal = turned * vertex.normal;
      }
      return Corner({
        position: cameras.toClip(world),
        world,
        normal,
        local: vertex.position,
        iron: std.select(d.u32(0), d.u32(1), iron),
      });
    },
    pixelOf: (corner) => {
      'use gpu';
      const normal = std.normalize(corner.normal);
      const shape = corner.local / SIZE;
      const grain = perlin3d.sample(shape * 9);
      const stain = perlin3d.sample(shape * 1.3);
      const patina = std.smoothstep(0.02, 0.43, stain + grain * 0.18);
      let color = std.mix(d.vec3f(0.68, 0.36, 0.12), d.vec3f(0.12, 0.2, 0.15), patina * 0.5);
      let roughness = 0.31 + patina * 0.18;
      let metallic = d.f32(0.9);
      let emission = d.vec3f();
      // A polished rim, and above it the pierced lid
      const height = shape.y + LENGTH / SIZE;
      if (height > 0.1 && height < 0.22) {
        color = d.vec3f(0.85, 0.57, 0.22);
        roughness = 0.2;
      }
      if (height > 0.26 && height < 0.74) {
        // Lancet vents in the lid, lit from inside by the embers
        const around = std.fract(std.atan2(shape.z, shape.x) * (14 / (2 * Math.PI)));
        const taper = std.saturate(1 - std.abs(height - 0.48) / 0.2);
        const vent = 1 - std.smoothstep(0.12, 0.16, std.abs(around - 0.5) / std.max(taper, 0.01));
        color = std.mix(color, d.vec3f(0.02, 0.012, 0.008), vent);
        emission = d.vec3f(1, 0.32, 0.07) * (vent * (1.4 + stain));
      }
      if (corner.iron === 1) {
        color = d.vec3f(0.34, 0.32, 0.3);
        metallic = 0.85;
        roughness = 0.3;
      }
      const material = materials.glossy(color, roughness);
      material.metallic = metallic;
      material.emission = d.vec3f(emission);
      const lit = shading.shade(
        materials.antialiased(material, normal),
        shading.pointAt(corner.world, normal),
      );
      return Seen({ color: d.vec4f(lit, 1) });
    },
    color: passes.shaded,
    draw: (pass) => {
      pass.draw(body.indexCount);
      pass.draw(link.indexCount, total, body.indexCount);
    },
  };

  return {
    drawing,
    /** Lays the chain's links where its particles are now. */
    update: lay,
    destroy() {
      mesh.destroy();
      links.buffer.destroy();
    },
  };
}
