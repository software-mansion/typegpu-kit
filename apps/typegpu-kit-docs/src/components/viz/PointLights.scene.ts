import {
  instances,
  lighting,
  materials,
  meshPool,
  table,
  type Frame,
  type RowId,
  type Surface,
} from '@typegpu/kit';
import { meshes } from '@typegpu/kit/geometry';
import { d, std, tgpu, type TgpuRoot } from 'typegpu';
import { linear, litStage, SKY } from './stage.ts';

export const MAX = 16;
export const START = 6;

/** The range every light starts with, in world units. */
export const RANGE = 4;

/** How high the lights circle above the floor. */
const HEIGHT = 0.6;

const WORKGROUP = 64;

/** The lights' colours, in turn. */
const HUES = [
  [0.5, 0.66, 0.98],
  [0.97, 0.71, 0.31],
  [0.42, 0.82, 0.65],
  [0.8, 0.65, 0.95],
] as const;

const fraction = (x: number) => x - Math.floor(x);

/**
 * A ball that carries a light: the light's slot in the lights' table, and its circle: radius, radians
 * a second and the angle it starts at.
 */
const Lamp = instances.define('Lamp', {
  host: { light: d.u32, orbit: d.vec3f },
  transform: 'device',
});

export async function createScene(root: TgpuRoot, surface: Surface) {
  const pool = meshPool.create(root, { vertices: 2048, indices: 8192, capacity: 2 });
  const box = pool.add(meshes.box());
  const ball = pool.add(meshes.sphere());

  const things = instances.create(root, pool, { capacity: 4 });
  things.add(box, {
    position: [0, -0.1, 0],
    scale: [14, 0.2, 14],
    ...materials.glossy(linear(0.3, 0.31, 0.33), 0.45),
  });
  const chalk = materials.matte(linear(0.8, 0.8, 0.8));
  things.add(ball, { position: [-1.6, 0.5, 0.4], ...chalk });
  things.add(ball, { position: [1.4, 0.35, -1.2], scale: 0.7, ...chalk });
  things.add(ball, { position: [0.9, 0.75, 2], scale: 1.5, ...chalk });

  const lamps = instances.create(root, pool, { kind: Lamp, capacity: MAX });
  const { MovingPointLight } = lighting;
  const lights = table.create(root, MovingPointLight, { capacity: MAX });
  const light = lighting.create(root, {
    settings: { ...SKY, skyIntensity: 0.2, sunDirection: [-0.3, 1, 0.5], sunIntensity: 0.15 },
    points: { table: lights },
  });

  const time = root.createUniform(d.f32);
  const circle = tgpu.computeFn({
    workgroupSize: [WORKGROUP],
    in: { gid: d.builtin.globalInvocationId },
  })(({ gid }) => {
    'use gpu';
    const i = gid.x;
    if (!Lamp.alive(i)) return;

    const orbit = Lamp.orbit(i);
    const angle = time.$ * orbit.y + orbit.z;
    const position = d.vec3f(std.cos(angle) * orbit.x, HEIGHT, std.sin(angle) * orbit.x);
    Lamp.device.$[i].position = d.vec3f(position);
    MovingPointLight.device.$[Lamp.light(i)].position = d.vec3f(position);
  });
  const circling = root
    .pipe(lamps.inject())
    .pipe(lights.inject())
    .createComputePipeline({ compute: circle });

  const stage = litStage(root, surface, {
    camera: { target: [0, 0, 0.3], distance: 11, pitch: 0.8, yaw: 0.5 },
  });
  const { view, screen, final } = stage;
  const scene = stage.lit(light);
  const drawn = [scene.add(things.renderer()), scene.add(lamps.renderer())];

  const carried: { light: RowId<typeof MovingPointLight>; lamp: instances.Id<typeof Lamp> }[] = [];
  let range = RANGE;

  /** Adds or removes lights, each with the ball that carries it, until there are `wanted`. */
  const count = (wanted: number) => {
    while (carried.length > wanted) {
      const last = carried.pop()!;
      lights.remove(last.light);
      lamps.remove(last.lamp);
    }

    while (carried.length < wanted) {
      const i = carried.length;
      const [r, g, b] = HUES[i % HUES.length];
      const id = lights.add({ color: [r, g, b], intensity: 0.8, range });
      const radius = 1 + 4 * fraction(i * 0.618 + 0.3);
      const speed = (0.3 + 0.4 * fraction(i * 0.382)) * (i % 2 === 0 ? 1 : -1);
      const lamp = lamps.add(ball, {
        light: lights.slotOf(id),
        orbit: [radius, speed, i * 2.4],
        scale: 0.14,
        ...materials.glow(d.vec3f(r, g, b), 3),
      });
      carried.push({ light: id, lamp });
    }
  };

  count(START);
  await Promise.all([scene.initAsync(), circling.initAsync(), final.initAsync()]);

  return {
    count,

    /** Sets where every light's falloff reaches zero. */
    range(next: number) {
      range = next;
      for (const each of carried) lights.fields.range.write(each.light, range);
    },

    frame(frame: Frame) {
      view.update(frame.dt);
      screen.fit();
      time.write(frame.time);
      things.flush();
      lamps.flush();
      lights.flush();

      if (lamps.span > 0) {
        const compute = frame.encoder.beginComputePass();
        compute.setPipeline(circling);
        compute.dispatchWorkgroups(Math.ceil(lamps.span / WORKGROUP));
        compute.end();
      }

      scene.record(frame, drawn);
      stage.present(frame);
    },

    destroy() {
      things.destroy();
      lamps.destroy();
      pool.destroy();
      light.destroy();
      lights.destroy();
      time.buffer.destroy();
      stage.destroy();
    },
  };
}
