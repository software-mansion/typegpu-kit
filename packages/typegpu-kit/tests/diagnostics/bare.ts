import { d, type TgpuRoot } from 'typegpu';
import { cameras, instances, meshPool, particles, passes, targets } from '../../src/index.ts';

declare const root: TgpuRoot;
declare const pool: meshPool.MeshPool;

// A parameter typed by a bare name takes every instance
declare function fit(screen: targets.Targets): void;
declare function flush(set: instances.Instances): void;
declare function warm(description: passes.Description): void;
declare function step(system: particles.Particles): void;

const screen = targets.create(
  root,
  { width: 1, height: 1 },
  { hdr: { format: 'rgba16float', sampled: true }, depth: { format: 'depth24plus' } },
);
fit(screen);
flush(instances.create(root, pool, { capacity: 8 }));
warm(passes.create(root, screen, { hdr: passes.color, depth: passes.depth }));

// And one of a kind or a schema of your own
declare const view: cameras.OrbitCamera;
const Tinted = instances.define('Tinted', { host: { tint: d.vec3f } });
flush(instances.create(root, pool, { kind: Tinted, capacity: 8 }));
const sparks = particles.create(root, view, { capacity: 64, step: 1 / 60 });
const embers = particles.create(root, view, {
  capacity: 64,
  step: 1 / 60,
  particle: particles.schema('Ember', { heat: d.f32 }),
});
step(sparks);
step(embers);
