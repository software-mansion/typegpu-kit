import { d, type TgpuRoot } from 'typegpu';
import { cameras, instances, lighting, particles } from '../../src/index.ts';

declare const root: TgpuRoot;
declare const view: cameras.OrbitCamera;

// An inline hook needs no annotation, and a field its schema lacks is an error at the access
const Ember = particles.schema('Ember', { heat: d.f32 });
particles.create(root, view, {
  capacity: 64,
  step: 1 / 60,
  particle: Ember,
  update: (ember) => {
    'use gpu';
    ember.size *= ember.heat + ember.glow;
    return ember;
  },
});

instances.define('Fish', {
  host: { belly: d.vec3f },
  visible: (fish) => {
    'use gpu';
    return fish.belly.x > fish.fins;
  },
});

lighting.create(root, {
  shading: {
    ambient: (material, point) => {
      'use gpu';
      return material.color * point.shade;
    },
  },
});
