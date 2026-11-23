import { cameras, passes, targets } from '@typegpu/kit';
import { recordingRoot } from '@typegpu/kit/testing';
import { d, std, tgpu } from 'typegpu';
import { describe, expect, it } from 'vitest';
import { environment, lighting, materials, shading, images } from '../../src/index.ts';
import { recorded } from '../recorded.ts';

const glow = (direction: d.v3f) => {
  'use gpu';
  return std.abs(direction);
};

describe('environment', () => {
  it('bakes in one compute pass of a dispatch per level of both cubes and one for the harmonics, making nothing again', async () => {
    const recording = recordingRoot();
    const { root } = recording;
    const sky = environment.create(root);
    const added = sky.add(glow);
    await sky.initAsync();
    // Five shared kernels and the function's
    expect(recording.log.pipelines).toBe(6);
    // A panorama costs a bind group alone
    const panorama = { width: 8, height: 4, pixels: new Uint16Array(128) };
    for (let i = 0; i < 2; i++) sky.add(images.create(root, panorama, { mips: false }));
    await sky.initAsync();
    expect(recording.log.pipelines).toBe(6);
    const bake = () => {
      const encoder = root.createCommandEncoder();
      sky.bake(encoder, added);
      encoder.submit();
    };
    bake();
    const counts = recording.log.counts();
    // The source, 8 halvings of a 256 cube, 5 rougher reflection levels and the harmonics
    expect(recorded(recording.log, bake)).toEqual([
      'begin-compute',
      ...Array.from({ length: 15 }, () => 'dispatch'),
      'end-compute',
      'submit',
    ]);
    expect(recording.log.counts()).toEqual(counts);
  });

  it('reads a panorama or a cube and rejects other layer counts and foreign sources', () => {
    const { root } = recordingRoot();
    const sky = environment.create(root, { size: 32 });
    const face = { width: 4, height: 4, pixels: new Uint8Array(64) };

    sky.add(images.create(root, { width: 8, height: 4, pixels: new Uint8Array(128) }));
    sky.add(
      images.cube(
        root,
        Array.from({ length: 6 }, () => face),
      ),
    );
    expect(() => sky.add(images.array(root, [face, face, face]))).toThrow('3 layers');

    const other = environment.create(root, { size: 16 }).add(glow);
    const encoder = root.createCommandEncoder();
    expect(() => sky.bake(encoder, other)).toThrow('this environment');
  });

  it('clips a panorama and returns its strongest source as a sun', () => {
    const width = 64;
    const pixels = new Uint16Array(width * 32 * 4).fill(0x3c00);
    // 2 × 2 texels of 62 facing +x, and dimmer ones of 46 facing −x
    for (const [x, value] of [
      [31, 0x53c0],
      [0, 0x51c0],
    ] as const) {
      for (const at of [15 * width + x, 15 * width + x + 1, 16 * width + x, 16 * width + x + 1]) {
        pixels.fill(value, at * 4, at * 4 + 3);
      }
    }
    const { image, direction, color, intensity } = environment.clip({ width, height: 32, pixels });
    expect(Math.max(...image.pixels)).toBe(0x4f80);
    expect(direction[0]).toBeCloseTo(1, 2);
    expect(color).toEqual([1, 1, 1]);
    // 4 texels of 32 over the ceiling, each (2π / 64)(π / 32) across, over π
    expect(intensity).toBeCloseTo(0.39, 2);
    expect(pixels[(15 * width + 31) * 4]).toBe(0x53c0);
  });

  it('turns a direction on the CPU as its rotation turns the environment, +X towards +Z', () => {
    const turned = environment.turn(d.vec3f(1, 0.5, 0), Math.PI / 2);
    expect(turned.x).toBeCloseTo(0, 6);
    expect(turned.y).toBe(0.5);
    expect(turned.z).toBeCloseTo(1, 6);
  });

  it('rejects sizes that are not powers of two from 16 to 2048, and no samples', () => {
    const { root } = recordingRoot();
    for (const size of [8, 100, 4096]) {
      expect(() => environment.create(root, { size })).toThrow('power of two');
    }
    expect(() => environment.create(root, { samples: 0 })).toThrow('samples');
  });

  it('lights shaded surfaces through the ambient hook, adding uniforms, a cube and a sampler', () => {
    const { root, log } = recordingRoot();
    const sky = environment.create(root, { size: 16 });
    const light = lighting.create(root, { shading: { ambient: sky.ambient } });
    const size = { width: 8, height: 8 };
    const view = cameras.orbit(root, { size, target: [0, 0, 0], distance: 4 });
    const screen = targets.create(root, size, {
      hdr: { format: 'rgba16float' },
      depth: { format: 'depth24plus' },
    });
    const scene = passes
      .create(root, screen, { hdr: passes.color, depth: passes.depth })
      .pipe(view.inject())
      .pipe(light.inject());
    const behind = scene.add(sky.background());
    const encoder = root.createCommandEncoder();
    scene.record(encoder, behind);
    encoder.submit();
    expect(log.shaders.some((wgsl) => wgsl.includes('texture_cube<f32>'))).toBe(true);

    const color = root.createMutable(d.vec3f);
    const kernel = tgpu.computeFn({ workgroupSize: [1] })(() => {
      'use gpu';
      const point = shading.pointAt(d.vec3f(), d.vec3f(0, 1, 0));
      color.$ = shading.shade(materials.matte(d.vec3f(1)), point);
    });
    root
      .pipe(view.inject())
      .pipe(light.inject())
      .createComputePipeline({ compute: kernel })
      .initSync();
    const shaded = log.shaders.at(-1)!;
    // The camera, the lighting, the environment's settings and harmonics
    expect(shaded.match(/var<uniform>/g)).toHaveLength(4);
    expect(shaded.match(/texture_cube<f32>/g)).toHaveLength(1);
    expect(shaded.match(/var<storage/g)).toHaveLength(1);
  });

  it('puts the noon sun straight up at the equator on an equinox, and the 6h sun to the east', () => {
    const suns = [12, 6, 18].map((hour) => [...environment.sunAt(hour, 0, 0)]);
    const toward = [
      [0, 1, 0],
      [1, 0, 0],
      [-1, 0, 0],
    ];

    suns.flat().forEach((c, i) => expect(c).toBeCloseTo(toward.flat()[i], 5));
  });

  it('finds stars in cells of the size given, sizing discs by derivatives', () => {
    const wgsl = tgpu.resolve([
      tgpu.fn([d.vec3f], environment.Star)(environment.stars({ cells: 37 })),
    ]);

    expect(wgsl).toContain('37');
    expect(wgsl).toContain('dpdx');
  });
});
