import { cameras, frames, picker } from '@typegpu/kit';
import { offscreenSurface } from '@typegpu/kit/testing';
import { gpuRoot } from '@typegpu/kit/testing/dawn';
import { d, tgpu, type TgpuRoot } from 'typegpu';
import { expect, it } from 'vitest';
import { resolve } from '../src/assets.ts';
import { assets } from '../src/stampede/assets.ts';
import {
  aroundAt,
  createHerd,
  Horse,
  LANE_SPREAD,
  metresAround,
  MIXES,
  PHASES,
  RESTING_POSES,
  STEP_SECONDS,
  trackAt,
  trackHeading,
} from '../src/stampede/herd.ts';
import { createPoses } from '../src/stampede/horses.ts';
import { heightAt, loopAt } from '../src/stampede/land.ts';
import { createScene } from '../src/stampede/scene.ts';
import { layout } from '../src/stampede/scenery.ts';
import { createPicking } from '../src/stampede/selection.ts';
import { fromDisk, settle } from './utils/examples.ts';

const START = 1.85;
const STEPS_PER_FRAME = 6;

/** Repeatable dice for the herd's traits. */
const dice = (seed: number) => () => {
  seed = (seed * 1664525 + 1013904223) >>> 0;
  return seed / 2 ** 32;
};

/** A herd on its own, run in frames of six steps, and a copy of its horses to read. */
async function herdOf(root: TgpuRoot, count: number) {
  const files = await resolve('stampede', assets, fromDisk);
  const posed = createPoses(root, files.horse);
  const herd = createHerd(root, {
    count,
    middle: START,
    handOff: 50,
    obstacles: layout().obstacles,
    poses: posed.ids,
    random: dice(7),
  });
  const copy = root.createMutable(d.arrayOf(Horse, count));
  const copying = root.createComputePipeline({
    compute: tgpu.computeFn({ workgroupSize: [64], in: { gid: d.builtin.globalInvocationId } })(
      ({ gid }) => {
        'use gpu';
        if (gid.x >= count) return;
        copy.$[gid.x] = Horse(herd.horses.$[gid.x]);
      },
    ),
  });
  const state = { urgency: 0.85, clapAge: 1e3 };
  const loop = frames.create(root, (frame) => {
    const seconds = STEPS_PER_FRAME * STEP_SECONDS;
    state.clapAge += seconds;
    herd.advance(seconds, state.urgency);
    herd.threat.patch({ clapAge: state.clapAge });
    const pass = frame.encoder.beginComputePass();
    for (let i = 0; i < STEPS_PER_FRAME; i++) herd.step(pass);
    pass.setPipeline(copying);
    pass.dispatchWorkgroups(Math.ceil(count / 64));
    pass.end();
  });
  let time = 0;
  return {
    herd,
    posed,
    files,
    state,
    run(seconds: number) {
      for (let t = 0; t < seconds; t += STEPS_PER_FRAME * STEP_SECONDS) {
        loop.run((time += STEPS_PER_FRAME * STEP_SECONDS));
      }
    },
    read: () => copy.buffer.read(),
    gaits: async () => (await herd.gaits.rows.read()).slice(0, count),
    destroy() {
      loop.destroy();
      copy.buffer.destroy();
      herd.destroy();
      posed.poses.destroy();
    },
  };
}

const share = <T>(items: readonly T[], test: (item: T) => boolean) =>
  items.filter(test).length / items.length;

it('keeps the herd galloping clockwise on its track with its middle, and frightens it from a thunderclap', async () => {
  const root = await gpuRoot();
  const run = await herdOf(root, 1024);
  run.run(30);
  const running = await run.read();
  expect(
    share(
      running,
      (horse) => Math.abs(loopAt(horse.position.x, horse.position.z) - 1) < LANE_SPREAD + 0.06,
    ),
  ).toBeGreaterThan(0.97);
  expect(
    share(running, (horse) => {
      const [x, z] = trackHeading(aroundAt(horse.position.x, horse.position.z));
      const speed = Math.hypot(horse.velocity.x, horse.velocity.y);
      return horse.velocity.x * x + horse.velocity.y * z > speed * 0.7;
    }),
  ).toBeGreaterThan(0.95);
  const gaits = await run.gaits();
  expect(share(gaits, (row) => row.state === run.herd.gaits.definition.galloping)).toBeGreaterThan(
    0.95,
  );
  const galloping = new Set(run.posed.ids.slice((MIXES - 1) * PHASES, MIXES * PHASES));
  expect(share(running, (horse) => galloping.has(horse.pose))).toBeGreaterThan(0.95);
  // The herd's middle, by the angle around the oval, stays with the clock that leads it.
  const offsets = running
    .map((horse) => {
      const offset = aroundAt(horse.position.x, horse.position.z) - run.herd.middle;
      return offset - Math.round(offset / (Math.PI * 2)) * Math.PI * 2;
    })
    .toSorted((a, b) => a - b);
  expect(Math.abs(offsets[offsets.length >> 1] * metresAround(run.herd.middle))).toBeLessThan(40);

  const [x, z] = trackAt(run.herd.middle);
  run.herd.threat.patch({ clap: [x, heightAt(x, z), z] });
  run.state.clapAge = 0;
  const away = (horse: d.Infer<typeof Horse>) =>
    Math.hypot(horse.position.x - x, horse.position.z - z);
  const near = [...running.keys()].filter((i) => away(running[i]) < 40);
  run.run(1);
  const struck = await run.read();
  // Those it struck run out from it, whichever way the herd carried them.
  const fled = near.reduce((sum, i) => sum + away(struck[i]) - away(running[i]), 0);
  expect(fled / near.length).toBeGreaterThan(4);
  expect(share(near, (i) => struck[i].panic > 0.5)).toBeGreaterThan(0.8);

  run.destroy();
  root.destroy();
});

it('slows from a gallop to a walk through every mix between, some stopping to graze on their own poses', async () => {
  const root = await gpuRoot();
  const run = await herdOf(root, 256);
  run.run(4);
  const moving = run.posed.ids.slice(0, PHASES * MIXES);
  const levelOf = (horse: d.Infer<typeof Horse>) => Math.floor(moving.indexOf(horse.pose) / PHASES);
  let last = (await run.read()).map(levelOf);
  expect(share(last, (level) => level === MIXES - 1)).toBeGreaterThan(0.9);

  run.state.urgency = 0;
  let largest = 0;
  for (let frame = 0; frame < 60; frame++) {
    run.run(STEPS_PER_FRAME * STEP_SECONDS);
    const levels = (await run.read()).map(levelOf);
    levels.forEach((level, i) => {
      if (level >= 0 && last[i] >= 0) largest = Math.max(largest, Math.abs(level - last[i]));
    });
    last = levels;
  }
  expect(largest).toBeLessThanOrEqual(1);
  // Some stop to graze or rest on poses of their own; the rest walk.
  const stillMoving = last.filter((level) => level >= 0);
  expect(stillMoving.length / last.length).toBeGreaterThan(0.5);
  expect(share(stillMoving, (level) => level === 0)).toBeGreaterThan(0.95);
  const [calm, gaits] = await Promise.all([run.read(), run.gaits()]);
  const grazing = calm.filter((_, i) => gaits[i].state === run.herd.gaits.definition.grazing);
  expect(grazing.length / calm.length).toBeGreaterThan(0.03);
  const grazingPoses = new Set(run.posed.ids.slice(PHASES * MIXES + RESTING_POSES));
  expect(share(grazing, (horse) => grazingPoses.has(horse.pose))).toBeGreaterThan(0.9);
  run.destroy();
  root.destroy();
});

it('picks a galloping horse by the middle of its moving body, and nothing above it', async () => {
  const root = await gpuRoot();
  const run = await herdOf(root, 1);
  run.run(2);
  const [horse] = await run.read();
  const surface = offscreenSurface(root, { width: 640, height: 360, format: 'rgba8unorm' });
  const view = cameras.orbit(root, {
    size: surface,
    target: [horse!.position.x, horse!.position.y + 1.2, horse!.position.z],
    distance: 12,
    yaw: 0.6,
    pitch: 0.15,
  });
  const picked = createPicking(root, view, {
    model: run.files.horse,
    herd: run.herd,
    posed: run.posed,
    surface,
  });
  const seen = root.createMutable(d.u32);
  const marking = root.pipe(picked.inject()).createComputePipeline({
    compute: tgpu.computeFn({ workgroupSize: [1] })(() => {
      'use gpu';
      seen.$ = picked.found();
    }),
  });
  const aim = [0, 0];
  const loop = frames.create(root, (frame) => {
    view.update(frame.dt);
    const pass = frame.encoder.beginComputePass();
    run.posed.poses.animate(pass, 0);
    picked.pick(pass, aim[0], aim[1]);
    pass.setPipeline(marking);
    pass.dispatchWorkgroups(1);
    pass.end();
  });
  const pickAt = async (point: readonly number[]) => {
    const projected = [0, 0, 0];
    expect(view.project(point, projected)).toBe(true);
    aim[0] = (projected[0] + 1) / 2;
    aim[1] = (1 - projected[1]) / 2;
    loop.run(0);
    await settle();
    return seen.read();
  };
  const { x, y, z } = horse!.position;
  expect(await pickAt([x, y + 1.2, z])).toBe(0);
  expect(await pickAt([x, y + 4.5, z])).toBe(picker.none);
  loop.destroy();
  picked.destroy();
  seen.buffer.destroy();
  surface.destroy();
  run.destroy();
  root.destroy();
});

it('counts every horse once, skinned near the eye or a card farther off', async () => {
  const root = await gpuRoot({ features: ['indirect-first-instance'] });
  const surface = offscreenSurface(root, { width: 256, height: 144, format: 'rgba8unorm' });
  const files = await resolve('stampede', assets, fromDisk);
  const scene = await createScene(root, surface, { ...files, count: 256, random: dice(3) });
  const loop = frames.create(root, scene.frame);
  let seconds = 0;
  const draw = async (times: number) => {
    for (let i = 0; i < times; i++) {
      loop.run((seconds += 0.1));
      await root.device.queue.onSubmittedWorkDone();
      await settle();
    }
  };
  const counts = () => [scene.settings.readouts!.skinned!(), scene.settings.readouts!.impostors!()];
  await draw(4);
  const [skinned, carded] = counts() as [number, number];
  expect(skinned + carded).toBe(256);
  expect(skinned).toBeGreaterThan(0);

  scene.settings.values.handOff = 10;
  scene.settings.write!('handOff');
  await draw(4);
  const [fewer, more] = counts() as [number, number];
  expect(fewer + more).toBe(256);
  expect(fewer).toBeLessThan(skinned);

  scene.settings.values.count = 1024;
  scene.settings.write!('count');
  await draw(4);
  const [near, far] = counts() as [number, number];
  expect(near + far).toBe(1024);
  loop.destroy();
  scene.destroy();
  surface.destroy();
  root.destroy();
});
