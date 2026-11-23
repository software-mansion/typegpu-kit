import { readFileSync } from 'node:fs';
import { frames, type Frames } from '@typegpu/kit';
import type { CreateScene, Scene } from '@typegpu/kit/hosting';
import { offscreenSurface, recordingRoot, type RecordingLog } from '@typegpu/kit/testing';
import { describe, expect, it } from 'vitest';
import type { Loading } from 'typegpu-kit-examples/contract';
import * as aoHorizons from '../src/components/viz/AoHorizons.program.ts';
import * as bindGroups from '../src/components/viz/BindGroups.program.ts';
import * as bloomChain from '../src/components/viz/BloomChain.program.ts';
import * as cadences from '../src/components/viz/Cadence.program.ts';
import * as cascades from '../src/components/viz/Cascades.program.ts';
import * as coroutineSteps from '../src/components/viz/CoroutineSteps.program.ts';
import * as coroutineTimeline from '../src/components/viz/CoroutineTimeline.program.ts';
import * as countersTimeline from '../src/components/viz/CountersTimeline.program.ts';
import * as depthPrecision from '../src/components/viz/DepthPrecision.program.ts';
import * as exposureAdapt from '../src/components/viz/ExposureAdapt.program.ts';
import * as fixedStep from '../src/components/viz/FixedStep.program.ts';
import * as frameAnatomy from '../src/components/viz/FrameAnatomy.program.ts';
import * as gaitMachine from '../src/components/viz/GaitMachine.program.ts';
import * as gltfRead from '../src/components/viz/GltfRead.program.ts';
import * as historyBuffers from '../src/components/viz/HistoryBuffers.program.ts';
import * as impostorsFigure from '../src/components/viz/Impostors.scene.ts';
import * as inject from '../src/components/viz/Inject.program.ts';
import * as meshBuild from '../src/components/viz/MeshBuild.program.ts';
import * as meshPool from '../src/components/viz/MeshPool.program.ts';
import * as mipLevels from '../src/components/viz/MipLevels.program.ts';
import * as moduleLife from '../src/components/viz/ModuleLife.program.ts';
import * as neighbourReads from '../src/components/viz/NeighbourReads.program.ts';
import * as orbitCamera from '../src/components/viz/OrbitCamera.program.ts';
import * as readbackTimeline from '../src/components/viz/ReadbackTimeline.program.ts';
import { KEPT } from '../src/components/viz/timeline.ts';
import { seeded } from '../src/components/viz/svg.ts';
import * as recordingLog from '../src/components/viz/RecordingLog.program.ts';
import * as runtimeLanes from '../src/components/viz/RuntimeLanes.program.ts';
import * as skinWeights from '../src/components/viz/SkinWeights.program.ts';
import * as slotAllocation from '../src/components/viz/SlotAllocation.program.ts';
import * as surfaceSize from '../src/components/viz/SurfaceSize.program.ts';
import * as taaSamples from '../src/components/viz/TaaSamples.program.ts';
import * as tableFlush from '../src/components/viz/TableFlush.program.ts';
import * as tableParts from '../src/components/viz/TableParts.program.ts';
import * as targetsResize from '../src/components/viz/TargetsResize.program.ts';
import * as warmupBudget from '../src/components/viz/WarmupBudget.program.ts';
import { fromDisk } from 'typegpu-kit-examples/testing';

/**
 * Every figure's scene, by its file; each must run with no options, or with what its `load` reads,
 * as an example's `assets.ts` does.
 */
const scenes = import.meta.glob<{
  createScene: CreateScene<object | undefined, Scene>;
  load?: (loading: Loading) => Promise<object>;
}>('../src/components/viz/*.scene.ts', { eager: true });

/** Every figure's program, by its file; each must run with no options. */
const programs = import.meta.glob<{ createProgram: () => unknown }>(
  '../src/components/viz/*.program.ts',
  { eager: true },
);

/** A recording root and the surface a figure's scene draws to on it. */
function sceneRoot() {
  const recording = recordingRoot({ features: ['indirect-first-instance'] });
  const surface = offscreenSurface(recording.root, { width: 64, height: 40, format: 'bgra8unorm' });
  return { ...recording, surface };
}

/** Runs three frames, and checks that the third creates nothing the second did not. */
function expectSteady(loop: Frames, log: RecordingLog) {
  loop.run(0);
  loop.run(1 / 60);
  const steady = log.counts();
  loop.run(2 / 60);
  expect(log.counts()).toEqual(steady);
}

describe.each(Object.entries(scenes))('%s', (_path, { createScene, load }) => {
  it('submits a frame at a time and creates nothing in a steady one', async () => {
    const options = await load?.(fromDisk);
    const { root, log, surface } = sceneRoot();
    const scene = await createScene(root, surface, options);
    const loop = frames.create(root, scene.frame);
    log.events.length = 0;

    expectSteady(loop, log);
    expect(log.events.filter((event) => event === 'submit')).toHaveLength(3);

    loop.destroy();
    scene.destroy?.();
  });
});

it('frees each atlas it captures again, and creates nothing in a steady frame of the atlas', async () => {
  const { root, log, surface } = sceneRoot();
  const before = log.alive({ uniforms: false });
  const scene = await impostorsFigure.createScene(root, surface);
  const loop = frames.create(root, scene.frame);

  await scene.capture(4);
  await scene.read('parallax');
  await scene.capture(6);
  scene.showAtlas(true);
  expectSteady(loop, log);

  loop.destroy();
  scene.destroy();
  expect(log.alive({ uniforms: false })).toEqual(before);
});

it.each(Object.entries(programs))('runs %s with no options', async (_path, { createProgram }) => {
  const program = (await createProgram()) as { destroy?: () => void } | undefined;
  program?.destroy?.();
});

describe('the bind groups', () => {
  const { createProgram, move, OPENING, OWNERS } = bindGroups;

  it('puts fixed bindings in the automatic group, a share in a group each, and opens one over', () => {
    const opening = createProgram();
    expect(opening).toMatchObject({
      groups: [['output'], ['shadow'], ['fog'], ['palette'], ['tint']],
      count: 5,
      limit: 4,
      fits: false,
    });
    expect(createProgram([]).groups).toEqual([['shadow', 'fog', 'palette', 'tint', 'output']]);
    expect(createProgram([OWNERS]).groups).toEqual([
      ['output'],
      ['shadow', 'fog', 'palette', 'tint'],
    ]);

    const joined = createProgram(move(OPENING, 'tint', ['shadow']));
    expect(joined).toMatchObject({
      groups: [['output'], ['shadow', 'tint'], ['fog'], ['palette']],
      count: 4,
      fits: true,
    });
  });

  it('swaps a shared resource with a bind group, and a fixed one with a pipeline', () => {
    const { swaps } = createProgram(move(OPENING, 'tint', ['output']));
    expect(swaps.fog).toEqual({ pipelines: 0, bindGroups: 1 });
    expect(swaps.tint.pipelines).toBe(1);
    expect(swaps.output.pipelines).toBe(1);
  });
});

describe('the bloom chain', () => {
  it('goes down a level a pass and back up, and stops where a small source runs out', () => {
    const program = bloomChain.createProgram();
    const chain = program.frame();

    expect(chain.rows).toHaveLength(7);
    expect(chain.rows[1]).toEqual({ width: 960, height: 540 });
    expect(chain.passes).toBe(11);
    expect(chain.arrows.filter((arrow) => arrow.clears).map((arrow) => arrow.to)).toEqual([
      1, 2, 3, 4, 5, 6,
    ]);
    expect(chain.arrows.filter((arrow) => !arrow.clears).map((arrow) => arrow.to)).toEqual([
      5, 4, 3, 2, 1,
    ]);

    Object.assign(program.size, { width: 40, height: 24 });
    const small = program.frame();
    expect(small.rows.at(-1)).toEqual({ width: 1, height: 1 });
    expect(small.rows.length - 1).toBeLessThan(6);
    expect(small.passes).toBe(2 * (small.rows.length - 1) - 1);
    program.destroy();
  });
});

describe('the frame anatomy', () => {
  it('submits a frame, and nothing from a recorder that throws', () => {
    const program = frameAnatomy.createProgram();
    const all = { field: true, particles: true, readback: true, burst: true };

    const frame = program.run({ ...all, fail: false });
    expect(frame.submitted).toBe(true);
    expect(frame.events.map((event) => event.kind)).toContain('submit');

    const thrown = program.run({ ...all, fail: true });
    expect(thrown.submitted).toBe(false);
    expect(thrown.events.map((event) => event.kind)).not.toContain('submit');
  });
});

describe('the history buffers', () => {
  it('keeps last frame in previous, where a write straight into current is copied aside too', () => {
    const program = historyBuffers.createProgram();
    program.step();
    program.step();
    const third = program.step();

    expect(third).toEqual({
      index: 2,
      written: [61, 17],
      kept: { current: [61, 17], previous: [32, 29] },
      straight: { current: [61, 17], previous: [61, 17] },
    });
    expect(historyBuffers.moved(third.kept)).toBeCloseTo(Math.hypot(29, 12));
    expect(historyBuffers.moved(third.straight)).toBe(0);

    program.writes = 'on click';
    const idle = program.step();
    program.write();
    expect([idle, program.step()]).toEqual([
      {
        index: 3,
        written: undefined,
        kept: { current: [61, 17], previous: [61, 17] },
        straight: { current: [61, 17], previous: [61, 17] },
      },
      {
        index: 4,
        written: [100, 12],
        kept: { current: [100, 12], previous: [61, 17] },
        straight: { current: [100, 12], previous: [100, 12] },
      },
    ]);
    program.destroy();
  });
});

describe('members and placeholders', () => {
  it('binds both members whatever is piped, the last piped definition or hook, and names what nothing filled', () => {
    const { outcome } = inject.createProgram();

    expect(outcome('member', 'none')).toEqual({ bound: [0, 1] });
    expect(outcome('member', 'ba')).toEqual({ bound: [0, 1] });
    expect(outcome('definition', 'ab')).toEqual({ bound: [1] });
    expect(outcome('definition', 'ba')).toEqual({ bound: [0] });
    expect(outcome('hook', 'ab')).toEqual({ bound: [1] });
    expect(outcome('hook', 'none').error).toBe(
      'The pipeline is missing a fill for cameras.current.',
    );
  });
});

describe('the mesh pool', () => {
  it('hands each mesh the next ranges, and says why a shape does not fit', () => {
    const program = meshPool.createProgram();
    let refused: string | undefined;
    while (refused === undefined) refused = program.add('sphere');

    const [first, second] = program.meshes;
    expect(second!.baseVertex).toBe(first!.vertexCount);
    expect(second!.firstIndex).toBe(first!.indexCount);
    expect(program.names).toEqual(program.meshes.map(() => 'sphere'));
    expect(refused).toMatch(/^The mesh pool has room for/);
    program.destroy();
  });
});

/** A model file of the examples. */
const modelFile = (path: string) =>
  new Uint8Array(
    readFileSync(new URL(`../../../packages/typegpu-kit-examples/src/${path}`, import.meta.url)),
  );

describe('the glTF reader', () => {
  const windmill = modelFile('hex-isle/models/windmill.glb');

  it('reads the windmill as one mesh, and its top as a part about its own origin', () => {
    const whole = gltfRead.read(windmill);
    expect(whole.nodes.map((node) => [node.name, node.depth])).toEqual([
      ['building_windmill_blue', 0],
      ['building_windmill_top_blue', 1],
    ]);
    expect(whole.model?.meshes).toHaveLength(1);
    expect(whole.model?.meshes[0]?.textures).toEqual([['color', 0]]);

    const split = gltfRead.read(windmill, ['building_windmill_top_blue']);
    const [part] = split.model!.parts;
    expect(part?.meshes[0]?.vertices).toBe(697);
    expect(split.model!.meshes[0].vertices + part!.meshes[0].vertices).toBe(
      whole.model!.meshes[0].vertices,
    );
  });

  it('reads the rogue as two meshes on a skin, and the clips file as clips alone', () => {
    const rogue = gltfRead.read(modelFile('skeletons/models/rogue.glb')).model!;
    expect(rogue.meshes.map((mesh) => [mesh.material, mesh.skinned])).toEqual([
      ['skeleton', true],
      ['Glow', true],
    ]);
    expect(rogue.meshes[1]?.metallic).toBe(1); // no factor in the file: glTF's default
    expect(rogue.joints).toHaveLength(24);

    const clips = gltfRead.read(modelFile('skeletons/models/melee-clips.glb')).model!;
    expect(clips.meshes).toHaveLength(0);
    expect(clips.clips.map((clip) => clip.name)).toEqual([
      'Melee_1H_Attack_Chop',
      'Melee_1H_Attack_Stab',
      'Melee_2H_Attack_Chop',
    ]);
  });

  it('shows what read throws for a file cut short, a web page and a .gltf', () => {
    expect(gltfRead.read(windmill.slice(0, 60_000)).error).toBe(
      `The binary glTF is incomplete. Its header declares ${windmill.byteLength} bytes, but only 60000 are available.`,
    );
    expect(gltfRead.read(gltfRead.broken.page()).error).toBe("The bytes aren't a binary glTF.");
    expect(gltfRead.read(gltfRead.broken.json()).error).toMatch(/gltf-transform copy in.gltf/);
  });
});

describe('the module life', () => {
  it('frees the buffers the field made on destroy, and leaves the surface it borrows', async () => {
    const lives = await moduleLife.createProgram();
    const destroy = moduleLife.STEPS.indexOf('destroy');

    expect(lives.find((life) => life.borrowed)!.to).toBeUndefined();
    expect(lives.filter((life) => life.to === destroy).length).toBeGreaterThan(0);
    expect(lives.every((life) => life.borrowed || life.from >= 0)).toBe(true);
  });
});

describe('the recording log', () => {
  it('creates nothing in a steady frame', async () => {
    const program = recordingLog.createProgram();
    const counts = () => recordingLog.COUNTERS.map((counter) => program.count(counter));
    await program.initAsync();
    program.frame(2);
    const steady = counts().slice(0, 5);
    program.frame(2);

    expect(steady[0]).toBeGreaterThan(0);
    expect(counts().slice(0, 5)).toEqual(steady);
    program.destroy();
  });
});

describe('the surface size', () => {
  it('caps the pixel ratio at maxPixelRatio', () => {
    const phone = { width: 1170, height: 2532, pixelRatio: 3 };
    expect(surfaceSize.createProgram()).toEqual({
      buffer: { width: 780, height: 1688, pixelRatio: 2 },
      screen: phone,
    });
    expect(surfaceSize.createProgram({ maxPixelRatio: 3 }).buffer).toEqual(phone);
    const monitor = { width: 1920, height: 1080, pixelRatio: 1 };
    expect(surfaceSize.createProgram(monitor).buffer).toEqual(monitor);
  });
});

describe('the table flush', () => {
  it('uploads every host row between two writes, each device field written, and a new row whole', () => {
    const program = tableFlush.createProgram();
    for (let i = 0; i < 9; i++) program.add();
    program.flush();
    program.remove(2);
    program.remove(6);
    program.flush();
    program.write(1, 'tint');
    program.write(7, 'tint');
    program.write(4, 'position');
    program.write(5, 'health');
    program.add();
    program.flush();

    expect(program.uploads).toEqual([
      ['host', 'slots 1 to 7', '', '112 B'],
      ['device', 'slot 4', 'position', '12 B'],
      ['device', 'slot 5', 'health', '4 B'],
      ['device', 'slot 2', 'new', '16 B'],
      ['live', 'word 0', '', '4 B'],
    ]);
    expect(program.uploaded('3:tint')).toBe(true);
    expect(program.changed('1:tint')).toBe(false);
    program.destroy();
  });
});

const late = (run: warmupBudget.Run) =>
  run.frames.filter((frame) => frame.busy > warmupBudget.VSYNC);
const starting = (run: warmupBudget.Run) => run.frames.filter((frame) => frame.starts.length);

describe('the warm-up budget', () => {
  const { createProgram } = warmupBudget;

  it('spreads the starts over frames on time, and misses a frame with a budget past one', async () => {
    const small = await createProgram({ budget: 1 });
    const large = await createProgram({ budget: 32 });

    expect(starting(small)).toHaveLength(10);
    expect(late(small)).toHaveLength(0);
    expect(starting(large)).toHaveLength(1);
    expect(late(large)).toHaveLength(1);
    for (const run of [small, large]) {
      expect(run.compiles).toHaveLength(10);
      expect(run.progress.at(-1)!.fraction).toBe(1);
    }
  });

  it('gives performance.now back', async () => {
    const now = performance.now;
    await createProgram();
    expect(performance.now).toBe(now);
  });
});

describe('the coroutine timeline', () => {
  const { createProgram, FRAMES, LANES } = coroutineTimeline;
  const lane = (name: (typeof LANES)[number]) => LANES.indexOf(name);
  const notes = (run: coroutineTimeline.Run, name: (typeof LANES)[number]) =>
    run.notes.filter((note) => note.lane === lane(name)).map((note) => note.label);

  it('runs every coroutine to its end or the last frame, without a stop', () => {
    const run = createProgram();

    expect(notes(run, 'game')).toEqual(['got 7']);
    expect(notes(run, 'pick')).toEqual(['return 7']);
    expect(notes(run, 'loader')).toEqual(['loaded']);
    expect(run.ends[lane('game')]).toBe(FRAMES);
    expect(run.ends[lane('spawner')]).toBe(FRAMES);
    // The loader does two chunks a frame within its budget of 2 ms at 0.7 ms a chunk.
    expect(Math.max(...run.runs[lane('loader')])).toBe(3);
  });

  it('runs the finally blocks of game and the pick it waits on, on cancel', () => {
    const run = createProgram({ stopAt: 7, stop: 'cancel' });

    expect(notes(run, 'game')).toEqual(['finally']);
    expect(notes(run, 'pick')).toEqual(['finally']);
    expect(notes(run, 'spawner')).toEqual([]);
    expect(run.ends[lane('game')]).toBe(7);
    expect(run.ends[lane('spawner')]).toBe(FRAMES);
  });

  it('runs every finally block on destroy, and starts nothing after', () => {
    const run = createProgram({ stopAt: 7, stop: 'destroy' });

    for (const name of ['game', 'pick', 'spawner'] as const) {
      expect(notes(run, name)[0]).toBe('finally');
      expect(run.ends[lane(name)]).toBe(7);
    }
    expect(run.runs[lane('loader')].every((count) => count === 0)).toBe(true);
  });
});

/** Runs `count` of the program's frames. */
async function runFrames(program: { step(): Promise<void> }, count: number) {
  for (let i = 0; i < count; i++) await program.step();
}

/** The frames a readback timeline copied in, and those it skipped while every buffer mapped. */
const copied = (program: readbackTimeline.Program) =>
  program.columns.filter((column) => column.copy).map((column) => column.index);
const skipped = (program: readbackTimeline.Program) =>
  program.columns.filter((column) => column.full && column.wanted).map((column) => column.index);

describe('the readback timeline', () => {
  const { createProgram } = readbackTimeline;

  it('copies at its rate into the next free buffer, and holds the frame a copy came from', async () => {
    const program = createProgram({ hz: 30, latency: 3, slots: 3 });
    await runFrames(program, KEPT + 6);

    const { columns } = program;
    expect(columns).toHaveLength(KEPT);
    expect(copied(program).every((index) => index % 2 === 0)).toBe(true);
    expect(new Set(columns.map((column) => column.copy?.slot))).toEqual(new Set([0, 1, undefined]));
    expect(skipped(program)).toEqual([]);

    // The CPU holds the index of the frame it was copied in, three or four frames back
    const last = columns.at(-1)!;
    expect(last.value).toBe(last.latest);
    expect(last.index - last.latest).toBeGreaterThanOrEqual(3);
    expect(last.index - last.latest).toBeLessThanOrEqual(4);
    program.destroy();
  });

  it('wants a copy in the frames clock.ts picks, and skips them while every buffer maps', async () => {
    // 10 Hz ticks every 6 frames; one buffer maps for 8, and a late copy keeps to the schedule
    const program = createProgram({ hz: 10, latency: 8, slots: 1 });
    await runFrames(program, 20);
    expect(copied(program)).toEqual([0, 8, 16]);
    expect(skipped(program)).toEqual([6, 7, 12, 13, 14, 15]);

    // The mirror agrees with the readback in every frame with a free buffer
    for (const hz of [60, 30, 10]) {
      const check = createProgram({ hz, latency: 5, slots: 2 });
      await runFrames(check, 40);
      for (const column of check.columns) {
        if (!column.full) expect(column.wanted).toBe(column.copy !== undefined);
      }
      check.destroy();
    }
    program.destroy();
  });

  it('answers a read with the next copy, as it lands', async () => {
    const program = createProgram({ hz: 0, latency: 3, slots: 1 });
    await runFrames(program, 4);
    expect(copied(program)).toEqual([]);

    program.ask();
    await runFrames(program, 8);
    const asked = program.columns.find((column) => column.asked)!;
    const answer = program.columns.find((column) => column.copy?.answers)!;
    expect(answer.index).toBe(asked.index + 1);
    const resumed = program.columns.find((column) => column.resumed.includes(asked.index))!;
    expect(resumed.index).toBe(answer.copy!.until);
    expect(resumed.latest).toBe(answer.index);
    program.destroy();
  });
});

describe('the gait machine', () => {
  const { createProgram, GALLOP_FROM, HORSES, STEP, WALK_FROM } = gaitMachine;
  const seconds = (program: gaitMachine.Program, time: number) => {
    for (let t = 0; t < time; t += STEP) program.step();
  };

  it('walks below the band and gallops above it, each horse keeping its gait inside it', () => {
    const program = createProgram(seeded(3));
    const { Gait, horses, pace } = program;
    seconds(program, 10);
    for (let i = 0; i < HORSES; i++) {
      if (pace[i] > GALLOP_FROM + 0.1) expect(horses.state(i)).toBe(Gait.galloping);
      if (pace[i] < WALK_FROM - 0.1) expect(horses.state(i)).toBe(Gait.walking);
    }
    expect(program.counts().reduce((sum, count) => sum + count)).toBe(HORSES);
  });

  it('grazes while calm, and bolts every horse at a startle, galloping two seconds later', () => {
    const program = createProgram(seeded(5));
    const { Gait } = program;
    program.urgency = 0.1;
    seconds(program, 30);
    expect(program.counts()[Gait.grazing]).toBeGreaterThan(0);

    program.startle();
    program.step();
    expect(program.counts()[Gait.bolting]).toBe(HORSES);
    expect(program.lit.some((left) => left > 0)).toBe(true);
    seconds(program, 2.1);
    expect(program.counts()[Gait.bolting]).toBe(0);
  });
});

describe('the coroutine steps', () => {
  const { createProgram, FRAMES } = coroutineSteps;

  it('runs nothing before the switch, then opens the door after half a second', () => {
    const program = createProgram();
    program.step();
    program.press();
    while (!program.finished) program.step();

    const [idle, start, ...rest] = program.steps;
    expect(program.steps).toHaveLength(FRAMES);
    expect(idle!.started).toBe(false);
    expect(start!.ran).toEqual([1, 2]);
    expect(start!.paused).toBe(2);

    // Four frames of the wait, at 8 a second, then a quarter of the door a frame.
    expect(rest.slice(0, 3).every((step) => step.ran.length === 0)).toBe(true);
    const opened = rest.findIndex((step) => step.open === 1);
    expect(rest[opened].done).toBe(true);
    expect(rest.at(-1)!.open).toBe(1);
    program.destroy();
  });
});

describe('the table parts', () => {
  it('puts back the stale x in one struct, and never in a table', () => {
    const program = tableParts.createProgram();
    for (let i = 0; i < 30; i++) program.frame(1 / 60);
    program.hit(1, 6);
    program.frame(1 / 60);

    const [one, two] = program.layouts;
    expect(one!.overwritten).toBeGreaterThan(0);
    expect(two!.overwritten).toBe(0);
    expect(one!.uploads.at(-1)).toMatchObject({ rows: [1, 6], bytes: 192 });
    expect(two!.uploads.at(-1)).toMatchObject({ rows: [1, 6], bytes: 96 });
    program.destroy();
  });
});

describe('the neighbour reads', () => {
  it('gives the same rows in any order from the copy, and other rows in place', () => {
    const program = neighbourReads.createProgram();
    const inTwoOrders = (reads: neighbourReads.Reads) => {
      program.reads = reads;
      program.restart();
      const first = program.rows;
      program.shuffle();
      program.restart();
      return [first, program.rows];
    };

    const [one, other] = inTwoOrders('in place');
    expect(one).not.toEqual(other);
    expect(program.step.threads.some((thread) => thread.late)).toBe(true);

    const [kept, again] = inTwoOrders('copy');
    const { before } = program.step;
    expect(before).toEqual([1, 2, 3, 4, 5, 6, 7, 8]);
    expect(kept).toEqual(before.map((_, row) => before.at(row - 1)));
    expect(again).toEqual(kept);
    program.destroy();
  });
});

describe('the orbit camera', () => {
  it('reads the eye from its uniform, and writes it only while it moves', () => {
    const program = orbitCamera.createProgram();
    const [x, y, z] = program.eye();
    const flat = 7 * Math.cos(0.5);
    expect(x).toBeCloseTo(flat * Math.sin(0.7), 4);
    expect(y).toBeCloseTo(7 * Math.sin(0.5), 4);
    expect(z).toBeCloseTo(flat * Math.cos(0.7), 4);

    program.goal.yaw += Math.PI / 2;
    program.frame(1 / 60);
    expect(program.moving).toBe(true);
    for (let i = 0; i < 240; i++) program.frame(1 / 60);
    expect(program.moving).toBe(false);

    const writes = program.writes;
    program.frame(1 / 60);
    expect(program.writes).toBe(writes);

    // A new damping is a new camera, where the old one was.
    program.damp(4);
    expect(program.eye()[0]).toBeCloseTo(flat * Math.sin(0.7 + Math.PI / 2), 3);
    program.destroy();
  });
});

describe('the targets', () => {
  it('rebuilds once per change of drawing buffer, with a bind group from memo for each', () => {
    const program = targetsResize.createProgram();
    program.frame();
    program.frame();
    expect(program.rebuilds).toBe(0);
    expect(program.groups).toBe(1);
    expect(program.stale).toBe(false);

    // A lower resolution shrinks the drawn region and rebuilds nothing
    program.settings.resolution = 0.5;
    program.frame();
    expect(program.rebuilds).toBe(0);
    expect(program.screen.regions.hdr).toMatchObject({ width: 640, height: 360 });

    program.settings.width = 320;
    program.frame();
    program.frame();
    // A ratio past the cap leaves the drawing buffer as it was
    program.settings.ratio = 3;
    program.frame();
    expect(program.rebuilds).toBe(1);
    program.settings.ratio = 1;
    program.frame();

    expect(program.rebuilds).toBe(2);
    expect(program.groups).toBe(3);
    expect(program.stale).toBe(true);
    expect(program.screen.textures.bloom.props.size).toEqual([160, 180]);
    expect(program.once.props.size).toEqual([1280, 720]);
    program.destroy();
  });
});

describe('the slots', () => {
  it('keeps the table denser than the free list over the waves', () => {
    const program = slotAllocation.createProgram();
    program.add(20);
    for (let i = 0; i < 300; i++) program.wave();

    const { kit, lifo } = program;
    expect(kit.count).toBe(lifo.count);
    expect(kit.mean!).toBeGreaterThan(lifo.mean!);
    program.destroy();
  });

  it('fills the lowest free slot first', () => {
    const program = slotAllocation.createProgram();
    program.add(20);
    program.remove(6);
    program.add(1);

    const { live, span } = program.kit;
    const lowestFree = [...Array(span).keys()].find((slot) => !live.has(slot));
    const filled = [...live].filter((slot) => slot < (lowestFree ?? span));
    expect(filled.length).toBeGreaterThan(0);
    expect(program.kit.count).toBe(15);
    program.destroy();
  });
});

describe('the mesh build', () => {
  type Vec3 = meshBuild.Vertex['position'];
  const minus = (a: Vec3, b: Vec3) => ({ x: a.x - b.x, y: a.y - b.y, z: a.z - b.z });
  const dot = (a: Vec3, b: Vec3) => a.x * b.x + a.y * b.y + a.z * b.z;
  const cross = (a: Vec3, b: Vec3) => ({
    x: a.y * b.z - a.z * b.y,
    y: a.z * b.x - a.x * b.z,
    z: a.x * b.y - a.y * b.x,
  });

  it('evaluates a corner per index, each triangle counter-clockwise seen from outside', () => {
    const program = meshBuild.createProgram();
    const box = program.build('box');
    expect([box.uses.length, box.indices.length, box.corners.length]).toEqual([24, 36, 36]);
    expect(program.build('box')).toBe(box);

    for (const name of meshBuild.SHAPES) {
      const { corners, indices, uses } = program.build(name);
      expect(corners).toHaveLength(indices.length);
      expect(uses.reduce((sum, each) => sum + each, 0)).toBe(indices.length);

      for (let t = 0; t < corners.length; t += 3) {
        const triangle = corners.slice(t, t + 3);
        const [a, b, c] = triangle.map((corner) => corner.position) as [Vec3, Vec3, Vec3];
        const face = cross(minus(b, a), minus(c, a));
        const outward = triangle.reduce((sum, corner) => sum + dot(face, corner.normal), 0);
        expect(outward, `${name}, triangle ${t / 3}`).toBeGreaterThan(-1e-6);
      }
    }
  });
});

describe('the counters timeline', () => {
  it('loses no stomp while copies are skipped, and drops the copies a clear overtakes', async () => {
    // Three buffers mapping for 7 frames each outlast the copies every other frame at 30 Hz
    const program = countersTimeline.createProgram({ latency: 7, hz: 30 });
    for (let i = 0; i < 12; i++) {
      program.add('stomps', 1);
      await runFrames(program, 1);
    }
    expect(program.columns.some((column) => column.full && column.wanted)).toBe(true);

    await runFrames(program, 12);
    const taken = program.columns.reduce((sum, column) => sum + (column.taken?.stomps ?? 0), 0);
    expect(program.columns.at(-1)!.latest?.stomps).toBe(12);
    expect(taken).toBe(12);

    program.add('stomps', 1);
    await runFrames(program, 2);
    program.clear();
    await runFrames(program, 1);
    const cleared = program.columns.at(-1)!;
    expect(cleared.counters.stomps).toBe(0);
    expect(program.columns.some((column) => column.copy?.dropped)).toBe(true);
    expect(cleared.latest).toBeUndefined();
    program.destroy();
  });
});

describe('the runtime lanes', () => {
  const { createProgram, VSYNC } = runtimeLanes;

  /** Two seconds of the model: what its last second held, and the time steps it saw. */
  const run = (options: runtimeLanes.Options) => {
    const program = createProgram(options);
    program.advance(2000);
    const stats = program.stats(1000);
    const dts = program.blocks.flatMap((block) => (block.dt === undefined ? [] : [block.dt]));
    program.destroy();
    return { ...stats, dts };
  };

  it('holds 120 Hz off the JS thread, where only the UI runtime drops app frames', () => {
    const worker = run({ on: 'worker' });
    expect(worker.hz).toBeGreaterThan(115);
    expect([worker.dropped, worker.missed]).toEqual([0, 0]);
    expect(Math.max(...worker.dts)).toBeCloseTo(VSYNC / 1000, 4);

    const ui = run({ on: 'ui' });
    expect(ui.hz).toBeGreaterThan(110);
    expect(ui.dropped).toBeGreaterThan(0);
    expect(ui.dropped).toBeLessThan(ui.shown * 0.1);
  });

  it('misses vsyncs on a worker, and drops app frames on the UI runtime, when frames run long', () => {
    const worker = run({ on: 'worker', frame: 12 });
    expect(worker.hz).toBeLessThan(65);
    expect(worker.missed).toBeGreaterThan(40);
    expect(worker.dropped).toBe(0);

    const ui = run({ on: 'ui', frame: 7 });
    expect(ui.hz).toBeLessThan(65);
    expect(ui.dropped).toBeGreaterThan(ui.shown * 0.4);
    expect(ui.missed).toBe(0);
  });

  it('runs at 60 Hz at most on the JS thread', () => {
    const js = run({ on: 'js' });
    expect(js.hz).toBeGreaterThan(55);
    expect(js.hz).toBeLessThan(62);
    expect(js.dropped).toBe(0);
    expect(Math.min(...js.dts)).toBeCloseTo((2 * VSYNC) / 1000, 4);
  });

  it('runs sent closures in order, each before the next frame that reads it', () => {
    const program = createProgram({ on: 'worker', frame: 6 });
    program.advance(50);
    program.send();
    program.advance(1);
    program.send();
    program.advance(40);

    const [first, second] = program.sends;
    expect(first!.ran).toBeGreaterThan(first!.at);
    expect(second!.ran).toBeGreaterThan(first!.ran!);
    for (const sent of [first!, second!]) expect(sent.read).toBeGreaterThan(sent.ran!);
    program.destroy();
  });
});

describe('the skin weights', () => {
  it('reads a chain of three joints and mixes a vertex between the two it names', () => {
    const { skin, mesh, pose } = skinWeights.createProgram();
    expect(skin.joints.map(({ name, parent }) => [name, parent])).toEqual([
      ['shoulder', -1],
      ['elbow', 0],
      ['wrist', 1],
    ]);

    // The vertex a quarter past the elbow, as the file stores it
    const vertex = 10;
    expect([...mesh.positions.subarray(vertex * 3, vertex * 3 + 3)]).toEqual([1.25, -0.375, 0]);
    expect([...mesh.joints!.subarray(vertex * 4, vertex * 4 + 4)]).toEqual([0, 1, 0, 0]);
    expect([...mesh.weights!.subarray(vertex * 4, vertex * 4 + 4)]).toEqual([0.25, 0.75, 0, 0]);

    const rest = pose([0, 0, 0]);
    rest.positions.forEach((value, i) => expect(value).toBeCloseTo(mesh.positions[i], 5));

    // The elbow turns a quarter: the shoulder leaves the vertex where it was, the elbow swings it
    const bent = pose([0, Math.PI / 2, 0]);
    const byShoulder = bent.carried(vertex, 0);
    const byElbow = bent.carried(vertex, 1);
    expect([...byShoulder].map((x) => +x.toFixed(4))).toEqual([1.25, -0.375, 0]);
    expect([...byElbow].map((x) => +x.toFixed(4))).toEqual([1.375, 0.25, 0]);
    for (let axis = 0; axis < 3; axis++) {
      expect(bent.positions[vertex * 3 + axis]).toBeCloseTo(
        0.25 * byShoulder[axis] + 0.75 * byElbow[axis],
        5,
      );
    }
    expect([...bent.place(2)].map((x) => +x.toFixed(4))).toEqual([1, 1, 0]);
  });
});

describe('the adapting exposure', () => {
  it('takes its first target at once, then closes the stops left at rise or fall, whatever the frames', () => {
    const { createProgram, LIGHTS, screen } = exposureAdapt;
    const eye = createProgram();
    const { key, rise, fall, highest } = eye.settings;
    /** Stops between the exposure and the target of `light`, after `count` frames filling a second. */
    const left = (light: exposureAdapt.Light, count: number) => {
      for (let i = 0; i < count; i++) eye.record(LIGHTS[light], 1 / count);
      return Math.abs(Math.log2(eye.exposure / eye.target(LIGHTS[light])));
    };

    expect(eye.record(LIGHTS.bright, 1 / 60)).toBeCloseTo(key / LIGHTS.bright);
    expect(left('dark', 60)).toBeCloseTo(4 * Math.exp(-rise));
    for (let i = 0; i < 20; i++) eye.record(LIGHTS.dark, 1);
    expect(screen(LIGHTS.dark * eye.exposure)).toBeCloseTo(screen(key), 4);
    expect(left('bright', 7)).toBeCloseTo(4 * Math.exp(-fall));

    expect(createProgram().record(1e-3, 0)).toBe(highest);
  });
});

describe('the cascades', () => {
  it('fits each slice a square that holds it, the same texels over more ground', () => {
    const program = cascades.createProgram();
    const { cascades: placed, size, far } = program.fit;
    expect(placed.map(({ farther }) => farther / far)).toEqual([1 / 9, 4 / 9, 1]);
    // The snap to whole texels moves a square up to half a texel off its slice
    for (const { corners, x, y, radius, density } of placed) {
      for (const [cornerX, cornerY] of corners) {
        const out = Math.max(Math.abs(cornerX - x), Math.abs(cornerY - y)) - radius;
        expect(out).toBeLessThanOrEqual(0.5 / density);
      }
    }
    const densities = placed.map(({ radius }) => size / (radius * 2));
    expect(placed.map(({ density }) => density)).toEqual(densities);
    expect(densities).toEqual(densities.toSorted((a, b) => b - a));

    // Half as far, the same texels cover half the ground
    program.far(60);
    expect(program.fit.cascades[0].density).toBeCloseTo(densities[0] * 2);
    program.shape({ cascades: 4, even: true });
    expect(program.fit.cascades.map(({ near }) => near)).toEqual([0, 15, 30, 45]);
    program.destroy();
  });
});

describe('the mip levels', () => {
  it('fills a level for each dispatch, the farthest depth or the mean of each block', () => {
    const program = mipLevels.createProgram();
    const largest = program.levels('largest');
    expect(largest.map((level) => level.length)).toEqual([8, 4, 2, 1]);
    // The one-texel pole is gone from level 1, and the box from level 2
    expect(largest[1]).toEqual([
      [9, 9, 9, 9],
      [9, 2, 9, 9],
      [7, 2, 7, 7],
      [5, 4, 5, 5],
    ]);
    expect(largest[2]).toEqual([
      [9, 9],
      [7, 7],
    ]);
    // Means of means over whole blocks are the mean of level 0
    const [first, , , [[last]]] = program.levels('mean');
    expect(last).toBeCloseTo(first.flat().reduce((sum, depth) => sum + depth, 0) / 64);
    // A block past the edge of a one-texel level repeats its last texel
    expect(mipLevels.blockOf({ level: 1, x: 0, y: 0 }, [[5]])).toEqual(
      Array.from({ length: 4 }, () => ({ level: 0, x: 0, y: 0 })),
    );
  });
});

describe('the fixed step', () => {
  const STEP = 1 / 60;

  it('runs a step for each frame a step long, ending with the frame', () => {
    const program = fixedStep.createProgram({ step: STEP });
    for (let i = 0; i < 4; i++) {
      const ran = program.frame(i * STEP, (i + 1) * STEP);
      expect(ran.steps).toHaveLength(1);
      expect(ran.steps[0]).toBeCloseTo((i + 1) * STEP, 9);
      expect(ran.capped).toBe(false);
    }
  });

  it('runs maxSteps of a stall, carries as many again and drops the rest', () => {
    const program = fixedStep.createProgram({ step: STEP, maxSteps: 4 });
    const stall = program.frame(0, 0.15);
    expect(stall.steps.map((at) => at / STEP)).toEqual(
      [1, 2, 3, 4].map((n) => expect.closeTo(n, 6)),
    );
    expect(stall.capped).toBe(true);
    expect(stall.carried).toBeCloseTo(4 * STEP, 9);
    expect(stall.dropped).toBeCloseTo(8 * STEP, 9);
    expect(program.sim.debt).toBeCloseTo(4 * STEP, 9);

    // The next frame steps the carried time, which lies in the stall
    const next = program.frame(0.15, 0.15 + STEP);
    expect(next.steps.map((at) => at / STEP)).toEqual(
      [5, 6, 7, 8].map((n) => expect.closeTo(n, 6)),
    );
    expect(next.from).toBeCloseTo(stall.carried, 9);
  });

  it('keeps as much time owed as the clock has debt, and loses none it does not drop', () => {
    const program = fixedStep.createProgram({ step: 1 / 120, maxSteps: 3 });
    const lengths = [0.007, 0.016, 0.051, 0.004, 0.15, 0.009, 0.033, 0.017];
    let start = 0;
    let accounted = 0;
    for (let i = 0; i < 40; i++) {
      const end = start + lengths[i % lengths.length];
      const ran = program.frame(start, end);
      accounted += ran.steps.length * program.sim.step + (end - ran.dropped);
      expect(program.owed).toBeCloseTo(program.sim.debt, 9);
      start = end;
    }
    expect(accounted + program.owed).toBeCloseTo(start, 9);
  });
});

describe('the cadences', () => {
  const FRAME = 1 / 60;
  /** Asks a program about `count` frames of `FRAME` from `from`. */
  const run = (program: ReturnType<typeof cadences.createProgram>, count: number, from = 0) => {
    const ran = [];
    for (let i = 0; i < count; i++) {
      const end = from + (i + 1) * FRAME;
      ran.push({ end, ...program.frame(from + i * FRAME, end) });
    }
    return ran;
  };

  it('runs every nth frame by count, and in the frame closest to each tick by time', () => {
    const ran = run(cadences.createProgram({ every: 4, hz: 25 }), 30);
    expect(ran.flatMap(({ counted }, i) => (counted ? [i] : []))).toEqual([
      0, 4, 8, 12, 16, 20, 24, 28,
    ]);

    const runs = ran.flatMap(({ timed, end }, i) => (timed ? [{ i, end, ...timed }] : []));
    // At 60 fps, 25 Hz runs every 2.4 frames, in gaps of two and three
    expect(runs.slice(1).map(({ i }, n) => i - runs[n].i)).toEqual([
      2, 3, 2, 3, 2, 2, 3, 2, 3, 2, 2, 3,
    ]);
    for (const [n, tick] of runs.entries()) {
      expect(tick.at).toBeCloseTo(FRAME + n / 25, 9);
      expect(Math.abs(tick.at - tick.end)).toBeLessThanOrEqual(FRAME / 2 + 1e-9);
      expect(tick.missed).toEqual([]);
    }
  });

  it('gives up the ticks a stall covers and starts the schedule again at its end', () => {
    const program = cadences.createProgram({ every: 4, hz: 25 });
    run(program, 3);
    // The last run answered the tick at FRAME + 1 / 25, so 0.0967 s is next
    const stall = program.frame(3 * FRAME, 3 * FRAME + 0.15);
    expect(stall.timed?.at).toBeCloseTo(3 * FRAME + 0.15, 9);
    expect(stall.timed?.missed).toEqual(
      [0.0967, 0.1367, 0.1767].map((at) => expect.closeTo(at, 3)),
    );

    const after = 3 * FRAME + 0.15;
    const ran = run(program, 6, after);
    expect(ran.find(({ timed }) => timed)?.timed?.at).toBeCloseTo(after + 1 / 25, 9);
  });

  it('runs both in the frame after reset(), and starts the schedule there', () => {
    const program = cadences.createProgram({ every: 4, hz: 25 });
    run(program, 2);
    program.reset();
    const ran = program.frame(2 * FRAME, 3 * FRAME);
    expect(ran.counted).toBe(true);
    expect(ran.timed).toEqual({ at: expect.closeTo(3 * FRAME, 9), missed: [] });
  });
});

describe('the depth precision', () => {
  const program = depthPrecision.createProgram({ near: 0.1, far: 1000 });

  it('projects the near plane to 0 and the far plane to 1, and the reverse when reversed', () => {
    expect(program.depth('standard', 0.1)).toBeCloseTo(0, 6);
    expect(program.depth('standard', 1000)).toBeCloseTo(1, 6);
    expect(program.depth('reversed', 0.1)).toBeCloseTo(1, 6);
    expect(program.depth('reversed', 1000)).toBeCloseTo(0, 6);
  });

  it('steps standard depth by the float spacing below 1, over the slope of its projection', () => {
    // At 400 m standard depth lies in [0.5, 1), where floats are 2^-24 apart
    const slope = (0.1 * 1000) / (999.9 * 400 ** 2);
    expect(program.step('standard', 400) / (2 ** -24 / slope)).toBeCloseTo(1, 2);
  });

  it('tells far surfaces apart under reversed depth by orders of magnitude finer gaps', () => {
    expect(program.step('standard', 800) / program.step('reversed', 800)).toBeGreaterThan(1000);
    expect(program.apart('standard', 400, 0.02)).toBe(false);
    expect(program.apart('reversed', 400, 0.02)).toBe(true);
  });

  it('crowds standard depth at the eye and spreads reversed depth to the far plane', () => {
    const [standard, reversed] = depthPrecision.KINDS.map(
      (kind) => program.ticks(kind, 1_000_000).filter((at) => at > 10).length,
    );
    expect(standard).toBe(0);
    expect(reversed).toBeGreaterThan(500);
  });
});

/** Steps `count` jitter cycles of the TAA samples, calling `each` after every frame. */
const cycles = (program: taaSamples.Program, count: number, each?: () => void) => {
  for (let i = 0; i < count * program.phases; i++) {
    program.step();
    each?.();
  }
};

describe('the TAA samples', () => {
  const { createProgram, shade, THROWN, WIDTH } = taaSamples;

  it("takes the kit's 8 jitter phases, and 18 drawn at two thirds", () => {
    expect(createProgram().phases).toBe(8);
    expect(createProgram({ scaled: true })).toMatchObject({
      phases: 18,
      drawn: { width: 6, height: 4 },
    });
  });

  it('resolves a pixel that half its samples cover to a half over a cycle', () => {
    // Four of the eight samples in each pixel of column 4 lie left of 4.55
    const program = createProgram({ edge: { at: 4.55, slope: 0 } });
    cycles(program, 10);
    const sums = [0, 0, 0];
    cycles(program, 1, () => {
      for (let k = 0; k < 3; k++) sums[k]! += shade(program.resolved[2 * WIDTH + 3 + k]!);
    });
    expect(sums.map((sum) => sum / program.phases)).toEqual([
      expect.closeTo(1, 3),
      expect.closeTo(0.5, 3),
      expect.closeTo(0, 3),
    ]);
  });

  it('keeps history at rest, and throws out what a moved edge leaves stale', () => {
    const program = createProgram();
    cycles(program, 4);
    let most = 0;
    cycles(program, 1, () => (most = Math.max(most, ...program.rejected)));
    expect(most).toBeLessThan(THROWN);

    program.edge.at += 2;
    program.step();
    const thrown = [...program.rejected].filter((cut) => cut >= THROWN);
    expect(thrown.length).toBeGreaterThan(4);
    // A pixel the edge now covers was dark: unclamped, one frame's blend would lift it to 0.1
    expect(shade(program.resolved[4]!)).toBeGreaterThan(0.3);
  });
});

describe('the ambient occlusion horizons', () => {
  const { createProgram } = aoHorizons;

  it('leaves open ground its whole sky', () => {
    const flat = createProgram({
      profile: [
        [0, 0],
        [3, 0],
      ],
    });
    expect(flat.trace(60).visibility).toBeCloseTo(1, 6);
    expect(flat.kept(60)).toBeCloseTo(1, 6);
  });

  it('takes about half the sky at the foot of a wall within reach, with any jitter', () => {
    const wall = createProgram({
      profile: [
        [0, 0],
        [1.5, 0],
        [1.5, 0.25],
        [3, 0.25],
      ],
    });
    for (const jitter of [0, 0.5, 1]) expect(wall.trace(59, jitter).visibility).toBeCloseTo(0.5, 1);
    expect(wall.trace(59).left.highest).toBeUndefined();
    expect(wall.trace(59).right.steps).toHaveLength(3);
  });

  it('narrows the crease, and closes the gap until the radius is smaller than its walls', () => {
    const program = createProgram();
    expect(program.trace(31).visibility).toBeLessThan(0.4);
    expect(program.kept(84)).toBeLessThan(0.4);
    expect(createProgram({ settings: { radius: 0.15 } }).kept(84)).toBe(1);
  });
});
