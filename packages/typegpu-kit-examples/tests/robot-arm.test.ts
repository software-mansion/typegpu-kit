import { expect, it } from 'vitest';
import { mat4, vec3, type Vec3Arg } from 'wgpu-matrix';
import { createArm, JOINTS, TIP, WRIST } from '../src/robot-arm/arm.ts';
import { createBlocks, HOVER, type Block } from '../src/robot-arm/blocks.ts';
import { createScene } from '../src/robot-arm/scene.ts';
import { BLOOM, describeExample, pass } from './utils/examples.ts';

/**
 * A draw for each mesh of each instance set: the arm's nine shapes, the blocks, the bench top, the
 * pegboard and the five meshes of the lamp, tools and shelf.
 */
const meshes = Array<string>(9 + 1 + 1 + 1 + 5).fill('draw-indexed');

/**
 * A frame's events: the lamp's shadow, the depth prepass, the occlusion's trace and denoise, the
 * scene, the bloom, the final pass and FXAA.
 */
const frame = [
  ...pass(...meshes),
  ...pass(...meshes),
  ...pass('draw'),
  ...pass('draw'),
  ...pass(...meshes),
  ...BLOOM,
  ...pass('draw'),
  ...pass('draw'),
  'submit',
];

/** A gripper whose fingertips jump to where the blocks send them, pointing straight down. */
const jumping = () => {
  const tip = vec3.create(2, 1, 0);
  const wrist = mat4.identity();
  const follow = (target: Vec3Arg) => {
    vec3.copy(target, tip);
    mat4.translation([target[0], target[1] + TIP[1], target[2]], wrist);
    mat4.rotateX(wrist, Math.PI, wrist);
  };
  follow(tip);
  return { tip, wrist, follow };
};

/** A 0.4 cube of `colour` with its footprint centred at x, z and its bottom at y. */
const cube = (colour: number, x: number, y: number, z: number): Block => {
  const size = 0.4;
  return { colour, size, height: size, x, y, z, yaw: 0 };
};

/** Where a block stands, to four decimals. */
const standing = ({ x, y, z }: Block) => [x, y, z].map((value) => Math.round(value * 1e4) / 1e4);

describeExample(
  'robot-arm',
  createScene,
  (build) => {
    it('draws the shadow, the occlusion and the scene, blooms, finishes and smooths', async () => {
      const { log, next } = await build();
      next();
      const loaded = log.events.length;
      next();
      expect(log.events.slice(loaded)).toEqual(frame);
    });

    it('picks up a block, lets its stack fall and sets it down apart from the rest', () => {
      const stack: Block[] = [
        { colour: 0, size: 0.6, height: 0.4, x: 1.8, y: 0, z: 0, yaw: 0 },
        { colour: 0, size: 0.5, height: 0.3, x: 1.8, y: 1, z: 0, yaw: 0 },
        { colour: 0, size: 0.5, height: 0.4, x: -1.8, y: 0, z: 0, yaw: 0 },
        { colour: 0, size: 0.6, height: 0.4, x: -1.8, y: 0, z: 0.1, yaw: 0 },
      ];
      const [bottom, top, left, right] = stack as [Block, Block, Block, Block];
      const gripper = jumping();
      const blocks = createBlocks(stack, gripper, [[0, 1.6]]);
      const run = (seconds: number) => {
        for (let t = 0; t < seconds; t += 1 / 60) {
          blocks.step(1 / 60, [2, 0]);
          gripper.follow(blocks.target);
        }
      };

      // A block falls onto the one under it, and blocks side by side are pushed apart
      expect(top.y).toBeCloseTo(bottom.height);
      expect(Math.hypot(left.x - right.x, left.z - right.z)).toBeGreaterThanOrEqual(0.66 - 1e-6);

      // A tap from above finds the top of the stack
      expect(blocks.hit([bottom.x, 5, bottom.z], [0, -1, 0])).toBe(1);
      run(0.1);
      blocks.grab(0);
      run(2);
      expect(blocks.holding).toBe(true);
      expect(bottom.y).toBeGreaterThan(0.3);
      expect(top.y).toBe(0);

      const spot = [Math.cos(0.75) * 1.8, Math.sin(0.75) * 1.8] as const;
      blocks.drop(spot[0], spot[1]);
      run(2);
      expect(blocks.holding).toBe(false);
      expect(Math.hypot(bottom.x - spot[0], bottom.z - spot[1])).toBeLessThan(0.1);
      expect(bottom.y).toBe(0);
    });

    it('hovers over the towers while steered and dips once for each grab or drop', () => {
      const block = cube(0, 1.8, 0, 0);
      const gripper = jumping();
      const blocks = createBlocks([block], gripper, [[0, 1.6]]);
      /** Steps `seconds` with the pointer at bench point 2, 0.5 and counts the target's dips. */
      const dips = (seconds: number) => {
        let count = 0;
        let low = false;
        for (let t = 0; t < seconds; t += 1 / 60) {
          blocks.step(1 / 60, [2, 0.5]);
          gripper.follow(blocks.target);
          if (!low && blocks.target[1] < HOVER) count++;
          low = blocks.target[1] < HOVER;
        }
        return count;
      };

      expect(dips(1)).toBe(0);
      expect([...blocks.target]).toEqual([2, HOVER, 0.5]);
      // A press on the bench with nothing held leaves the hand where it is
      blocks.drop(1.8, 0);
      expect(dips(1)).toBe(0);

      blocks.grab(0);
      expect(dips(2)).toBe(1);
      expect(blocks.holding).toBe(true);
      blocks.drop(0, 1.8);
      expect(dips(2)).toBe(1);
      expect(blocks.holding).toBe(false);
      expect(blocks.target[1]).toBe(HOVER);
    });

    it('sorts each loose block onto its colour’s tower, then scatters the tops to sort again', () => {
      const homes = [
        [1.6, 0],
        [-1.6, 0],
      ] as const;
      // A red tower with a blue block on top, and a loose red block
      const stack = [cube(0, 1.6, 0, 0), cube(1, 1.6, 0.4, 0), cube(0, 0, 0, 1.6)];
      const [tower, blue, red] = stack as [Block, Block, Block];
      const gripper = jumping();
      const blocks = createBlocks(stack, gripper, homes);
      /** Steps attract mode until `done` holds, for at most ten seconds. */
      const until = (done: () => boolean) => {
        for (let t = 0; t < 10 && !done(); t += 1 / 60) {
          blocks.auto();
          blocks.step(1 / 60, [0, -2]);
          gripper.follow(blocks.target);
        }
        return done();
      };
      const sorted = () =>
        standing(blue).join() === '-1.6,0,0' && standing(red).join() === '1.6,0.4,0';
      const scattered = () => stack.some(({ x, y }) => y === 0 && Math.abs(x) < 0.5);

      expect(until(sorted)).toBe(true);
      expect(standing(tower)).toEqual([1.6, 0, 0]);

      // Sorted, it sets a tower's top on the bench between the homes, then sorts it back
      expect(until(scattered)).toBe(true);
      expect(until(sorted)).toBe(true);
    });

    it('brings the fingers down onto a goal in reach, within each joint’s range', () => {
      const arm = createArm();

      for (const goal of [
        [2, 0.3, 0],
        [-1.5, 0.3, -1.5],
        [0.8, 0.3, 0.2],
        [0.4, 0.3, 2.6],
      ] as const) {
        // A second of frames.
        for (let step = 0; step < 60; step++) {
          arm.solve([goal[0], goal[1] + TIP[1], goal[2]], WRIST, 4);
          arm.aim(WRIST, [0, -1, 0]);
        }

        const miss = Math.hypot(...goal.map((at, k) => at - arm.tip[k]));
        expect(miss).toBeLessThan(0.01);
        // The wrist's frame points its +y, towards the tip, straight down.
        expect(arm.frames[WRIST][5]).toBeCloseTo(-1, 3);
        for (const [i, angle] of arm.angles.entries()) {
          const [low, high] = JOINTS[i].range;
          expect(angle).toBeGreaterThanOrEqual(low);
          expect(angle).toBeLessThanOrEqual(high);
        }
      }
    });
  },
  {
    // A bake for each of the pool's fifteen meshes, the colour, lamp-shadow and prepass pipelines
    // of the five instance sets, the occlusion's two passes, three bloom passes, the final and FXAA
    pipelines: 15 + 5 * 3 + 2 + 3 + 1 + 1,
    gap: 0.05,
  },
);
