import { coroutines, frames } from '@typegpu/kit';
import { d } from 'typegpu';
import { afterEach, expect, it, vi } from 'vitest';
import { CLEAR_TIME, createBlocks } from '../src/blockfall/blocks.ts';
import { CELLS, createBlockfall, PIECE } from '../src/blockfall/game.ts';
import { createScene, type SceneOptions } from '../src/blockfall/scene.ts';
import { BLOOM, compute, describeExample, pass } from './utils/examples.ts';

afterEach(() => {
  vi.restoreAllMocks();
});

/** The blocks, the plinth and the stone chips, drawn in each pass that shapes or shades them. */
const SOLIDS = ['draw-indexed', 'draw-indexed', 'draw-indexed'];

/**
 * A frame's events: in one compute pass, the steps and live lists of the embers and the dust, and
 * any bursts they spawn; then the depth prepass of the blocks, the plinth and the chips, the
 * occlusion's two passes, the same three in the fill's shadow and in the key's, then lit, with the
 * backdrop, the well, the ghosts, the embers and the dust, the bloom's six levels down and five up,
 * and the final pass. The blocks are listed on the CPU.
 */
const frame = (steps: number, bursts = 0) => [
  ...compute(2 * steps + bursts + 4),
  ...pass(...SOLIDS),
  ...pass('draw'),
  ...pass('draw'),
  ...pass(...SOLIDS),
  ...pass(...SOLIDS),
  ...pass(...SOLIDS, 'draw', 'draw', 'draw-indexed', 'draw-indirect', 'draw-indirect'),
  ...BLOOM,
  ...pass('draw'),
  'submit',
];

describeExample(
  'blockfall',
  createScene,
  (build) => {
    /** The scene, with the lines of each score it hears. */
    const setup = async (options: SceneOptions = {}) => {
      const lines: number[] = [];
      const built = await build({ onScore: (score) => lines.push(score.lines), ...options });
      return { ...built, lines };
    };

    it('steps the embers and the dust', async () => {
      const { log, next } = await setup();
      expect(log.shaders.filter((code) => code.includes('@vertex'))).toHaveLength(19);

      // The mesh pool bakes each of its three shapes once, as the scene is made.
      next();
      next();
      expect(log.events).toEqual([
        ...Array.from({ length: 3 }, () => [...compute(), 'submit']).flat(),
        ...frame(0, 1),
        ...frame(2),
      ]);
    });

    it('moves the piece along the grid the camera faces, and turns the camera a quarter', async () => {
      const { scene } = await setup();
      const moves: number[][] = [];
      vi.spyOn(scene.game, 'move').mockImplementation((...by) => void moves.push(by));
      const { controls } = scene.ui;
      expect(scene.ui.score()).toBe('0\nlines 0 · level 1');

      const yaw = scene.view.goal.yaw;
      controls.right.press();
      controls.away.press();
      controls.turnRight.press();
      expect(scene.view.goal.yaw).toBeCloseTo(yaw + Math.PI / 2);
      controls.right.press();

      // Right and away are perpendicular, and after a quarter turn right points where away did.
      // Adding 0 turns -0 into 0
      const [right, away, turned] = moves.map(([x = 0, z = 0]) => [x + 0, z + 0]);
      expect(Math.abs(right![0] * away![0] + right![1] * away![1])).toBe(0);
      expect(turned).toEqual(away);
    });

    it('writes the rows of the blocks that start to move, and none while they ease', async () => {
      const { log, scene, next } = await setup();
      const rowWrites = () =>
        log.writes.filter(({ buffer }) => buffer.label === 'Block_host').length;

      // The new piece starts to ease into the well from above it, and its ghosts to where it lands.
      next();
      expect(rowWrites()).toBe(2);
      next(0.02);
      next(0.02);
      expect(rowWrites()).toBe(2);

      // A move uploads the piece's rows and the ghosts' once, and they ease on the GPU.
      scene.game.move(1, 0);
      next(0.02);
      next(0.02);
      next(0.02);
      expect(rowWrites()).toBe(4);
    });

    it('clears a full layer once it has flashed, and breaks every block of it', async () => {
      // Every piece is the 2x2 square, which spawns over x 2..3, z 2..3.
      vi.spyOn(Math, 'random').mockReturnValue(1.5 / 7);
      const { root } = await build();
      const lines: number[] = [];
      const broken: number[][] = [];
      const scheduler = coroutines.create();
      const blocks = createBlocks(root, {
        time: root.createUniform(d.f32),
        capacity: CELLS + PIECE,
        ghosts: PIECE,
        plinth: { at: [0, 0, 0], size: [1, 1, 1], color: d.vec3f(0.5) },
        grid: [1, 1],
      });
      const game = createBlockfall({
        blocks,
        scheduler,
        onScore: (score) => lines.push(score.lines),
        onBreak: (...block) => broken.push(block),
      });
      const loop = frames.create(root, (now) => {
        game.update(now);
        scheduler.run(now);
      });

      let time = 0;
      const advance = (seconds: number) => {
        for (const end = time + seconds; time < end; time += 1 / 60) loop.run(time);
      };

      advance(0.1);
      for (const x of [-2, 0, 2]) {
        for (const z of [-2, 0, 2]) {
          game.move(x, 0);
          game.move(0, z);
          game.drop();
          advance(0.05);
        }
      }

      // Nine squares fill the bottom layer, which flashes and then goes.
      expect(broken).toHaveLength(0);

      advance(CLEAR_TIME + 0.1);
      expect(lines).toEqual([0, 1]);

      // Every block of the layer breaks, at height 0, as the square's kind.
      expect(broken).toHaveLength(36);
      expect(broken.every(([, y, , kind]) => y === 0 && kind === 1)).toBe(true);
    });
  },
  {
    // The bakes of the rounded box, the plinth and the chips' box; the blocks, chips, backdrop, well
    // and ghosts; the blocks and chips in the prepass and both shadows; the occlusion's two passes;
    // the spawn, step, draw and live lists of the embers and the dust; three bloom passes and the
    // final pass.
    pipelines: 30,
  },
);
