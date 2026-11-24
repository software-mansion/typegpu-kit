import { gpuRoot } from '@typegpu/kit/testing/dawn';
import { d } from 'typegpu';
import { describe, expect, it } from 'vitest';
import { backends, passes } from '../../src/index.ts';
import { createCulling } from '../../src/render/culling.ts';

const MESHES = 4;
const CAPACITY = 1000;

/**
 * 900 rows over several workgroups: mesh 0 lists rows 0 to 599, mesh 1 is drawn but has none, mesh
 * 2 lists rows 600 to 899 backwards, and mesh 3 was never drawn. The camera sees rows not divisible
 * by 3.
 */
async function setup(
  firstInstance: boolean,
  layers = 1,
  seen = (row: number) => {
    'use gpu';
    return row % 3 !== 0;
  },
  active = (_layer: number) => true,
  compute: 'gpu' | 'cpu' = 'gpu',
) {
  const root = await gpuRoot({ features: firstInstance ? ['indirect-first-instance'] : [] });
  // A device that says its compute runs on the CPU gets the walk, which a GPU runs just as well
  if (compute === 'cpu') {
    Object.assign(root.device, {
      [backends.brand]: { ...backends.of(root), compute } satisfies backends.Backend,
    });
  }
  const first = Array.from({ length: 600 }, (_, i) => i);
  const second = Array.from({ length: 300 }, (_, i) => 899 - i);
  const header = [MESHES, MESHES + 600, MESHES + 600, 0];
  const listing = [...header, ...first, ...second];

  const listed = root.createReadonly(d.arrayOf(d.u32, MESHES + CAPACITY), [
    ...listing,
    ...Array<number>(MESHES + CAPACITY - listing.length).fill(0),
  ]);
  const meshes = root.createReadonly(
    d.arrayOf(d.u32, CAPACITY),
    Array.from({ length: CAPACITY }, (_, row) => (row < 600 ? 0 : 2)),
  );
  const entries = root.createUniform(d.u32, first.length + second.length);

  const culling = createCulling(root, root, {
    meshes: MESHES,
    capacity: CAPACITY,
    layers,
    firstInstance,
    listed,
    entries,
    meshOf: (row) => {
      'use gpu';
      return meshes.$[row];
    },
    seen,
    active: (layer) => active(layer),
  });

  const frame = async () => {
    const encoder = root.createCommandEncoder();
    const pass = encoder.beginComputePass();
    culling.record(pass, first.length + second.length);
    pass.end();
    encoder.submit();
    return Promise.all([culling.culled.read(), culling.draws.read()]);
  };

  return { root, frame, seenFirst: first.filter((row) => row % 3 !== 0), second };
}

describe('culling', () => {
  it.each([true, false])(
    'lists the rows seen in the listing’s order, each mesh after the last, frame after frame (firstInstance %s)',
    async (firstInstance) => {
      const { root, frame, seenFirst, second } = await setup(firstInstance);
      const seenSecond = second.filter((row) => row % 3 !== 0);

      for (let i = 0; i < 2; i++) {
        const [culled, draws] = await frame();
        const start = MESHES + seenFirst.length;

        expect(culled.slice(MESHES, start)).toEqual(seenFirst);
        expect(culled.slice(start, start + seenSecond.length)).toEqual(seenSecond);
        // Each mesh's start and count; the empty mesh and the one never drawn keep their draws.
        expect([culled[0], culled[2]]).toEqual([MESHES, start]);
        expect([draws[1], draws[6], draws[11], draws[16]]).toEqual([400, 0, 200, 0]);
        expect([draws[4], draws[14]]).toEqual(firstInstance ? [MESHES, start] : [0, 0]);
      }

      root.destroy();
    },
  );
});

describe('culling into layers', () => {
  it.each([true, false])(
    'lists each layer’s rows after the lists of the layers before it (firstInstance %s)',
    async (firstInstance) => {
      // Layer 0 sees rows not divisible by 3, and layer 1 the even rows
      const { root, frame, seenFirst, second } = await setup(firstInstance, 2, (row) => {
        'use gpu';
        if (passes.layer.$ === 0) return row % 3 !== 0;
        return row % 2 === 0;
      });
      const span = MESHES + CAPACITY;
      const record = MESHES * 5;
      const evenFirst = Array.from({ length: 300 }, (_, i) => i * 2);
      const evenSecond = second.filter((row) => row % 2 === 0);

      for (let i = 0; i < 2; i++) {
        const [culled, draws] = await frame();
        const start = MESHES + seenFirst.length;
        expect(culled.slice(MESHES, start)).toEqual(seenFirst);
        expect([draws[1], draws[11]]).toEqual([400, 200]);

        const at = span + MESHES;
        expect(culled.slice(at, at + evenFirst.length)).toEqual(evenFirst);
        expect(culled.slice(at + 300, at + 300 + evenSecond.length)).toEqual(evenSecond);
        // Start offsets index the whole buffer, so a draw needs no layer to find its rows.
        expect([culled[span], culled[span + 2]]).toEqual([at, at + 300]);
        expect([draws[record + 1], draws[record + 11]]).toEqual([300, 150]);
        expect([draws[record + 4], draws[record + 14]]).toEqual(
          firstInstance ? [at, at + 300] : [0, 0],
        );
      }

      root.destroy();
    },
  );
});

// Layer 0 sees rows not divisible by 3, layer 1 every row, and layer 2 the even rows
const layered = (row: number) => {
  'use gpu';
  if (passes.layer.$ === 0) return row % 3 !== 0;
  if (passes.layer.$ === 1) return true;
  return row % 2 === 0;
};

describe('culling active layers', () => {
  it('culls only the active layers, each into its own list, and every layer once all are', async () => {
    let skipped = 1;
    const { root, frame } = await setup(true, 3, layered, (layer) => layer !== skipped);
    const span = MESHES + CAPACITY;
    const record = MESHES * 5;

    let [culled, draws] = await frame();
    expect([draws[1], draws[11]]).toEqual([400, 200]);
    expect([draws[record + 1], draws[record + 11]]).toEqual([0, 0]);
    expect([draws[2 * record + 1], draws[2 * record + 11]]).toEqual([300, 150]);
    expect(culled[2 * span]).toBe(2 * span + MESHES);

    skipped = -1;
    [culled, draws] = await frame();
    expect([draws[record + 1], draws[record + 11]]).toEqual([600, 300]);
    expect([draws[2 * record + 1], draws[2 * record + 11]]).toEqual([300, 150]);

    root.destroy();
  });
});

describe('culling where compute runs on the CPU', () => {
  it.each([true, false])(
    'walks each layer into the lists and draws the scan writes (firstInstance %s)',
    async (firstInstance) => {
      let skipped = 1;
      const active = (layer: number) => layer !== skipped;
      const scanned = await setup(firstInstance, 3, layered, active);
      const walked = await setup(firstInstance, 3, layered, active, 'cpu');
      const span = MESHES + CAPACITY;

      for (let i = 0; i < 3; i++) {
        if (i === 2) skipped = -1;
        const [expectedCulled, expectedDraws] = await scanned.frame();
        const [culled, draws] = await walked.frame();

        expect(draws).toEqual(expectedDraws);
        for (let k = 0; k < 3; k++) {
          if (!active(k)) continue;
          const seenInLayer =
            (expectedDraws[k * MESHES * 5 + 1] ?? 0) + (expectedDraws[k * MESHES * 5 + 11] ?? 0);
          const end = k * span + MESHES + seenInLayer;
          expect(culled.slice(k * span, end)).toEqual(expectedCulled.slice(k * span, end));
        }
      }

      scanned.root.destroy();
      walked.root.destroy();
    },
  );
});
