import { frames } from '@typegpu/kit';
import { offscreenSurface, recordingRoot } from '@typegpu/kit/testing';
import { describe, expect, it } from 'vitest';
import { extractContracts } from '../src/contracts/extract.ts';
import {
  counted,
  delta,
  overClaim,
  ownedBy,
  type Counted,
  type Fixture,
  type Fixtures,
  type Others,
} from './conformance/harness.ts';
import { kit } from './conformance/kit.ts';
import { fixtures as particles } from './conformance/particles.ts';
import { fixtures as finish } from './conformance/finish.ts';
import { fixtures as render } from './conformance/render.ts';
import { fixtures as text } from './conformance/text.ts';

// Runs every module a contract describes on `recordingRoot()`, and checks what its tags say against
// what it does: what it keeps against `@owns`, a steady frame against `@allocates`, its submits
// against `@submits`, and that `destroy()` frees what it made but its uniforms.

const { contracts } = extractContracts();
const fixtures: Fixtures = { ...kit, ...particles, ...finish, ...render, ...text };

/** Modules the checks cover: those that borrow a root, but tools and vendored copies. */
const checked = contracts.filter((contract) => contract.kind === 'module' && !contract.vendored);

/** `@owns` items the counts can't read, each as `module: item`, so a new one is a choice. */
const vague = [
  'kit/passes: the pipelines `add` builds',
  'kit/passes: until the drawing or description is destroyed',
  // Modules that make a passes description own what it builds
  'kit/lighting: the pipelines `add` builds',
  'kit/lighting: until the drawing or description is destroyed',
];

const settle = () => new Promise((resolve) => setTimeout(resolve, 0));

/** TypeGPU's own catch-all groups, which it makes and caches for a pipeline's loose resources. */
const automatic = (label: string | undefined) => label?.endsWith('Automatic Bind Group & Layout');

/** What `after` lists beyond `before`, one for one, since unnamed objects share a name. */
const since = (before: readonly string[], after: readonly string[]) => {
  const left = [...before];
  return after.filter((each) => {
    const at = left.indexOf(each);
    if (at >= 0) left.splice(at, 1);
    return at < 0;
  });
};

/** `[Buffer "label"]` and the like, by the counts' key. */
const kindOf = (each: string): Counted =>
  each.startsWith('[Buffer') ? 'buffers' : each.startsWith('[Texture') ? 'textures' : 'querySets';

async function measure(fixture: Fixture) {
  const { root, log } = recordingRoot(fixture.features ? { features: fixture.features } : {});
  const surface = offscreenSurface(root, { width: 64, height: 36, format: 'bgra8unorm' });
  const built = await fixture.setup(root, surface);

  // What the counts leave out of the module's: TypeGPU's automatic groups, and what `others` makes.
  const excluded: Partial<Record<Counted, number>> = {};
  const exclude = (key: Counted, n: number) => (excluded[key] = (excluded[key] ?? 0) + n);
  const createBindGroup = root.device.createBindGroup.bind(root.device);
  root.device.createBindGroup = (descriptor) => {
    if (automatic(descriptor.label)) exclude('bindGroups', 1);
    return createBindGroup(descriptor);
  };
  const others: Others = (make) => {
    const from = log.counts();
    const result = make();
    const after = () => {
      for (const [key, n] of Object.entries(delta(log.counts(), from))) {
        if (n > 0 && (counted as readonly string[]).includes(key)) exclude(key as Counted, n);
      }
    };
    if (result instanceof Promise) return result.finally(after) as typeof result;
    after();
    return result;
  };

  const before = log.counts();
  const alive = log.alive();
  const submitted = log.submissions;
  const instance = await built.create(others);
  const created = log.submissions - submitted;

  const loop = frames.create(root, (frame) => instance.frame(frame));
  let time = 0;
  const run = async (count: number) => {
    for (let i = 0; i < count; i++) {
      loop.run(time);
      time += 1 / 60;
      await settle();
    }
  };

  await run(3);
  // What it keeps: the buffers, textures and query sets still alive, and every sampler, pipeline
  // and bind group made, which WebGPU can't destroy.
  const kept = since(alive, log.alive());
  const made = delta(log.counts(), before);
  const owned = Object.fromEntries(
    counted.map((key) => [
      key,
      ['buffers', 'textures', 'querySets'].includes(key)
        ? kept.filter((each) => kindOf(each) === key).length - (excluded[key] ?? 0)
        : made[key] - (excluded[key] ?? 0),
    ]),
  ) as Record<Counted, number>;

  const steadyFrom = log.counts();
  const automaticFrom = excluded.bindGroups ?? 0;
  const submittedFrom = log.submissions;
  await run(3);
  const grown = delta(log.counts(), steadyFrom);
  const steady = {
    ...grown,
    bindGroups: grown.bindGroups - ((excluded.bindGroups ?? 0) - automaticFrom),
  };
  const steadySubmits = log.submissions - submittedFrom - 3;

  instance.destroy?.();
  loop.destroy();
  return {
    owned,
    kept,
    steady,
    created,
    steadySubmits,
    // Uniforms are left to the garbage collector
    leaked: since(alive, log.alive({ uniforms: false })),
  };
}

describe('module contracts, run', () => {
  it('have a fixture for every module that borrows a root', () => {
    const missing = checked.map((contract) => contract.id).filter((id) => !(id in fixtures));
    expect(missing).toEqual([]);
  });

  it('count every `@owns` item but the few listed', () => {
    const found = checked.flatMap((contract) =>
      ownedBy(contract, contracts).vague.map((item) => `${contract.id}: ${item}`),
    );
    expect(found.toSorted()).toEqual(vague.toSorted());
  });

  for (const contract of checked) {
    const fixture = fixtures[contract.id];
    if (!fixture) continue;
    if ('skip' in fixture) {
      it.skip(`${contract.id}: ${fixture.skip}`, () => {});
      continue;
    }

    it(`${contract.id} does what its tags say`, async () => {
      const result = await measure(fixture);
      const owns = ownedBy(contract, contracts);
      // Bind groups count only where `@owns` names them: TypeGPU makes and caches its own.
      const kinds = counted.filter((key) => key !== 'bindGroups' || key in owns.ranges);

      expect(
        overClaim(owns, result.owned, kinds),
        `@owns; alive: ${result.kept.join(', ')}`,
      ).toEqual([]);
      if (contract.allocates === undefined) {
        expect(result.steady, 'a steady frame allocates nothing').toEqual(
          Object.fromEntries(Object.keys(result.steady).map((key) => [key, 0])),
        );
      }
      if (contract.submits === undefined) {
        expect(result.created, 'no submit while created').toBe(0);
        expect(result.steadySubmits, 'no submit of its own in a frame').toBe(0);
      } else if (contract.submits.when === 'setup') {
        expect(result.steadySubmits, 'no submit of its own in a steady frame').toBe(0);
      }
      if (contract.lifecycle.destroy)
        expect(result.leaked, 'destroy() frees what it made').toEqual([]);
    });
  }
});
