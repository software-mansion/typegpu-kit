import {
  font,
  instances,
  picker,
  poses,
  text,
  type Baked,
  type CameraBase,
  type Picker,
  type Surface,
} from '@typegpu/kit';
import { d, std, tgpu, type TgpuComputePass, type TgpuRoot } from 'typegpu';
import { mat4 } from 'wgpu-matrix';
import type { Model } from '../models.ts';
import { CAPACITY, HORSE_SCALE, type Herd } from './herd.ts';
import type { Posed } from './horses.ts';

/** Picker target at the body centre in model-bound coordinates. */
const BODY_MIDDLE = [0, -0.006, 0.03] as const;
const PICK_RADIUS_METRES = 1.6;
export const PICK_REACH_POINTS = 3;

/** Horse names, each shared by a group of indices. */
const NAMES = `Comet Juniper Saffron Biscuit Thistle Maverick Clover Dune Ember Willow Pepper Sable
  Tumbleweed Marigold Rook Sorrel Nutmeg Gale Bramble Cinder Honey Flint Mistral Pippin Ochre Quill
  Rosemary Sundance Tansy Umber Vesper Wren`.split(/\s+/);
const LONGEST = Math.max(...NAMES.map((name) => name.length));
const EM_POINTS = 26;
const LABEL_METRES = 2.5;
const INK = d.vec3f(1, 0.9, 0.7);
const OUTLINE_PIXELS = 0.8;
const OUTLINE = d.vec3f(0.03, 0.02, 0.015);

interface PickingOptions {
  readonly model: Model;
  readonly herd: Herd;
  readonly posed: Posed;
  readonly surface: Surface;
}

/** The joint of the horse's skin bound nearest `BODY_MIDDLE`. */
function middleJointOf(model: Model) {
  let middle = { joint: 0, distance: Infinity };
  for (const [joint, { inverseBind }] of model.skin!.joints.entries()) {
    const [x, y, z] = mat4.getTranslation(mat4.inverse(inverseBind));
    const distance =
      (x! - BODY_MIDDLE[0]) ** 2 + (y! - BODY_MIDDLE[1]) ** 2 + (z! - BODY_MIDDLE[2]) ** 2;
    if (distance < middle.distance) middle = { joint, distance };
  }
  return middle.joint;
}

/** Body-sized picking target centred on a moving body joint. */
export function createPicking(root: TgpuRoot, view: CameraBase, options: PickingOptions) {
  const { model, herd, posed, surface } = options;
  const middleJoint = middleJointOf(model);
  return picker.create(root, view, {
    capacity: CAPACITY,
    size: surface,
    rank: picker.byEye,
    settings: { reach: PICK_REACH_POINTS },
    alive: (member) => {
      'use gpu';
      return member < herd.controls.$.count;
    },
    position: tgpu
      .fn((member: number) => {
        'use gpu';
        const horse = herd.horses.$[member];
        const at = poses.jointPosition(horse.pose, middleJoint);
        const facing = instances.toward(d.vec3f(horse.facing.x, 0, horse.facing.y));
        const turned = instances.rotate(facing, at);
        return herd.drawnAt(member) + turned * (herd.sizeOf(member) * HORSE_SCALE);
      })
      .pipe(posed.poses.inject()),
    radius: (member) => {
      'use gpu';
      return PICK_RADIUS_METRES * herd.sizeOf(member);
    },
  });
}

/** Fills a glyph with its colour inside a dark outline. */
const outlined = (fragment: text.Fragment) => {
  'use gpu';
  const fill = std.saturate(fragment.distance + 0.5);
  const edge = std.saturate(fragment.round + OUTLINE_PIXELS + 0.5);
  return d.vec4f(
    std.mix(OUTLINE, fragment.color.xyz, fill) * edge * fragment.color.w,
    edge * fragment.color.w,
  );
};

interface LabelOptions {
  readonly font: Baked;
  readonly picked: Picker;
  readonly herd: Herd;
}

/**
 * Writes the hovered horse's label on the GPU in the same frame as picking, copying its name from
 * a set laid out once and never drawn.
 */
export function createLabel(root: TgpuRoot, view: CameraBase, options: LabelOptions) {
  const { font: baked, picked, herd } = options;
  const sans = font.create(root, baked);
  const names = text.create(root, sans, { capacity: NAMES.length, length: LONGEST });
  for (const [i, name] of NAMES.entries()) {
    names.write(i, name, { em: EM_POINTS, baseline: 'alphabetic' });
  }
  const label = text.create(root, sans, {
    camera: view,
    capacity: 1,
    length: LONGEST,
    place: text.pinned,
    shade: outlined,
  });
  const writeName = tgpu.computeFn({ workgroupSize: [1] })(() => {
    'use gpu';
    const member = picked.found();
    if (member === picker.none) {
      label.erase(0);
      return;
    }
    for (let k = d.u32(0); k < LONGEST; k++)
      label.setQuad(0, k, names.quadAt(member % NAMES.length, k));
    label.setLabel(
      0,
      text.Label({
        position: herd.drawnAt(member) + d.vec3f(0, LABEL_METRES * herd.sizeOf(member), 0),
        color: d.vec4f(INK, 1),
      }),
    );
  });
  const writing = root.createComputePipeline({ compute: writeName });
  return {
    drawing: label.drawing,
    initAsync: () => writing.initAsync(),
    write(pass: TgpuComputePass) {
      pass.setPipeline(writing);
      pass.dispatchWorkgroups(1);
    },

    destroy() {
      label.destroy();
      names.destroy();
      sans.destroy();
    },
  };
}
