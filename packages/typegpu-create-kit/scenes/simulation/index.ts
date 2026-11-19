import * as web from '@typegpu/kit/web';
import { createScene } from './scene.ts';

const canvas = document.querySelector('canvas')!;
web.host(createScene, {
  canvas,
  attach: ({ scene }) => web.orbitControls(scene.view, canvas),
});
