import * as web from '@typegpu/kit/web';
import { createScene } from './scene.ts';

web.host(createScene, { canvas: document.querySelector('canvas')! });
