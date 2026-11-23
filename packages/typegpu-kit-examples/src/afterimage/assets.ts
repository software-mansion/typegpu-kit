import { glb } from '../assets.ts';
import { PARTS } from './arcade.ts';

/** The hall and its pendulum, one Draco-compressed file made in Blender for the kit. */
export const assets = { model: glb('models/arcade.glb', { parts: PARTS, draco: true }) };
