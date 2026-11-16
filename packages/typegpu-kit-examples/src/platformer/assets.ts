import { image } from '../assets.ts';

/** The sprite sheets, frames 32 texels square in a row. */
export const assets = {
  /** The hero's frames, facing right: idle, running, jumping, landing, turning and sliding. */
  hero: image('textures/wafflelad.webp'),
  /** The sponge's frames, facing left: lying still, blinking, then a wiggle. */
  sponge: image('textures/sponge.webp'),
  /** A bubble's frames, 8 texels square: whole, then popping. */
  bubble: image('textures/bubble.webp'),
};
