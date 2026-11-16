import { buttons, sounds, text } from '../ui.ts';
import type { Sound } from './game.ts';

const file = (sound: Sound) => `sounds/${sound}.wav`;

export const ui = {
  score: text({ corner: 'top-right' }),
  sounds: sounds({
    jump: file('jump'),
    backflip: file('backflip'),
    slide: file('slide'),
    coin: file('coin'),
    hurt: file('hurt'),
    grass1: file('grass1'),
    grass2: file('grass2'),
    grass3: file('grass3'),
    grass4: file('grass4'),
    fall: file('fall'),
    heel: file('heel'),
    peppergrind: file('peppergrind'),
    fork_jump: file('fork_jump'),
    fork_land: file('fork_land'),
    fork_death: file('fork_death'),
    sponge_walk: file('sponge_walk'),
    death_layer: file('death_layer'),
    sponge_death1: file('sponge_death1'),
    sponge_death2: file('sponge_death2'),
  } satisfies Record<Sound, string>),
  controls: buttons(
    {
      left: { label: '◀', name: 'left', keys: ['ArrowLeft', 'KeyA'], held: true },
      right: { label: '▶', name: 'right', keys: ['ArrowRight', 'KeyD'], held: true },
      down: {
        label: '▼',
        name: 'duck',
        keys: ['ArrowDown', 'KeyS', 'ShiftLeft', 'ShiftRight'],
        held: true,
      },
      jump: { label: 'jump', keys: ['ArrowUp', 'KeyW', 'Space'], held: true },
    },
    { left: [['left', 'down', 'right']], right: [['jump']] },
  ),
};
