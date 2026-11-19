import { buttons, text } from '../ui.ts';

export const ui = {
  score: text({ corner: 'top-right' }),
  // Dragging the board orbits it, so the pads leave out the turns
  controls: buttons(
    {
      away: { label: '▲', name: 'away', keys: ['ArrowUp'] },
      near: { label: '▼', name: 'near', keys: ['ArrowDown'] },
      left: { label: '◀', name: 'left', keys: ['ArrowLeft'] },
      right: { label: '▶', name: 'right', keys: ['ArrowRight'] },
      spin: { label: 'spin', keys: ['KeyZ'] },
      roll: { label: 'roll', keys: ['KeyX'] },
      tip: { label: 'tip', keys: ['KeyC'] },
      drop: { label: 'drop', keys: ['Space'] },
      turnLeft: { label: 'Q', name: 'turn left', keys: ['KeyQ'] },
      turnRight: { label: 'E', name: 'turn right', keys: ['KeyE'] },
    },
    {
      left: [
        [null, 'away', null],
        ['left', 'near', 'right'],
      ],
      right: [['spin', 'roll', 'tip'], ['drop']],
    },
  ),
};
