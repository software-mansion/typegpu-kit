import { d, type TgpuRoot } from 'typegpu';
import { table } from '../../src/index.ts';

declare const root: TgpuRoot;

const Enemy = table.define('Enemy', { device: { position: d.vec3f } });
const Shot = table.define('Shot', { device: { position: d.vec3f } });
const enemies = table.create(root, Enemy, { capacity: 8 });
const shots = table.create(root, Shot, { capacity: 8 });

// A row id names its definition
enemies.remove(shots.add());
