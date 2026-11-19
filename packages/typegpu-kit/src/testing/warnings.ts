import { afterAll } from 'vitest';
import { failOnWarnings } from './failOnWarnings.ts';

const watching = failOnWarnings();
afterAll(() => watching[Symbol.dispose]());
