import { describe, expect, it } from 'vitest';
import { vec3 } from 'wgpu-matrix';
import { straightOrientation } from './arcball.ts';

// camera space looks down -z with +y up
const look = (q: Float32Array) => [...vec3.transformQuat([0, 0, -1], q)];
const up = (q: Float32Array) => [...vec3.transformQuat([0, 1, 0], q)];
const close = (a: number[], b: number[]) => a.forEach((x, k) => expect(x).toBeCloseTo(b[k], 6));

describe('flipping a straight-on view', () => {
  it.each([
    ['top', [0, 0, -1], [0, 1, 0]],
    ['front', [0, 1, 0], [0, 0, 1]],
    ['side', [-1, 0, 0], [0, 0, 1]],
  ] as const)('%s: looks from the opposite side, up stays up', (view, direction, screenUp) => {
    const normal = straightOrientation(view, false);
    const flipped = straightOrientation(view, true);
    close(look(normal), [...direction]);
    close(look(flipped), direction.map((x) => -x));
    close(up(normal), [...screenUp]);
    close(up(flipped), [...screenUp]);
  });
});
