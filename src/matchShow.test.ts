import { describe, expect, it } from 'vitest';
import { vec3 } from 'wgpu-matrix';
import { applyFit, type BestFit } from './alignment.ts';
import { ease, showModels } from './matchShow.ts';

const fit: BestFit = {
  rotation: [0, 0, Math.sin(Math.PI / 8), Math.cos(Math.PI / 8)], // 45° around z
  learnerCentre: [0.2, -0.1, 0.5],
  targetCentre: [0.1, 0.3, 0.6],
};
const OFFSET = 2.8;
const close = (a: ArrayLike<number>, b: ArrayLike<number>) => [0, 1, 2].forEach((k) => expect(a[k]).toBeCloseTo(b[k], 6));

describe('the Hand-in animation', () => {
  it('starts with both surfaces where they are', () => {
    const { learner, target } = showModels(0, 0, fit, OFFSET);
    close(vec3.transformMat4([1, 2, 3], learner), [1, 2, 3]);
    close(vec3.transformMat4([1, 2, 3], target), [1, 2, 3]);
  });

  it('brings both together in the middle, then puts the learner into its best fit', () => {
    const { learner, target } = showModels(1, 1, fit, OFFSET);
    const p: [number, number, number] = [0.7, -0.4, 0.9];
    const fitted = applyFit(fit, p);
    close(vec3.transformMat4(p, learner), [fitted[0] + OFFSET / 2, fitted[1], fitted[2]]);
    // the Target is drawn OFFSET to the right; it ends at the same place as the fitted learner
    close(vec3.transformMat4([fitted[0] + OFFSET, fitted[1], fitted[2]], target), [fitted[0] + OFFSET / 2, fitted[1], fitted[2]]);
  });

  it('eases in and out', () => {
    expect(ease(0)).toBe(0);
    expect(ease(0.5)).toBeCloseTo(0.5, 10);
    expect(ease(1)).toBe(1);
    expect(ease(2)).toBe(1);
  });
});
