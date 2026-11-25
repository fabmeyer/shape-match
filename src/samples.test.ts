import { describe, expect, it } from 'vitest';
import { CREASE_GAP, samplePositions } from './samples.ts';
import { clampedUniformKnots } from './controlNet.ts';
import { repeatedKnots } from './shapes.ts';

const increasing = (t: number[]) => t.every((x, i) => i === 0 || x > t[i - 1]);

describe('samplePositions', () => {
  it('is the uniform grid when every interior knot lies on it', () => {
    const knots = [0, 0, 0, 0, 0.25, 0.5, 0.75, 1, 1, 1, 1];
    expect(samplePositions(knots, 3, 5)).toEqual([0, 0.25, 0.5, 0.75, 1]);
  });

  it('adds interior knots that the grid steps over', () => {
    const knots = clampedUniformKnots(6, 3); // interior knots 1/3, 2/3
    const t = samplePositions(knots, 3, 128);
    expect(t).toHaveLength(130);
    expect(t).toContain(1 / 3);
    expect(t).toContain(2 / 3);
  });

  it('starts at 0, ends at 1 and is strictly increasing', () => {
    for (const knots of [clampedUniformKnots(9, 2), repeatedKnots(5, 2, 2), clampedUniformKnots(4, 1)]) {
      const t = samplePositions(knots, 2, 40);
      expect(t[0]).toBe(0);
      expect(t[t.length - 1]).toBe(1);
      expect(increasing(t)).toBe(true);
    }
  });

  it('puts a second sample just before a crease (multiplicity >= degree)', () => {
    const knots = repeatedKnots(4, 2, 2); // degree 2, double knots at 1/4, 1/2, 3/4
    const t = samplePositions(knots, 2, 10);
    for (const k of [0.25, 0.5, 0.75]) {
      expect(t).toContain(k);
      expect(t).toContain(k - CREASE_GAP);
    }
  });

  it('adds no crease sample where the surface is smooth (multiplicity < degree)', () => {
    const t = samplePositions(clampedUniformKnots(6, 3), 3, 128);
    expect(t.some((x) => Math.abs(x - (1 / 3 - CREASE_GAP)) < 1e-12)).toBe(false);
  });

  it('shifts the uniform values by a fraction of a cell, keeping both ends', () => {
    const t = samplePositions([0, 0, 1, 1], 1, 5, 0.5);
    expect(t).toEqual([0, 0.125, 0.375, 0.625, 0.875, 1]);
  });

  it('treats every knot of a degree 1 surface as a crease', () => {
    const t = samplePositions(clampedUniformKnots(4, 1), 1, 7); // knots 1/3, 2/3 lie on the grid i/6
    expect(t).toContain(1 / 3);
    expect(t).toContain(1 / 3 - CREASE_GAP);
    expect(t).toHaveLength(9);
  });
});
