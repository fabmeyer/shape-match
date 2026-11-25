import { describe, expect, it } from 'vitest';
import { allShapes } from './shapes.ts';
import { carWheelError, checkLayout, checkShape, CHECK_RES, CHECK_TOL, roundness, sampleGrid } from './shapeChecks.ts';

const shapes = allShapes();

describe('shape library', () => {
  it('has 11 Targets in difficulty order', () => {
    expect(shapes.map((s) => s.difficulty)).toEqual([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]);
  });

  describe.each(shapes)('$id', (shape) => {
    it('has a valid Layout', () => {
      expect(checkLayout(shape.net)).toEqual([]);
    });

    it('passes every check that applies (seam, caps, roundness)', () => {
      expect(checkShape(shape)).toEqual([]);
    });
  });

  it.each(shapes.filter((s) => s.construction.startsWith('revolution')))('$id is round to 1e-5', (shape) => {
    expect(roundness(sampleGrid(shape.net, CHECK_RES))).toBeLessThan(CHECK_TOL);
  });

  it('has round car wheels to 1e-5', () => {
    expect(carWheelError(shapes.find((s) => s.id === 'car')!.net)).toBeLessThan(CHECK_TOL);
  });
});
