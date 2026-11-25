// the CPU evaluation against the reference values of the course test models (test-models/README.md)
import { describe, expect, it } from 'vitest';
import { courseModels } from './courseModels.ts';
import { evaluate } from './nurbs.ts';
import { checkLayout } from './shapeChecks.ts';

describe.skipIf(courseModels.length === 0)('course test models', () => {
  describe.each(courseModels)('$name', (model) => {
    it('has a valid Layout', () => {
      expect(checkLayout(model.net)).toEqual([]);
    });

    it('hits every reference value of S(u, v)', () => {
      for (const { u, v, point } of model.references) {
        const s = evaluate(model.net, u, v).point;
        expect(Math.hypot(s[0] - point[0], s[1] - point[1], s[2] - point[2])).toBeLessThanOrEqual(model.tolerance);
      }
    });
  });
});
