import { describe, expect, it } from 'vitest';
import { basisFuns, evaluate, findSpan } from './nurbs.ts';
import { clampedUniformKnots, flatControlNet, setPoint, type ControlNet } from './controlNet.ts';

// the worked example from the lecture: cubic, 7 control points
const U = [0, 0, 0, 0, 0.25, 0.5, 0.75, 1, 1, 1, 1];

describe('lecture example (cubic, 7 points, u = 0.42)', () => {
  it('finds span 4', () => {
    expect(findSpan(6, 3, 0.42, U)).toBe(4);
  });

  it('gives the basis [0.0082, 0.3588, 0.5807, 0.0524]', () => {
    const N = basisFuns(4, 0.42, 3, U);
    [0.0082, 0.3588, 0.5807, 0.0524].forEach((expected, k) => expect(N[k]).toBeCloseTo(expected, 4));
  });
});

describe('findSpan', () => {
  it('puts u = 1 into the last non-empty span', () => {
    expect(findSpan(6, 3, 1, U)).toBe(6);
  });

  it('puts a knot value into the span that starts there', () => {
    expect(findSpan(6, 3, 0.5, U)).toBe(5);
  });
});

describe('basisFuns', () => {
  it('adds up to 1 everywhere (partition of unity)', () => {
    for (let u = 0; u <= 1; u += 0.05) {
      const N = basisFuns(findSpan(6, 3, u, U), u, 3, U);
      expect(N.reduce((a, b) => a + b)).toBeCloseTo(1, 12);
      for (const n of N) expect(n).toBeGreaterThanOrEqual(0);
    }
  });
});

describe('evaluate', () => {
  // degree 1 x 1 with 2 x 2 points is the bilinear patch through its corners
  function bilinear(): ControlNet {
    const corners = [[0, 0, 0], [2, 0, 1], [0, 3, 2], [2, 3, 5]];
    return {
      countU: 2, countV: 2, p: 1, q: 1,
      knotsU: clampedUniformKnots(2, 1), knotsV: clampedUniformKnots(2, 1),
      points: new Float32Array(corners.flatMap(([x, y, z]) => [x, y, z, 1])),
    };
  }

  it('interpolates the corners of a clamped net', () => {
    const net = flatControlNet();
    setPoint(net, 0, [-1, -1, 0.7]);
    expect(evaluate(net, 0, 0).point).toEqual([-1, -1, expect.closeTo(0.7, 6)]);
    expect(evaluate(net, 1, 1).point.map((c) => Number(c.toFixed(6)))).toEqual([1, 1, 0]);
  });

  it('is the bilinear patch for degree 1 x 1', () => {
    const [x, y, z] = evaluate(bilinear(), 0.25, 0.5).point;
    expect(x).toBeCloseTo(0.5, 12);
    expect(y).toBeCloseTo(1.5, 12);
    // z: (1-u)(1-v) 0 + u(1-v) 1 + (1-u)v 2 + uv 5
    expect(z).toBeCloseTo(0.75 * 0.5 * 0 + 0.25 * 0.5 * 1 + 0.75 * 0.5 * 2 + 0.25 * 0.5 * 5, 12);
  });

  it('does not change when all weights are multiplied by the same factor', () => {
    const net = flatControlNet();
    setPoint(net, 14, [0, 0, 1]);
    const scaled = { ...net, points: net.points.map((c) => c * 3) };
    for (const [u, v] of [[0.1, 0.2], [0.42, 0.7], [0.9, 0.5]]) {
      const a = evaluate(net, u, v).point;
      const b = evaluate(scaled, u, v).point;
      a.forEach((c, i) => expect(b[i]).toBeCloseTo(c, 5));
    }
  });

  it('gives rational basis functions that add up to 1', () => {
    const net = flatControlNet();
    net.points.set([0, 0, 0, 2.5], 14 * 4); // a heavy point at the origin
    const { R } = evaluate(net, 0.4, 0.45);
    expect(R.flat().reduce((a, b) => a + b)).toBeCloseTo(1, 6);
  });
});
