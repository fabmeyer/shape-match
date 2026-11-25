import { describe, expect, it } from 'vitest';
import { flatControlNet, getPoint, setPoint, type ControlNet } from './controlNet.ts';
import { allShapes } from './shapes.ts';
import { pointTriangleDistance2, sampleSurface, shapeDistance } from './shapeDistance.ts';
import { cloneNet } from './target.ts';

const RES = 16; // small grids keep the brute force fast

function moved(net: ControlNet, dx: number, dy: number, dz: number): ControlNet {
  const out = cloneNet(net);
  for (let k = 0; k < out.countU * out.countV; k++) {
    const [x, y, z] = getPoint(out, k);
    setPoint(out, k, [x + dx, y + dy, z + dz]);
  }
  return out;
}

const ball = allShapes().find((s) => s.id === 'ball')!.net;

describe('pointTriangleDistance2', () => {
  const tri = [0, 0, 0, 1, 0, 0, 0, 1, 0] as const;
  const d = (x: number, y: number, z: number) => Math.sqrt(pointTriangleDistance2(x, y, z, ...tri));

  it('measures to the inside, the edges and the corners', () => {
    expect(d(0.2, 0.2, 0.5)).toBeCloseTo(0.5, 12);  // above the inside
    expect(d(0.5, -1, 0)).toBeCloseTo(1, 12);        // beside edge ab
    expect(d(-3, -4, 0)).toBeCloseTo(5, 12);         // beyond corner a
    expect(d(1, 1, 0)).toBeCloseTo(Math.SQRT1_2, 12); // beside edge bc
  });
});

describe('sampleSurface', () => {
  it('gives weights that add up to the surface area', () => {
    const s = sampleSurface(flatControlNet(), RES); // a 2 x 2 square
    expect(s.weights.reduce((a, b) => a + b)).toBeCloseTo(4, 10);
  });
});

describe('shapeDistance', () => {
  it('is 0 for the same surface on the same grid', () => {
    const d = shapeDistance(ball, ball, { res: RES });
    expect(d.rms).toBe(0);
    expect(d.hausdorff).toBe(0);
  });

  it('is d / diagonal for a surface moved by d', () => {
    const plane = flatControlNet();
    const d = shapeDistance(plane, moved(plane, 0, 0, 0.1), { res: RES });
    expect(d.diagonal).toBeCloseTo(Math.sqrt(8), 10);
    expect(d.rms).toBeCloseTo(0.1 / Math.sqrt(8), 6);
    expect(d.hausdorff).toBeCloseTo(0.1 / Math.sqrt(8), 6);
  });

  it('is symmetric: swapping the surfaces only changes which diagonal divides', () => {
    const other = moved(ball, 0.05, 0, 0.1);
    const a = shapeDistance(ball, other, { res: RES });
    const b = shapeDistance(other, ball, { res: RES });
    expect(a.rms * a.diagonal).toBeCloseTo(b.rms * b.diagonal, 10);
    expect(a.hausdorff * a.diagonal).toBeCloseTo(b.hausdorff * b.diagonal, 10);
  });

  it('does not change when all weights are multiplied by the same factor', () => {
    const scaled = { ...ball, points: ball.points.map((c) => c * 2.5) };
    const d = shapeDistance(ball, scaled, { res: RES });
    expect(d.rms).toBeLessThan(1e-6);
  });

  it('lets one local bump show in the Hausdorff distance much more than in the RMS', () => {
    const plane = flatControlNet(12, 12);
    const bump = cloneNet(plane);
    const k = 5 * 12 + 6;
    const [x, y] = getPoint(bump, k);
    setPoint(bump, k, [x, y, 0.2]);
    const d = shapeDistance(plane, bump, { res: 32 });
    expect(d.hausdorff / d.rms).toBeGreaterThan(4);
  });

  describe('Noise floor (the same surface on a grid shifted by half a cell)', () => {
    const floor = (res: number, mode: 'triangle' | 'point') =>
      shapeDistance(ball, ball, { res, mode, learnerShift: 0.5 }).rms;

    it('is much smaller with triangles than with points', () => {
      expect(floor(16, 'triangle')).toBeLessThan(floor(16, 'point') / 5);
    });

    it('falls about like h^2 with triangles and like h with points', () => {
      const triangleRatio = floor(12, 'triangle') / floor(24, 'triangle');
      const pointRatio = floor(12, 'point') / floor(24, 'point');
      expect(triangleRatio).toBeGreaterThan(3); // h^2: 4
      expect(pointRatio).toBeGreaterThan(1.6);  // h: 2
      expect(pointRatio).toBeLessThan(2.6);
    });
  });
});
