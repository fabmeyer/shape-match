import { describe, expect, it } from 'vitest';
import { flatControlNet, getPoint, setPoint, type ControlNet } from './controlNet.ts';
import { allShapes } from './shapes.ts';
import { pairedDistance, pointTriangleDistance2, sampleSurface, shapeDistance } from './shapeDistance.ts';
import { allTasks } from './target.ts';
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

// the app's grader: two surfaces that look the same get the same grade, however their control points sit
describe('shapeDistance lined up (nearest point after the best fit)', () => {
  const house = allTasks().find((t) => t.id === 'house')!;
  // the House with its hidden cap points moved inside the end faces, then turned and moved: the same shape
  const learner = cloneNet(house.target);
  const { countU, countV } = learner;
  const q = [Math.sin(0.2), 0, 0, Math.cos(0.2)] as const; // 0.4 rad around x
  for (let j = 0; j < countV; j++) {
    for (let i = 0; i < countU; i++) {
      let [x, y, z] = getPoint(house.target, j * countU + i);
      if (j === 0 || j === countV - 1) { x += 0.2; z -= 0.2; }
      const c = Math.cos(0.4), s = Math.sin(0.4);
      setPoint(learner, j * countU + i, [x + 0.3, c * y - s * z, s * y + c * z + 0.1]);
    }
  }

  it('grades the shape only: a moved cap point and a turn cost nothing', () => {
    expect(q[3]).toBeCloseTo(Math.cos(0.2)); // (the turn above, as a quaternion)
    const lined = shapeDistance(house.target, learner, { res: RES, align: true });
    expect(lined.rms).toBeLessThan(0.0005);       // a quarter of the Tolerance τ = 0.2 %
    expect(pairedDistance(house.target, learner, RES, true).rms).toBeGreaterThan(0.003);
  });

  it('measures without the best fit when the Layouts differ', () => {
    const vase = allShapes().find((s) => s.id === 'vase')!.net;
    const plane = flatControlNet();
    expect(shapeDistance(vase, plane, { res: RES, align: true }).rms).toBe(shapeDistance(vase, plane, { res: RES }).rms);
  });
});

describe('pairedDistance (the same grid point on both surfaces)', () => {
  it('is 0 for the Target against itself', () => {
    const d = pairedDistance(ball, ball, RES);
    expect(d.rms).toBe(0);
    expect(d.hausdorff).toBe(0);
  });

  it('is exactly the shift when the whole surface moves', () => {
    const d = pairedDistance(ball, moved(ball, 0, 0, 0.1), RES);
    expect(d.rms * d.diagonal).toBeCloseTo(0.1, 6);
    expect(d.hausdorff * d.diagonal).toBeCloseTo(0.1, 6);
  });

  it('is never smaller than the Shape distance: it never passes a wrong shape', () => {
    for (const task of allTasks()) {
      const paired = pairedDistance(task.target, task.start, RES);
      const nearest = shapeDistance(task.target, task.start, { res: RES });
      expect(paired.hausdorff).toBeGreaterThanOrEqual(nearest.hausdorff * (1 - 1e-9));
    }
  });

  // the known limit (docs/research/shape-distance-parameter-wise.md): the House's cap point can sit anywhere
  // inside its end face without changing the shape, but the pairs no longer line up
  it('counts a correct House with its cap points elsewhere as off', () => {
    const house = allTasks().find((t) => t.id === 'house')!;
    const learner = cloneNet(house.target);
    const { countU, countV } = learner;
    for (const j of [0, countV - 1]) {
      for (let i = 0; i < countU; i++) {
        const [x, y, z] = getPoint(house.target, j * countU + i);
        setPoint(learner, j * countU + i, [x + 0.2, y, z - 0.2]); // still inside the end face
      }
    }
    expect(shapeDistance(house.target, learner, { res: RES }).rms).toBeLessThan(1e-9);
    expect(pairedDistance(house.target, learner, RES).rms).toBeGreaterThan(0.01);
  });

  it('pairs every point of the Tent: no hidden cap point, so a correct Tent is exactly 0', () => {
    const tent = allTasks().find((t) => t.id === 'tent')!;
    expect(tent.target.countV).toBe(2);
    expect(pairedDistance(tent.target, cloneNet(tent.target), RES).rms).toBe(0);
  });

  it('needs the same Layout for both surfaces', () => {
    expect(() => pairedDistance(ball, flatControlNet(), RES)).toThrow(/same Layout/);
  });
});
