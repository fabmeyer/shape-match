import { describe, expect, it } from 'vitest';
import { getPoint, setPoint, type ControlNet } from './controlNet.ts';
import { applyFit, bestFit, rotate, symmetricEigen4 } from './alignment.ts';
import { pairedDistance } from './shapeDistance.ts';
import { allTasks, cloneNet } from './target.ts';

const RES = 24;

// the whole net turned by the quaternion q around the origin, then moved
function turned(net: ControlNet, q: [number, number, number, number], move: [number, number, number]): ControlNet {
  const out = cloneNet(net);
  for (let k = 0; k < out.countU * out.countV; k++) {
    const r = rotate(q, getPoint(out, k));
    setPoint(out, k, [r[0] + move[0], r[1] + move[1], r[2] + move[2]]);
  }
  return out;
}

const angle = Math.PI / 6; // 30° around a tilted axis
const axis = [1, 2, 2].map((c) => c / 3);
const q30 = [...axis.map((c) => c * Math.sin(angle / 2)), Math.cos(angle / 2)] as [number, number, number, number];

describe('symmetricEigen4', () => {
  it('finds eigenvalues and eigenvectors of a symmetric matrix', () => {
    const a = [[4, 1, 0, 2], [1, 3, 1, 0], [0, 1, 2, 1], [2, 0, 1, 1]];
    const { values, vectors } = symmetricEigen4(a);
    for (let k = 0; k < 4; k++) {
      const v = vectors.map((row) => row[k]);
      const av = a.map((row) => row.reduce((s, x, i) => s + x * v[i], 0));
      av.forEach((x, i) => expect(x).toBeCloseTo(values[k] * v[i], 10));
    }
  });
});

describe('bestFit', () => {
  it('finds the rotation and the move between turned point sets', () => {
    const pts = [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0.5];
    const moved: number[] = [];
    for (let k = 0; k < pts.length; k += 3) moved.push(...rotate(q30, [pts[k], pts[k + 1], pts[k + 2]]).map((c, a) => c + [0.3, -1, 2][a]));
    const fit = bestFit(pts, moved, [1, 1, 1, 1, 1]);
    fit.rotation.forEach((c, k) => expect(c).toBeCloseTo(q30[k], 8));
    for (let k = 0; k < pts.length; k += 3) {
      applyFit(fit, [pts[k], pts[k + 1], pts[k + 2]]).forEach((c, a) => expect(c).toBeCloseTo(moved[k + a], 8));
    }
  });
});

describe('the Paired distance with the best fit', () => {
  it.each(allTasks().map((t) => [t.id, t] as const))('%s: a turned and moved Target fits exactly', (_, task) => {
    const learner = turned(task.target, q30, [0.4, -0.2, 0.3]);
    expect(pairedDistance(task.target, learner, RES).rms).toBeGreaterThan(0.01);
    const aligned = pairedDistance(task.target, learner, RES, true);
    expect(aligned.rms).toBeLessThan(1e-6); // the control points are stored as f32
    expect(aligned.hausdorff).toBeLessThan(1e-6);
  });

  it('can not fix the shape: the starts stay far off', () => {
    for (const task of allTasks()) expect(pairedDistance(task.target, task.start, RES, true).rms).toBeGreaterThan(0.01);
  });
});
