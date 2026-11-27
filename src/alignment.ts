// the best fit before the Paired distance: turn and move the Learner surface so that it lies on the Target
// as well as possible (docs/research/shape-distance-alignment.md). Both surfaces share one Layout (ADR 0001),
// so sample (i, j) belongs to sample (i, j) and the best fit has a closed form, no search and no ML:
//   1. the centre of each surface: the area-weighted mean of its samples
//   2. the best rotation around those centres: Horn's method (Horn 1987, JOSA A 4(4)): the eigenvector of
//      the largest eigenvalue of a 4 x 4 matrix built from the cross-covariance of the pairs, as a quaternion
//   3. the move: the learner's centre goes onto the Target's centre
// align.wgsl does the same on the GPU; this is the CPU reference
import type { Point3 } from './controlNet.ts';

export interface BestFit {
  rotation: [number, number, number, number]; // unit quaternion (x, y, z, w), turns the learner around its centre
  learnerCentre: Point3;
  targetCentre: Point3;
}

export const NO_FIT: BestFit = { rotation: [0, 0, 0, 1], learnerCentre: [0, 0, 0], targetCentre: [0, 0, 0] };

// eigenvalues and eigenvectors of a symmetric 4 x 4 matrix (cyclic Jacobi rotations), the columns of v
export function symmetricEigen4(a: number[][]): { values: number[]; vectors: number[][] } {
  const m = a.map((row) => [...row]);
  const v = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]];
  for (let sweep = 0; sweep < 20; sweep++) {
    let off = 0;
    for (let p = 0; p < 4; p++) for (let q = p + 1; q < 4; q++) off += m[p][q] * m[p][q];
    if (off < 1e-30) break;
    for (let p = 0; p < 4; p++) {
      for (let q = p + 1; q < 4; q++) {
        if (Math.abs(m[p][q]) < 1e-30) continue;
        // the rotation in the (p, q) plane that makes m[p][q] zero
        const theta = (m[q][q] - m[p][p]) / (2 * m[p][q]);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1);
        const s = t * c;
        for (let k = 0; k < 4; k++) { // m = m J
          const mkp = m[k][p], mkq = m[k][q];
          m[k][p] = c * mkp - s * mkq;
          m[k][q] = s * mkp + c * mkq;
        }
        for (let k = 0; k < 4; k++) { // m = J^T m
          const mpk = m[p][k], mqk = m[q][k];
          m[p][k] = c * mpk - s * mqk;
          m[q][k] = s * mpk + c * mqk;
        }
        for (let k = 0; k < 4; k++) { // v = v J
          const vkp = v[k][p], vkq = v[k][q];
          v[k][p] = c * vkp - s * vkq;
          v[k][q] = s * vkp + c * vkq;
        }
      }
    }
  }
  return { values: [m[0][0], m[1][1], m[2][2], m[3][3]], vectors: v };
}

// the best fit of the pairs (learner[k] -> target[k]), each pair with its weight; points as x, y, z triples
export function bestFit(learner: ArrayLike<number>, target: ArrayLike<number>, weights: ArrayLike<number>): BestFit {
  const n = weights.length;
  let sw = 0;
  const cl = [0, 0, 0], ct = [0, 0, 0];
  for (let k = 0; k < n; k++) {
    const w = weights[k];
    sw += w;
    for (let a = 0; a < 3; a++) { cl[a] += w * learner[k * 3 + a]; ct[a] += w * target[k * 3 + a]; }
  }
  if (!(sw > 0)) return NO_FIT;
  for (let a = 0; a < 3; a++) { cl[a] /= sw; ct[a] /= sw; }

  // cross-covariance S[a][b] = sum of w (p_a - cl_a)(q_b - ct_b), p on the learner, q on the Target
  const S = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (let k = 0; k < n; k++) {
    const w = weights[k];
    for (let a = 0; a < 3; a++) {
      const pa = learner[k * 3 + a] - cl[a];
      for (let b = 0; b < 3; b++) S[a][b] += w * pa * (target[k * 3 + b] - ct[b]);
    }
  }
  const [[xx, xy, xz], [yx, yy, yz], [zx, zy, zz]] = S;
  // Horn's matrix; its top eigenvector is the quaternion (w, x, y, z) of the best rotation
  const N = [
    [xx + yy + zz, yz - zy, zx - xz, xy - yx],
    [yz - zy, xx - yy - zz, xy + yx, zx + xz],
    [zx - xz, xy + yx, -xx + yy - zz, yz + zy],
    [xy - yx, zx + xz, yz + zy, -xx - yy + zz],
  ];
  const { values, vectors } = symmetricEigen4(N);
  let best = 0;
  for (let k = 1; k < 4; k++) if (values[k] > values[best]) best = k;
  let [w, x, y, z] = vectors.map((row) => row[best]);
  if (w < 0) [w, x, y, z] = [-w, -x, -y, -z]; // the shorter way round, for the animation
  const len = Math.hypot(w, x, y, z) || 1;
  return {
    rotation: [x / len, y / len, z / len, w / len],
    learnerCentre: cl as Point3,
    targetCentre: ct as Point3,
  };
}

// turns a point by a unit quaternion (x, y, z, w)
export function rotate(q: BestFit['rotation'], p: Point3): Point3 {
  const [x, y, z, w] = q;
  // t = 2 (q.xyz × p), p' = p + w t + q.xyz × t
  const tx = 2 * (y * p[2] - z * p[1]);
  const ty = 2 * (z * p[0] - x * p[2]);
  const tz = 2 * (x * p[1] - y * p[0]);
  return [
    p[0] + w * tx + (y * tz - z * ty),
    p[1] + w * ty + (z * tx - x * tz),
    p[2] + w * tz + (x * ty - y * tx),
  ];
}

// where the best fit puts a point of the learner: turned around its centre, then onto the Target's centre
export function applyFit(fit: BestFit, p: Point3): Point3 {
  const c = fit.learnerCentre;
  const r = rotate(fit.rotation, [p[0] - c[0], p[1] - c[1], p[2] - c[2]]);
  return [r[0] + fit.targetCentre[0], r[1] + fit.targetCentre[1], r[2] + fit.targetCentre[2]];
}
