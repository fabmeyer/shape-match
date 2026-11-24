// NURBS evaluation on the CPU for a single (u, v)
// same algorithms as shaders/tessellate.wgsl (The NURBS Book: A2.1 FindSpan, A2.2 BasisFuns, A4.3 SurfacePoint)
// used for the parameter space view and the formula panel
import type { ControlNet, Point3 } from './controlNet.ts';

// A2.1: index of the knot span that contains u, i.e. U[span] <= u < U[span + 1]
// n = number of control points - 1
export function findSpan(n: number, p: number, u: number, U: number[]): number {
  if (u >= U[n + 1]) return n; // u = 1 belongs to the last span
  if (u <= U[p]) return p;
  let low = p;
  let high = n + 1;
  let span = Math.floor((low + high) / 2);
  while (u < U[span] || u >= U[span + 1]) {
    if (u < U[span]) high = span;
    else low = span;
    span = Math.floor((low + high) / 2);
  }
  return span;
}

// A2.2: the p + 1 basis functions that are non-zero at u: N[k] = N_{span-p+k, p}(u)
export function basisFuns(span: number, u: number, p: number, U: number[]): number[] {
  const N = new Array<number>(p + 1).fill(0);
  const left = new Array<number>(p + 1).fill(0);
  const right = new Array<number>(p + 1).fill(0);
  N[0] = 1;
  for (let j = 1; j <= p; j++) {
    left[j] = u - U[span + 1 - j];
    right[j] = U[span + j] - u;
    let saved = 0;
    for (let r = 0; r < j; r++) {
      const temp = N[r] / (right[r + 1] + left[j - r]);
      N[r] = saved + right[r + 1] * temp;
      saved = left[j - r] * temp;
    }
    N[j] = saved;
  }
  return N;
}

export interface SurfaceEval {
  u: number;
  v: number;
  spanU: number;   // active control points in u: spanU - p ... spanU
  spanV: number;   // active control points in v: spanV - q ... spanV
  Nu: number[];    // N_{spanU-p+k, p}(u), k = 0..p
  Nv: number[];    // N_{spanV-q+l, q}(v), l = 0..q
  R: number[][];   // R[l][k]: rational basis function of control point (spanU-p+k, spanV-q+l), all add up to 1
  walkers: number[][]; // stage a: walkers[l] = sum over k of Nu[k] * Pw of row spanV-q+l, homogeneous (w*x, w*y, w*z, w)
  point: Point3;   // S(u, v)
}

// A4.3 in its two stages (the two loops of the compute shader), plus the share R of every active control point
//   stage a, along u: one walker per active row of control points
//   stage b, along v: the walkers weighted with Nv give S(u, v)
export function evaluate(net: ControlNet, u: number, v: number): SurfaceEval {
  const { p, q, countU, countV, knotsU, knotsV, points } = net;
  const spanU = findSpan(countU - 1, p, u, knotsU);
  const spanV = findSpan(countV - 1, q, v, knotsV);
  const Nu = basisFuns(spanU, u, p, knotsU);
  const Nv = basisFuns(spanV, v, q, knotsV);

  const Sw = [0, 0, 0, 0];
  const R: number[][] = [];
  const walkers: number[][] = [];
  for (let l = 0; l <= q; l++) {
    R.push([]);
    const walker = [0, 0, 0, 0]; // stage a for row spanV - q + l (the inner loop)
    for (let k = 0; k <= p; k++) {
      const idx = ((spanV - q + l) * countU + spanU - p + k) * 4;
      for (let c = 0; c < 4; c++) walker[c] += Nu[k] * points[idx + c];
      R[l].push(Nu[k] * Nv[l] * points[idx + 3]); // N_i * N_j * w_ij, divided by the weight sum below
    }
    walkers.push(walker);
    for (let c = 0; c < 4; c++) Sw[c] += Nv[l] * walker[c]; // stage b (the outer loop)
  }
  for (const row of R) for (let k = 0; k <= p; k++) row[k] /= Sw[3];

  return { u, v, spanU, spanV, Nu, Nv, R, walkers, point: [Sw[0] / Sw[3], Sw[1] / Sw[3], Sw[2] / Sw[3]] };
}
