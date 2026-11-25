// the walkers of the lecture ("curves of curves", slide 26), for the current (u, v):
//   green:  every row j of control points is a curve C_j(u) along u; its point at u is the walker
//   orange: the walkers are the control points of a curve along v: S(u, v') for all v'
//           (an iso-parameter line, it lies on the surface); its point at v is S(u, v)
// everything is computed in homogeneous coordinates, so it is also correct with weights
import type { ControlNet } from './controlNet.ts';
import { basisFuns, findSpan } from './nurbs.ts';

const SAMPLES = 64;               // line segments per curve
export const MAX_SEGMENTS = 4096; // size of the GPU buffer: (rows + 1) x 64 curve pieces, enough for the car (25 rows)

// one line segment for curves.wgsl: a (xyz, width in px), b (xyz, 1 = dashed), colour (rgba)
export const SEGMENT_FLOATS = 12;

export const WALKER_ACTIVE = 8;  // flags for points.wgsl
export const WALKER_FAINT = 16;

const GREEN: [number, number, number] = [0.31, 0.84, 0.54];
const GREEN_FAINT: [number, number, number] = [0.2, 0.42, 0.3];
const ORANGE: [number, number, number] = [0.95, 0.54, 0.31];

export interface Walkers {
  segments: Float32Array;    // SEGMENT_FLOATS per segment
  segmentCount: number;
  points: Float32Array;      // the walkers as (x, y, z, 1), one per row
  flags: Uint32Array;        // WALKER_ACTIVE for the q + 1 rows that count at v, else WALKER_FAINT
}

// homogeneous point of the curve sum_i N_i(t) * pts[i] (pts: list of (w*x, w*y, w*z, w))
function curvePoint(pts: number[][], degree: number, knots: number[], t: number): number[] {
  const span = findSpan(pts.length - 1, degree, t, knots);
  const N = basisFuns(span, t, degree, knots);
  const out = [0, 0, 0, 0];
  for (let k = 0; k <= degree; k++) {
    for (let c = 0; c < 4; c++) out[c] += N[k] * pts[span - degree + k][c];
  }
  return out;
}

const toPoint = (h: number[]) => [h[0] / h[3], h[1] / h[3], h[2] / h[3]];

export function buildWalkers(net: ControlNet, u: number, v: number): Walkers {
  const { countU, countV, p, q, knotsU, knotsV } = net;
  const segments = new Float32Array(MAX_SEGMENTS * SEGMENT_FLOATS);
  let count = 0;
  const add = (a: number[], b: number[], width: number, color: number[], dashed = false) => {
    if (count >= MAX_SEGMENTS) return;
    segments.set([...a, width, ...b, dashed ? 1 : 0, ...color, 1], count * SEGMENT_FLOATS);
    count++;
  };
  // a curve given by a function t -> homogeneous point, as SAMPLES straight pieces
  const addCurve = (at: (t: number) => number[], width: number, color: number[]) => {
    let previous = toPoint(at(0));
    for (let s = 1; s <= SAMPLES; s++) {
      const next = toPoint(at(s / SAMPLES));
      add(previous, next, width, color);
      previous = next;
    }
  };

  // the rows of the control net, homogeneous
  const rows: number[][][] = [];
  for (let j = 0; j < countV; j++) {
    const row: number[][] = [];
    for (let i = 0; i < countU; i++) row.push(Array.from(net.points.subarray((j * countU + i) * 4, (j * countU + i) * 4 + 4)));
    rows.push(row);
  }

  // which rows count at v: spanV - q ... spanV
  const spanV = findSpan(countV - 1, q, v, knotsV);
  const active = (j: number) => j >= spanV - q && j <= spanV;

  // green: the row curves, the faint ones first so the active ones are drawn over them
  for (const pass of [false, true]) {
    for (let j = 0; j < countV; j++) {
      if (active(j) !== pass) continue;
      addCurve((t) => curvePoint(rows[j], p, knotsU, t), pass ? 3 : 1.5, pass ? GREEN : GREEN_FAINT);
    }
  }

  // the walkers: point at u on every row curve (stage a)
  const walkers = rows.map((row) => curvePoint(row, p, knotsU, u));

  // dashed orange: the control polygon of the orange curve (it connects the walkers)
  for (let j = 0; j + 1 < countV; j++) add(toPoint(walkers[j]), toPoint(walkers[j + 1]), 1.5, ORANGE, true);

  // orange: the curve with the walkers as control points (stage b), S(u, v') for all v'
  addCurve((t) => curvePoint(walkers, q, knotsV, t), 3, ORANGE);

  const points = new Float32Array(countV * 4);
  const flags = new Uint32Array(countV);
  walkers.forEach((h, j) => {
    points.set([...toPoint(h), 1], j * 4);
    flags[j] = active(j) ? WALKER_ACTIVE : WALKER_FAINT;
  });
  return { segments, segmentCount: count, points, flags };
}
