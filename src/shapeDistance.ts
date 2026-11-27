// Shape distance on the CPU: the reference for the GPU error pass (shaders/distance.wgsl) and the
// CPU baseline of the benchmark (ADR 0002)
//   from every sample of one surface to the nearest triangle of the other, in both directions
//   every sample counts with the area it stands for, so bunched samples don't count more
//   RMS and Hausdorff (the largest single distance), both divided by the diagonal of the Target's bounding box
// the GPU does the same steps in the same order, so both give the same numbers up to f32 rounding
import type { ControlNet } from './controlNet.ts';
import { evaluate } from './nurbs.ts';
import { samplePositions } from './samples.ts';
import { applyFit, bestFit, NO_FIT, type BestFit } from './alignment.ts';

// samples per direction before the knots are added (samples.ts); adjustable per call
export const ERROR_RES = 64;

// refinements of the best fit with the nearest points (ICP) before the nearest-point distance, see shapeDistance
export const ICP_STEPS = 8; // each step shrinks a misfit by about a third: 8 steps leave ~3 % of it

// a triangle with less than this squared doubled area is a point or a line (a collapsed cap, a tip):
// it is left out, its neighbours cover the same place
const DEGENERATE = 1e-20;

export interface SampledSurface {
  resU: number;
  resV: number;
  points: Float64Array;  // x, y, z per sample, index j * resU + i
  weights: Float64Array; // the area each sample stands for; together they add up to the surface area
}

export interface ShapeDistance {
  rms: number;                  // area-weighted RMS of both directions / diagonal
  hausdorff: number;            // largest distance in either direction / diagonal
  diagonal: number;             // of the Target's bounding box
  learnerDistances: Float64Array; // per sample of the Learner surface: distance to the Target (for the heatmap)
  targetDistances: Float64Array;  // per sample of the Target: distance to the Learner surface
}

// the sums a direction contributes; the GPU's reduction pass produces the same three numbers
export interface DistanceSums {
  sumWeightedSquares: number; // sum of w * d^2
  sumWeights: number;         // sum of w
  max: number;                // largest d
}

export interface DistanceOptions {
  res?: number;                    // samples per direction before the knots, default ERROR_RES
  mode?: 'triangle' | 'point';     // 'point' = nearest sample instead of nearest triangle, only for experiment E1
  learnerShift?: number;           // shifts the Learner's grid by this fraction of a cell (Noise floor: 0.5)
  align?: boolean;                 // line up first: the best fit from the pairs of the same grid point (alignment.ts)
}

// ---------------------------------------------------------
// Sampling
// ---------------------------------------------------------

// the surface on the error grid, with the area every sample stands for:
// every grid cell (two triangles) gives a quarter of its area to each of its four corners
export function sampleSurface(net: ControlNet, res = ERROR_RES, shift = 0): SampledSurface {
  const us = samplePositions(net.knotsU, net.p, res, shift);
  const vs = samplePositions(net.knotsV, net.q, res, shift);
  const resU = us.length;
  const resV = vs.length;
  const points = new Float64Array(resU * resV * 3);
  for (let j = 0; j < resV; j++) {
    for (let i = 0; i < resU; i++) points.set(evaluate(net, us[i], vs[j]).point, (j * resU + i) * 3);
  }

  const weights = new Float64Array(resU * resV);
  for (let j = 0; j + 1 < resV; j++) {
    for (let i = 0; i + 1 < resU; i++) {
      const [a, b, c, d] = cellCorners(resU, i, j);
      const quarter = (triangleArea(points, a, b, d) + triangleArea(points, a, d, c)) / 4;
      weights[a] += quarter;
      weights[b] += quarter;
      weights[c] += quarter;
      weights[d] += quarter;
    }
  }
  return { resU, resV, points, weights };
}

// the four corners of grid cell (i, j), split into the triangles (a, b, d) and (a, d, c) like the render grid
function cellCorners(resU: number, i: number, j: number): [number, number, number, number] {
  const a = j * resU + i;
  return [a, a + 1, a + resU, a + resU + 1];
}

function triangleArea(p: Float64Array, a: number, b: number, c: number): number {
  return Math.sqrt(doubledAreaSquared(p, a, b, c)) / 2;
}

function doubledAreaSquared(p: Float64Array, a: number, b: number, c: number): number {
  const ux = p[b * 3] - p[a * 3], uy = p[b * 3 + 1] - p[a * 3 + 1], uz = p[b * 3 + 2] - p[a * 3 + 2];
  const vx = p[c * 3] - p[a * 3], vy = p[c * 3 + 1] - p[a * 3 + 1], vz = p[c * 3 + 2] - p[a * 3 + 2];
  const cx = uy * vz - uz * vy, cy = uz * vx - ux * vz, cz = ux * vy - uy * vx;
  return cx * cx + cy * cy + cz * cz;
}

// the diagonal of the bounding box of the samples
export function boundingDiagonal(s: SampledSurface): number {
  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  for (let k = 0; k < s.points.length; k += 3) {
    for (let c = 0; c < 3; c++) {
      min[c] = Math.min(min[c], s.points[k + c]);
      max[c] = Math.max(max[c], s.points[k + c]);
    }
  }
  return Math.hypot(max[0] - min[0], max[1] - min[1], max[2] - min[2]);
}

// ---------------------------------------------------------
// Distances
// ---------------------------------------------------------

// squared distance from p to the triangle (a, b, c)
export function pointTriangleDistance2(
  px: number, py: number, pz: number,
  ax: number, ay: number, az: number,
  bx: number, by: number, bz: number,
  cx: number, cy: number, cz: number,
): number {
  const [qx, qy, qz] = closestOnTriangle(px, py, pz, ax, ay, az, bx, by, bz, cx, cy, cz);
  const dx = px - qx, dy = py - qy, dz = pz - qz;
  return dx * dx + dy * dy + dz * dz;
}

// the point of the triangle (a, b, c) nearest to p
// Ericson, Real-Time Collision Detection, 5.1.5: find the Voronoi region of p (a corner, an edge or the inside)
export function closestOnTriangle(
  px: number, py: number, pz: number,
  ax: number, ay: number, az: number,
  bx: number, by: number, bz: number,
  cx: number, cy: number, cz: number,
): [number, number, number] {
  const abx = bx - ax, aby = by - ay, abz = bz - az;
  const acx = cx - ax, acy = cy - ay, acz = cz - az;
  const apx = px - ax, apy = py - ay, apz = pz - az;
  const d1 = abx * apx + aby * apy + abz * apz;
  const d2 = acx * apx + acy * apy + acz * apz;
  let qx: number, qy: number, qz: number; // the closest point
  if (d1 <= 0 && d2 <= 0) {
    qx = ax; qy = ay; qz = az; // corner a
  } else {
    const bpx = px - bx, bpy = py - by, bpz = pz - bz;
    const d3 = abx * bpx + aby * bpy + abz * bpz;
    const d4 = acx * bpx + acy * bpy + acz * bpz;
    const cpx = px - cx, cpy = py - cy, cpz = pz - cz;
    const d5 = abx * cpx + aby * cpy + abz * cpz;
    const d6 = acx * cpx + acy * cpy + acz * cpz;
    const vc = d1 * d4 - d3 * d2;
    const vb = d5 * d2 - d1 * d6;
    const va = d3 * d6 - d5 * d4;
    if (d3 >= 0 && d4 <= d3) {
      qx = bx; qy = by; qz = bz; // corner b
    } else if (d6 >= 0 && d5 <= d6) {
      qx = cx; qy = cy; qz = cz; // corner c
    } else if (vc <= 0 && d1 >= 0 && d3 <= 0) {
      const t = d1 / (d1 - d3); // edge ab
      qx = ax + t * abx; qy = ay + t * aby; qz = az + t * abz;
    } else if (vb <= 0 && d2 >= 0 && d6 <= 0) {
      const t = d2 / (d2 - d6); // edge ac
      qx = ax + t * acx; qy = ay + t * acy; qz = az + t * acz;
    } else if (va <= 0 && d4 - d3 >= 0 && d5 - d6 >= 0) {
      const t = (d4 - d3) / (d4 - d3 + (d5 - d6)); // edge bc
      qx = bx + t * (cx - bx); qy = by + t * (cy - by); qz = bz + t * (cz - bz);
    } else {
      const denom = 1 / (va + vb + vc); // inside: barycentric coordinates
      const v = vb * denom;
      const w = vc * denom;
      qx = ax + abx * v + acx * w; qy = ay + aby * v + acy * w; qz = az + abz * v + acz * w;
    }
  }
  return [qx, qy, qz];
}

// the triangles of a sampled surface as flat corner coordinates (9 per triangle) with a bounding box each,
// degenerate ones left out
interface Triangles {
  count: number;
  corners: Float64Array; // ax ay az bx by bz cx cy cz
  boxes: Float64Array;   // minx miny minz maxx maxy maxz
}

function triangles(s: SampledSurface): Triangles {
  const corners: number[] = [];
  const boxes: number[] = [];
  const p = s.points;
  const add = (a: number, b: number, c: number) => {
    if (doubledAreaSquared(p, a, b, c) < DEGENERATE) return;
    const xyz = [a, b, c].flatMap((k) => [p[k * 3], p[k * 3 + 1], p[k * 3 + 2]]);
    corners.push(...xyz);
    for (let axis = 0; axis < 3; axis++) boxes.push(Math.min(xyz[axis], xyz[axis + 3], xyz[axis + 6]));
    for (let axis = 0; axis < 3; axis++) boxes.push(Math.max(xyz[axis], xyz[axis + 3], xyz[axis + 6]));
  };
  for (let j = 0; j + 1 < s.resV; j++) {
    for (let i = 0; i + 1 < s.resU; i++) {
      const [a, b, c, d] = cellCorners(s.resU, i, j);
      add(a, b, d);
      add(a, d, c);
    }
  }
  return { count: corners.length / 9, corners: new Float64Array(corners), boxes: new Float64Array(boxes) };
}

// squared distance from p to a box (0 inside): a lower bound for every triangle in it
function boxDistance2(b: Float64Array, k: number, px: number, py: number, pz: number): number {
  const dx = Math.max(b[k] - px, 0, px - b[k + 3]);
  const dy = Math.max(b[k + 1] - py, 0, py - b[k + 4]);
  const dz = Math.max(b[k + 2] - pz, 0, pz - b[k + 5]);
  return dx * dx + dy * dy + dz * dz;
}

// for every sample of `from`: the distance to the nearest triangle (or sample) of `to`
// brute force like the GPU, the boxes only skip triangles that can't be nearer
// closest (optional, 3 per sample): receives the nearest point itself (for the best fit's refinement)
export function oneSidedDistances(from: SampledSurface, to: SampledSurface, mode: 'triangle' | 'point' = 'triangle', closest?: Float64Array): Float64Array {
  const count = from.points.length / 3;
  const out = new Float64Array(count);
  const tris = mode === 'triangle' ? triangles(to) : undefined;
  for (let s = 0; s < count; s++) {
    const px = from.points[s * 3], py = from.points[s * 3 + 1], pz = from.points[s * 3 + 2];
    let best = Infinity;
    if (tris) {
      const { corners: t, boxes } = tris;
      for (let k = 0; k < tris.count; k++) {
        if (boxDistance2(boxes, k * 6, px, py, pz) >= best) continue;
        const o = k * 9;
        const q = closestOnTriangle(px, py, pz, t[o], t[o + 1], t[o + 2], t[o + 3], t[o + 4], t[o + 5], t[o + 6], t[o + 7], t[o + 8]);
        const d2 = (px - q[0]) ** 2 + (py - q[1]) ** 2 + (pz - q[2]) ** 2;
        if (d2 < best) { best = d2; closest?.set(q, s * 3); }
      }
    } else {
      const q = to.points;
      for (let k = 0; k < q.length; k += 3) {
        const dx = px - q[k], dy = py - q[k + 1], dz = pz - q[k + 2];
        best = Math.min(best, dx * dx + dy * dy + dz * dz);
      }
    }
    out[s] = Math.sqrt(best);
  }
  return out;
}

export function sums(distances: ArrayLike<number>, weights: ArrayLike<number>): DistanceSums {
  let sumWeightedSquares = 0, sumWeights = 0, max = 0;
  for (let k = 0; k < distances.length; k++) {
    sumWeightedSquares += weights[k] * distances[k] * distances[k];
    sumWeights += weights[k];
    max = Math.max(max, distances[k]);
  }
  return { sumWeightedSquares, sumWeights, max };
}

// both directions together, divided by the diagonal: shared by the CPU and the GPU path
export function combine(a: DistanceSums, b: DistanceSums, diagonal: number): { rms: number; hausdorff: number } {
  const rms = Math.sqrt((a.sumWeightedSquares + b.sumWeightedSquares) / (a.sumWeights + b.sumWeights));
  return { rms: rms / diagonal, hausdorff: Math.max(a.max, b.max) / diagonal };
}

// the Shape distance between the Learner surface and the Target
// with align: the Learner surface is first turned and moved onto the Target, then measured to the nearest
// point, so only the shape counts:
//   1. a first best fit from the pairs of the same grid point (as for the Paired distance)
//   2. ICP_STEPS refinements (ICP, Besl & McKay 1992): each learner sample is paired with its nearest point
//      on the Target, and the best fit of these pairs is the next one. The pairs of step 1 depend on how the
//      control points are spread along the surface (a hidden cap point, a ring slid along the profile);
//      the nearest points don't, so the fit ends up depending on the shape only
// Needs the same Layout for step 1; with another Layout (after a knot insertion) it measures without it
export function shapeDistance(target: ControlNet, learner: ControlNet, options: DistanceOptions = {}): ShapeDistance & { fit: BestFit } {
  const { res = ERROR_RES, mode = 'triangle', learnerShift = 0, align = false } = options;
  const t = sampleSurface(target, res);
  const l = sampleSurface(learner, res, learnerShift);
  let fit = NO_FIT;
  if (align && l.resU === t.resU && l.resV === t.resV) {
    const original = l.points.slice();
    const moveBy = (f: BestFit) => {
      for (let k = 0; k < original.length; k += 3) l.points.set(applyFit(f, [original[k], original[k + 1], original[k + 2]]), k);
    };
    fit = bestFit(original, t.points, t.weights);
    const closest = new Float64Array(original.length);
    for (let step = 0; step < ICP_STEPS; step++) {
      moveBy(fit);
      oneSidedDistances(l, t, mode, closest);
      fit = bestFit(original, closest, l.weights);
    }
    moveBy(fit);
  }
  const diagonal = boundingDiagonal(t);
  const learnerDistances = oneSidedDistances(l, t, mode);
  const targetDistances = oneSidedDistances(t, l, mode);
  const { rms, hausdorff } = combine(sums(learnerDistances, l.weights), sums(targetDistances, t.weights), diagonal);
  return { rms, hausdorff, diagonal, learnerDistances, targetDistances, fit };
}

// ---------------------------------------------------------
// The Paired distance: the teacher's simpler grader
// ---------------------------------------------------------
// both surfaces on the same (u, v) grid; sample (i, j) of the Learner surface is compared with sample (i, j)
// of the Target, no search. Each pair counts with the area its Target sample stands for. RMS and max,
// divided by the Target's diagonal like the Shape distance.
// It is never smaller than the geometric gap, so it never passes a wrong shape, but a correct shape whose
// points are spread differently along the surface (a cap point moved inside the end face, other weights
// for the same circle) counts as off (docs/research/shape-distance-parameter-wise.md).
// It needs one Layout for both surfaces (ADR 0001), so both grids have the same parameter values.

// which grader measures the Shape distance: nearest point on triangles (ADR 0002) or the Paired distance
export type Metric = 'nearest' | 'paired';
export const PAIRED_RES = 128; // the teacher's grid: cheap enough for 128 x 128

// align: turn and move the Learner surface onto the Target first (the best fit, alignment.ts), so only
// the shape counts, not where it stands or how it is turned
export function pairedDistance(target: ControlNet, learner: ControlNet, res = PAIRED_RES, align = false): ShapeDistance & { fit: BestFit } {
  const t = sampleSurface(target, res);
  const l = sampleSurface(learner, res);
  if (t.resU !== l.resU || t.resV !== l.resV) throw new Error('The Paired distance needs the same Layout for both surfaces');
  const fit = align ? bestFit(l.points, t.points, t.weights) : NO_FIT;
  const diagonal = boundingDiagonal(t);
  const d = new Float64Array(t.resU * t.resV);
  for (let s = 0; s < d.length; s++) {
    const k = s * 3;
    const p = align ? applyFit(fit, [l.points[k], l.points[k + 1], l.points[k + 2]]) : [l.points[k], l.points[k + 1], l.points[k + 2]];
    d[s] = Math.hypot(t.points[k] - p[0], t.points[k + 1] - p[1], t.points[k + 2] - p[2]);
  }
  const both = sums(d, t.weights);
  const { rms, hausdorff } = combine(both, both, diagonal);
  return { rms, hausdorff, diagonal, learnerDistances: d, targetDistances: d, fit };
}
