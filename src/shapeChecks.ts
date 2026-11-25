// checks for the Targets of src/shapes.ts: is the Layout valid, do closed shapes close, are circles round
// used by the tests (src/shapes.test.ts) and by scripts/shapes-preview.ts
import { evaluate } from './nurbs.ts';
import { CAR_WHEELS, type Shape } from './shapes.ts';
import type { ControlNet, Point3 } from './controlNet.ts';

export const CHECK_RES = 40;  // samples per direction
export const CHECK_TOL = 1e-5; // the control points are stored as f32

const dist = (a: Point3, b: Point3) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);

// grid[j][i] = S(u_i, v_j) on a uniform res x res grid
export function sampleGrid(net: ControlNet, res: number): Point3[][] {
  const grid: Point3[][] = [];
  for (let j = 0; j < res; j++) {
    const row: Point3[] = [];
    for (let i = 0; i < res; i++) row.push(evaluate(net, i / (res - 1), j / (res - 1)).point);
    grid.push(row);
  }
  return grid;
}

// the Layout: degrees, knot counts, clamped and non-decreasing knots, weights > 0
export function checkLayout(net: ControlNet): string[] {
  const errors: string[] = [];
  const knots: [number[], number, number, string][] = [[net.knotsU, net.countU, net.p, 'U'], [net.knotsV, net.countV, net.q, 'V']];
  for (const [U, count, deg, name] of knots) {
    if (deg < 1 || deg > 3) errors.push(`degree ${name} = ${deg} is outside 1..3 (MAX_DEGREE in tessellate.wgsl)`);
    if (U.length !== count + deg + 1) errors.push(`knots ${name}: ${U.length} values, expected ${count + deg + 1}`);
    if (U.some((k, i) => i > 0 && k < U[i - 1])) errors.push(`knots ${name} decrease`);
    if (U.slice(0, deg + 1).some((k) => k !== 0) || U.slice(-deg - 1).some((k) => k !== 1)) errors.push(`knots ${name} are not clamped to [0, 1]`);
  }
  if (net.points.length !== net.countU * net.countV * 4) errors.push('wrong number of control points');
  for (let k = 3; k < net.points.length; k += 4) if (!(net.points[k] > 0)) { errors.push('a weight is not > 0'); break; }
  return errors;
}

// the max distance between the samples of two rows/columns
function maxGap(a: Point3[], b: Point3[]): number {
  return Math.max(...a.map((p, i) => dist(p, b[i])));
}

// how much the distance from the z axis varies along each row (0 for exact circles around z)
export function roundness(grid: Point3[][]): number {
  let worst = 0;
  for (const row of grid) {
    const radii = row.map(([x, y]) => Math.hypot(x, y));
    worst = Math.max(worst, Math.max(...radii) - Math.min(...radii));
  }
  return worst;
}

// car: every point of the side outline below an axle, within a wheel's radius of its centre,
// lies on that wheel's circle. The outline is the middle of the extrusion (v = 0.5)
export function carWheelError(net: ControlNet, res = 2000): number {
  const { radius, axle, centres } = CAR_WHEELS;
  let worst = 0;
  for (let i = 0; i <= res; i++) {
    const [, y, z] = evaluate(net, i / res, 0.5).point;
    if (z > axle - 1e-9) continue;
    for (const yc of centres) {
      if (Math.abs(y - yc) <= radius + 1e-9) worst = Math.max(worst, Math.abs(Math.hypot(y - yc, z - axle) - radius));
    }
  }
  return worst;
}

// every check that applies to the shape; an empty list means it is fine
export function checkShape(shape: Shape, grid = sampleGrid(shape.net, CHECK_RES)): string[] {
  const errors = checkLayout(shape.net);
  const last = grid.length - 1;
  const column = (i: number) => grid.map((row) => row[i]);
  const kind = shape.construction;
  // closed shapes: the seam u = 0 and u = 1 must be the same curve
  if (/^(revolution|loft|extrusion)/.test(kind)) {
    const gap = maxGap(column(0), column(grid[0].length - 1));
    if (gap > CHECK_TOL) errors.push(`seam is open: ${gap.toExponential(2)}`);
  }
  if (kind.startsWith('revolution')) {
    const worst = roundness(grid);
    if (worst > CHECK_TOL) errors.push(`not round: radius varies by ${worst.toExponential(2)}`);
  }
  if (shape.id === 'car') {
    const worst = carWheelError(shape.net);
    if (worst > CHECK_TOL) errors.push(`wheels are not round: radius off by ${worst.toExponential(2)}`);
  }
  // extrusions: the first and last rows are closed caps (a single point)
  if (kind.startsWith('extrusion')) {
    for (const row of [grid[0], grid[last]]) if (maxGap(row, row.map(() => row[0])) > CHECK_TOL) errors.push('a cap is not closed');
  }
  return errors;
}
