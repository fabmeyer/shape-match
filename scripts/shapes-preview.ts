// checks every shape of src/shapes.ts, writes one OBJ per shape and a contact sheet
// run: node scripts/shapes-preview.ts   (Node 22.18+ runs TypeScript directly)
// writes docs/shapes/obj/<id>.obj and docs/shapes/preview.bmp (convert to PNG with sips on macOS)
import { mkdirSync, writeFileSync } from 'node:fs';
import { evaluate } from '../src/nurbs.ts';
import { allShapes, type Shape } from '../src/shapes.ts';
import type { ControlNet, Point3 } from '../src/controlNet.ts';

const RES = 40; // samples per direction
const OUT = new URL('../docs/shapes/', import.meta.url);

type V3 = [number, number, number];
const sub = (a: V3, b: V3): V3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot = (a: V3, b: V3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a: V3, b: V3): V3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = (a: V3): V3 => { const l = Math.hypot(...a); return [a[0] / l, a[1] / l, a[2] / l]; };
const dist = (a: V3, b: V3) => Math.hypot(...sub(a, b));

function sampleGrid(net: ControlNet, res: number): Point3[][] {
  const grid: Point3[][] = [];
  for (let j = 0; j < res; j++) {
    const row: Point3[] = [];
    for (let i = 0; i < res; i++) row.push(evaluate(net, i / (res - 1), j / (res - 1)).point);
    grid.push(row);
  }
  return grid; // grid[j][i] = S(u_i, v_j)
}

// ---------------------------------------------------------
// Checks
// ---------------------------------------------------------

const TOL = 1e-5; // the control points are stored as f32

function checkStructure(net: ControlNet): string[] {
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

function checkShape(shape: Shape, grid: Point3[][]): string[] {
  const errors = checkStructure(shape.net);
  const column = (i: number) => grid.map((row) => row[i]);
  const closedAround = shape.construction.startsWith('revolution') || shape.construction.startsWith('loft') || shape.construction.startsWith('extrusion');
  // closed shapes: the seam u = 0 and u = 1 must be the same curve
  if (closedAround && maxGap(column(0), column(RES - 1)) > TOL) errors.push(`seam is open: ${maxGap(column(0), column(RES - 1)).toExponential(2)}`);
  // revolutions: every sample at the same v has the same distance from the z axis (an exact circle)
  if (shape.construction.startsWith('revolution')) {
    let worst = 0;
    for (const row of grid) {
      const radii = row.map(([x, y]) => Math.hypot(x, y));
      worst = Math.max(worst, Math.max(...radii) - Math.min(...radii));
    }
    if (worst > TOL) errors.push(`not round: radius varies by ${worst.toExponential(2)}`);
  }
  // car: the lowest point of every section over a wheel lies on a circle (radius 0.27 below the sill at 0.3)
  if (shape.id === 'car') {
    let worst = 0;
    for (let j = 0; j <= 400; j++) {
      let low: Point3 | undefined;
      for (let i = 0; i <= 400; i++) {
        const p = evaluate(shape.net, i / 400, j / 400).point;
        if (!low || p[2] < low[2]) low = p;
      }
      const [, y, z] = low!;
      for (const yc of [-0.82, 0.85]) {
        if (Math.abs(y - yc) < 0.25) worst = Math.max(worst, Math.abs(Math.hypot(y - yc, z - 0.3) - 0.27));
      }
    }
    if (worst > TOL) errors.push(`wheels are not round: radius off by ${worst.toExponential(2)}`);
  }
  // extrusions: the first and last rows are closed caps (a single point)
  if (shape.construction.startsWith('extrusion')) {
    for (const row of [grid[0], grid[RES - 1]]) if (maxGap(row, row.map(() => row[0])) > TOL) errors.push('a cap is not closed');
  }
  return errors;
}

// ---------------------------------------------------------
// OBJ export (for Quick Look, Blender, ...)
// ---------------------------------------------------------

function toObj(shape: Shape, grid: Point3[][]): string {
  const lines = [`# ${shape.order} (${shape.construction}), sampled ${RES} x ${RES} from src/shapes.ts`, `o ${shape.id}`];
  for (const row of grid) for (const [x, y, z] of row) lines.push(`v ${x.toFixed(5)} ${y.toFixed(5)} ${z.toFixed(5)}`);
  for (let j = 0; j < RES - 1; j++) {
    for (let i = 0; i < RES - 1; i++) {
      const a = j * RES + i + 1; // OBJ counts from 1
      lines.push(`f ${a} ${a + 1} ${a + RES + 1} ${a + RES}`);
    }
  }
  return lines.join('\n') + '\n';
}

// ---------------------------------------------------------
// Contact sheet: a tiny z-buffer rasteriser, one tile per shape
// ---------------------------------------------------------

const TILE = 300;
const COLS = 4;
const BG: V3 = [244, 238, 226];   // paper
const WOOD: V3 = [205, 140, 85];  // surface colour

// digits as 3 x 5 bitmaps, for the number in the corner of each tile
const DIGITS = ['111101101101111', '010110010010111', '111001111100111', '111001111001111', '101101111001001',
  '111100111001111', '111100111101111', '111001001001001', '111101111101111', '111101111001111'];

function contactSheet(shapes: Shape[], grids: Point3[][][]): { width: number; height: number; rgb: Uint8Array } {
  const rows = Math.ceil(shapes.length / COLS);
  const width = COLS * TILE;
  const height = rows * TILE;
  const rgb = new Uint8Array(width * height * 3);
  for (let k = 0; k < width * height; k++) rgb.set(BG, k * 3);
  const depth = new Float32Array(width * height).fill(Infinity);

  // view from the front left and a bit above, orthographic
  const az = (60 * Math.PI) / 180;
  const el = (25 * Math.PI) / 180;
  const eye = norm([Math.sin(az) * Math.cos(el), -Math.cos(az) * Math.cos(el), Math.sin(el)]);
  const forward: V3 = [-eye[0], -eye[1], -eye[2]];
  const right = norm(cross(forward, [0, 0, 1]));
  const up = cross(right, forward);
  const light = norm([0.2 * eye[0] - 0.5, 0.2 * eye[1] - 0.6, 0.2 * eye[2] + 0.7]);

  grids.forEach((grid, s) => {
    const ox = (s % COLS) * TILE;
    const oy = Math.floor(s / COLS) * TILE;
    const flat = grid.flat();
    const sx = flat.map((p) => dot(p, right));
    const sy = flat.map((p) => dot(p, up));
    const scale = (TILE * 0.8) / Math.max(Math.max(...sx) - Math.min(...sx), Math.max(...sy) - Math.min(...sy));
    const mx = (Math.max(...sx) + Math.min(...sx)) / 2;
    const my = (Math.max(...sy) + Math.min(...sy)) / 2;
    const project = (p: Point3): V3 => [ox + TILE / 2 + (dot(p, right) - mx) * scale, oy + TILE / 2 - (dot(p, up) - my) * scale, dot(p, forward)];

    const triangle = (a: Point3, b: Point3, c: Point3) => {
      const n = cross(sub(b, a), sub(c, a));
      if (Math.hypot(...n) < 1e-12) return; // collapsed (cap or tip)
      const shade = 0.25 + 0.75 * Math.abs(dot(norm(n), light)); // both sides lit
      const [A, B, C] = [project(a), project(b), project(c)];
      const x0 = Math.max(ox, Math.floor(Math.min(A[0], B[0], C[0])));
      const x1 = Math.min(ox + TILE - 1, Math.ceil(Math.max(A[0], B[0], C[0])));
      const y0 = Math.max(oy, Math.floor(Math.min(A[1], B[1], C[1])));
      const y1 = Math.min(oy + TILE - 1, Math.ceil(Math.max(A[1], B[1], C[1])));
      const area = (B[0] - A[0]) * (C[1] - A[1]) - (B[1] - A[1]) * (C[0] - A[0]);
      if (Math.abs(area) < 1e-9) return;
      for (let y = y0; y <= y1; y++) {
        for (let x = x0; x <= x1; x++) {
          const px = x + 0.5, py = y + 0.5;
          const w0 = ((B[0] - px) * (C[1] - py) - (B[1] - py) * (C[0] - px)) / area;
          const w1 = ((C[0] - px) * (A[1] - py) - (C[1] - py) * (A[0] - px)) / area;
          const w2 = 1 - w0 - w1;
          if (w0 < 0 || w1 < 0 || w2 < 0) continue;
          const z = w0 * A[2] + w1 * B[2] + w2 * C[2];
          const k = y * width + x;
          if (z >= depth[k]) continue;
          depth[k] = z;
          rgb.set(WOOD.map((c) => Math.min(255, c * shade)), k * 3);
        }
      }
    };
    for (let j = 0; j < RES - 1; j++) {
      for (let i = 0; i < RES - 1; i++) {
        triangle(grid[j][i], grid[j][i + 1], grid[j + 1][i + 1]);
        triangle(grid[j][i], grid[j + 1][i + 1], grid[j + 1][i]);
      }
    }

    // the shape's number in the top left corner
    String(shapes[s].difficulty).split('').forEach((d, n) => {
      const bits = DIGITS[Number(d)];
      for (let by = 0; by < 5; by++) for (let bx = 0; bx < 3; bx++) {
        if (bits[by * 3 + bx] !== '1') continue;
        for (let yy = 0; yy < 4; yy++) for (let xx = 0; xx < 4; xx++) {
          const x = ox + 10 + n * 16 + bx * 4 + xx;
          const y = oy + 10 + by * 4 + yy;
          rgb.set([90, 60, 40], (y * width + x) * 3);
        }
      }
    });
  });
  return { width, height, rgb };
}

function bmp({ width, height, rgb }: { width: number; height: number; rgb: Uint8Array }): Uint8Array {
  const rowSize = Math.ceil((width * 3) / 4) * 4;
  const size = 54 + rowSize * height;
  const out = new Uint8Array(size);
  const view = new DataView(out.buffer);
  out.set([0x42, 0x4d]);
  view.setUint32(2, size, true);
  view.setUint32(10, 54, true);
  view.setUint32(14, 40, true);
  view.setInt32(18, width, true);
  view.setInt32(22, height, true); // positive: rows go bottom up
  view.setUint16(26, 1, true);
  view.setUint16(28, 24, true);
  view.setUint32(34, rowSize * height, true);
  for (let y = 0; y < height; y++) {
    const row = 54 + (height - 1 - y) * rowSize;
    for (let x = 0; x < width; x++) {
      const k = (y * width + x) * 3;
      out.set([rgb[k + 2], rgb[k + 1], rgb[k]], row + x * 3); // BGR
    }
  }
  return out;
}

// ---------------------------------------------------------

const shapes = allShapes();
const grids = shapes.map((s) => sampleGrid(s.net, RES));
mkdirSync(new URL('obj/', OUT), { recursive: true });

let failed = 0;
shapes.forEach((shape, k) => {
  const errors = checkShape(shape, grids[k]);
  const { countU, countV, p, q } = shape.net;
  console.log(`${errors.length ? 'FAIL' : 'ok  '} ${String(shape.difficulty).padStart(2)} ${shape.id.padEnd(9)} ${countU}x${countV} points, degree ${p}x${q}, ${shape.construction}`);
  for (const e of errors) console.log(`       ${e}`);
  failed += errors.length ? 1 : 0;
  writeFileSync(new URL(`obj/${shape.id}.obj`, OUT), toObj(shape, grids[k]));
});
writeFileSync(new URL('preview.bmp', OUT), bmp(contactSheet(shapes, grids)));
console.log(failed ? `${failed} shape(s) failed` : 'all shapes ok');
process.exitCode = failed ? 1 : 0;
