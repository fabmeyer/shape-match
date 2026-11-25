// where a surface is sampled along one parameter direction: a uniform grid plus every distinct interior knot
// a knot is where the pieces of a NURBS meet; with multiplicity >= degree the surface has a crease there
// (only C0), which a uniform grid would step over. So the knots are always samples, and at a crease the
// knot gets a second sample just before it, so the triangles on both sides get their own normal

// distance of the extra sample before a crease, in parameter units (f32 resolves about 6e-8 near 1)
export const CREASE_GAP = 1e-5;

// two values closer than this count as the same sample
const SAME = 1e-7;

// distinct knot values with their multiplicity
export function knotMultiplicities(knots: number[]): Map<number, number> {
  const counts = new Map<number, number>();
  for (const knot of knots) counts.set(knot, (counts.get(knot) ?? 0) + 1);
  return counts;
}

// sorted parameter values in [0, 1] (clamped knot vectors span [0, 1]):
//   the n uniform values i / (n - 1), or with a shift s the values (i + s) / (n - 1) that lie inside,
//   with 0 and 1 always included (shift = 0.5 is the grid shifted by half a cell, for the Noise floor),
//   plus every distinct interior knot that is not already one of them,
//   plus knot - CREASE_GAP for every interior knot with multiplicity >= degree
// so the count is n + (interior knots not on the grid) + (crease knots)
export function samplePositions(knots: number[], degree: number, n: number, shift = 0): number[] {
  if (n < 2) throw new Error('need at least 2 samples');
  const values: number[] = [0, 1];
  for (let i = 0; i < n; i++) {
    const t = (i + shift) / (n - 1);
    if (t > 0 && t < 1) values.push(t);
  }
  const first = knots[0];
  const last = knots[knots.length - 1];
  for (const [knot, count] of knotMultiplicities(knots)) {
    if (knot <= first || knot >= last) continue; // the ends are already samples
    values.push(knot);
    if (count >= degree) values.push(knot - CREASE_GAP);
  }
  values.sort((a, b) => a - b);

  // drop uniform values that fall on a knot; keep the knot itself (exact) and the crease sample
  const isKnot = new Set(knots);
  const out: number[] = [];
  for (const t of values) {
    const prev = out[out.length - 1];
    if (prev !== undefined && t - prev < SAME) {
      if (isKnot.has(t)) out[out.length - 1] = t;
      continue;
    }
    out.push(t);
  }
  return out;
}
