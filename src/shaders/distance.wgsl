// compute pass: one direction of the Shape distance (ADR 0002), one thread per sample of the "from" surface
//   distance: to the nearest triangle of the "to" surface (brute force over all its triangles)
//   weight:   the area the sample stands for, a quarter of each grid cell around it
//   closest:  the nearest point itself, for the best fit's refinement (ICP, see errorPass.ts)
// both surfaces come from tessellate.wgsl; the grid cells are split into the triangles (a, b, d) and (a, d, c)
// the same steps as shapeDistance.ts on the CPU

// a triangle with a smaller squared doubled area is a point or a line (collapsed cap, tip): left out
const DEGENERATE : f32 = 1e-20;

struct Params {
  fromResU : u32,
  fromResV : u32,
  toResU   : u32,
  toResV   : u32,
};

@group(0) @binding(0) var<uniform> params : Params;
@group(0) @binding(1) var<storage, read> fromPositions : array<vec4f>;
@group(0) @binding(2) var<storage, read> toPositions : array<vec4f>;
@group(0) @binding(3) var<storage, read_write> distances : array<f32>;
@group(0) @binding(4) var<storage, read_write> weights : array<f32>;
@group(0) @binding(5) var<storage, read_write> closest : array<vec4f>;

// the point of the triangle (a, b, c) nearest to p
// Ericson, Real-Time Collision Detection, 5.1.5: find the Voronoi region of p (a corner, an edge or the inside)
fn closestOnTriangle(p : vec3f, a : vec3f, b : vec3f, c : vec3f) -> vec3f {
  let ab = b - a;
  let ac = c - a;
  let ap = p - a;
  let d1 = dot(ab, ap);
  let d2 = dot(ac, ap);
  if (d1 <= 0.0 && d2 <= 0.0) { return a; } // corner a

  let bp = p - b;
  let d3 = dot(ab, bp);
  let d4 = dot(ac, bp);
  if (d3 >= 0.0 && d4 <= d3) { return b; } // corner b

  let cp = p - c;
  let d5 = dot(ab, cp);
  let d6 = dot(ac, cp);
  if (d6 >= 0.0 && d5 <= d6) { return c; } // corner c

  let vc = d1 * d4 - d3 * d2;
  if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) { return a + d1 / (d1 - d3) * ab; } // edge ab
  let vb = d5 * d2 - d1 * d6;
  if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) { return a + d2 / (d2 - d6) * ac; } // edge ac
  let va = d3 * d6 - d5 * d4;
  if (va <= 0.0 && d4 - d3 >= 0.0 && d5 - d6 >= 0.0) { // edge bc
    return b + (d4 - d3) / ((d4 - d3) + (d5 - d6)) * (c - b);
  }
  let denom = 1.0 / (va + vb + vc); // inside: barycentric coordinates
  return a + ab * (vb * denom) + ac * (vc * denom);
}

fn nearer(p : vec3f, q : vec3f, best : ptr<function, f32>, bestPoint : ptr<function, vec3f>) {
  let d = dot(p - q, p - q);
  if (d < *best) { *best = d; *bestPoint = q; }
}

fn isDegenerate(a : vec3f, b : vec3f, c : vec3f) -> bool {
  let n = cross(b - a, c - a);
  return dot(n, n) < DEGENERATE;
}

fn toPoint(i : u32, j : u32) -> vec3f {
  return toPositions[j * params.toResU + i].xyz;
}

fn fromPoint(i : u32, j : u32) -> vec3f {
  return fromPositions[j * params.fromResU + i].xyz;
}

// area of grid cell (i, j) of the "from" surface: its two triangles
fn cellArea(i : u32, j : u32) -> f32 {
  let a = fromPoint(i, j);
  let b = fromPoint(i + 1u, j);
  let c = fromPoint(i, j + 1u);
  let d = fromPoint(i + 1u, j + 1u);
  return 0.5 * (length(cross(b - a, d - a)) + length(cross(d - a, c - a)));
}

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) id : vec3u) {
  let count = params.fromResU * params.fromResV;
  let s = id.x;
  if (s >= count) { return; }
  let i = s % params.fromResU;
  let j = s / params.fromResU;
  let p = fromPositions[s].xyz;

  var best = 3.0e38;
  var bestPoint = p;
  for (var tj = 0u; tj + 1u < params.toResV; tj++) {
    for (var ti = 0u; ti + 1u < params.toResU; ti++) {
      let a = toPoint(ti, tj);
      let b = toPoint(ti + 1u, tj);
      let c = toPoint(ti, tj + 1u);
      let d = toPoint(ti + 1u, tj + 1u);
      if (!isDegenerate(a, b, d)) { nearer(p, closestOnTriangle(p, a, b, d), &best, &bestPoint); }
      if (!isDegenerate(a, d, c)) { nearer(p, closestOnTriangle(p, a, d, c), &best, &bestPoint); }
    }
  }
  distances[s] = sqrt(best);
  closest[s] = vec4f(bestPoint, 1.0);

  // a quarter of every cell this sample is a corner of
  var area = 0.0;
  if (i + 1u < params.fromResU && j + 1u < params.fromResV) { area += cellArea(i, j); }
  if (i > 0u && j + 1u < params.fromResV) { area += cellArea(i - 1u, j); }
  if (i + 1u < params.fromResU && j > 0u) { area += cellArea(i, j - 1u); }
  if (i > 0u && j > 0u) { area += cellArea(i - 1u, j - 1u); }
  weights[s] = 0.25 * area;
}
