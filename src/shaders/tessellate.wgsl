// compute pass: evaluates the NURBS surface S(u, v) and its normal on a resU x resV grid,
// one thread per (u, v) sample. Both results are used as vertex buffers
// the parameter values come from samplePositions in samples.ts (a uniform grid plus the knots)
// The NURBS Book: A2.1 FindSpan, A2.3 DersBasisFuns (first derivatives only), A4.3 / A4.4

const MAX_DEGREE : u32 = 3u;
const MAX_P1 : u32 = MAX_DEGREE + 1u; // at most p + 1 basis functions are non-zero

struct Params {
  countU : u32, // control points in u
  countV : u32, // control points in v
  p      : u32, // degree in u
  q      : u32, // degree in v
  resU   : u32, // samples in u
  resV   : u32, // samples in v
  offV   : u32, // index where the v knots start in knots
  _pad   : u32, // unused, keeps the struct at 32 bytes
};

@group(0) @binding(0) var<uniform> params : Params;
@group(0) @binding(1) var<storage, read> controlPoints : array<vec4f>; // (wx, wy, wz, w)
@group(0) @binding(2) var<storage, read> knots : array<f32>;           // U, then V from offV
@group(0) @binding(3) var<storage, read_write> positions : array<vec4f>;
@group(0) @binding(4) var<storage, read_write> normals : array<vec4f>;
@group(0) @binding(5) var<storage, read> samples : array<f32>;         // resU values of u, then resV values of v

fn knot(off : u32, i : u32) -> f32 {
  return knots[off + i];
}

// binary search over the knots
fn findSpan(n : u32, p : u32, u : f32, off : u32) -> u32 {
  if (u >= knot(off, n + 1u)) { return n; } // u = 1 belongs to the last span
  if (u <= knot(off, p)) { return p; }
  var low = p;
  var high = n + 1u;
  var span = (low + high) / 2u;
  while (u < knot(off, span) || u >= knot(off, span + 1u)) {
    if (u < knot(off, span)) { high = span; } else { low = span; }
    span = (low + high) / 2u;
  }
  return span;
}

// the p + 1 non-zero basis functions at u and their first derivatives:
// N[k] = N_{span-p+k, p}(u), dN[k] = N'_{span-p+k, p}(u)
struct Basis {
  N  : array<f32, MAX_P1>,
  dN : array<f32, MAX_P1>,
};

// A2.3 for the first derivative. ndu holds the basis functions of all degrees
// (upper triangle, ndu[r][j] = N of degree j) and the knot differences (lower triangle)
fn dersBasisFuns(span : u32, u : f32, p : u32, off : u32) -> Basis {
  var ndu : array<array<f32, MAX_P1>, MAX_P1>;
  var left : array<f32, MAX_P1>;
  var right : array<f32, MAX_P1>;
  ndu[0][0] = 1.0;
  for (var j = 1u; j <= p; j++) {
    left[j] = u - knot(off, span + 1u - j);
    right[j] = knot(off, span + j) - u;
    var saved = 0.0;
    for (var r = 0u; r < j; r++) {
      ndu[j][r] = right[r + 1u] + left[j - r]; // knot difference
      let temp = ndu[r][j - 1u] / ndu[j][r];
      ndu[r][j] = saved + right[r + 1u] * temp;
      saved = left[j - r] * temp;
    }
    ndu[j][j] = saved;
  }

  var b : Basis;
  for (var k = 0u; k <= p; k++) {
    b.N[k] = ndu[k][p];
    // derivative from two basis functions of degree p - 1:
    // N'_{i,p} = p * (N_{i,p-1} / (u_{i+p} - u_i) - N_{i+1,p-1} / (u_{i+p+1} - u_{i+1}))
    var d = 0.0;
    if (k >= 1u) { d += ndu[k - 1u][p - 1u] / ndu[p][k - 1u]; }
    if (k < p) { d -= ndu[k][p - 1u] / ndu[p][k]; }
    b.dN[k] = f32(p) * d;
  }
  return b;
}

struct SurfaceSample {
  point  : vec3f,
  normal : vec3f,
};

// weighted sum of the active control points (in homogeneous coordinates),
// the same sum with the derivatives gives the two tangents
fn surfaceSample(u : f32, v : f32) -> SurfaceSample {
  let p = params.p;
  let q = params.q;
  let uspan = findSpan(params.countU - 1u, p, u, 0u);
  let vspan = findSpan(params.countV - 1u, q, v, params.offV);
  let bu = dersBasisFuns(uspan, u, p, 0u);
  let bv = dersBasisFuns(vspan, v, q, params.offV);

  var Sw = vec4f(0.0);  // the point (w*x, w*y, w*z, w)
  var Swu = vec4f(0.0); // its derivative in u
  var Swv = vec4f(0.0); // its derivative in v
  for (var l = 0u; l <= q; l++) {
    let row = (vspan - q + l) * params.countU; // start of control point row j
    for (var k = 0u; k <= p; k++) {
      let Pw = controlPoints[row + uspan - p + k];
      Sw += bu.N[k] * bv.N[l] * Pw;
      Swu += bu.dN[k] * bv.N[l] * Pw;
      Swv += bu.N[k] * bv.dN[l] * Pw;
    }
  }

  // back to 3D: S = A / w, and by the quotient rule S_u = (A_u - w_u * S) / w
  let w = Sw.w;
  let S = Sw.xyz / w;
  let Su = (Swu.xyz - Swu.w * S) / w; // tangent in u direction
  let Sv = (Swv.xyz - Swv.w * S) / w; // tangent in v direction

  // u runs along +x and v along +y on the start plane, so this normal points to +z there
  let n = cross(Su, Sv);
  let len = length(n);
  // a tangent of length 0 (e.g. two equal control points at the border) gives no direction
  return SurfaceSample(S, select(vec3f(0.0, 0.0, 1.0), n / len, len > 1e-12));
}

@compute @workgroup_size(8, 8)
fn cs_main(@builtin(global_invocation_id) id : vec3u) {
  // the grid is rounded up to whole workgroups: skip threads outside it
  if (id.x >= params.resU || id.y >= params.resV) { return; }

  let u = samples[id.x];
  let v = samples[params.resU + id.y];
  let sample = surfaceSample(u, v);
  let index = id.y * params.resU + id.x;
  positions[index] = vec4f(sample.point, 1.0);
  normals[index] = vec4f(sample.normal, 0.0);
}
