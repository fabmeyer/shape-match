// compute pass: the best fit of the Learner surface onto the Target, in a single workgroup
// (the same steps as bestFit() in alignment.ts on the CPU; Horn 1987, closed-form absolute orientation)
//   1. the centre of each surface: the mean of its samples, weighted by the Target's sample areas
//   2. the cross-covariance of the pairs around those centres (3 x 3)
//   3. thread 0: Horn's 4 x 4 matrix, its top eigenvector (Jacobi rotations) = the best rotation as a quaternion
// writes the rotation, both centres and the transform that paired.wgsl applies to the Learner surface
// sample s of the Learner surface belongs to sample s of the Target: both are on the same (u, v) grid

const THREADS : u32 = 256u;

struct Params {
  count : u32, // samples
};

struct Alignment {
  rotation      : vec4f,   // unit quaternion (x, y, z, w)
  learnerCentre : vec4f,
  targetCentre  : vec4f,
  transform     : mat4x4f, // p -> R (p - learnerCentre) + targetCentre
};

@group(0) @binding(0) var<uniform> params : Params;
@group(0) @binding(1) var<storage, read> learnerPositions : array<vec4f>;
@group(0) @binding(2) var<storage, read> targetPositions : array<vec4f>;
@group(0) @binding(3) var<storage, read> weights : array<f32>;
@group(0) @binding(4) var<storage, read_write> alignment : Alignment;

// three vec4 per thread, used twice: first for the centres, then for the 9 covariance sums
var<workgroup> a : array<vec4f, THREADS>;
var<workgroup> b : array<vec4f, THREADS>;
var<workgroup> c : array<vec4f, THREADS>;
var<workgroup> centres : array<vec3f, 2>;

fn sumUp(t : u32) {
  for (var stride = THREADS / 2u; stride > 0u; stride /= 2u) {
    if (t < stride) {
      a[t] += a[t + stride];
      b[t] += b[t + stride];
      c[t] += c[t + stride];
    }
    workgroupBarrier();
  }
}

// one Jacobi rotation in the (p, q) plane of the symmetric matrix m; v collects the eigenvectors (columns)
fn jacobi(m : ptr<function, array<array<f32, 4>, 4>>, v : ptr<function, array<array<f32, 4>, 4>>, p : u32, q : u32) {
  let apq = (*m)[p][q];
  if (abs(apq) < 1e-20) { return; }
  let theta = ((*m)[q][q] - (*m)[p][p]) / (2.0 * apq);
  var sgn = 1.0;
  if (theta < 0.0) { sgn = -1.0; }
  let t = sgn / (abs(theta) + sqrt(theta * theta + 1.0));
  let cs = 1.0 / sqrt(t * t + 1.0);
  let sn = t * cs;
  for (var k = 0u; k < 4u; k++) {
    let mkp = (*m)[k][p];
    let mkq = (*m)[k][q];
    (*m)[k][p] = cs * mkp - sn * mkq;
    (*m)[k][q] = sn * mkp + cs * mkq;
  }
  for (var k = 0u; k < 4u; k++) {
    let mpk = (*m)[p][k];
    let mqk = (*m)[q][k];
    (*m)[p][k] = cs * mpk - sn * mqk;
    (*m)[q][k] = sn * mpk + cs * mqk;
  }
  for (var k = 0u; k < 4u; k++) {
    let vkp = (*v)[k][p];
    let vkq = (*v)[k][q];
    (*v)[k][p] = cs * vkp - sn * vkq;
    (*v)[k][q] = sn * vkp + cs * vkq;
  }
}

fn writeIdentity() {
  alignment.rotation = vec4f(0.0, 0.0, 0.0, 1.0);
  alignment.learnerCentre = vec4f(0.0);
  alignment.targetCentre = vec4f(0.0);
  alignment.transform = mat4x4f(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
}

@compute @workgroup_size(256)
fn cs_main(@builtin(local_invocation_index) t : u32) {
  // 1. the weighted sums of the positions: a = (w, w p), b = (w q, 0)
  var sa = vec4f(0.0);
  var sb = vec4f(0.0);
  for (var k = t; k < params.count; k += THREADS) {
    let w = weights[k];
    sa += vec4f(w, w * learnerPositions[k].xyz);
    sb += vec4f(w * targetPositions[k].xyz, 0.0);
  }
  a[t] = sa;
  b[t] = sb;
  c[t] = vec4f(0.0);
  workgroupBarrier();
  sumUp(t);
  if (t == 0u) {
    let sw = max(a[0].x, 1e-30);
    centres[0] = a[0].yzw / sw;
    centres[1] = b[0].xyz / sw;
  }
  workgroupBarrier();
  let totalWeight = a[0].x; // every thread reads it before the arrays are reused
  workgroupBarrier();

  // 2. the cross-covariance S[i][j] = sum of w (p_i - cl_i)(q_j - ct_j)
  //    a = (Sxx, Sxy, Sxz, Syx), b = (Syy, Syz, Szx, Szy), c = (Szz, 0, 0, 0)
  var ca = vec4f(0.0);
  var cb = vec4f(0.0);
  var cc = vec4f(0.0);
  for (var k = t; k < params.count; k += THREADS) {
    let w = weights[k];
    let p = learnerPositions[k].xyz - centres[0];
    let q = targetPositions[k].xyz - centres[1];
    ca += w * vec4f(p.x * q.x, p.x * q.y, p.x * q.z, p.y * q.x);
    cb += w * vec4f(p.y * q.y, p.y * q.z, p.z * q.x, p.z * q.y);
    cc += w * vec4f(p.z * q.z, 0.0, 0.0, 0.0);
  }
  a[t] = ca;
  b[t] = cb;
  c[t] = cc;
  workgroupBarrier();
  sumUp(t);
  if (t != 0u) { return; }

  // 3. Horn's matrix and its top eigenvector
  if (!(totalWeight > 0.0)) { writeIdentity(); return; }
  let xx = a[0].x; let xy = a[0].y; let xz = a[0].z;
  let yx = a[0].w; let yy = b[0].x; let yz = b[0].y;
  let zx = b[0].z; let zy = b[0].w; let zz = c[0].x;
  var m = array<array<f32, 4>, 4>(
    array<f32, 4>(xx + yy + zz, yz - zy, zx - xz, xy - yx),
    array<f32, 4>(yz - zy, xx - yy - zz, xy + yx, zx + xz),
    array<f32, 4>(zx - xz, xy + yx, -xx + yy - zz, yz + zy),
    array<f32, 4>(xy - yx, zx + xz, yz + zy, -xx - yy + zz),
  );
  var v = array<array<f32, 4>, 4>(
    array<f32, 4>(1.0, 0.0, 0.0, 0.0),
    array<f32, 4>(0.0, 1.0, 0.0, 0.0),
    array<f32, 4>(0.0, 0.0, 1.0, 0.0),
    array<f32, 4>(0.0, 0.0, 0.0, 1.0),
  );
  for (var sweep = 0; sweep < 12; sweep++) {
    jacobi(&m, &v, 0u, 1u); jacobi(&m, &v, 0u, 2u); jacobi(&m, &v, 0u, 3u);
    jacobi(&m, &v, 1u, 2u); jacobi(&m, &v, 1u, 3u); jacobi(&m, &v, 2u, 3u);
  }
  var best = 0u;
  for (var k = 1u; k < 4u; k++) { if (m[k][k] > m[best][best]) { best = k; } }
  var quat = vec4f(v[1][best], v[2][best], v[3][best], v[0][best]); // (x, y, z, w)
  if (quat.w < 0.0) { quat = -quat; } // the shorter way round
  quat = normalize(quat);

  // the rotation matrix of the quaternion, column by column
  let x = quat.x; let y = quat.y; let z = quat.z; let w = quat.w;
  let r = mat3x3f(
    1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z), 2.0 * (x * z - w * y),
    2.0 * (x * y - w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + w * x),
    2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y),
  );
  let shift = centres[1] - r * centres[0];
  alignment.rotation = quat;
  alignment.learnerCentre = vec4f(centres[0], 1.0);
  alignment.targetCentre = vec4f(centres[1], 1.0);
  alignment.transform = mat4x4f(vec4f(r[0], 0.0), vec4f(r[1], 0.0), vec4f(r[2], 0.0), vec4f(shift, 1.0));
}
