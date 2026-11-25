// compute pass: sums one direction of the Shape distance in a single workgroup
//   results[slot] = (sum of w * d^2, sum of w, largest d, unused)
// every thread first adds up every 256th sample, then the 256 partial results are added in a tree

const THREADS : u32 = 256u;

struct Params {
  count : u32, // samples
  slot  : u32, // which vec4 of results to write
};

@group(0) @binding(0) var<uniform> params : Params;
@group(0) @binding(1) var<storage, read> distances : array<f32>;
@group(0) @binding(2) var<storage, read> weights : array<f32>;
@group(0) @binding(3) var<storage, read_write> results : array<vec4f>;

var<workgroup> partial : array<vec3f, THREADS>;

@compute @workgroup_size(256)
fn cs_main(@builtin(local_invocation_index) t : u32) {
  var sum = vec3f(0.0); // sum of w * d^2, sum of w, max d
  for (var k = t; k < params.count; k += THREADS) {
    let d = distances[k];
    let w = weights[k];
    sum = vec3f(sum.x + w * d * d, sum.y + w, max(sum.z, d));
  }
  partial[t] = sum;
  workgroupBarrier();

  for (var stride = THREADS / 2u; stride > 0u; stride /= 2u) {
    if (t < stride) {
      let a = partial[t];
      let b = partial[t + stride];
      partial[t] = vec3f(a.x + b.x, a.y + b.y, max(a.z, b.z));
    }
    workgroupBarrier();
  }
  if (t == 0u) { results[params.slot] = vec4f(partial[0], 0.0); }
}
