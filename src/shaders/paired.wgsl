// compute pass: the Paired distance, one thread per sample. Both surfaces come from tessellate.wgsl on the
// same (u, v) grid, so sample s of the Learner surface belongs to sample s of the Target: no search
//   distance: |learner[s] - target[s]|, the learner first moved by alignment.transform: the best fit from
//             align.wgsl, or no change (the identity) when the grader does not line up the surfaces
//   weight:   the area the Target sample stands for, a quarter of each Target grid cell around it
// the same steps as pairedDistance() in shapeDistance.ts on the CPU; reduce.wgsl sums the result

struct Params {
  resU : u32,
  resV : u32,
};

@group(0) @binding(0) var<uniform> params : Params;
@group(0) @binding(1) var<storage, read> learnerPositions : array<vec4f>;
@group(0) @binding(2) var<storage, read> targetPositions : array<vec4f>;
@group(0) @binding(3) var<storage, read_write> distances : array<f32>;
@group(0) @binding(4) var<storage, read_write> weights : array<f32>;

struct Alignment {
  rotation      : vec4f,
  learnerCentre : vec4f,
  targetCentre  : vec4f,
  transform     : mat4x4f,
};
@group(0) @binding(5) var<storage, read> alignment : Alignment;

fn targetPoint(i : u32, j : u32) -> vec3f {
  return targetPositions[j * params.resU + i].xyz;
}

// area of Target grid cell (i, j): its two triangles
fn cellArea(i : u32, j : u32) -> f32 {
  let a = targetPoint(i, j);
  let b = targetPoint(i + 1u, j);
  let c = targetPoint(i, j + 1u);
  let d = targetPoint(i + 1u, j + 1u);
  return 0.5 * (length(cross(b - a, d - a)) + length(cross(d - a, c - a)));
}

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) id : vec3u) {
  let count = params.resU * params.resV;
  let s = id.x;
  if (s >= count) { return; }
  let i = s % params.resU;
  let j = s / params.resU;
  let learner = (alignment.transform * vec4f(learnerPositions[s].xyz, 1.0)).xyz;
  distances[s] = distance(learner, targetPositions[s].xyz);

  var area = 0.0;
  if (i + 1u < params.resU && j + 1u < params.resV) { area += cellArea(i, j); }
  if (i > 0u && j + 1u < params.resV) { area += cellArea(i - 1u, j); }
  if (i + 1u < params.resU && j > 0u) { area += cellArea(i, j - 1u); }
  if (i > 0u && j > 0u) { area += cellArea(i - 1u, j - 1u); }
  weights[s] = 0.25 * area;
}
