// compute pass: moves the Learner surface's samples by the best fit from align.wgsl, so the nearest-point
// distance (distance.wgsl) measures the surfaces lined up: only the shape counts, not where it stands or how
// it is turned. One thread per sample; the Learner surface itself (its control points) does not change
struct Params {
  count : u32,
};

struct Alignment {
  rotation      : vec4f,
  learnerCentre : vec4f,
  targetCentre  : vec4f,
  transform     : mat4x4f,
};

@group(0) @binding(0) var<uniform> params : Params;
@group(0) @binding(1) var<storage, read> positions : array<vec4f>;
@group(0) @binding(2) var<storage, read> alignment : Alignment;
@group(0) @binding(3) var<storage, read_write> moved : array<vec4f>;

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) id : vec3u) {
  let s = id.x;
  if (s >= params.count) { return; }
  moved[s] = vec4f((alignment.transform * vec4f(positions[s].xyz, 1.0)).xyz, 1.0);
}
