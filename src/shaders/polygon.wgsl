// control polygon: dashed lines between neighbouring control points
// line-list, one instance per line, 2 vertices each

struct Camera {
  viewProjection : mat4x4f,
  eye            : vec4f,
  viewport       : vec4f, // width, height in pixels
};

@group(0) @binding(0) var<uniform> camera : Camera;
@group(0) @binding(1) var<storage, read> controlPoints : array<vec4f>; // (wx, wy, wz, w)
@group(0) @binding(2) var<storage, read> segments : array<vec2u>;      // the two point indices of every line

const DASH_PX : f32 = 6.0; // length of a dash and of a gap

struct VertexOut {
  @builtin(position) position : vec4f,
  // distance from the start of the line in pixels
  // linear = interpolated on screen (not perspective-correct), so all dashes have the same length
  @location(0) @interpolate(linear) dist : f32,
};

fn toClip(index : u32) -> vec4f {
  let Pw = controlPoints[index];
  return camera.viewProjection * vec4f(Pw.xyz / Pw.w, 1.0);
}

@vertex
fn vs_main(@builtin(vertex_index) vi : u32, @builtin(instance_index) ii : u32) -> VertexOut {
  let segment = segments[ii];
  let a = toClip(segment.x);
  let b = toClip(segment.y);
  // length of the line on screen
  let pixels = length((b.xy / b.w - a.xy / a.w) * camera.viewport.xy * 0.5);

  var out : VertexOut;
  out.position = select(a, b, vi == 1u);
  out.dist = select(0.0, pixels, vi == 1u);
  return out;
}

@fragment
fn fs_main(in : VertexOut) -> @location(0) vec4f {
  if (fract(in.dist / (2.0 * DASH_PX)) > 0.5) { discard; } // the gaps
  return vec4f(0.85, 0.87, 0.92, 1.0);
}
