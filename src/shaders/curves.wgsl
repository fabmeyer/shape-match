// thick lines (walkers: row curves, walker polygon, iso-curve), built in screen space
// one instance per segment, 6 vertices: a quad of the given width in pixels
// drawn twice: parts in front of the surface normally, parts behind it darker (HIDDEN)

struct Camera {
  viewProjection : mat4x4f,
  eye            : vec4f,
  viewport       : vec4f, // width, height in pixels
};

struct Segment {
  a     : vec4f, // start (xyz), width in pixels (w)
  b     : vec4f, // end (xyz), 1 = dashed (w)
  color : vec4f,
};

@group(0) @binding(0) var<uniform> camera : Camera;
@group(0) @binding(1) var<storage, read> segments : array<Segment>;

override HIDDEN : bool = false; // second pipeline: only where the line is behind the surface

const LIFT : f32 = 0.01;  // moves lines a bit towards the camera: the orange curve lies exactly on the surface
const DASH_PX : f32 = 5.0;

struct VertexOut {
  @builtin(position) position : vec4f,
  @location(0) @interpolate(linear) dist : f32, // pixels from the segment start (for dashes)
  @location(1) @interpolate(flat) dashed : f32,
  @location(2) @interpolate(flat) color : vec3f,
};

fn toClip(p : vec3f) -> vec4f {
  let toEye = normalize(camera.eye.xyz - p);
  return camera.viewProjection * vec4f(p + toEye * LIFT, 1.0);
}

@vertex
fn vs_main(@builtin(vertex_index) vi : u32, @builtin(instance_index) ii : u32) -> VertexOut {
  let s = segments[ii];
  let a = toClip(s.a.xyz);
  let b = toClip(s.b.xyz);

  var out : VertexOut;
  out.color = s.color.rgb;
  out.dashed = s.b.w;
  if (a.w <= 0.0 || b.w <= 0.0) { // an end behind the camera: skip the segment
    out.position = vec4f(2.0, 2.0, 2.0, 1.0);
    return out;
  }

  // both ends in pixels, relative to the screen centre
  let halfSize = camera.viewport.xy * 0.5;
  let pa = a.xy / a.w * halfSize;
  let pb = b.xy / b.w * halfSize;
  let len = length(pb - pa);
  let along = select(vec2f(1.0, 0.0), (pb - pa) / len, len > 1e-4);
  let halfWidth = s.a.w * 0.5;
  let side = vec2f(-along.y, along.x) * halfWidth;
  let ext = along * halfWidth; // a bit longer at both ends: closes the gaps where segments meet

  var corners = array<vec2f, 6>(pa - ext - side, pa - ext + side, pb + ext + side,
                                pa - ext - side, pb + ext + side, pb + ext - side);
  let atB = vi == 2u || vi == 4u || vi == 5u;
  let c = select(a, b, atB); // depth and w of that end
  out.position = vec4f(corners[vi] / halfSize * c.w, c.z, c.w);
  out.dist = select(-halfWidth, len + halfWidth, atB);
  return out;
}

@fragment
fn fs_main(in : VertexOut) -> @location(0) vec4f {
  if (in.dashed > 0.5 && fract(in.dist / (2.0 * DASH_PX)) > 0.5) { discard; }
  var color = in.color;
  if (HIDDEN) { color = mix(color, vec3f(0.1, 0.11, 0.13), 0.55); } // behind the surface: darker
  return vec4f(color, 1.0);
}
