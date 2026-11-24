// three arrows at the selection: red = x, green = y, blue = z
// one instance per arrow, 9 vertices: a thin quad (shaft) and a triangle (head)
// built in screen space, so the arrows have a constant thickness in pixels

struct Camera {
  viewProjection : mat4x4f,
  eye            : vec4f,
  viewport       : vec4f, // width, height in pixels
};

struct Gizmo {
  origin : vec3f, // centre of the selection
  size   : f32,   // arrow length in world units
  axis   : i32,   // highlighted arrow (0 = x, 1 = y, 2 = z), -1 = none
};

@group(0) @binding(0) var<uniform> camera : Camera;
@group(0) @binding(1) var<uniform> gizmo : Gizmo;

const SHAFT_PX : f32 = 1.5; // half width of the shaft
const HEAD_PX  : f32 = 6.0; // half width of the head

struct VertexOut {
  @builtin(position) position : vec4f,
  @location(0) color : vec3f,
};

@vertex
fn vs_main(@builtin(vertex_index) vi : u32, @builtin(instance_index) ii : u32) -> VertexOut {
  var axes = array<vec3f, 3>(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0), vec3f(0.0, 0.0, 1.0));
  var colors = array<vec3f, 3>(vec3f(0.92, 0.22, 0.2), vec3f(0.25, 0.8, 0.3), vec3f(0.3, 0.5, 1.0));

  let a = camera.viewProjection * vec4f(gizmo.origin, 1.0);
  let b = camera.viewProjection * vec4f(gizmo.origin + axes[ii] * gizmo.size, 1.0);

  var out : VertexOut;
  out.color = colors[ii];
  if (i32(ii) == gizmo.axis) { out.color = mix(out.color, vec3f(1.0), 0.55); }

  // an end behind the camera: move the arrow off screen
  if (a.w <= 0.0 || b.w <= 0.0) {
    out.position = vec4f(2.0, 2.0, 2.0, 1.0);
    return out;
  }

  // both ends in pixels, relative to the screen centre
  let halfSize = camera.viewport.xy * 0.5;
  let start = a.xy / a.w * halfSize;
  let tip = b.xy / b.w * halfSize;
  let neck = mix(start, tip, 0.75);                 // where the shaft ends and the head starts
  let along = normalize(tip - start + vec2f(1e-6)); // direction of the arrow on screen
  let side = vec2f(-along.y, along.x);

  var corners = array<vec2f, 9>(
    start - side * SHAFT_PX, start + side * SHAFT_PX, neck + side * SHAFT_PX, // shaft
    start - side * SHAFT_PX, neck + side * SHAFT_PX, neck - side * SHAFT_PX,
    neck - side * HEAD_PX, neck + side * HEAD_PX, tip,                      // head
  );
  out.position = vec4f(corners[vi] / halfSize, 0.0, 1.0);
  return out;
}

@fragment
fn fs_main(in : VertexOut) -> @location(0) vec4f {
  return vec4f(in.color, 1.0);
}
