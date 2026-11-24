// render pass for the tessellated NURBS surface, lit with the normals from the compute pass

struct Camera {
  viewProjection : mat4x4f,
  eye            : vec4f,
  viewport       : vec4f,
};

@group(0) @binding(0) var<uniform> camera : Camera;

struct VertexOut {
  @builtin(position) position : vec4f,
  @location(0) world : vec3f,
  @location(1) normal : vec3f,
};

@vertex
fn vs_main(@location(0) position : vec4f, @location(1) normal : vec4f) -> VertexOut {
  var out : VertexOut;
  out.position = camera.viewProjection * vec4f(position.xyz, 1.0);
  out.world = position.xyz;
  out.normal = normal.xyz;
  return out;
}

@fragment
fn fs_main(in : VertexOut) -> @location(0) vec4f {
  // interpolation between the vertices shortens the normal: make it length 1 again
  var n = normalize(in.normal);
  let toEye = normalize(camera.eye.xyz - in.world);
  if (dot(n, toEye) < 0.0) { n = -n; } // lit from both sides

  // light near the camera, a bit from above (like a lamp on the camera)
  let light = normalize(toEye + vec3f(0.3, 0.3, 0.6));
  let diffuse = max(dot(n, light), 0.0);

  // small highlight (Blinn-Phong): makes bumps and dents easier to see
  let halfway = normalize(light + toEye);
  let specular = 0.25 * pow(max(dot(n, halfway), 0.0), 40.0);

  let base = vec3f(0.45, 0.55, 0.75);
  return vec4f(base * (0.3 + 0.7 * diffuse) + specular, 1.0);
}
