// render pass for the tessellated NURBS surface, lit with the normals from the compute pass

struct Camera {
  viewProjection : mat4x4f,
  eye            : vec4f, // w = 1: position (perspective); w = 0: direction to the camera (orthographic)
  viewport       : vec4f,
};

@group(0) @binding(0) var<uniform> camera : Camera;
@group(0) @binding(1) var<uniform> surfaceColor : vec4f; // blue for the learner, orange for the target

struct VertexOut {
  @builtin(position) position : vec4f,
  @location(0) world : vec3f,
  @location(1) normal : vec3f,
  @location(2) uv : vec2f, // parameters of the surface point (tessellate.wgsl puts them in the 4th components)
};

// direction from p to the camera: the same for every point when the camera is orthographic
fn toCamera(p : vec3f) -> vec3f {
  if (camera.eye.w == 0.0) { return normalize(camera.eye.xyz); }
  return normalize(camera.eye.xyz - p);
}

@vertex
fn vs_main(@location(0) position : vec4f, @location(1) normal : vec4f) -> VertexOut {
  var out : VertexOut;
  out.position = camera.viewProjection * vec4f(position.xyz, 1.0);
  out.world = position.xyz;
  out.normal = normal.xyz;
  out.uv = vec2f(position.w, normal.w);
  return out;
}

// lighting for a surface colour: a lamp on the camera, a bit from above
fn lit(in : VertexOut, base : vec3f) -> vec4f {
  // interpolation between the vertices shortens the normal: make it length 1 again
  var n = normalize(in.normal);
  let toEye = toCamera(in.world);
  if (dot(n, toEye) < 0.0) { n = -n; } // lit from both sides

  // light near the camera, a bit from above (like a lamp on the camera)
  let light = normalize(toEye + vec3f(0.3, 0.3, 0.6));
  let diffuse = max(dot(n, light), 0.0);

  // small highlight (Blinn-Phong): makes bumps and dents easier to see
  let halfway = normalize(light + toEye);
  let specular = 0.25 * pow(max(dot(n, halfway), 0.0), 40.0);

  return vec4f(base * (0.3 + 0.7 * diffuse) + specular, 1.0);
}

@fragment
fn fs_main(in : VertexOut) -> @location(0) vec4f {
  return lit(in, surfaceColor.rgb);
}

// ---------------------------------------------------------
// Heatmap (second pipeline, fragment entry fs_heat): how far the Learner surface is from the Target here.
// The error pass measured the distance at the samples of the error grid; this interpolates it at (u, v).
// The error grid is not uniform (it contains the knots), so the cell is found by binary search
// ---------------------------------------------------------

struct Heat {
  resU : u32,
  resV : u32,
  cold : f32, // distances up to here (τ) keep the surface colour
  hot  : f32, // from here on (the 1★ limit) full red
};

@group(1) @binding(0) var<uniform> heat : Heat;
@group(1) @binding(1) var<storage, read> heatSamples : array<f32>;   // u values, then v values
@group(1) @binding(2) var<storage, read> heatDistances : array<f32>; // per sample, index j * resU + i

// the cell [k, k + 1] of the n sample values starting at off that contains t, and where t lies in it (0 ... 1)
fn findCell(t : f32, off : u32, n : u32) -> vec2f {
  var lo = 0u;
  var hi = n - 1u;
  while (hi - lo > 1u) {
    let mid = (lo + hi) / 2u;
    if (heatSamples[off + mid] <= t) { lo = mid; } else { hi = mid; }
  }
  let a = heatSamples[off + lo];
  let b = heatSamples[off + lo + 1u];
  return vec2f(f32(lo), clamp((t - a) / max(b - a, 1e-7), 0.0, 1.0));
}

fn heatDistance(uv : vec2f) -> f32 {
  let cu = findCell(uv.x, 0u, heat.resU);
  let cv = findCell(uv.y, heat.resU, heat.resV);
  let i = u32(cu.x);
  let j = u32(cv.x);
  let at = j * heat.resU + i;
  let d00 = heatDistances[at];
  let d10 = heatDistances[at + 1u];
  let d01 = heatDistances[at + heat.resU];
  let d11 = heatDistances[at + heat.resU + 1u];
  return mix(mix(d00, d10, cu.y), mix(d01, d11, cu.y), cv.y);
}

// 0 = within τ (the surface colour), 0.5 = yellow, 1 = red at the 1★ limit; log scale like the Match meter
fn heatColor(d : f32, base : vec3f) -> vec3f {
  let t = clamp(log(max(d, heat.cold) / heat.cold) / log(heat.hot / heat.cold), 0.0, 1.0);
  let yellow = vec3f(0.98, 0.82, 0.25);
  let red = vec3f(0.92, 0.22, 0.18);
  if (t < 0.5) { return mix(base, yellow, t * 2.0); }
  return mix(yellow, red, t * 2.0 - 1.0);
}

@fragment
fn fs_heat(in : VertexOut) -> @location(0) vec4f {
  return lit(in, heatColor(heatDistance(in.uv), surfaceColor.rgb));
}
