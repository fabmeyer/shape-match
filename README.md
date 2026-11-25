# Shape Match

Learning NURBS surfaces by reshaping target shapes in WebGPU.
Applied Computer Graphics, Fall 2026, course project.

This first commit draws a flat plane and its 6 × 6 control points with WebGPU.
Drag to rotate the view (arcball camera) and scroll to zoom. There are no other
controls yet.

## Run

```bash
npm install
npm run dev        # open http://localhost:5173 in Chrome or Edge
npm run typecheck
npm test           # vitest in Node: NURBS, sample positions, shapes, Tasks, Shape distance
npm run test:gpu   # vitest in headless Chromium with WebGPU: GPU passes against the CPU reference
```

### GPU tests

`npm run test:gpu` runs every `*.gpu.test.ts` in headless Chromium through Playwright
(vitest browser mode), with `--enable-unsafe-webgpu`. Each test runs a GPU pass, reads the
result back and compares it with the CPU reference at a fixed f32 tolerance. It needs:

- the Playwright browser, once: `npx playwright install chromium`
- a machine with a GPU that Chromium can use headless (macOS with Metal works). If the
  tests fail with "No WebGPU adapter", the browser has no GPU access, for example on a CI
  runner without a GPU.

This is the same Vite + TypeScript setup as the torus-knot examples. Shaders
are imported with `?raw`.

## Files

| File | Content |
| --- | --- |
| `src/main.ts` | Buffers, pipelines and the frame loop |
| `src/controlNet.ts` | The flat 6 × 6 control net |
| `src/arcball.ts` | Arcball camera |
| `src/webgpu-init.ts` | Adapter and device setup, canvas resize (from the course examples) |
| `src/shaders/plane.wgsl` | The plane through the corner control points |
| `src/shaders/points.wgsl` | Control points as round dots, read from the storage buffer |

The control points sit in a storage buffer as `array<vec4f>` (w·x, w·y, w·z, w)
with index `j * countU + i`, so the tessellation compute pass can read them next.
