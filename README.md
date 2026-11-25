# Shape Match

Learning NURBS surfaces by reshaping target shapes in WebGPU.
Applied Computer Graphics, Fall 2026, course project.

The learner reshapes a NURBS surface (blue) until it matches a Target (orange). The editor
evaluates the surfaces on the GPU, and the Shape distance between them is measured on the GPU
too. The plan is in [docs/PLAN.md](docs/PLAN.md), the terms in [GLOSSARY.md](GLOSSARY.md),
the decisions in [docs/adr](docs/adr/). Pick a Level in the side panel or open one directly,
for example `?task=house`.

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
| `src/main.ts` | Pipelines, editing, loading a Task, the frame loop |
| `src/target.ts` | Tasks: Target, starting Learner surface, allowed tools (one Layout per Level, ADR 0001) |
| `src/shapes.ts` | The library of 11 Targets, from leaf to car |
| `src/controlNet.ts`, `src/nurbs.ts` | The control net and the NURBS evaluation on the CPU |
| `src/samples.ts` | Where surfaces are sampled: a uniform grid plus the knots |
| `src/surfaceGrid.ts` | One surface evaluated on the GPU on a grid (`shaders/tessellate.wgsl`) |
| `src/surfaceGpu.ts` | A drawn surface: render grid, control points, polygon, stems |
| `src/shapeDistance.ts` | Shape distance on the CPU, the reference (ADR 0002) |
| `src/errorPass.ts` | Shape distance on the GPU (`shaders/distance.wgsl`, `shaders/reduce.wgsl`) |
| `src/shapeChecks.ts` | Checks for the shape library (also run by `npm test`) |
| `src/pointDrag.ts`, `src/arcball.ts` | Picking and dragging control points, the camera |
| `src/paramSpace.ts`, `src/partition.ts`, `src/mathPanel.ts`, `src/walkers.ts` | The side panel views |
| `*.test.ts`, `*.gpu.test.ts` | Tests in Node and in headless Chromium with WebGPU |

The control points sit in a storage buffer as `array<vec4f>` (w·x, w·y, w·z, w)
with index `j * countU + i`.
