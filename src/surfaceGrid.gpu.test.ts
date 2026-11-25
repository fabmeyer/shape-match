// the tessellation compute pass against the CPU evaluation (nurbs.ts), at the same f32 parameter values
import { beforeAll, describe, expect, it } from 'vitest';
import { clampedUniformKnots, type ControlNet } from './controlNet.ts';
import { evaluate } from './nurbs.ts';
import { samplePositions } from './samples.ts';
import { allShapes } from './shapes.ts';
import { createTessellatePipeline, SurfaceGrid } from './surfaceGrid.ts';
import { readBuffer, requestTestDevice } from './gpuTesting.ts';

// f32 rounding: points of size ~1 agree to a few 1e-7; this leaves room for longer sums and rational weights
const TOLERANCE = 1e-5;
const RES = 16;

// the lecture's curve (cubic, 7 points, U = [0,0,0,0, .25, .5, .75, 1,1,1,1]) as the u direction of a surface
function lectureNet(): ControlNet {
  const countU = 7, countV = 4;
  const points = new Float32Array(countU * countV * 4);
  for (let j = 0; j < countV; j++) {
    for (let i = 0; i < countU; i++) {
      const w = 1 + 0.25 * ((i + j) % 3); // some weights, so the rational part is tested too
      points.set([w * (i / 3 - 1), w * (j / 3 - 0.5), w * Math.sin(i + 2 * j) * 0.4, w], (j * countU + i) * 4);
    }
  }
  return { countU, countV, p: 3, q: 3, knotsU: [0, 0, 0, 0, 0.25, 0.5, 0.75, 1, 1, 1, 1], knotsV: clampedUniformKnots(4, 3), points };
}

const shapes = allShapes();
const nets: [string, ControlNet][] = [
  ['lecture example', lectureNet()],
  ['ball (weights)', shapes.find((s) => s.id === 'ball')!.net],
  ['vase (closed)', shapes.find((s) => s.id === 'vase')!.net],
  ['house (creases)', shapes.find((s) => s.id === 'house')!.net],
];

let device: GPUDevice;
let pipeline: GPUComputePipeline;
beforeAll(async () => {
  device = await requestTestDevice();
  pipeline = createTessellatePipeline(device);
});

describe('tessellation pass', () => {
  it.each(nets)('matches the CPU evaluation: %s', async (_, net) => {
    const grid = new SurfaceGrid(device, pipeline, net, RES);
    const encoder = device.createCommandEncoder();
    const compute = encoder.beginComputePass();
    grid.tessellate(compute);
    compute.end();
    device.queue.submit([encoder.finish()]);
    const gpu = new Float32Array(await readBuffer(device, grid.positions));
    grid.destroy();

    const us = samplePositions(net.knotsU, net.p, RES).map(Math.fround); // the shader gets them as f32
    const vs = samplePositions(net.knotsV, net.q, RES).map(Math.fround);
    let worst = 0;
    for (let j = 0; j < vs.length; j++) {
      for (let i = 0; i < us.length; i++) {
        const cpu = evaluate(net, us[i], vs[j]).point;
        const k = (j * us.length + i) * 4;
        for (let c = 0; c < 3; c++) worst = Math.max(worst, Math.abs(gpu[k + c] - cpu[c]));
      }
    }
    expect(worst).toBeLessThan(TOLERANCE);
  });
});
