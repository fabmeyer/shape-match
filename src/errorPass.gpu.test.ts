// the GPU error pass against the CPU reference (shapeDistance.ts)
import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { flatControlNet, getPoint, setPoint, type ControlNet } from './controlNet.ts';
import { ErrorPass } from './errorPass.ts';
import { shapeDistance } from './shapeDistance.ts';
import { allShapes } from './shapes.ts';
import { allTasks, cloneNet } from './target.ts';
import { readBuffer, requestTestDevice } from './gpuTesting.ts';

const RES = 16; // the CPU reference is brute force, small grids keep it fast

// f32 on the GPU against f64 on the CPU, relative to the diagonal
const SUM_TOLERANCE = 2e-3;     // relative error of RMS and Hausdorff
const SAMPLE_TOLERANCE = 2e-5;  // per-sample distance, in units of the diagonal

const shape = (id: string) => allShapes().find((s) => s.id === id)!.net;

function moved(net: ControlNet, dx: number, dz: number): ControlNet {
  const out = cloneNet(net);
  for (let k = 0; k < out.countU * out.countV; k++) {
    const [x, y, z] = getPoint(out, k);
    setPoint(out, k, [x + dx, y, z + dz]);
  }
  return out;
}

const level1 = allTasks()[0];
const house = allTasks().find((t) => t.id === 'house')!;
const pairs: [string, ControlNet, ControlNet][] = [
  ['level 1: target against the flat start', level1.target, level1.start],
  ['ball against the ball moved', shape('ball'), moved(shape('ball'), 0.1, 0.05)],
  ['house against its squashed start (creases)', house.target, house.start],
  ['different Layouts: vase against the flat plane', shape('vase'), flatControlNet()],
  ['different Layouts: cup against the mushroom', shape('cup'), shape('mushroom')],
];

let device: GPUDevice;
let pass: ErrorPass;
beforeAll(async () => {
  device = await requestTestDevice();
  pass = new ErrorPass(device);
  pass.resolution = RES;
});
afterAll(() => pass.destroy());

function expectClose(gpu: number, cpu: number) {
  expect(Math.abs(gpu - cpu)).toBeLessThanOrEqual(SUM_TOLERANCE * Math.max(cpu, 1e-6));
}

describe('error pass', () => {
  it.each(pairs)('matches the CPU reference: %s', async (_, target, learner) => {
    pass.setTarget(target);
    pass.setLearner(learner);
    const gpu = await pass.measure();
    const cpu = shapeDistance(target, learner, { res: RES });
    expect(gpu.diagonal).toBeCloseTo(cpu.diagonal, 10);
    expectClose(gpu.rms, cpu.rms);
    expectClose(gpu.hausdorff, cpu.hausdorff);

    // the per-sample distances, which the heatmap will read
    const learnerGpu = new Float32Array(await readBuffer(device, pass.learnerDistances!));
    const targetGpu = new Float32Array(await readBuffer(device, pass.targetDistances!));
    const worst = (a: Float32Array, b: Float64Array) => Math.max(...b.map((d, k) => Math.abs(d - a[k])));
    expect(learnerGpu.length).toBe(cpu.learnerDistances.length);
    expect(worst(learnerGpu, cpu.learnerDistances) / cpu.diagonal).toBeLessThan(SAMPLE_TOLERANCE);
    expect(worst(targetGpu, cpu.targetDistances) / cpu.diagonal).toBeLessThan(SAMPLE_TOLERANCE);
  });

  it('is 0 for the Target against itself', async () => {
    pass.setTarget(shape('ball'));
    pass.setLearner(shape('ball'));
    const d = await pass.measure();
    expect(d.rms).toBeLessThan(1e-6);
    expect(d.hausdorff).toBeLessThan(1e-5);
  });

  it('follows the learner after its control points change', async () => {
    const start = flatControlNet();
    pass.setTarget(level1.target);
    pass.setLearner(start);
    const before = await pass.measure();
    pass.uploadLearnerPoints(level1.target.points); // the learner moves the right point
    const after = await pass.measure();
    expect(before.rms).toBeGreaterThan(0.01);
    expect(after.rms).toBeLessThan(1e-6);
  });

  it('can change the grid size at run time', async () => {
    const [target, learner] = [shape('ball'), moved(shape('ball'), 0.1, 0)];
    pass.setTarget(target);
    pass.setLearner(learner);
    for (const res of [12, 24]) {
      pass.resolution = res;
      const gpu = await pass.measure();
      expect(gpu.res).toBe(res);
      expectClose(gpu.rms, shapeDistance(target, learner, { res }).rms);
    }
    pass.resolution = RES;
  });
});
