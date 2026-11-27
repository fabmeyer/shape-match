// the GPU error pass against the CPU reference (shapeDistance.ts)
import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { flatControlNet, getPoint, setPoint, type ControlNet } from './controlNet.ts';
import { ErrorPass, measureLevel } from './errorPass.ts';
import { pairedDistance, PAIRED_RES, shapeDistance } from './shapeDistance.ts';
import { allShapes } from './shapes.ts';
import { allTasks, cloneNet } from './target.ts';
import { calibrate, grade, gradingConfig } from './grading.ts';
import { ERROR_RES } from './shapeDistance.ts';
import { readBuffer, requestTestDevice } from './gpuTesting.ts';
import { createTessellatePipeline } from './surfaceGrid.ts';
import { NO_FIT, rotate } from './alignment.ts';

const RES = 16; // the CPU reference is brute force, small grids keep it fast

// f32 on the GPU against f64 on the CPU, relative to the diagonal
const SUM_TOLERANCE = 2e-3;     // relative error of RMS and Hausdorff
const SAMPLE_TOLERANCE = 2e-5;  // per-sample distance, in units of the diagonal
const LINED_UP_TOLERANCE = 2e-2; // relative, nearest point after the best fit and its refinements

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

// relative for normal values, absolute near 0 (a flat surface has a Noise floor of exactly 0 on the CPU, f32 noise on the GPU)
const ZERO_TOLERANCE = 1e-6;
function expectClose(gpu: number, cpu: number) {
  expect(Math.abs(gpu - cpu)).toBeLessThanOrEqual(Math.max(SUM_TOLERANCE * cpu, ZERO_TOLERANCE));
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

  it('measures the Noise floor and the start of a Level like the CPU', async () => {
    const level = await measureLevel(device, createTessellatePipeline(device), house.target, house.start, RES);
    const noise = shapeDistance(house.target, house.target, { res: RES, learnerShift: 0.5 });
    const start = shapeDistance(house.target, house.start, { res: RES });
    expectClose(level.noise.rms, noise.rms);
    expectClose(level.start.rms, start.rms);
    const ball = shape('ball');
    const ballNoise = await measureLevel(device, createTessellatePipeline(device), ball, ball, RES);
    expectClose(ballNoise.noise.rms, shapeDistance(ball, ball, { res: RES, learnerShift: 0.5 }).rms);
    expectClose(ballNoise.noise.hausdorff, shapeDistance(ball, ball, { res: RES, learnerShift: 0.5 }).hausdorff);
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

// every Level as the app grades it: on the GPU, on the app's error grid (the Noise floor, and so τ,
// depends on the grid: on coarser grids the start of Level 1 or Level 2 (weights) would already get a Star)
describe('every Task', () => {
  let tessellate: GPUComputePipeline;
  beforeAll(() => { tessellate = createTessellatePipeline(device); });
  it.each(allTasks())('$id: the start gives 0 Stars, the Target 3 Stars', async (task) => {
    const config = gradingConfig(task.grading);
    const level = await measureLevel(device, tessellate, task.target, task.start, ERROR_RES);
    const calibration = calibrate(level.noise, level.start, config);
    expect(grade(level.start, calibration, config).stars).toBe(0);
    pass.resolution = ERROR_RES;
    pass.setTarget(task.target);
    pass.setLearner(task.target);
    expect(grade(await pass.measure(), calibration, config).stars).toBe(3);
  });
});

// the teacher's grader: the same grid point on both surfaces
describe('paired error pass', () => {
  const samePairs = pairs.filter(([name]) => !name.startsWith('different Layouts'));
  beforeAll(() => { pass.metric = 'paired'; });
  afterAll(() => { pass.metric = 'nearest'; pass.resolution = RES; });

  it.each(samePairs)('matches the CPU reference: %s', async (_, target, learner) => {
    pass.resolution = RES;
    pass.setTarget(target);
    pass.setLearner(learner);
    const gpu = await pass.measure();
    const cpu = pairedDistance(target, learner, RES);
    expectClose(gpu.rms, cpu.rms);
    expectClose(gpu.hausdorff, cpu.hausdorff);
    const heat = new Float32Array(await readBuffer(device, pass.learnerDistances!));
    expect(Math.max(...cpu.learnerDistances.map((d, k) => Math.abs(d - heat[k]))) / cpu.diagonal).toBeLessThan(SAMPLE_TOLERANCE);
  });

  it('has no Noise floor: the Target against itself is 0', async () => {
    const level = await measureLevel(device, createTessellatePipeline(device), house.target, house.start, RES, 'paired');
    expect(level.noise.rms).toBeLessThan(1e-7);
    expectClose(level.start.rms, pairedDistance(house.target, house.start, RES).rms);
  });

  it.each(allTasks())('$id at 128: the start gives 0 Stars, the Target 3 Stars', async (task) => {
    const config = gradingConfig(task.grading);
    const level = await measureLevel(device, createTessellatePipeline(device), task.target, task.start, PAIRED_RES, 'paired');
    const calibration = calibrate(level.noise, level.start, config);
    expect(grade(level.start, calibration, config).stars).toBe(0);
    pass.resolution = PAIRED_RES;
    pass.setTarget(task.target);
    pass.setLearner(task.target);
    expect(grade(await pass.measure(), calibration, config).stars).toBe(3);
  });
});

// the Paired distance after the best fit (align.wgsl): turn and move the learner onto the Target first
describe('lined-up error pass', () => {
  const angle = Math.PI / 6;
  const axis = [1, 2, 2].map((c) => c / 3);
  const q30 = [...axis.map((c) => c * Math.sin(angle / 2)), Math.cos(angle / 2)] as [number, number, number, number];
  const turned = (net: ControlNet) => {
    const out = cloneNet(net);
    for (let k = 0; k < out.countU * out.countV; k++) {
      const r = rotate(q30, getPoint(out, k));
      setPoint(out, k, [r[0] + 0.4, r[1] - 0.2, r[2] + 0.3]);
    }
    return out;
  };
  beforeAll(() => { pass.metric = 'paired'; pass.align = true; pass.resolution = RES; });
  afterAll(() => { pass.align = false; pass.metric = 'nearest'; pass.resolution = RES; });

  it.each([...pairs.filter(([name]) => !name.startsWith('different Layouts')), ['ball against itself turned and moved', shape('ball'), turned(shape('ball'))] as const])(
    'matches the CPU reference: %s', async (_, target, learner) => {
      pass.setTarget(target);
      pass.setLearner(learner);
      const gpu = await pass.measure();
      const cpu = pairedDistance(target, learner, RES, true);
      expectClose(gpu.rms, cpu.rms);
      expectClose(gpu.hausdorff, cpu.hausdorff);
      gpu.fit!.rotation.forEach((c, k) => expect(c).toBeCloseTo(cpu.fit.rotation[k], 4));
      gpu.fit!.targetCentre.forEach((c, k) => expect(c).toBeCloseTo(cpu.fit.targetCentre[k], 4));
    });

  it('finds the turn back: a turned Target fits to 0', async () => {
    pass.setTarget(shape('house'));
    pass.setLearner(turned(shape('house')));
    const d = await pass.measure();
    expect(d.rms).toBeLessThan(1e-5);
    // the rotation that undoes q30 is its inverse
    [-q30[0], -q30[1], -q30[2], q30[3]].forEach((c, k) => expect(d.fit!.rotation[k]).toBeCloseTo(c, 4));
  });

  it.each(allTasks())('$id at 128: the start gives 0 Stars, the Target 3 Stars', async (task) => {
    const config = gradingConfig(task.grading);
    const level = await measureLevel(device, createTessellatePipeline(device), task.target, task.start, PAIRED_RES, 'paired', true);
    const calibration = calibrate(level.noise, level.start, config);
    expect(grade(level.start, calibration, config).stars).toBe(0);
    pass.resolution = PAIRED_RES;
    pass.setTarget(task.target);
    pass.setLearner(turned(task.target)); // the right shape in the wrong place counts as right
    expect(grade(await pass.measure(), calibration, config).stars).toBe(3);
  });
});

// the app's grader: the best fit from the pairs of the same grid point, then the nearest-point distance
describe('nearest point, lined up', () => {
  const angle = Math.PI / 6;
  const axis = [1, 2, 2].map((c) => c / 3);
  const q30 = [...axis.map((c) => c * Math.sin(angle / 2)), Math.cos(angle / 2)] as [number, number, number, number];
  const turned = (net: ControlNet) => {
    const out = cloneNet(net);
    for (let k = 0; k < out.countU * out.countV; k++) {
      const r = rotate(q30, getPoint(out, k));
      setPoint(out, k, [r[0] + 0.4, r[1] - 0.2, r[2] + 0.3]);
    }
    return out;
  };
  beforeAll(() => { pass.metric = 'nearest'; pass.align = true; pass.resolution = RES; });
  afterAll(() => { pass.align = false; pass.resolution = RES; });

  // the different Layouts can't be paired: both measure without the best fit
  it.each([...pairs, ['ball against itself turned and moved', shape('ball'), turned(shape('ball'))] as const])(
    'matches the CPU reference: %s', async (_, target, learner) => {
      pass.setTarget(target);
      pass.setLearner(learner);
      const gpu = await pass.measure();
      const cpu = shapeDistance(target, learner, { res: RES, align: true });
      // 8 refinements in f32 drift a little from f64 when the shapes are far apart (a start, cup against mushroom)
      const close = (g: number, c: number) => expect(Math.abs(g - c)).toBeLessThanOrEqual(Math.max(LINED_UP_TOLERANCE * c, ZERO_TOLERANCE));
      close(gpu.rms, cpu.rms);
      close(gpu.hausdorff, cpu.hausdorff);
      if (cpu.fit === NO_FIT) expect(gpu.fit).toBeUndefined(); // different Layouts: no pairs, no best fit
      // the turn is only well defined when the shapes match; far apart (a squashed start) f32 and f64 can end
      // a few thousandths apart, which changes nothing there: such a surface gets 0 Stars either way
      else gpu.fit!.rotation.forEach((c, k) => expect(Math.abs(c - cpu.fit.rotation[k])).toBeLessThan(cpu.rms < 0.01 ? 1e-4 : 1e-2));
    });

  it('finds the turn back: a turned Target fits to 0', async () => {
    pass.setTarget(shape('house'));
    pass.setLearner(turned(shape('house')));
    const d = await pass.measure();
    expect(d.rms).toBeLessThan(1e-5);
    [-q30[0], -q30[1], -q30[2], q30[3]].forEach((c, k) => expect(d.fit!.rotation[k]).toBeCloseTo(c, 4));
  });

  it.each(allTasks())('$id at 64: the start gives 0 Stars, the Target 3 Stars', async (task) => {
    const config = gradingConfig(task.grading);
    const level = await measureLevel(device, createTessellatePipeline(device), task.target, task.start, ERROR_RES, 'nearest', true);
    const calibration = calibrate(level.noise, level.start, config);
    expect(grade(level.start, calibration, config).stars).toBe(0);
    pass.resolution = ERROR_RES;
    pass.setTarget(task.target);
    pass.setLearner(turned(task.target));
    expect(grade(await pass.measure(), calibration, config).stars).toBe(3);
  });
});
