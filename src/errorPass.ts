// the Shape distance on the GPU (ADR 0002): the Learner surface and the Target on their own error grid,
// separate from the render grid, then both directions of the distance and their sums
//   1. tessellate.wgsl: both surfaces on the error grid (the Target only once per Level)
//   2. distance.wgsl:   per sample, the distance to the nearest triangle of the other surface and its area
//   3. reduce.wgsl:     per direction, sum of w * d^2, sum of w and the largest d
// only the six sums come back to the CPU; combine() in shapeDistance.ts turns them into RMS and Hausdorff
// shapeDistance.ts is the CPU reference: same samples, same triangles, same weights
import type { ControlNet } from './controlNet.ts';
import { boundingDiagonal, combine, ERROR_RES, sampleSurface, type DistanceSums } from './shapeDistance.ts';
import { BufferSet } from './gpuBuffer.ts';
import { createTessellatePipeline, SurfaceGrid } from './surfaceGrid.ts';
import distanceCode from './shaders/distance.wgsl?raw';
import reduceCode from './shaders/reduce.wgsl?raw';

const DISTANCE_WORKGROUP = 64; // @workgroup_size in distance.wgsl

// compiling the compute pipelines takes about a second, so every device compiles them once
// and every ErrorPass on it (the live one, one per Level calibration) shares them
interface ErrorPipelines {
  distance: GPUComputePipeline;
  reduce: GPUComputePipeline;
}
const pipelineCache = new WeakMap<GPUDevice, ErrorPipelines>();

function errorPipelines(device: GPUDevice): ErrorPipelines {
  let cached = pipelineCache.get(device);
  if (!cached) {
    const pipeline = (label: string, code: string) => device.createComputePipeline({
      label,
      layout: 'auto',
      compute: { module: device.createShaderModule({ label, code }), entryPoint: 'cs_main' },
    });
    cached = { distance: pipeline('distance', distanceCode), reduce: pipeline('reduce distances', reduceCode) };
    pipelineCache.set(device, cached);
  }
  return cached;
}

export interface GpuShapeDistance {
  rms: number;       // / diagonal, like shapeDistance()
  hausdorff: number; // / diagonal
  diagonal: number;  // of the Target's bounding box
  res: number;       // the error grid it was measured on
}

// one direction: from the samples of one grid to the triangles of the other
interface Direction {
  distances: GPUBuffer; // f32 per sample of `from`
  weights: GPUBuffer;   // f32 per sample of `from`
  distanceGroup: GPUBindGroup;
  reduceGroup: GPUBindGroup;
  sampleCount: number;
  buffers: BufferSet;
}

export class ErrorPass {
  private device: GPUDevice;
  private tessellatePipeline: GPUComputePipeline;
  private distancePipeline: GPUComputePipeline;
  private reducePipeline: GPUComputePipeline;
  private results: GPUBuffer; // two vec4f: learner -> target, target -> learner

  private res = ERROR_RES;
  private targetNet?: ControlNet;
  private learnerNet?: ControlNet;
  private learnerShift = 0;
  private target?: SurfaceGrid;
  private learner?: SurfaceGrid;
  private diagonal = 1;
  private directions: Direction[] = [];

  constructor(device: GPUDevice, tessellatePipeline = createTessellatePipeline(device)) {
    this.device = device;
    this.tessellatePipeline = tessellatePipeline;
    const { distance, reduce } = errorPipelines(device);
    this.distancePipeline = distance;
    this.reducePipeline = reduce;
    this.results = device.createBuffer({
      label: 'distance sums',
      size: 2 * 16,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
  }

  // samples per direction of the error grid (before the knots are added); can change at any time
  get resolution(): number {
    return this.res;
  }

  set resolution(res: number) {
    if (res === this.res) return;
    this.res = res;
    if (this.targetNet) this.setTarget(this.targetNet);
    if (this.learnerNet) this.setLearner(this.learnerNet, this.learnerShift);
  }

  // a new Level: the Target only changes here
  setTarget(net: ControlNet): void {
    this.targetNet = net;
    this.target?.destroy();
    this.target = new SurfaceGrid(this.device, this.tessellatePipeline, net, this.res);
    this.diagonal = boundingDiagonal(sampleSurface(net, this.res)); // same samples as the GPU grid
    this.rebuildDirections();
  }

  // a new Learner surface, possibly with another Layout (Level load, knot insertion)
  // shift moves its grid by a fraction of a cell: the Target with shift 0.5 gives the Noise floor
  setLearner(net: ControlNet, shift = 0): void {
    this.learnerNet = net;
    this.learnerShift = shift;
    this.learner?.destroy();
    this.learner = new SurfaceGrid(this.device, this.tessellatePipeline, net, this.res, { shift });
    this.rebuildDirections();
  }

  // the learner's control points changed (drag, weight, undo): same Layout, new positions
  uploadLearnerPoints(points: Float32Array): void {
    this.learner?.uploadPoints(points);
  }

  // per sample of the Learner surface on the error grid: its distance to the Target (for the heatmap)
  get learnerDistances(): GPUBuffer | undefined {
    return this.directions[0]?.distances;
  }

  // per sample of the Target on the error grid: its distance to the Learner surface
  get targetDistances(): GPUBuffer | undefined {
    return this.directions[1]?.distances;
  }

  private rebuildDirections(): void {
    for (const d of this.directions) d.buffers.destroy();
    this.directions = [];
    if (!this.learner || !this.target) return;
    this.directions = [this.direction(this.learner, this.target, 0), this.direction(this.target, this.learner, 1)];
  }

  private direction(from: SurfaceGrid, to: SurfaceGrid, slot: number): Direction {
    const device = this.device;
    const { STORAGE, UNIFORM, COPY_SRC } = GPUBufferUsage;
    const count = from.sampleCount;
    const buffers = new BufferSet(device);
    const distances = buffers.create('sample distances', STORAGE | COPY_SRC, count * 4);
    const weights = buffers.create('sample weights', STORAGE | COPY_SRC, count * 4);
    const distanceParams = buffers.create('distance params', UNIFORM, new Uint32Array([from.resU, from.resV, to.resU, to.resV]));
    const reduceParams = buffers.create('reduce params', UNIFORM, new Uint32Array([count, slot]));

    const group = (pipeline: GPUComputePipeline, buffers: GPUBuffer[]) => device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
    });
    return {
      distances,
      weights,
      sampleCount: count,
      distanceGroup: group(this.distancePipeline, [distanceParams, from.positions, to.positions, distances, weights]),
      reduceGroup: group(this.reducePipeline, [reduceParams, distances, weights, this.results]),
      buffers,
    };
  }

  // the Shape distance between the current Learner surface and the Target
  async measure(): Promise<GpuShapeDistance> {
    if (!this.learner || !this.target) throw new Error('ErrorPass needs a Target and a Learner surface first');
    const device = this.device;
    const encoder = device.createCommandEncoder({ label: 'error pass' });
    const compute = encoder.beginComputePass({ label: 'error pass' });
    this.target.tessellate(compute);  // only if it changed (once per Level)
    this.learner.tessellate(compute); // only if the control points changed
    compute.setPipeline(this.distancePipeline);
    for (const d of this.directions) {
      compute.setBindGroup(0, d.distanceGroup);
      compute.dispatchWorkgroups(Math.ceil(d.sampleCount / DISTANCE_WORKGROUP));
    }
    compute.setPipeline(this.reducePipeline);
    for (const d of this.directions) {
      compute.setBindGroup(0, d.reduceGroup);
      compute.dispatchWorkgroups(1);
    }
    compute.end();

    // each measurement gets its own staging buffer, so two can overlap
    const staging = device.createBuffer({ size: this.results.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    encoder.copyBufferToBuffer(this.results, 0, staging, 0, this.results.size);
    device.queue.submit([encoder.finish()]);
    const res = this.res;
    const diagonal = this.diagonal;
    await staging.mapAsync(GPUMapMode.READ);
    const r = new Float32Array(staging.getMappedRange().slice(0));
    staging.destroy();

    const toSums = (k: number): DistanceSums => ({ sumWeightedSquares: r[k], sumWeights: r[k + 1], max: r[k + 2] });
    return { ...combine(toSums(0), toSums(4), diagonal), diagonal, res };
  }

  destroy(): void {
    this.target?.destroy();
    this.learner?.destroy();
    for (const d of this.directions) d.buffers.destroy();
    this.results.destroy();
  }
}

// what a Level needs before it can be graded (grading.ts), measured once when it loads:
//   noise: the Target against itself on a grid shifted by half a cell (the Noise floor)
//   start: the starting Learner surface against the Target (0 % Match)
// uses its own ErrorPass, so the learner can edit meanwhile
export async function measureLevel(device: GPUDevice, tessellatePipeline: GPUComputePipeline,
  target: ControlNet, start: ControlNet, res = ERROR_RES): Promise<{ noise: GpuShapeDistance; start: GpuShapeDistance }> {
  const pass = new ErrorPass(device, tessellatePipeline);
  try {
    pass.resolution = res;
    pass.setTarget(target);
    pass.setLearner(target, 0.5);
    const noise = await pass.measure();
    pass.setLearner(start);
    return { noise, start: await pass.measure() };
  } finally {
    pass.destroy();
  }
}
