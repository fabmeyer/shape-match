// the Shape distance on the GPU (ADR 0002): the Learner surface and the Target on their own error grid,
// separate from the render grid, then both directions of the distance and their sums
//   1. tessellate.wgsl: both surfaces on the error grid (the Target only once per Level)
//   2. distance.wgsl:   per sample, the distance to the nearest triangle of the other surface and its area
//   3. reduce.wgsl:     per direction, sum of w * d^2, sum of w and the largest d
// only the six sums come back to the CPU; combine() in shapeDistance.ts turns them into RMS and Hausdorff
// shapeDistance.ts is the CPU reference: same samples, same triangles, same weights
// with metric 'paired' step 2 is paired.wgsl instead: the Paired distance, sample against the same sample;
// with align on, align.wgsl first finds the best fit (turn and move the learner onto the Target) and the
// Paired distance is measured after it. paired.wgsl runs once before it, for the sample areas align.wgsl needs
import type { ControlNet } from './controlNet.ts';
import { boundingDiagonal, combine, ERROR_RES, sampleSurface, type DistanceSums, type Metric } from './shapeDistance.ts';
import { BufferSet } from './gpuBuffer.ts';
import { createTessellatePipeline, SurfaceGrid } from './surfaceGrid.ts';
import distanceCode from './shaders/distance.wgsl?raw';
import pairedCode from './shaders/paired.wgsl?raw';
import alignCode from './shaders/align.wgsl?raw';
import type { BestFit } from './alignment.ts';
import reduceCode from './shaders/reduce.wgsl?raw';

const DISTANCE_WORKGROUP = 64; // @workgroup_size in distance.wgsl and paired.wgsl

// compiling the compute pipelines takes about a second, so every device compiles them once
// and every ErrorPass on it (the live one, one per Level calibration) shares them
interface ErrorPipelines {
  distance: GPUComputePipeline;
  paired: GPUComputePipeline;
  align: GPUComputePipeline;
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
    cached = {
      distance: pipeline('distance', distanceCode),
      paired: pipeline('paired distance', pairedCode),
      align: pipeline('best fit', alignCode),
      reduce: pipeline('reduce distances', reduceCode),
    };
    pipelineCache.set(device, cached);
  }
  return cached;
}

// what the heatmap needs: the learner's error grid and the distance of every sample in it
export interface HeatSource {
  distances: GPUBuffer; // f32 per sample, index j * resU + i
  samples: GPUBuffer;   // the grid's u values, then its v values
  resU: number;
  resV: number;
}

export interface GpuShapeDistance {
  rms: number;       // / diagonal, like shapeDistance()
  hausdorff: number; // / diagonal
  diagonal: number;  // of the Target's bounding box
  res: number;       // the error grid it was measured on
  fit?: BestFit;     // how the Learner surface was turned and moved before measuring (align on)
}

// the alignment buffer (Alignment in align.wgsl and paired.wgsl): rotation, two centres, transform
const ALIGNMENT_FLOATS = 4 + 4 + 4 + 16;
const IDENTITY_ALIGNMENT = new Float32Array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);

// one direction: from the samples of one grid to the triangles of the other
// (the Paired distance has a single one, summed into both result slots: both directions are the same)
interface Direction {
  distances: GPUBuffer; // f32 per sample of `from`
  weights: GPUBuffer;   // f32 per sample of `from`
  pipeline: GPUComputePipeline;
  distanceGroup: GPUBindGroup;
  alignGroup?: GPUBindGroup; // paired only: the best fit from these pairs
  reduceGroups: GPUBindGroup[];
  sampleCount: number;
  buffers: BufferSet;
}

export class ErrorPass {
  private device: GPUDevice;
  private tessellatePipeline: GPUComputePipeline;
  private pipelines: ErrorPipelines;
  private results: GPUBuffer; // two vec4f: learner -> target, target -> learner
  private alignment: GPUBuffer; // the best fit, or the identity when not lining up

  private res = ERROR_RES;
  private kind: Metric = 'nearest';
  private lineUp = false;
  private targetNet?: ControlNet;
  private learnerNet?: ControlNet;
  private learnerShift = 0;
  private target?: SurfaceGrid;
  private learner?: SurfaceGrid;
  private diagonal = 1;
  private directions: Direction[] = [];
  // counts up whenever the buffers of the heat source are replaced (new Level, Layout or grid)
  revision = 0;

  constructor(device: GPUDevice, tessellatePipeline = createTessellatePipeline(device)) {
    this.device = device;
    this.tessellatePipeline = tessellatePipeline;
    this.pipelines = errorPipelines(device);
    this.results = device.createBuffer({
      label: 'distance sums',
      size: 2 * 16,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    this.alignment = device.createBuffer({
      label: 'best fit',
      size: ALIGNMENT_FLOATS * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(this.alignment, 0, IDENTITY_ALIGNMENT);
  }

  // line up before measuring: the best fit (turn and move) of the Learner surface onto the Target, so only
  // the shape counts. Only with the Paired distance: it needs the pairs of the same grid
  get align(): boolean {
    return this.lineUp;
  }

  set align(on: boolean) {
    this.lineUp = on;
    if (!on) this.device.queue.writeBuffer(this.alignment, 0, IDENTITY_ALIGNMENT);
  }

  private get aligning(): boolean {
    return this.lineUp && this.kind === 'paired';
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

  // which grader: 'nearest' (point to triangle, ADR 0002) or 'paired' (the same grid point); can change at any time
  get metric(): Metric {
    return this.kind;
  }

  set metric(metric: Metric) {
    if (metric === this.kind) return;
    this.kind = metric;
    this.rebuildDirections();
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
    return (this.directions[1] ?? this.directions[0])?.distances; // paired: the same distances
  }

  get heatSource(): HeatSource | undefined {
    if (!this.learner || !this.directions[0]) return undefined;
    const { samples, resU, resV } = this.learner;
    return { distances: this.directions[0].distances, samples, resU, resV };
  }

  private rebuildDirections(): void {
    for (const d of this.directions) d.buffers.destroy();
    this.directions = [];
    this.revision++;
    if (!this.learner || !this.target) return;
    if (this.kind === 'paired') {
      // two grids of different size can't be paired: no directions until the other surface arrives too
      // (setTarget comes before setLearner on a Level load); measure() says why if it stays that way
      if (this.learner.resU !== this.target.resU || this.learner.resV !== this.target.resV) return;
      this.directions = [this.direction(this.learner, this.target, [0, 1], this.pipelines.paired)];
      return;
    }
    const { distance } = this.pipelines;
    this.directions = [this.direction(this.learner, this.target, [0], distance), this.direction(this.target, this.learner, [1], distance)];
  }

  private direction(from: SurfaceGrid, to: SurfaceGrid, slots: number[], pipeline: GPUComputePipeline): Direction {
    const device = this.device;
    const { STORAGE, UNIFORM, COPY_SRC } = GPUBufferUsage;
    const count = from.sampleCount;
    const buffers = new BufferSet(device);
    const distances = buffers.create('sample distances', STORAGE | COPY_SRC, count * 4);
    const weights = buffers.create('sample weights', STORAGE | COPY_SRC, count * 4);
    const distanceParams = buffers.create('distance params', UNIFORM, new Uint32Array([from.resU, from.resV, to.resU, to.resV]));
    const reduceParams = slots.map((slot) => buffers.create('reduce params', UNIFORM, new Uint32Array([count, slot])));

    const group = (pipeline: GPUComputePipeline, buffers: GPUBuffer[]) => device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
    });
    const reduce = this.pipelines.reduce;
    const paired = pipeline === this.pipelines.paired;
    const alignParams = paired ? buffers.create('align params', UNIFORM, new Uint32Array([count, 0, 0, 0])) : undefined;
    return {
      distances,
      weights,
      sampleCount: count,
      pipeline,
      distanceGroup: group(pipeline, paired
        ? [distanceParams, from.positions, to.positions, distances, weights, this.alignment]
        : [distanceParams, from.positions, to.positions, distances, weights]),
      alignGroup: alignParams && group(this.pipelines.align, [alignParams, from.positions, to.positions, weights, this.alignment]),
      reduceGroups: reduceParams.map((params) => group(reduce, [params, distances, weights, this.results])),
      buffers,
    };
  }

  // the Shape distance between the current Learner surface and the Target
  async measure(): Promise<GpuShapeDistance> {
    if (!this.learner || !this.target) throw new Error('ErrorPass needs a Target and a Learner surface first');
    if (this.directions.length === 0) throw new Error('The Paired distance needs the same Layout for the Learner surface and the Target');
    const device = this.device;
    const encoder = device.createCommandEncoder({ label: 'error pass' });
    const compute = encoder.beginComputePass({ label: 'error pass' });
    this.target.tessellate(compute);  // only if it changed (once per Level)
    this.learner.tessellate(compute); // only if the control points changed
    for (const d of this.directions) {
      compute.setPipeline(d.pipeline);
      compute.setBindGroup(0, d.distanceGroup);
      compute.dispatchWorkgroups(Math.ceil(d.sampleCount / DISTANCE_WORKGROUP));
      if (this.aligning && d.alignGroup) {
        // the best fit needs the sample areas of the pass above; then the distances again, lined up
        compute.setPipeline(this.pipelines.align);
        compute.setBindGroup(0, d.alignGroup);
        compute.dispatchWorkgroups(1);
        compute.setPipeline(d.pipeline);
        compute.setBindGroup(0, d.distanceGroup);
        compute.dispatchWorkgroups(Math.ceil(d.sampleCount / DISTANCE_WORKGROUP));
      }
    }
    compute.setPipeline(this.pipelines.reduce);
    for (const d of this.directions) {
      for (const group of d.reduceGroups) {
        compute.setBindGroup(0, group);
        compute.dispatchWorkgroups(1);
      }
    }
    compute.end();

    // each measurement gets its own staging buffer, so two can overlap
    // the six sums, then the best fit: (rotation, two centres) is all the animation needs of it
    const resultBytes = this.results.size;
    const staging = device.createBuffer({ size: resultBytes + 48, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    encoder.copyBufferToBuffer(this.results, 0, staging, 0, resultBytes);
    encoder.copyBufferToBuffer(this.alignment, 0, staging, resultBytes, 48);
    device.queue.submit([encoder.finish()]);
    const res = this.res;
    const diagonal = this.diagonal;
    const aligned = this.aligning;
    await staging.mapAsync(GPUMapMode.READ);
    const r = new Float32Array(staging.getMappedRange().slice(0));
    staging.destroy();

    const toSums = (k: number): DistanceSums => ({ sumWeightedSquares: r[k], sumWeights: r[k + 1], max: r[k + 2] });
    const fit: BestFit | undefined = aligned
      ? { rotation: [r[8], r[9], r[10], r[11]], learnerCentre: [r[12], r[13], r[14]], targetCentre: [r[16], r[17], r[18]] }
      : undefined;
    return { ...combine(toSums(0), toSums(4), diagonal), diagonal, res, fit };
  }

  destroy(): void {
    this.target?.destroy();
    this.learner?.destroy();
    for (const d of this.directions) d.buffers.destroy();
    this.results.destroy();
    this.alignment.destroy();
  }
}

// what a Level needs before it can be graded (grading.ts), measured once when it loads:
//   noise: the Target against itself on a grid shifted by half a cell (the Noise floor);
//          the Paired distance compares the same grid points, so there it is the Target against itself: 0
//   start: the starting Learner surface against the Target (0 % Match)
// uses its own ErrorPass, so the learner can edit meanwhile
export async function measureLevel(device: GPUDevice, tessellatePipeline: GPUComputePipeline,
  target: ControlNet, start: ControlNet, res = ERROR_RES, metric: Metric = 'nearest', align = false,
): Promise<{ noise: GpuShapeDistance; start: GpuShapeDistance }> {
  const pass = new ErrorPass(device, tessellatePipeline);
  try {
    pass.resolution = res;
    pass.metric = metric;
    pass.align = align;
    pass.setTarget(target);
    pass.setLearner(target, metric === 'paired' ? 0 : 0.5);
    const noise = await pass.measure();
    pass.setLearner(start);
    return { noise, start: await pass.measure() };
  } finally {
    pass.destroy();
  }
}
