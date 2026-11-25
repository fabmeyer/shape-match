// one NURBS surface evaluated on the GPU on a grid of (u, v) samples (shaders/tessellate.wgsl)
// used for drawing (render grid, SurfaceGpu) and for the Shape distance (error grid, ErrorPass)
// everything here depends on the Layout and the grid, so a new Layout means a new SurfaceGrid (ADR 0001)
import type { ControlNet } from './controlNet.ts';
import { samplePositions } from './samples.ts';
import tessellateCode from './shaders/tessellate.wgsl?raw';

const WORKGROUP = 8; // @workgroup_size(8, 8) in tessellate.wgsl

export function createTessellatePipeline(device: GPUDevice): GPUComputePipeline {
  return device.createComputePipeline({
    label: 'tessellate',
    layout: 'auto',
    compute: {
      module: device.createShaderModule({ label: 'tessellate', code: tessellateCode }),
      entryPoint: 'cs_main',
    },
  });
}

export class SurfaceGrid {
  readonly resU: number; // samples in u (uniform grid plus knots, samples.ts)
  readonly resV: number;
  readonly sampleCount: number;
  readonly controlPoints: GPUBuffer; // (w*x, w*y, w*z, w), index j * countU + i
  readonly positions: GPUBuffer;     // vec4f per sample, index j * resU + i (also a vertex buffer)
  readonly normals: GPUBuffer;       // vec4f per sample

  // set when the control points changed; the next tessellate() evaluates the surface again
  dirty = true;

  private device: GPUDevice;
  private pipeline: GPUComputePipeline;
  private buffers: GPUBuffer[] = [];
  private bindGroup: GPUBindGroup;

  // res: uniform samples per direction before the knots; shift moves the uniform grid by a fraction of a cell
  // points: the control points to upload; usually net.points, the drawn Target passes a shifted copy
  constructor(device: GPUDevice, pipeline: GPUComputePipeline, net: ControlNet, res: number,
    { points = net.points, shift = 0 }: { points?: Float32Array; shift?: number } = {}) {
    this.device = device;
    this.pipeline = pipeline;

    const samplesU = samplePositions(net.knotsU, net.p, res, shift);
    const samplesV = samplePositions(net.knotsV, net.q, res, shift);
    this.resU = samplesU.length;
    this.resV = samplesV.length;
    this.sampleCount = this.resU * this.resV;

    const { STORAGE, UNIFORM, VERTEX, COPY_SRC } = GPUBufferUsage;
    this.controlPoints = this.buffer('control points', STORAGE, points);
    // struct Params in tessellate.wgsl, 8 x u32
    const params = this.buffer('surface params', UNIFORM,
      new Uint32Array([net.countU, net.countV, net.p, net.q, this.resU, this.resV, net.knotsU.length, 0]));
    // both knot vectors in one buffer: U first, V starts at index knotsU.length
    const knots = this.buffer('knots', STORAGE, new Float32Array([...net.knotsU, ...net.knotsV]));
    // the parameter values of the grid: resU values of u, then resV values of v
    const samples = this.buffer('sample positions', STORAGE, new Float32Array([...samplesU, ...samplesV]));
    // written by the compute pass, read by the render pass or the distance pass (COPY_SRC: tests read them back)
    this.positions = this.buffer('surface positions', STORAGE | VERTEX | COPY_SRC, this.sampleCount * 16);
    this.normals = this.buffer('surface normals', STORAGE | VERTEX | COPY_SRC, this.sampleCount * 16);

    this.bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [params, this.controlPoints, knots, this.positions, this.normals, samples]
        .map((buffer, binding) => ({ binding, resource: { buffer } })),
    });
  }

  // a buffer filled with data, or an empty one of the given size in bytes
  private buffer(label: string, usage: number, content: ArrayBufferView | number): GPUBuffer {
    const size = typeof content === 'number' ? content : content.byteLength;
    const buffer = this.device.createBuffer({ label, size, usage: usage | GPUBufferUsage.COPY_DST });
    if (typeof content !== 'number') this.device.queue.writeBuffer(buffer, 0, content as GPUAllowSharedBufferSource);
    this.buffers.push(buffer);
    return buffer;
  }

  uploadPoints(points: Float32Array): void {
    this.device.queue.writeBuffer(this.controlPoints, 0, points as GPUAllowSharedBufferSource);
    this.dirty = true;
  }

  // evaluates the surface again if the control points changed
  tessellate(compute: GPUComputePassEncoder): void {
    if (!this.dirty) return;
    compute.setPipeline(this.pipeline);
    compute.setBindGroup(0, this.bindGroup);
    compute.dispatchWorkgroups(Math.ceil(this.resU / WORKGROUP), Math.ceil(this.resV / WORKGROUP));
    this.dirty = false;
  }

  destroy(): void {
    for (const buffer of this.buffers) buffer.destroy();
  }
}
