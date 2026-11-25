// the GPU side of one NURBS surface (the Learner surface or the Target): its buffers and bind groups
// everything here depends on the Layout, so a new Layout means a new SurfaceGpu (ADR 0001):
// a Level brings its own Layout, and knot insertion changes the Layout inside a Level
import type { ControlNet } from './controlNet.ts';
import { samplePositions } from './samples.ts';

// samples per direction before the knots are added (samples.ts)
export const RENDER_RES = 128;

// the pipelines are created once and shared by all surfaces
export interface SurfacePipelines {
  tessellate: GPUComputePipeline;
  surface: GPURenderPipeline;
  points: GPURenderPipeline;
  polygon: GPURenderPipeline;
  stems: GPURenderPipeline;
}

export class SurfaceGpu {
  readonly countU: number;
  readonly countV: number;
  readonly pointCount: number;
  readonly resU: number; // samples in u (uniform grid plus knots)
  readonly resV: number;

  readonly controlPoints: GPUBuffer; // (w*x, w*y, w*z, w) per control point
  readonly flags: GPUBuffer;         // one u32 per control point for points.wgsl

  // set when the control points changed; the next frame evaluates the surface again
  dirty = true;

  private device: GPUDevice;
  private pipelines: SurfacePipelines;
  private buffers: GPUBuffer[] = [];
  private positions: GPUBuffer;
  private normals: GPUBuffer;
  private indices: GPUBuffer;
  private indexCount: number;
  private segmentCount: number;
  private tessellateGroup: GPUBindGroup;
  private surfaceGroup: GPUBindGroup;
  private pointsGroup: GPUBindGroup;
  private polygonGroup: GPUBindGroup;
  private stemsGroup: GPUBindGroup;

  // points: the control points to upload; usually net.points, the Target passes a shifted copy
  constructor(device: GPUDevice, pipelines: SurfacePipelines, camera: GPUBuffer, color: GPUBuffer,
    net: ControlNet, points: Float32Array = net.points) {
    this.device = device;
    this.pipelines = pipelines;
    this.countU = net.countU;
    this.countV = net.countV;
    this.pointCount = net.countU * net.countV;

    const samplesU = samplePositions(net.knotsU, net.p, RENDER_RES);
    const samplesV = samplePositions(net.knotsV, net.q, RENDER_RES);
    this.resU = samplesU.length;
    this.resV = samplesV.length;

    this.controlPoints = this.buffer('control points', points.byteLength, GPUBufferUsage.STORAGE, points);
    this.flags = this.buffer('point flags', this.pointCount * 4, GPUBufferUsage.STORAGE);

    // params for the compute shader (struct Params in tessellate.wgsl, 8 x u32)
    const params = new Uint32Array([net.countU, net.countV, net.p, net.q, this.resU, this.resV, net.knotsU.length, 0]);
    const paramBuffer = this.buffer('surface params', params.byteLength, GPUBufferUsage.UNIFORM, params);
    // both knot vectors in one buffer: U first, V starts at index knotsU.length
    const knots = new Float32Array([...net.knotsU, ...net.knotsV]);
    const knotBuffer = this.buffer('knots', knots.byteLength, GPUBufferUsage.STORAGE, knots);
    // the parameter values of the grid: resU values of u, then resV values of v
    const samples = new Float32Array([...samplesU, ...samplesV]);
    const sampleBuffer = this.buffer('sample positions', samples.byteLength, GPUBufferUsage.STORAGE, samples);

    // written by the compute pass (STORAGE), read by the render pass (VERTEX), one vec4f per sample
    const sampleCount = this.resU * this.resV;
    this.positions = this.buffer('surface positions', sampleCount * 16, GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX);
    this.normals = this.buffer('surface normals', sampleCount * 16, GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX);

    // two triangles per grid cell; the patch is open, so no wrap-around like the torus knot
    const { resU, resV } = this;
    const indices = new Uint32Array((resU - 1) * (resV - 1) * 6);
    for (let j = 0, k = 0; j < resV - 1; j++) {
      for (let i = 0; i < resU - 1; i++, k += 6) {
        const a = j * resU + i; // (i, j)
        const b = a + 1;        // (i + 1, j)
        const c = a + resU;     // (i, j + 1)
        const d = c + 1;        // (i + 1, j + 1)
        indices.set([a, b, d, a, d, c], k);
      }
    }
    this.indexCount = indices.length;
    this.indices = this.buffer('surface indices', indices.byteLength, GPUBufferUsage.INDEX, indices);

    // control polygon: one line between every two neighbouring control points (in u and in v)
    const segments: number[] = [];
    for (let j = 0; j < net.countV; j++) {
      for (let i = 0; i < net.countU; i++) {
        const k = j * net.countU + i;
        if (i + 1 < net.countU) segments.push(k, k + 1);
        if (j + 1 < net.countV) segments.push(k, k + net.countU);
      }
    }
    const segmentData = new Uint32Array(segments);
    this.segmentCount = segmentData.length / 2;
    const segmentBuffer = this.buffer('polygon segments', segmentData.byteLength, GPUBufferUsage.STORAGE, segmentData);

    const group = (pipeline: GPUPipelineBase, buffers: GPUBuffer[]) => device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
    });
    this.tessellateGroup = group(pipelines.tessellate,
      [paramBuffer, this.controlPoints, knotBuffer, this.positions, this.normals, sampleBuffer]);
    this.surfaceGroup = group(pipelines.surface, [camera, color]);
    this.pointsGroup = group(pipelines.points, [camera, this.controlPoints, this.flags]);
    this.polygonGroup = group(pipelines.polygon, [camera, this.controlPoints, segmentBuffer]);
    this.stemsGroup = group(pipelines.stems, [camera, this.controlPoints]);
  }

  private buffer(label: string, size: number, usage: number, data?: ArrayBufferView): GPUBuffer {
    const buffer = this.device.createBuffer({ label, size, usage: usage | GPUBufferUsage.COPY_DST });
    if (data) this.device.queue.writeBuffer(buffer, 0, data as GPUAllowSharedBufferSource);
    this.buffers.push(buffer);
    return buffer;
  }

  uploadPoints(points: Float32Array): void {
    this.device.queue.writeBuffer(this.controlPoints, 0, points);
    this.dirty = true;
  }

  uploadFlags(flags: Uint32Array): void {
    this.device.queue.writeBuffer(this.flags, 0, flags);
  }

  // evaluates the surface again if the control points changed
  tessellate(compute: GPUComputePassEncoder): void {
    if (!this.dirty) return;
    compute.setPipeline(this.pipelines.tessellate);
    compute.setBindGroup(0, this.tessellateGroup);
    compute.dispatchWorkgroups(Math.ceil(this.resU / 8), Math.ceil(this.resV / 8));
    this.dirty = false;
  }

  drawSurface(pass: GPURenderPassEncoder): void {
    pass.setPipeline(this.pipelines.surface);
    pass.setBindGroup(0, this.surfaceGroup);
    pass.setVertexBuffer(0, this.positions);
    pass.setVertexBuffer(1, this.normals);
    pass.setIndexBuffer(this.indices, 'uint32');
    pass.drawIndexed(this.indexCount);
  }

  drawPoints(pass: GPURenderPassEncoder): void {
    pass.setPipeline(this.pipelines.points);
    pass.setBindGroup(0, this.pointsGroup);
    pass.draw(6, this.pointCount);
  }

  drawPolygon(pass: GPURenderPassEncoder): void {
    pass.setPipeline(this.pipelines.polygon);
    pass.setBindGroup(0, this.polygonGroup);
    pass.draw(2, this.segmentCount);
  }

  // red line from each control point down to z = 0 (the starting height)
  drawStems(pass: GPURenderPassEncoder): void {
    pass.setPipeline(this.pipelines.stems);
    pass.setBindGroup(0, this.stemsGroup);
    pass.draw(2, this.pointCount);
  }

  destroy(): void {
    for (const buffer of this.buffers) buffer.destroy();
  }
}
