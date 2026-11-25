// the GPU side of one drawn NURBS surface (the Learner surface or the Target): its render grid and
// what is drawn on top (control points, control polygon, stems)
// everything here depends on the Layout, so a new Layout means a new SurfaceGpu (ADR 0001):
// a Level brings its own Layout, and knot insertion changes the Layout inside a Level
import type { ControlNet } from './controlNet.ts';
import { SurfaceGrid } from './surfaceGrid.ts';

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
  readonly pointCount: number;
  readonly grid: SurfaceGrid;

  private device: GPUDevice;
  private pipelines: SurfacePipelines;
  private buffers: GPUBuffer[] = [];
  private flags: GPUBuffer; // one u32 per control point for points.wgsl
  private indices: GPUBuffer;
  private indexCount: number;
  private segmentCount: number;
  private surfaceGroup: GPUBindGroup;
  private pointsGroup: GPUBindGroup;
  private polygonGroup: GPUBindGroup;
  private stemsGroup: GPUBindGroup;

  // points: the control points to upload; usually net.points, the Target passes a shifted copy
  constructor(device: GPUDevice, pipelines: SurfacePipelines, camera: GPUBuffer, color: GPUBuffer,
    net: ControlNet, points: Float32Array = net.points) {
    this.device = device;
    this.pipelines = pipelines;
    this.pointCount = net.countU * net.countV;
    this.grid = new SurfaceGrid(device, pipelines.tessellate, net, RENDER_RES, { points });
    this.flags = this.buffer('point flags', GPUBufferUsage.STORAGE, this.pointCount * 4);

    // two triangles per grid cell; the patch is open, so no wrap-around like the torus knot
    const { resU, resV } = this.grid;
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
    this.indices = this.buffer('surface indices', GPUBufferUsage.INDEX, indices);

    // control polygon: one line between every two neighbouring control points (in u and in v)
    const segments: number[] = [];
    for (let j = 0; j < net.countV; j++) {
      for (let i = 0; i < net.countU; i++) {
        const k = j * net.countU + i;
        if (i + 1 < net.countU) segments.push(k, k + 1);
        if (j + 1 < net.countV) segments.push(k, k + net.countU);
      }
    }
    this.segmentCount = segments.length / 2;
    const segmentBuffer = this.buffer('polygon segments', GPUBufferUsage.STORAGE, new Uint32Array(segments));

    const controlPoints = this.grid.controlPoints;
    const group = (pipeline: GPUPipelineBase, buffers: GPUBuffer[]) => device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
    });
    this.surfaceGroup = group(pipelines.surface, [camera, color]);
    this.pointsGroup = group(pipelines.points, [camera, controlPoints, this.flags]);
    this.polygonGroup = group(pipelines.polygon, [camera, controlPoints, segmentBuffer]);
    this.stemsGroup = group(pipelines.stems, [camera, controlPoints]);
  }

  // a buffer filled with data, or an empty one of the given size in bytes
  private buffer(label: string, usage: number, content: ArrayBufferView | number): GPUBuffer {
    const size = typeof content === 'number' ? content : content.byteLength;
    const buffer = this.device.createBuffer({ label, size, usage: usage | GPUBufferUsage.COPY_DST });
    if (typeof content !== 'number') this.device.queue.writeBuffer(buffer, 0, content as GPUAllowSharedBufferSource);
    this.buffers.push(buffer);
    return buffer;
  }

  get dirty(): boolean {
    return this.grid.dirty;
  }

  uploadPoints(points: Float32Array): void {
    this.grid.uploadPoints(points);
  }

  uploadFlags(flags: Uint32Array): void {
    this.device.queue.writeBuffer(this.flags, 0, flags as GPUAllowSharedBufferSource);
  }

  // evaluates the surface again if the control points changed
  tessellate(compute: GPUComputePassEncoder): void {
    this.grid.tessellate(compute);
  }

  drawSurface(pass: GPURenderPassEncoder): void {
    pass.setPipeline(this.pipelines.surface);
    pass.setBindGroup(0, this.surfaceGroup);
    pass.setVertexBuffer(0, this.grid.positions);
    pass.setVertexBuffer(1, this.grid.normals);
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
    this.grid.destroy();
    for (const buffer of this.buffers) buffer.destroy();
  }
}
