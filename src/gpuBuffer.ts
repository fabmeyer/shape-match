// creating GPU buffers: one filled with data, or an empty one of a given size in bytes
// COPY_DST is always added, so every buffer can be updated with writeBuffer

export function createBuffer(device: GPUDevice, label: string, usage: number, content: ArrayBufferView | number): GPUBuffer {
  const size = typeof content === 'number' ? content : content.byteLength;
  const buffer = device.createBuffer({ label, size, usage: usage | GPUBufferUsage.COPY_DST });
  if (typeof content !== 'number') device.queue.writeBuffer(buffer, 0, content as GPUAllowSharedBufferSource);
  return buffer;
}

// keeps every buffer it creates, so an object can free all of them at once
export class BufferSet {
  private device: GPUDevice;
  private buffers: GPUBuffer[] = [];

  constructor(device: GPUDevice) {
    this.device = device;
  }

  create(label: string, usage: number, content: ArrayBufferView | number): GPUBuffer {
    const buffer = createBuffer(this.device, label, usage, content);
    this.buffers.push(buffer);
    return buffer;
  }

  destroy(): void {
    for (const buffer of this.buffers) buffer.destroy();
    this.buffers = [];
  }
}
