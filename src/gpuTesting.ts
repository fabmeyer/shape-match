// helpers for the GPU tests (*.gpu.test.ts), which run in headless Chromium with WebGPU

export async function requestTestDevice(): Promise<GPUDevice> {
  if (!navigator.gpu) throw new Error('WebGPU is not available in this browser (see README: npm run test:gpu)');
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error('No WebGPU adapter: the test browser has no GPU access');
  const device = await adapter.requestDevice();
  device.addEventListener('uncapturederror', (e) => { throw (e as GPUUncapturedErrorEvent).error; });
  return device;
}

// copies a buffer (it needs COPY_SRC) to the CPU
export async function readBuffer(device: GPUDevice, buffer: GPUBuffer, size = buffer.size): Promise<ArrayBuffer> {
  const staging = device.createBuffer({ size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(buffer, 0, staging, 0, size);
  device.queue.submit([encoder.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const copy = staging.getMappedRange().slice(0);
  staging.destroy();
  return copy;
}
