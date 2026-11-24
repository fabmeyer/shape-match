import { quat } from 'wgpu-matrix';
import { initWebGPU } from './webgpu-init.ts';
import { ArcballCamera } from './arcball.ts';
import { flatControlNet, getPoint, getWeight, setWeight } from './controlNet.ts';
import { PointDragger } from './pointDrag.ts';
import { History } from './history.ts';
import { evaluate } from './nurbs.ts';
import { ParamSpaceView } from './paramSpace.ts';
import { updateMathPanel } from './mathPanel.ts';
import tessellateCode from './shaders/tessellate.wgsl?raw';
import surfaceCode from './shaders/surface.wgsl?raw';
import pointsCode from './shaders/points.wgsl?raw';
import stemsCode from './shaders/stems.wgsl?raw';
import polygonCode from './shaders/polygon.wgsl?raw';
import gizmoCode from './shaders/gizmo.wgsl?raw';

const canvas = document.getElementById('gpuCanvas') as HTMLCanvasElement;
const { device, context, format } = await initWebGPU(canvas);

// ---------------------------------------------------------
// Depth buffer and camera
// ---------------------------------------------------------

let depthTexture: GPUTexture | undefined;

function getDepthTexture(): GPUTexture {
  if (depthTexture && depthTexture.width === canvas.width && depthTexture.height === canvas.height) {
    return depthTexture;
  }
  depthTexture?.destroy();
  depthTexture = device.createTexture({
    size: [canvas.width, canvas.height],
    format: 'depth24plus',
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
  });
  return depthTexture;
}

const camera = new ArcballCamera(canvas, {
  target: [0, 0, 0],
  distance: 4,
  orientation: quat.fromAxisAngle([1, 0, 0], (50 * Math.PI) / 180),
});

const cameraData = new Float32Array(24);
const cameraBuffer = device.createBuffer({
  label: 'camera',
  size: cameraData.byteLength,
  usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
});

// ---------------------------------------------------------
// Control points
// ---------------------------------------------------------

const net = flatControlNet();
const pointCount = net.countU * net.countV;

// storage buffer, read by the compute pass and the point/line shaders
const controlPointBuffer = device.createBuffer({
  label: 'control points',
  size: net.points.byteLength,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
});
device.queue.writeBuffer(controlPointBuffer, 0, net.points);

// one flag per control point for points.wgsl: 1 = selected, 2 = active at the current (u, v)
const FLAG_SELECTED = 1;
const FLAG_ACTIVE = 2;
const FLAG_MARKER = 4;
const flagData = new Uint32Array(pointCount);
const flagBuffer = device.createBuffer({
  label: 'point flags',
  size: flagData.byteLength,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
});

// the surface point S(u, v) of the parameter space view, drawn like a control point
const markerData = new Float32Array(4);
const markerBuffer = device.createBuffer({
  label: 'marker',
  size: markerData.byteLength,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
});
const markerFlagBuffer = device.createBuffer({
  label: 'marker flag',
  size: 4,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
});
device.queue.writeBuffer(markerFlagBuffer, 0, new Uint32Array([FLAG_MARKER]));

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
const segmentCount = segmentData.length / 2;
const segmentBuffer = device.createBuffer({
  label: 'polygon segments',
  size: segmentData.byteLength,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
});
device.queue.writeBuffer(segmentBuffer, 0, segmentData);

// the three arrows at the selection (struct Gizmo in gizmo.wgsl: origin, size, axis)
const gizmoData = new ArrayBuffer(32);
const gizmoFloats = new Float32Array(gizmoData);
const gizmoInts = new Int32Array(gizmoData);
const gizmoBuffer = device.createBuffer({
  label: 'gizmo',
  size: gizmoData.byteLength,
  usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
});

// ---------------------------------------------------------
// Editing: selection, moving, weights, undo
// ---------------------------------------------------------

type Mode = 'rotate' | 'select';
let mode: Mode = 'rotate';
let showPolygon = true;
let showParam = true;  // parameter space section expanded
let showMath = false;  // formula section expanded

// copy of the initial control points, for the reset button
const initialPoints = net.points.slice();
const undoSteps = new History();

// the surface is only re-evaluated when the control points changed
let surfaceDirty = true;

const undoButton = document.getElementById('undo') as HTMLButtonElement;
const selectionInfo = document.getElementById('selectionInfo') as HTMLParagraphElement;
const weightSlider = document.getElementById('weight') as HTMLInputElement;
const weightNumber = document.getElementById('weightNumber') as HTMLInputElement;
const paramPanel = document.getElementById('paramPanel') as HTMLDetailsElement;
const mathPanel = document.getElementById('mathPanel') as HTMLDetailsElement;
const mathValues = document.getElementById('mathValues') as HTMLElement;

const dragger = new PointDragger(canvas, camera, net, {
  onSelect: selectionChanged,
  onBeginEdit: beginEdit,
  onEdit: pointsChanged,
});
const paramView = new ParamSpaceView(document.getElementById('uvCanvas') as HTMLCanvasElement, net, updateMath);

// the selection stays when rotating, but it is only shown (and editable) in select mode
const NO_SELECTION = new Set<number>();
function visibleSelection(): Set<number> {
  return mode === 'select' ? dragger.selected : NO_SELECTION;
}

// call before every change of the control points, so it can be undone
function beginEdit() {
  undoSteps.push(net.points);
  undoButton.disabled = false;
}

// call after every change of the control points
function pointsChanged() {
  device.queue.writeBuffer(controlPointBuffer, 0, net.points);
  surfaceDirty = true;
  updateMath();
  updateWeightPanel();
}

function selectionChanged() {
  uploadFlags();
  updateWeightPanel();
}

// S(u, v) on the CPU: for the marker, the active control points and the formula panel
let surfaceEval = evaluate(net, paramView.u, paramView.v);
function updateMath() {
  surfaceEval = evaluate(net, paramView.u, paramView.v);
  markerData.set([...surfaceEval.point, 1]);
  device.queue.writeBuffer(markerBuffer, 0, markerData);
  uploadFlags();
  if (showMath) updateMathPanel(mathValues, net, surfaceEval);
}

function uploadFlags() {
  flagData.fill(0);
  for (const k of visibleSelection()) flagData[k] |= FLAG_SELECTED;
  if (showParam || showMath) {
    for (let l = 0; l <= net.q; l++) {
      for (let k = 0; k <= net.p; k++) {
        flagData[(surfaceEval.spanV - net.q + l) * net.countU + surfaceEval.spanU - net.p + k] |= FLAG_ACTIVE;
      }
    }
  }
  device.queue.writeBuffer(flagBuffer, 0, flagData);
}

function undo() {
  const previous = undoSteps.pop();
  if (!previous) return;
  net.points.set(previous);
  undoButton.disabled = undoSteps.size === 0;
  pointsChanged();
}

function resetPoints() {
  beginEdit();
  net.points.set(initialPoints);
  dragger.selected.clear();
  selectionChanged();
  pointsChanged();
}

undoButton.addEventListener('click', undo);
document.getElementById('reset')!.addEventListener('click', resetPoints);
window.addEventListener('keydown', (e) => {
  const typing = e.target instanceof HTMLInputElement && e.target.type === 'number'; // Ctrl+Z undoes typing there
  if ((e.ctrlKey || e.metaKey) && !e.shiftKey && e.key.toLowerCase() === 'z' && !typing) {
    e.preventDefault();
    undo();
  }
  if (e.key === 'Escape') dragger.clear();
});

// ---------------------------------------------------------
// Weights
// ---------------------------------------------------------

const WEIGHT_MIN = 0.1; // weights must stay > 0
const WEIGHT_MAX = 5;

function updateWeightPanel() {
  const selected = [...visibleSelection()];
  weightSlider.disabled = weightNumber.disabled = selected.length === 0;
  if (selected.length === 0) {
    selectionInfo.textContent = mode === 'select'
      ? 'Click a control point. Shift+click adds one, dragging on empty space selects with a rectangle.'
      : 'Switch to Select to pick control points.';
    weightSlider.value = '1';
    weightNumber.value = '';
    return;
  }
  const weights = selected.map((k) => getWeight(net, k));
  if (selected.length === 1) {
    const k = selected[0];
    const [x, y, z] = getPoint(net, k).map((c) => c.toFixed(2));
    selectionInfo.textContent = `P(${k % net.countU}, ${Math.floor(k / net.countU)}) at (${x}, ${y}, ${z})`;
  } else {
    const same = weights.every((w) => Math.abs(w - weights[0]) < 1e-6);
    selectionInfo.textContent = `${selected.length} points selected` + (same ? '' : ', with different weights');
  }
  weightSlider.value = String(weights[0]);
  if (document.activeElement !== weightNumber) weightNumber.value = weights[0].toFixed(2); // don't overwrite while typing
}

// one drag of the slider (or one edit of the number) is one undo step
let weightEditing = false;
function setWeights(value: number) {
  if (!Number.isFinite(value)) return;
  if (!weightEditing) {
    beginEdit();
    weightEditing = true;
  }
  const w = Math.min(WEIGHT_MAX, Math.max(WEIGHT_MIN, value));
  for (const k of visibleSelection()) setWeight(net, k, w);
  pointsChanged();
}

for (const input of [weightSlider, weightNumber]) {
  input.addEventListener('input', () => setWeights(Number(input.value)));
  input.addEventListener('change', () => {
    weightEditing = false;
    updateWeightPanel();
  });
}

// ---------------------------------------------------------
// Mouse mode and toggles
// ---------------------------------------------------------

const modeButtons = document.querySelectorAll<HTMLButtonElement>('#mode button');

function setMode(next: Mode) {
  mode = next;
  camera.enabled = mode === 'rotate';
  dragger.enabled = mode === 'select';
  canvas.dataset.mode = mode;
  selectionChanged(); // no highlight while rotating
  modeButtons.forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.mode === mode)));
}

modeButtons.forEach((b) => b.addEventListener('click', () => setMode(b.dataset.mode as Mode)));
setMode('rotate');

// the control polygon has an on/off switch (a checkbox) instead of a button
const polygonSwitch = document.getElementById('togglePolygon') as HTMLInputElement;
polygonSwitch.addEventListener('change', () => { showPolygon = polygonSwitch.checked; });
// the two side panel sections can be collapsed; while a section is collapsed,
// its work (drawing, numbers, marker and active highlight) is skipped
paramPanel.addEventListener('toggle', () => {
  showParam = paramPanel.open;
  paramView.draw(); // the canvas had no size while collapsed
  updateMath();
});
mathPanel.addEventListener('toggle', () => {
  showMath = mathPanel.open;
  updateMath();
});
showParam = paramPanel.open;
showMath = mathPanel.open;
updateMath();

// ---------------------------------------------------------
// Surface
// ---------------------------------------------------------

// samples per direction: the surface is a RES_U x RES_V grid of points
const RES_U = 128;
const RES_V = 128;

// params for the compute shader (struct Params in tessellate.wgsl, 8 x u32)
const paramsData = new Uint32Array([
  net.countU, net.countV, net.p, net.q, RES_U, RES_V, net.knotsU.length, 0,
]);
const paramsBuffer = device.createBuffer({
  label: 'surface params',
  size: paramsData.byteLength,
  usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
});
device.queue.writeBuffer(paramsBuffer, 0, paramsData);

// both knot vectors in one buffer: U first, V starts at index knotsU.length
const knotData = new Float32Array([...net.knotsU, ...net.knotsV]);
const knotBuffer = device.createBuffer({
  label: 'knots',
  size: knotData.byteLength,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
});
device.queue.writeBuffer(knotBuffer, 0, knotData);

// written by the compute pass (STORAGE), read by the render pass (VERTEX)
const positionBuffer = device.createBuffer({
  label: 'surface positions',
  size: RES_U * RES_V * 16, // one vec4f (16 bytes) per sample
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX,
});
// the normal at every sample, for the lighting (same layout as the positions)
const normalBuffer = device.createBuffer({
  label: 'surface normals',
  size: RES_U * RES_V * 16,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX,
});

// two triangles per grid cell; the patch is open, so no wrap-around like the torus knot
const surfaceIndices = new Uint32Array((RES_U - 1) * (RES_V - 1) * 6);
for (let j = 0, k = 0; j < RES_V - 1; j++) {
  for (let i = 0; i < RES_U - 1; i++, k += 6) {
    const a = j * RES_U + i; // (i, j)
    const b = a + 1;         // (i + 1, j)
    const c = a + RES_U;     // (i, j + 1)
    const d = c + 1;         // (i + 1, j + 1)
    surfaceIndices.set([a, b, d, a, d, c], k);
  }
}
const surfaceIndexBuffer = device.createBuffer({
  label: 'surface indices',
  size: surfaceIndices.byteLength,
  usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
});
device.queue.writeBuffer(surfaceIndexBuffer, 0, surfaceIndices);

// ---------------------------------------------------------
// Pipelines
// ---------------------------------------------------------

const tessellatePipeline = device.createComputePipeline({
  label: 'tessellate',
  layout: 'auto',
  compute: {
    module: device.createShaderModule({ label: 'tessellate', code: tessellateCode }),
    entryPoint: 'cs_main',
  },
});

const surfaceModule = device.createShaderModule({ label: 'surface', code: surfaceCode });
const surfacePipeline = device.createRenderPipeline({
  label: 'surface',
  layout: 'auto',
  vertex: {
    module: surfaceModule,
    entryPoint: 'vs_main',
    buffers: [
      { arrayStride: 16, attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x4' }] }, // positions
      { arrayStride: 16, attributes: [{ shaderLocation: 1, offset: 0, format: 'float32x4' }] }, // normals
    ],
  },
  fragment: { module: surfaceModule, entryPoint: 'fs_main', targets: [{ format }] },
  primitive: { topology: 'triangle-list', cullMode: 'none' },
  depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
});

const pointsModule = device.createShaderModule({ label: 'points', code: pointsCode });
const pointsPipeline = device.createRenderPipeline({
  label: 'control points',
  layout: 'auto',
  vertex: { module: pointsModule, entryPoint: 'vs_main' },
  fragment: { module: pointsModule, entryPoint: 'fs_main', targets: [{ format }] },
  primitive: { topology: 'triangle-list' },
  depthStencil: { format: 'depth24plus', depthWriteEnabled: false, depthCompare: 'always' },
});

// red line from each point down to z = 0 (the starting height)
const stemsModule = device.createShaderModule({ label: 'stems', code: stemsCode });
const stemsPipeline = device.createRenderPipeline({
  label: 'stems',
  layout: 'auto',
  vertex: { module: stemsModule, entryPoint: 'vs_main' },
  fragment: { module: stemsModule, entryPoint: 'fs_main', targets: [{ format }] },
  primitive: { topology: 'line-list' },
  depthStencil: { format: 'depth24plus', depthWriteEnabled: false, depthCompare: 'less' },
});

// dashed lines between neighbouring control points, drawn on top like the points
// (at the start they lie exactly in the surface, a depth test would flicker)
const polygonModule = device.createShaderModule({ label: 'polygon', code: polygonCode });
const polygonPipeline = device.createRenderPipeline({
  label: 'control polygon',
  layout: 'auto',
  vertex: { module: polygonModule, entryPoint: 'vs_main' },
  fragment: { module: polygonModule, entryPoint: 'fs_main', targets: [{ format }] },
  primitive: { topology: 'line-list' },
  depthStencil: { format: 'depth24plus', depthWriteEnabled: false, depthCompare: 'always' },
});

const gizmoModule = device.createShaderModule({ label: 'gizmo', code: gizmoCode });
const gizmoPipeline = device.createRenderPipeline({
  label: 'gizmo',
  layout: 'auto',
  vertex: { module: gizmoModule, entryPoint: 'vs_main' },
  fragment: { module: gizmoModule, entryPoint: 'fs_main', targets: [{ format }] },
  primitive: { topology: 'triangle-list' },
  depthStencil: { format: 'depth24plus', depthWriteEnabled: false, depthCompare: 'always' },
});

const tessellateBindGroup = device.createBindGroup({
  layout: tessellatePipeline.getBindGroupLayout(0),
  entries: [
    { binding: 0, resource: { buffer: paramsBuffer } },
    { binding: 1, resource: { buffer: controlPointBuffer } },
    { binding: 2, resource: { buffer: knotBuffer } },
    { binding: 3, resource: { buffer: positionBuffer } },
    { binding: 4, resource: { buffer: normalBuffer } },
  ],
});
const surfaceBindGroup = device.createBindGroup({
  layout: surfacePipeline.getBindGroupLayout(0),
  entries: [{ binding: 0, resource: { buffer: cameraBuffer } }],
});
const pointsBindGroup = device.createBindGroup({
  layout: pointsPipeline.getBindGroupLayout(0),
  entries: [
    { binding: 0, resource: { buffer: cameraBuffer } },
    { binding: 1, resource: { buffer: controlPointBuffer } },
    { binding: 2, resource: { buffer: flagBuffer } },
  ],
});
// same pipeline as the control points, but with the one surface point S(u, v)
const markerBindGroup = device.createBindGroup({
  layout: pointsPipeline.getBindGroupLayout(0),
  entries: [
    { binding: 0, resource: { buffer: cameraBuffer } },
    { binding: 1, resource: { buffer: markerBuffer } },
    { binding: 2, resource: { buffer: markerFlagBuffer } },
  ],
});
const polygonBindGroup = device.createBindGroup({
  layout: polygonPipeline.getBindGroupLayout(0),
  entries: [
    { binding: 0, resource: { buffer: cameraBuffer } },
    { binding: 1, resource: { buffer: controlPointBuffer } },
    { binding: 2, resource: { buffer: segmentBuffer } },
  ],
});
const gizmoBindGroup = device.createBindGroup({
  layout: gizmoPipeline.getBindGroupLayout(0),
  entries: [
    { binding: 0, resource: { buffer: cameraBuffer } },
    { binding: 1, resource: { buffer: gizmoBuffer } },
  ],
});
const stemsBindGroup = device.createBindGroup({
  layout: stemsPipeline.getBindGroupLayout(0),
  entries: [
    { binding: 0, resource: { buffer: cameraBuffer } },
    { binding: 1, resource: { buffer: controlPointBuffer } },
  ],
});

// ---------------------------------------------------------
// Frame loop
// ---------------------------------------------------------

function frame() {
  cameraData.set(camera.viewProjection(canvas.width / canvas.height), 0);
  cameraData.set([...camera.eye, 1], 16);
  cameraData.set([canvas.width, canvas.height, 0, 0], 20);
  device.queue.writeBuffer(cameraBuffer, 0, cameraData);

  // the arrows follow the selection and keep their size on screen, so they are updated every frame
  const gizmo = dragger.gizmo();
  if (gizmo) {
    gizmoFloats.set([...gizmo.origin, gizmo.length]);
    gizmoInts[4] = gizmo.axis;
    device.queue.writeBuffer(gizmoBuffer, 0, gizmoData);
  }

  const encoder = device.createCommandEncoder();

  // evaluate the surface again, but only if a control point changed
  if (surfaceDirty) {
    const compute = encoder.beginComputePass();
    compute.setPipeline(tessellatePipeline);
    compute.setBindGroup(0, tessellateBindGroup);
    compute.dispatchWorkgroups(Math.ceil(RES_U / 8), Math.ceil(RES_V / 8));
    compute.end();
    surfaceDirty = false;
  }

  const pass = encoder.beginRenderPass({
    colorAttachments: [{
      view: context.getCurrentTexture().createView(),
      clearValue: { r: 0.07, g: 0.07, b: 0.09, a: 1 },
      loadOp: 'clear',
      storeOp: 'store',
    }],
    depthStencilAttachment: {
      view: getDepthTexture().createView(),
      depthClearValue: 1.0,
      depthLoadOp: 'clear',
      depthStoreOp: 'store',
    },
  });

  pass.setPipeline(surfacePipeline);
  pass.setBindGroup(0, surfaceBindGroup);
  pass.setVertexBuffer(0, positionBuffer);
  pass.setVertexBuffer(1, normalBuffer);
  pass.setIndexBuffer(surfaceIndexBuffer, 'uint32');
  pass.drawIndexed(surfaceIndices.length);

  pass.setPipeline(stemsPipeline);
  pass.setBindGroup(0, stemsBindGroup);
  pass.draw(2, pointCount);

  if (showPolygon) {
    pass.setPipeline(polygonPipeline);
    pass.setBindGroup(0, polygonBindGroup);
    pass.draw(2, segmentCount);
  }

  pass.setPipeline(pointsPipeline);
  pass.setBindGroup(0, pointsBindGroup);
  pass.draw(6, pointCount);

  if (showParam || showMath) {
    pass.setBindGroup(0, markerBindGroup);
    pass.draw(6, 1);
  }

  if (gizmo) {
    pass.setPipeline(gizmoPipeline);
    pass.setBindGroup(0, gizmoBindGroup);
    pass.draw(9, 3);
  }

  pass.end();
  device.queue.submit([encoder.finish()]);

  requestAnimationFrame(frame);
}

requestAnimationFrame(frame);