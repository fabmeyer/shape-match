import { quat } from 'wgpu-matrix';
import { initWebGPU } from './webgpu-init.ts';
import { ArcballCamera, type ViewName } from './arcball.ts';
import { getPoint, getWeight, setWeight, type ControlNet } from './controlNet.ts';
import { PointDragger } from './pointDrag.ts';
import { History } from './history.ts';
import { evaluate } from './nurbs.ts';
import { ParamSpaceView } from './paramSpace.ts';
import { PartitionView } from './partition.ts';
import { updateMathPanel } from './mathPanel.ts';
import { allTasks, cloneNet, sameLayout, shiftedPoints, TARGET_OFFSET, type Task } from './target.ts';
import { SurfaceGpu, type SurfacePipelines } from './surfaceGpu.ts';
import { buildWalkers, MAX_SEGMENTS, SEGMENT_FLOATS } from './walkers.ts';
import tessellateCode from './shaders/tessellate.wgsl?raw';
import surfaceCode from './shaders/surface.wgsl?raw';
import pointsCode from './shaders/points.wgsl?raw';
import stemsCode from './shaders/stems.wgsl?raw';
import polygonCode from './shaders/polygon.wgsl?raw';
import gizmoCode from './shaders/gizmo.wgsl?raw';
import curvesCode from './shaders/curves.wgsl?raw';

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

// far enough back that both surfaces (together about 5 wide) fit across the view;
// a narrow window needs a larger distance (fovY = 45 degrees, as in the camera)
function startDistance(): number {
  const aspect = canvas.clientWidth / Math.max(1, canvas.clientHeight);
  return Math.max(5, 3.4 / (Math.tan(Math.PI / 8) * Math.min(aspect, 1.6)));
}

const camera = new ArcballCamera(canvas, {
  target: [TARGET_OFFSET / 2, 0, 0], // between the two surfaces
  distance: startDistance(),
  orientation: quat.fromAxisAngle([1, 0, 0], (50 * Math.PI) / 180),
});

const cameraData = new Float32Array(24);
const cameraBuffer = device.createBuffer({
  label: 'camera',
  size: cameraData.byteLength,
  usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
});

// ---------------------------------------------------------
// Pipelines (shared by both surfaces, independent of the Layout)
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

// walkers: two pipelines from the same shader, for the parts in front of and behind the surface
const curvesModule = device.createShaderModule({ label: 'curves', code: curvesCode });
function curvesPipeline(hidden: boolean): GPURenderPipeline {
  return device.createRenderPipeline({
    label: hidden ? 'curves behind' : 'curves',
    layout: 'auto',
    vertex: { module: curvesModule, entryPoint: 'vs_main' },
    fragment: { module: curvesModule, entryPoint: 'fs_main', targets: [{ format }], constants: { HIDDEN: hidden ? 1 : 0 } },
    primitive: { topology: 'triangle-list' },
    depthStencil: { format: 'depth24plus', depthWriteEnabled: false, depthCompare: hidden ? 'greater' : 'less-equal' },
  });
}
const curvesFrontPipeline = curvesPipeline(false);
const curvesBehindPipeline = curvesPipeline(true);

const pipelines: SurfacePipelines = {
  tessellate: tessellatePipeline,
  surface: surfacePipeline,
  points: pointsPipeline,
  polygon: polygonPipeline,
  stems: stemsPipeline,
};

// surface colours (surfaceColor in surface.wgsl)
function colorBuffer(label: string, rgb: [number, number, number]): GPUBuffer {
  const buffer = device.createBuffer({ label, size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(buffer, 0, new Float32Array([...rgb, 1]));
  return buffer;
}
const learnerColorBuffer = colorBuffer('learner colour', [0.45, 0.55, 0.75]);
const targetColorBuffer = colorBuffer('target colour', [0.85, 0.58, 0.38]);

// ---------------------------------------------------------
// Buffers that don't depend on the Layout
// ---------------------------------------------------------

// flags for points.wgsl: 1 = selected, 2 = active at the current (u, v)
const FLAG_SELECTED = 1;
const FLAG_ACTIVE = 2;
const FLAG_MARKER = 4;

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
const markerBindGroup = device.createBindGroup({
  layout: pointsPipeline.getBindGroupLayout(0),
  entries: [
    { binding: 0, resource: { buffer: cameraBuffer } },
    { binding: 1, resource: { buffer: markerBuffer } },
    { binding: 2, resource: { buffer: markerFlagBuffer } },
  ],
});

// walkers (slide 26): line segments of the curves (fixed size), and one dot per row of control points (per Layout)
const walkerSegmentBuffer = device.createBuffer({
  label: 'walker segments',
  size: MAX_SEGMENTS * SEGMENT_FLOATS * 4,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
});
let walkerSegmentCount = 0;
// 'auto' layouts belong to one pipeline, so each curves pipeline gets its own bind group
const curvesBindGroups = [curvesFrontPipeline, curvesBehindPipeline].map((pipeline) => device.createBindGroup({
  layout: pipeline.getBindGroupLayout(0),
  entries: [
    { binding: 0, resource: { buffer: cameraBuffer } },
    { binding: 1, resource: { buffer: walkerSegmentBuffer } },
  ],
}));

// the three arrows at the selection (struct Gizmo in gizmo.wgsl: origin, size, axis)
const gizmoData = new ArrayBuffer(32);
const gizmoFloats = new Float32Array(gizmoData);
const gizmoInts = new Int32Array(gizmoData);
const gizmoBuffer = device.createBuffer({
  label: 'gizmo',
  size: gizmoData.byteLength,
  usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
});
const gizmoBindGroup = device.createBindGroup({
  layout: gizmoPipeline.getBindGroupLayout(0),
  entries: [
    { binding: 0, resource: { buffer: cameraBuffer } },
    { binding: 1, resource: { buffer: gizmoBuffer } },
  ],
});

// ---------------------------------------------------------
// The Task and the two surfaces (rebuilt by loadTask and setLearner)
// ---------------------------------------------------------

const tasks = allTasks();
const requested = new URLSearchParams(location.search).get('task');
let task: Task = tasks.find((t) => t.id === requested) ?? tasks[0];

let net: ControlNet = cloneNet(task.start); // the Learner surface
let learner: SurfaceGpu | undefined;
let target: SurfaceGpu | undefined;
let flagData = new Uint32Array(0);
let walkerPointBuffer: GPUBuffer | undefined;
let walkerFlagBuffer: GPUBuffer | undefined;
let walkerPointsBindGroup: GPUBindGroup | undefined;

// ---------------------------------------------------------
// Editing: selection, moving, weights, undo
// ---------------------------------------------------------

type Mode = 'rotate' | 'select';
let mode: Mode = 'rotate';
let showPolygon = true;
let showMath = false;  // formula section expanded
let showSolution = false; // control points of the target
let showWalkers = false;  // row curves, walkers and iso-curve at the current (u, v)
let showMarker = true;    // the pink dot S(u, v) on the surface

const undoSteps = new History();

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

// partition of unity: one plot per parameter direction; dragging in it moves u or v
const partitionPanel = document.getElementById('partitionPanel') as HTMLDetailsElement;
const partitionU = new PartitionView(document.getElementById('partitionU') as HTMLCanvasElement, 'u',
  net.countU, net.p, net.knotsU, (u) => { paramView.u = u; paramView.draw(); updateMath(); });
const partitionV = new PartitionView(document.getElementById('partitionV') as HTMLCanvasElement, 'v',
  net.countV, net.q, net.knotsV, (v) => { paramView.v = v; paramView.draw(); updateMath(); });

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
  learner!.uploadPoints(net.points);
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
  if (partitionPanel.open) {
    partitionU.t = paramView.u;
    partitionV.t = paramView.v;
    partitionU.draw();
    partitionV.draw();
  }
  markerData.set([...surfaceEval.point, 1]);
  device.queue.writeBuffer(markerBuffer, 0, markerData);
  uploadFlags();
  if (showMath) updateMathPanel(mathValues, net, surfaceEval);
  if (showWalkers) uploadWalkers();
}

// the walkers depend on the control points and on (u, v), so they are rebuilt with the math
function uploadWalkers() {
  const walkers = buildWalkers(net, paramView.u, paramView.v);
  walkerSegmentCount = walkers.segmentCount;
  device.queue.writeBuffer(walkerSegmentBuffer, 0, walkers.segments, 0, walkerSegmentCount * SEGMENT_FLOATS);
  device.queue.writeBuffer(walkerPointBuffer!, 0, walkers.points);
  device.queue.writeBuffer(walkerFlagBuffer!, 0, walkers.flags);
}

function uploadFlags() {
  flagData.fill(0);
  for (const k of visibleSelection()) flagData[k] |= FLAG_SELECTED;
  // the awake control points belong to the surface point: marked only while it is shown
  if (showMarker) {
    for (let l = 0; l <= net.q; l++) {
      for (let k = 0; k <= net.p; k++) {
        flagData[(surfaceEval.spanV - net.q + l) * net.countU + surfaceEval.spanU - net.p + k] |= FLAG_ACTIVE;
      }
    }
  }
  learner!.uploadFlags(flagData);
}

function undo() {
  const previous = undoSteps.pop();
  if (!previous) return;
  net.points.set(previous);
  undoButton.disabled = undoSteps.size === 0;
  pointsChanged();
}

// back to the starting Learner surface of the Task
function resetPoints() {
  if (!sameLayout(net, task.start)) { // the Layout changed (knot insertion): start over
    setLearner(cloneNet(task.start));
    return;
  }
  beginEdit();
  net.points.set(task.start.points);
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
  weightSlider.disabled = weightNumber.disabled = selected.length === 0 || !task.tools.weights;
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
  if (!Number.isFinite(value) || !task.tools.weights) return;
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
// Loading a Task, changing the Layout
// ---------------------------------------------------------

// a new Learner surface, possibly with another Layout: everything that depends on the Layout is rebuilt
// (also needed inside a Level when knot insertion changes the Layout)
function setLearner(next: ControlNet) {
  net = next;
  learner?.destroy();
  learner = new SurfaceGpu(device, pipelines, cameraBuffer, learnerColorBuffer, net);
  flagData = new Uint32Array(learner.pointCount);

  walkerPointBuffer?.destroy();
  walkerFlagBuffer?.destroy();
  walkerPointBuffer = device.createBuffer({
    label: 'walker points',
    size: net.countV * 16,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  walkerFlagBuffer = device.createBuffer({
    label: 'walker flags',
    size: net.countV * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  walkerPointsBindGroup = device.createBindGroup({
    layout: pointsPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: cameraBuffer } },
      { binding: 1, resource: { buffer: walkerPointBuffer } },
      { binding: 2, resource: { buffer: walkerFlagBuffer } },
    ],
  });

  // the old undo steps and selection belong to the old Layout
  undoSteps.clear();
  undoButton.disabled = true;
  dragger.net = net;
  dragger.selected.clear();
  paramView.net = net;
  paramView.draw();
  partitionU.setBasis(net.countU, net.p, net.knotsU);
  partitionV.setBasis(net.countV, net.q, net.knotsV);
  updateMath();
  updateWeightPanel();
}

function loadTask(next: Task) {
  task = next;
  document.getElementById('taskTitle')!.textContent = task.title;
  document.getElementById('taskText')!.textContent = task.text;
  dragger.allowMove = task.tools.movePoints;

  target?.destroy();
  target = new SurfaceGpu(device, pipelines, cameraBuffer, targetColorBuffer, task.target,
    shiftedPoints(task.target, TARGET_OFFSET));
  // the target's control points (solution switch) are never selected: all flags stay 0
  setLearner(cloneNet(task.start));
}

// Level menu: a stand-in for the Board, which comes with the game screens
const taskSelect = document.getElementById('taskSelect') as HTMLSelectElement;
for (const t of tasks) taskSelect.add(new Option(t.title, t.id));
taskSelect.value = task.id;
taskSelect.addEventListener('change', () => {
  const next = tasks.find((t) => t.id === taskSelect.value)!;
  loadTask(next);
  const url = new URL(location.href);
  url.searchParams.set('task', next.id);
  history.replaceState(null, '', url); // a reload stays in the same Level
});

loadTask(task);

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

// views: perspective, or straight along an axis (top = xy-plane, front = xz-plane, side = yz-plane)
const viewButtons = document.querySelectorAll<HTMLButtonElement>('#view button');
function showView(view: ViewName) {
  viewButtons.forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.view === view)));
}
viewButtons.forEach((b) => b.addEventListener('click', () => {
  camera.setView(b.dataset.view as ViewName);
  showView(camera.view);
}));
camera.onViewChange = showView; // rotating a straight-on view makes it a perspective view again
setMode('rotate');

// the control polygon has an on/off switch (a checkbox) instead of a button
const polygonSwitch = document.getElementById('togglePolygon') as HTMLInputElement;
polygonSwitch.addEventListener('change', () => { showPolygon = polygonSwitch.checked; });
const solutionSwitch = document.getElementById('toggleSolution') as HTMLInputElement;
solutionSwitch.addEventListener('change', () => { showSolution = solutionSwitch.checked; });
const markerSwitch = document.getElementById('toggleMarker') as HTMLInputElement;
markerSwitch.addEventListener('change', () => {
  showMarker = markerSwitch.checked;
  uploadFlags(); // the cyan rings follow the switch
});
const walkersSwitch = document.getElementById('toggleWalkers') as HTMLInputElement;
walkersSwitch.addEventListener('change', () => {
  showWalkers = walkersSwitch.checked;
  updateMath();
});
// the two side panel sections can be collapsed; while a section is collapsed,
// its work (drawing, numbers, marker and active highlight) is skipped
paramPanel.addEventListener('toggle', () => {
  paramView.draw(); // the canvas had no size while collapsed
  updateMath();
});
partitionPanel.addEventListener('toggle', () => updateMath()); // the plots had no size while collapsed
mathPanel.addEventListener('toggle', () => {
  showMath = mathPanel.open;
  updateMath();
});
showMath = mathPanel.open;
updateMath();

// ---------------------------------------------------------
// Frame loop
// ---------------------------------------------------------

function frame() {
  const learnerGpu = learner!;
  const targetGpu = target!;
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

  // evaluate a surface again, but only if its control points changed
  if (learnerGpu.dirty || targetGpu.dirty) {
    const compute = encoder.beginComputePass();
    learnerGpu.tessellate(compute);
    targetGpu.tessellate(compute);
    compute.end();
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

  learnerGpu.drawSurface(pass);
  targetGpu.drawSurface(pass);

  if (showWalkers) {
    pass.setPipeline(curvesBehindPipeline);
    pass.setBindGroup(0, curvesBindGroups[1]);
    pass.draw(6, walkerSegmentCount);
    pass.setPipeline(curvesFrontPipeline);
    pass.setBindGroup(0, curvesBindGroups[0]);
    pass.draw(6, walkerSegmentCount);
  }

  learnerGpu.drawStems(pass);
  if (showPolygon) learnerGpu.drawPolygon(pass);

  if (showSolution) {
    if (showPolygon) targetGpu.drawPolygon(pass);
    targetGpu.drawPoints(pass);
  }

  learnerGpu.drawPoints(pass); // leaves the points pipeline set for the walkers and the marker

  if (showWalkers) {
    pass.setBindGroup(0, walkerPointsBindGroup!);
    pass.draw(6, net.countV);
  }

  if (showMarker) {
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
