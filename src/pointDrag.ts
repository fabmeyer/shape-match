import { mat4, vec3, type Mat4 } from 'wgpu-matrix';
import type { ArcballCamera } from './arcball.ts';
import { getPoint, setPoint, type ControlNet, type Point3 } from './controlNet.ts';

const PICK_RADIUS_PX = 12;   // how close the mouse must be to a point to pick it
const ARROW_RADIUS_PX = 9;   // how close the mouse must be to an arrow to grab it
const ARROW_LENGTH = 0.16;   // arrow length as a fraction of the distance to the camera (constant size on screen)
const LIMIT = 3;             // points stay inside [-LIMIT, LIMIT] in x, y and z
const AXES: Point3[] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];

export interface Gizmo {
  origin: Point3;  // centre of the selection
  length: number;  // arrow length in world units
  axis: number;    // arrow that is dragged or under the mouse (0 = x, 1 = y, 2 = z), -1 = none
}

export interface DragCallbacks {
  onSelect(): void;     // the selection changed
  onBeginEdit(): void;  // points are about to move (take the undo snapshot here)
  onEdit(): void;       // points moved
}

/**
 * selecting control points (click, shift+click, rectangle) and moving the selection
 * along one axis by dragging one of three arrows (x, y, z)
 * picking happens on the CPU: points and arrows are projected to the screen
 */
export class PointDragger {
  enabled = false;
  selected = new Set<number>(); // indices j * countU + i
  allowMove = true;             // false: points can be selected (for weights) but not moved
  net: ControlNet;              // replaced when a Level loads or the Layout changes

  private canvas: HTMLCanvasElement;
  private camera: ArcballCamera;
  private callbacks: DragCallbacks;

  private hoverAxis = -1;
  // drag of an arrow: s0 = position of the mouse along the axis at the start
  // pointUnder = control point under the mouse at the start (an arrow can lie over other points)
  private axisDrag: { axis: number; origin: Point3; s0: number; start: Map<number, Point3>; moved: boolean; pointUnder: number } | null = null;
  // drag of a selection rectangle, in client pixels
  private box: { x0: number; y0: number; x1: number; y1: number; add: boolean } | null = null;
  private boxElement: HTMLDivElement;

  constructor(canvas: HTMLCanvasElement, camera: ArcballCamera, net: ControlNet, callbacks: DragCallbacks) {
    this.canvas = canvas;
    this.camera = camera;
    this.net = net;
    this.callbacks = callbacks;

    this.boxElement = document.createElement('div');
    this.boxElement.style.cssText = 'position:fixed;border:1px dashed #fff;background:rgba(255,255,255,0.08);pointer-events:none;display:none';
    document.body.appendChild(this.boxElement);

    canvas.addEventListener('pointerdown', (e) => this.down(e));
    canvas.addEventListener('pointermove', (e) => this.move(e));
    canvas.addEventListener('pointerup', () => this.up());
    canvas.addEventListener('pointercancel', () => this.up());
  }

  // the three arrows, or null if there is nothing to show
  gizmo(): Gizmo | null {
    if (!this.enabled || !this.allowMove || this.selected.size === 0) return null;
    const origin = this.centre();
    return { origin, length: this.arrowLength(origin), axis: this.axisDrag ? this.axisDrag.axis : this.hoverAxis };
  }

  clear(): void {
    if (this.selected.size === 0) return;
    this.selected.clear();
    this.callbacks.onSelect();
  }

  // ---------------------------------------------------------
  // Pointer events
  // ---------------------------------------------------------

  private down(e: PointerEvent) {
    if (!this.enabled || e.button !== 0) return;
    this.canvas.setPointerCapture(e.pointerId);

    const k = this.pickPoint(e);

    // 1. an arrow of the current selection (with Shift the click is meant for a point)
    const axis = e.shiftKey && k >= 0 ? -1 : this.pickAxis(e);
    if (axis >= 0) {
      const origin = this.centre();
      const s0 = this.alongAxis(e, origin, axis);
      if (s0 !== null) {
        const start = new Map<number, Point3>();
        for (const k of this.selected) start.set(k, getPoint(this.net, k));
        this.axisDrag = { axis, origin, s0, start, moved: false, pointUnder: k };
      }
      return;
    }

    // 2. a control point
    if (k >= 0) {
      if (e.shiftKey) {
        if (!this.selected.delete(k)) this.selected.add(k); // toggle
      } else {
        this.selected.clear();
        this.selected.add(k);
      }
      this.callbacks.onSelect();
      return;
    }

    // 3. empty space: start a selection rectangle
    this.box = { x0: e.clientX, y0: e.clientY, x1: e.clientX, y1: e.clientY, add: e.shiftKey };
  }

  private move(e: PointerEvent) {
    if (!this.enabled) return;

    if (this.axisDrag) {
      const drag = this.axisDrag;
      const s = this.alongAxis(e, drag.origin, drag.axis);
      if (s === null) return; // arrow points at the camera: position along it is undefined
      // keep every selected point inside the limits, without changing the shape of the group
      let delta = s - drag.s0;
      for (const p of drag.start.values()) {
        delta = Math.min(LIMIT - p[drag.axis], Math.max(-LIMIT - p[drag.axis], delta));
      }
      if (!drag.moved) {
        if (delta === 0) return;
        drag.moved = true;
        this.callbacks.onBeginEdit();
      }
      for (const [k, p] of drag.start) {
        const moved: Point3 = [p[0], p[1], p[2]];
        moved[drag.axis] += delta;
        setPoint(this.net, k, moved);
      }
      this.callbacks.onEdit();
      return;
    }

    if (this.box) {
      this.box.x1 = e.clientX;
      this.box.y1 = e.clientY;
      const b = this.box;
      const s = this.boxElement.style;
      s.display = 'block';
      s.left = `${Math.min(b.x0, b.x1)}px`;
      s.top = `${Math.min(b.y0, b.y1)}px`;
      s.width = `${Math.abs(b.x1 - b.x0)}px`;
      s.height = `${Math.abs(b.y1 - b.y0)}px`;
      return;
    }

    this.hoverAxis = this.pickAxis(e); // highlight the arrow under the mouse
  }

  private up() {
    if (this.axisDrag) {
      // a click on an arrow without moving it, with a control point underneath: select that point
      const { moved, pointUnder } = this.axisDrag;
      this.axisDrag = null;
      if (!moved && pointUnder >= 0) {
        this.selected.clear();
        this.selected.add(pointUnder);
        this.callbacks.onSelect();
      }
      return;
    }
    if (!this.box) return;

    const b = this.box;
    this.box = null;
    this.boxElement.style.display = 'none';
    if (!b.add) this.selected.clear();
    const isClick = Math.abs(b.x1 - b.x0) < 3 && Math.abs(b.y1 - b.y0) < 3; // click on empty space: only clears
    if (!isClick) {
      const r = this.canvas.getBoundingClientRect();
      const count = this.net.countU * this.net.countV;
      for (let k = 0; k < count; k++) {
        const s = this.toScreen(getPoint(this.net, k));
        if (!s) continue;
        const x = s[0] + r.left;
        const y = s[1] + r.top;
        if (x >= Math.min(b.x0, b.x1) && x <= Math.max(b.x0, b.x1) && y >= Math.min(b.y0, b.y1) && y <= Math.max(b.y0, b.y1)) {
          this.selected.add(k);
        }
      }
    }
    this.callbacks.onSelect();
  }

  // ---------------------------------------------------------
  // Picking
  // ---------------------------------------------------------

  private viewProjection(): Mat4 {
    return this.camera.viewProjection(this.canvas.width / this.canvas.height);
  }

  // position on the canvas in CSS pixels and depth (0 = near), or null if the point is behind the camera
  private toScreen(p: Point3): [number, number, number] | null {
    const m = this.viewProjection();
    const cx = m[0] * p[0] + m[4] * p[1] + m[8] * p[2] + m[12];
    const cy = m[1] * p[0] + m[5] * p[1] + m[9] * p[2] + m[13];
    const cz = m[2] * p[0] + m[6] * p[1] + m[10] * p[2] + m[14];
    const cw = m[3] * p[0] + m[7] * p[1] + m[11] * p[2] + m[15];
    if (cw <= 0) return null;
    const r = this.canvas.getBoundingClientRect();
    return [(cx / cw * 0.5 + 0.5) * r.width, (0.5 - cy / cw * 0.5) * r.height, cz / cw];
  }

  private mouse(e: PointerEvent): [number, number] {
    const r = this.canvas.getBoundingClientRect();
    return [e.clientX - r.left, e.clientY - r.top];
  }

  // index of the control point nearest to the mouse, or -1
  private pickPoint(e: PointerEvent): number {
    const [mx, my] = this.mouse(e);
    const count = this.net.countU * this.net.countV;
    let best = -1;
    let bestDist = PICK_RADIUS_PX;
    let bestDepth = Infinity;
    for (let k = 0; k < count; k++) {
      const s = this.toScreen(getPoint(this.net, k));
      if (!s) continue;
      const d = Math.hypot(s[0] - mx, s[1] - my);
      // closer to the mouse wins; points (almost) on top of each other on screen: the front one wins
      const tie = Math.abs(d - bestDist) < 1;
      if ((d < bestDist && !tie) || (tie && d < PICK_RADIUS_PX && s[2] < bestDepth)) {
        bestDist = d;
        bestDepth = s[2];
        best = k;
      }
    }
    return best;
  }

  // arrow under the mouse (0 = x, 1 = y, 2 = z), or -1
  private pickAxis(e: PointerEvent): number {
    if (!this.allowMove || this.selected.size === 0) return -1;
    const o = this.centre();
    const length = this.arrowLength(o);
    const [mx, my] = this.mouse(e);

    let best = -1;
    let bestDist = ARROW_RADIUS_PX;
    for (let axis = 0; axis < 3; axis++) {
      const a = AXES[axis];
      // the first quarter is left out: there the three arrows meet and the control point sits
      const from = this.toScreen([o[0] + a[0] * length * 0.25, o[1] + a[1] * length * 0.25, o[2] + a[2] * length * 0.25]);
      const to = this.toScreen([o[0] + a[0] * length, o[1] + a[1] * length, o[2] + a[2] * length]);
      if (!from || !to) continue;
      // distance from the mouse to the segment from-to
      const dx = to[0] - from[0];
      const dy = to[1] - from[1];
      const len2 = dx * dx + dy * dy;
      const t = len2 > 0 ? Math.min(1, Math.max(0, ((mx - from[0]) * dx + (my - from[1]) * dy) / len2)) : 0;
      const d = Math.hypot(from[0] + t * dx - mx, from[1] + t * dy - my);
      if (d < bestDist) {
        bestDist = d;
        best = axis;
      }
    }
    return best;
  }

  // ---------------------------------------------------------
  // Geometry
  // ---------------------------------------------------------

  // centre of the selected points
  private centre(): Point3 {
    const c: Point3 = [0, 0, 0];
    for (const k of this.selected) {
      const p = getPoint(this.net, k);
      c[0] += p[0];
      c[1] += p[1];
      c[2] += p[2];
    }
    const n = Math.max(1, this.selected.size);
    return [c[0] / n, c[1] / n, c[2] / n];
  }

  // the arrows get longer with the distance to the camera, so they keep their size on screen
  private arrowLength(origin: Point3): number {
    const eye = this.camera.eye;
    return ARROW_LENGTH * Math.hypot(eye[0] - origin[0], eye[1] - origin[1], eye[2] - origin[2]);
  }

  // parameter s of the point on the line origin + s * axis that is closest to the ray under the mouse
  // null if the line and the ray are nearly parallel
  private alongAxis(e: PointerEvent, origin: Point3, axis: number): number | null {
    const r = this.canvas.getBoundingClientRect();
    const nx = ((e.clientX - r.left) / r.width) * 2 - 1;
    const ny = 1 - ((e.clientY - r.top) / r.height) * 2;
    const inv = mat4.inverse(this.viewProjection());
    const near = vec3.transformMat4([nx, ny, 0], inv); // depth 0 = near plane in WebGPU
    const far = vec3.transformMat4([nx, ny, 1], inv);
    const dir = vec3.normalize(vec3.subtract(far, near));

    const a = AXES[axis];
    const b = vec3.dot(a, dir);
    const denom = 1 - b * b; // both directions have length 1
    if (denom < 1e-3) return null;
    const w = vec3.subtract(origin, near);
    return (b * vec3.dot(dir, w) - vec3.dot(a, w)) / denom;
  }
}
