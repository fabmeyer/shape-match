import { mat4, quat, vec3, type Mat4, type Quat, type Vec3 } from 'wgpu-matrix';

// perspective: free arcball view; the others look straight along an axis, without perspective
export type ViewName = 'perspective' | 'top' | 'front' | 'side';

// camera orientations of the straight-on views (camera space looks down -z with +y up)
const STRAIGHT_VIEWS: Record<Exclude<ViewName, 'perspective'>, () => Quat> = {
  top: () => quat.identity(),                                 // looks down -z: x right, y up
  front: () => quat.fromAxisAngle([1, 0, 0], Math.PI / 2),    // looks along +y: x right, z up
  side: () => quat.multiply(                                  // looks along -x: y right, z up
    quat.fromAxisAngle([0, 0, 1], Math.PI / 2), quat.fromAxisAngle([1, 0, 0], Math.PI / 2)),
};

export class ArcballCamera {
  target: Vec3;
  distance: number;
  orientation: Quat;
  fovY = Math.PI / 4;
  enabled = true;
  // what dragging does: turn the view around the target, or move it (target and eye together) in the view plane
  drag: 'rotate' | 'move' = 'rotate';
  view: ViewName = 'perspective';
  onViewChange: (view: ViewName) => void = () => {}; // e.g. rotating leaves a straight-on view

  private dragging = false;
  private last: Vec3 = vec3.create();
  private lastPixel = { x: 0, y: 0 };
  private perspectiveOrientation: Quat = quat.identity(); // to come back to the last free view
  private home: { target: Vec3; orientation: Quat }; // the starting view, for reset()

  constructor(canvas: HTMLCanvasElement, opts: { target?: number[]; distance?: number; orientation?: Quat } = {}) {
    this.target = vec3.fromValues(...((opts.target ?? [0, 0, 0]) as [number, number, number]));
    this.distance = opts.distance ?? 5;
    this.orientation = opts.orientation ?? quat.identity();
    this.home = { target: vec3.clone(this.target), orientation: quat.clone(this.orientation) };

    canvas.addEventListener('pointerdown', (e) => {
      if (!this.enabled) return;
      canvas.setPointerCapture(e.pointerId);
      this.dragging = true;
      this.last = this.toBall(canvas, e);
      this.lastPixel = { x: e.clientX, y: e.clientY };
    });
    canvas.addEventListener('pointermove', (e) => {
      if (!this.dragging) return;
      if (this.drag === 'move') {
        this.move(canvas, e.clientX - this.lastPixel.x, e.clientY - this.lastPixel.y);
        this.lastPixel = { x: e.clientX, y: e.clientY };
        return; // moving keeps a straight-on view straight
      }
      // rotating a straight-on view turns it into a free (perspective) view, starting from where it is
      if (this.view !== 'perspective') {
        this.view = 'perspective';
        this.onViewChange(this.view);
      }
      const next = this.toBall(canvas, e);
      this.rotate(this.last, next);
      this.last = next;
    });
    const stop = () => { this.dragging = false; };
    canvas.addEventListener('pointerup', stop);
    canvas.addEventListener('pointercancel', stop);
    canvas.addEventListener('wheel', (e) => {
      e.preventDefault();
      this.distance = Math.min(50, Math.max(0.5, this.distance * Math.exp(e.deltaY * 0.001)));
    }, { passive: false });
  }

  private toBall(canvas: HTMLCanvasElement, e: PointerEvent): Vec3 {
    const r = canvas.getBoundingClientRect();
    const s = Math.min(r.width, r.height) / 2;
    const x = (e.clientX - r.left - r.width / 2) / s;
    const y = (r.top + r.height / 2 - e.clientY) / s;
    const d2 = x * x + y * y;
    const z = d2 <= 0.5 ? Math.sqrt(1 - d2) : 0.5 / Math.sqrt(d2);
    return vec3.normalize(vec3.fromValues(x, y, z));
  }

  private rotate(a: Vec3, b: Vec3) {
    const axis = vec3.cross(a, b);
    if (vec3.length(axis) < 1e-6) return;
    const angle = Math.acos(Math.min(1, Math.max(-1, vec3.dot(a, b))));
    const delta = quat.fromAxisAngle(vec3.normalize(axis), -angle);
    this.orientation = quat.normalize(quat.multiply(this.orientation, delta));
  }

  // moves the view by a mouse movement in CSS pixels: the surfaces follow the mouse
  // one pixel is the same distance at the target in the perspective and the straight-on views
  private move(canvas: HTMLCanvasElement, dx: number, dy: number) {
    const height = canvas.getBoundingClientRect().height || 1;
    const perPixel = (2 * this.distance * Math.tan(this.fovY / 2)) / height;
    const right = vec3.transformQuat([1, 0, 0], this.orientation);
    const up = vec3.transformQuat([0, 1, 0], this.orientation);
    this.target = vec3.add(this.target, vec3.add(vec3.scale(right, -dx * perPixel), vec3.scale(up, dy * perPixel)));
  }

  setView(view: ViewName): void {
    if (view === this.view) return;
    if (this.view === 'perspective') this.perspectiveOrientation = this.orientation;
    this.orientation = view === 'perspective' ? this.perspectiveOrientation : STRAIGHT_VIEWS[view]();
    this.view = view;
  }

  // back to the starting perspective view: its angle and target, at the given distance
  // (the caller picks the distance, so both surfaces fit the window as it is now)
  reset(distance: number): void {
    this.target = vec3.clone(this.home.target);
    this.orientation = quat.clone(this.home.orientation);
    this.perspectiveOrientation = quat.clone(this.home.orientation);
    this.distance = distance;
    if (this.view !== 'perspective') {
      this.view = 'perspective';
      this.onViewChange(this.view);
    }
  }

  get eye(): Vec3 {
    const back = vec3.transformQuat([0, 0, this.distance], this.orientation);
    return vec3.add(this.target, back);
  }

  viewProjection(aspect: number): Mat4 {
    const up = vec3.transformQuat([0, 1, 0], this.orientation);
    const view = mat4.lookAt(this.eye, this.target, up);
    let proj: Mat4;
    if (this.view === 'perspective') {
      proj = mat4.perspective(this.fovY, aspect, 0.05, 100);
    } else {
      // same visible height at the target as the perspective view, so switching keeps the zoom;
      // near < 0: nothing is cut off when zoomed in close
      const h = this.distance * Math.tan(this.fovY / 2);
      proj = mat4.ortho(-h * aspect, h * aspect, -h, h, -50, 100);
    }
    return mat4.multiply(proj, view);
  }
}
