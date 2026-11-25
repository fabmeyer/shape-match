// undo: every step is a copy of the control points before a change (the net is tiny)
const MAX_STEPS = 100;

export class History {
  private steps: Float32Array[] = [];

  get size(): number {
    return this.steps.length;
  }

  // call before changing the points
  push(points: Float32Array): void {
    this.steps.push(points.slice());
    if (this.steps.length > MAX_STEPS) this.steps.shift();
  }

  // a new Level or Layout: the old steps don't fit any more
  clear(): void {
    this.steps = [];
  }

  // the points before the last change, or undefined if there is nothing to undo
  pop(): Float32Array | undefined {
    return this.steps.pop();
  }
}
