// the target surface the learner tries to match
// all targets use the learner's net layout (6 x 6 points, same degrees and knots),
// so the GPU can share the parameter and knot buffers, and the solution is known
import { flatControlNet, getPoint, setPoint, type ControlNet } from './controlNet.ts';

// the target is drawn this far to the right of the learner's surface (both are 2 wide)
export const TARGET_OFFSET = 2.8;

export interface Task {
  title: string;
  text: string;
  target: ControlNet; // in the same coordinates as the learner's net (not shifted)
}

// level 1: one raised control point, to learn local support
export function level1(): Task {
  const target = flatControlNet();
  const k = 3 * target.countU + 2; // P(2, 3)
  const [x, y] = getPoint(target, k);
  setPoint(target, k, [x, y, 1.2]);
  return {
    title: 'Level 1 · Local support',
    text: 'Move one control point so that your surface (blue) matches the target (orange). Which point, and how far?',
    target,
  };
}

// copy of the control points, shifted along x
// moving all control points by the same vector moves the whole surface by it (affine invariance)
export function shiftedPoints(net: ControlNet, dx: number): Float32Array {
  const out = net.points.slice();
  for (let k = 0; k < out.length; k += 4) out[k] += dx * out[k + 3]; // stored as w * x
  return out;
}
