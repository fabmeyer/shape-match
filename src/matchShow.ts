// the Hand-in animation: before the Result, the learner sees how the Shape match check works
//   1. slide:  the Learner surface and the Target move towards each other and meet in the middle
//              (the Target turns see-through, so both stay visible)
//   2. align:  the Learner surface turns and moves into its best fit on the Target (alignment.ts, align.wgsl)
//   3. hold:   a short pause on the result, then the Result screen
// only the drawing moves (a model matrix per surface in surface.wgsl); the control points stay as they are
import { mat4, quat, type Mat4 } from 'wgpu-matrix';
import type { BestFit } from './alignment.ts';

export const SLIDE_MS = 1000;
export const ALIGN_MS = 1400;
export const HOLD_MS = 700;

// slow, fast, slow (cubic ease in and out), t in [0, 1]
export function ease(t: number): number {
  const x = Math.min(1, Math.max(0, t));
  return x < 0.5 ? 4 * x * x * x : 1 - (-2 * x + 2) ** 3 / 2;
}

// where both surfaces are drawn: slide in [0, 1] brings them together (the Target is drawn `offset` to the right,
// so each moves half of it); align in [0, 1] turns the learner around its centre into the best fit
export function showModels(slide: number, align: number, fit: BestFit | undefined, offset: number): { learner: Mat4; target: Mat4 } {
  const half = (offset / 2) * slide;
  const target = mat4.translation([-half, 0, 0]);
  let learner = mat4.translation([half, 0, 0]);
  if (fit && align > 0) {
    const [cx, cy, cz] = fit.learnerCentre;
    const [tx, ty, tz] = fit.targetCentre;
    const turn = quat.slerp(quat.identity(), fit.rotation, align);
    // move to the origin, turn part of the way, then go part of the way from the learner's centre to the Target's
    learner = mat4.multiply(learner, mat4.translation([cx + align * (tx - cx), cy + align * (ty - cy), cz + align * (tz - cz)]));
    learner = mat4.multiply(learner, mat4.fromQuat(turn));
    learner = mat4.multiply(learner, mat4.translation([-cx, -cy, -cz]));
  }
  return { learner, target };
}
