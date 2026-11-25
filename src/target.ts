// Tasks: what a Level asks for. Every Task brings its own Layout (ADR 0001): its Target and its
// starting Learner surface share that Layout, but different Tasks can have different Layouts
import { flatControlNet, getPoint, setPoint, type ControlNet } from './controlNet.ts';
import { allShapes, type Shape } from './shapes.ts';
import type { GradingConfig } from './grading.ts';

// the target is drawn this far to the right of the learner's surface (both are about 2 wide)
export const TARGET_OFFSET = 2.8;

// what the learner may change in a Level
export interface Tools {
  movePoints: boolean;
  weights: boolean;
  knotInsertion: boolean; // not built yet (issue #19)
}

export interface Task {
  id: string;           // stable name, also used in the URL (?task=ball)
  title: string;
  text: string;         // the Order: what the Customer wants and what to do
  target: ControlNet;   // in the same coordinates as the learner's net (not shifted)
  start: ControlNet;    // starting Learner surface, same Layout as the target
  tools: Tools;
  grading?: Partial<GradingConfig>; // changes to the grading factors for this Level
}

const ALL_TOOLS: Tools = { movePoints: true, weights: true, knotInsertion: false };

export function cloneNet(net: ControlNet): ControlNet {
  return { ...net, knotsU: [...net.knotsU], knotsV: [...net.knotsV], points: net.points.slice() };
}

// same number of control points, degrees and knots
export function sameLayout(a: ControlNet, b: ControlNet): boolean {
  return a.countU === b.countU && a.countV === b.countV && a.p === b.p && a.q === b.q
    && a.knotsU.join() === b.knotsU.join() && a.knotsV.join() === b.knotsV.join();
}

// level 1: one raised control point, to learn local support
export function level1(): Task {
  const target = flatControlNet();
  const k = 3 * target.countU + 2; // P(2, 3)
  const [x, y] = getPoint(target, k);
  setPoint(target, k, [x, y, 1.2]);
  return {
    id: 'level1',
    title: 'Level 1 · Local support',
    text: 'Move one control point so that your surface (blue) matches the target (orange). Which point, and how far?',
    target,
    start: flatControlNet(),
    tools: ALL_TOOLS,
  };
}

// the Target with its height squashed: same Layout and weights, so it fits ADR 0001
// a placeholder until each shape gets a designed starting surface (issue #22)
function squashed(net: ControlNet, factor = 0.25): ControlNet {
  const out = cloneNet(net);
  for (let k = 0; k < out.countU * out.countV; k++) {
    const [x, y, z] = getPoint(out, k);
    setPoint(out, k, [x, y, z * factor]);
  }
  return out;
}

// a shape of the library as a Task, for trying out Layouts until the Board exists
export function shapeTask(shape: Shape): Task {
  return {
    id: shape.id,
    title: `Shape ${shape.difficulty} · ${shape.id}`,
    text: `${shape.order}. ${shape.concept[0].toUpperCase()}${shape.concept.slice(1)}.`,
    target: shape.net,
    start: squashed(shape.net),
    tools: ALL_TOOLS,
  };
}

export function allTasks(): Task[] {
  return [level1(), ...allShapes().map(shapeTask)];
}

// copy of the control points, shifted along x
// moving all control points by the same vector moves the whole surface by it (affine invariance)
export function shiftedPoints(net: ControlNet, dx: number): Float32Array {
  const out = net.points.slice();
  for (let k = 0; k < out.length; k += 4) out[k] += dx * out[k + 3]; // stored as w * x
  return out;
}
