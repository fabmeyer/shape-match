// Tasks: what a Level asks for. Every Task brings its own Layout (ADR 0001): its Target and its
// starting Learner surface share that Layout, but different Tasks can have different Layouts
import { flatControlNet, getPoint, setPoint, setWeight, type ControlNet } from './controlNet.ts';
import { allShapes, revolve, type Shape } from './shapes.ts';
import type { GradingConfig } from './grading.ts';

// the target is drawn this far to the right of the learner's surface (both are about 2 wide)
export const TARGET_OFFSET = 2.8;

// in words, for the Task panel
export function toolsText(t: Tools): string {
  const can = [t.movePoints && 'move control points', t.weights && 'change weights', t.knotInsertion && 'insert knots'].filter(Boolean);
  return can.length ? `In this Level you can ${can.join(', ')}.` : 'In this Level you can only look.';
}

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

// level 2: a ridge across the sheet and a valley along it; where they cross, the two cancel out,
// so the learner sees that the effects of control points add up (superposition)
const LEVEL2_HEIGHT = 0.5;

export function level2(): Task {
  const target = flatControlNet();
  const { countU, countV } = target;
  const lift = (i: number, j: number, dz: number) => {
    const k = j * countU + i;
    const [x, y, z] = getPoint(target, k);
    setPoint(target, k, [x, y, z + dz]);
  };
  for (let i = 1; i < countU - 1; i++) lift(i, 2, LEVEL2_HEIGHT);  // the ridge: row j = 2
  for (let j = 1; j < countV - 1; j++) lift(3, j, -LEVEL2_HEIGHT); // the valley: column i = 3
  return {
    id: 'level2',
    title: 'Level 2 · Superposition',
    text: 'A ridge runs across the sheet, a valley runs along it. Where they cross, the sheet stays flat: '
      + 'up and down add up to nothing. Shape both.',
    target,
    start: flatControlNet(),
    tools: { movePoints: true, weights: false, knotInsertion: false },
  };
}

// level 3: a closed cylinder, degree 2 x 1 with 9 x 2 points (the circle of shapes.ts, pushed up).
// Its points already sit on the square around the circle; it is only round with the weights
// 1 at the middle of each side and √2/2 at the corners. The learner can change weights only, starting at 1
// with all weights 1 the pipe is only 1.15 % (5.8 τ) off round, which the default rules would give 1★ for:
// stricter Stars here, so the start gets 0★ and passing needs a corner weight of about 0.63 ... 0.82
// (3★: about 0.68 ... 0.75, around √2/2 ≈ 0.707; measured on the 64 error grid)
const PIPE = { radius: 0.8, height: 1.2 };
const PIPE_GRADING = { star2Factor: 1.5, star1Factor: 3 };

export function level3(): Task {
  const target = revolve([{ r: PIPE.radius, z: 0 }, { r: PIPE.radius, z: PIPE.height }], 1);
  const start = cloneNet(target);
  for (let k = 0; k < start.countU * start.countV; k++) setWeight(start, k, 1);
  return {
    id: 'level3',
    title: 'Level 3 · Weights',
    text: 'A pipe that must be perfectly round. The points are in the right place already: '
      + 'only the weights are wrong. Select points and change their weight until the pipe is round.',
    target,
    start,
    tools: { movePoints: false, weights: true, knotInsertion: false },
    grading: PIPE_GRADING,
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
    start: shape.start ?? squashed(shape.net),
    tools: ALL_TOOLS,
  };
}

export function allTasks(): Task[] {
  return [level1(), level2(), level3(), ...allShapes().map(shapeTask)];
}

// copy of the control points, shifted along x
// moving all control points by the same vector moves the whole surface by it (affine invariance)
export function shiftedPoints(net: ControlNet, dx: number): Float32Array {
  const out = net.points.slice();
  for (let k = 0; k < out.length; k += 4) out[k] += dx * out[k + 3]; // stored as w * x
  return out;
}
