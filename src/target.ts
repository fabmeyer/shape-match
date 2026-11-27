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

// the final Level: a ridge across the sheet and a valley along it; where they cross, the two cancel out,
// so the learner sees that the effects of control points add up (superposition). It comes last, after
// the library shapes: in playtests it was by far the hardest Level
const SUPERPOSITION_HEIGHT = 0.5;

export function superpositionLevel(): Task {
  const target = flatControlNet();
  const { countU, countV } = target;
  const lift = (i: number, j: number, dz: number) => {
    const k = j * countU + i;
    const [x, y, z] = getPoint(target, k);
    setPoint(target, k, [x, y, z + dz]);
  };
  for (let i = 1; i < countU - 1; i++) lift(i, 2, SUPERPOSITION_HEIGHT);  // the ridge: row j = 2
  for (let j = 1; j < countV - 1; j++) lift(3, j, -SUPERPOSITION_HEIGHT); // the valley: column i = 3
  return {
    id: 'superposition',
    title: 'Final · Superposition',
    text: 'A ridge runs across the sheet, a valley runs along it. Where they cross, the sheet stays flat: '
      + 'up and down add up to nothing. Shape both.',
    target,
    start: flatControlNet(),
    tools: { movePoints: true, weights: false, knotInsertion: false },
  };
}

// level 2: a closed cylinder, degree 2 x 1 with 9 x 2 points (the circle of shapes.ts, pushed up).
// Its points already sit on the square around the circle; it is only round with the weights
// 1 at the middle of each side and √2/2 at the corners. The learner can change weights only, starting at 1
// with all weights 1 the pipe is only 1.15 % (5.8 τ) off round, which the default rules would give 1★ for:
// stricter Stars here, so the start gets 0★ and passing needs a corner weight of about 0.63 ... 0.82
// (3★: about 0.68 ... 0.75, around √2/2 ≈ 0.707; measured on the 64 error grid)
const PIPE = { radius: 0.8, height: 1.2 };
const PIPE_GRADING = { star2Factor: 1.5, star1Factor: 3 };

export function weightsLevel(): Task {
  const target = revolve([{ r: PIPE.radius, z: 0 }, { r: PIPE.radius, z: PIPE.height }], 1);
  const start = cloneNet(target);
  for (let k = 0; k < start.countU * start.countV; k++) setWeight(start, k, 1);
  return {
    id: 'weights',
    title: 'Level 2 · Weights',
    text: 'A pipe that must be perfectly round. The points are in the right place already: '
      + 'only the weights are wrong. Select points and change their weight until the pipe is round.',
    target,
    start,
    tools: { movePoints: false, weights: true, knotInsertion: false },
    grading: PIPE_GRADING,
  };
}

// how a library shape's starting surface differs from its Target: always flattened, and from the Tent on
// it gets harder step by step, so the learner must fix more than one direction:
//   difficulty 2-4  (Tent, Star):                flattened, and stretched wider in x
//   difficulty 5-8  (House):                     ... and narrower in y
//   difficulty 9-11 (Boat, Car):                 ... and leaning backwards (along y, away from the Target)
// a revolution (Ball, Vase, Tree, Cup, Mushroom) is flattened and leans instead, from the Ball on:
// stretching in x or y would turn every ring into an ellipse of its own size, and the arrows can only move
// points along one axis, so each point of each ring would need its own exact drag (the Vase had 54).
// A lean moves every ring as a whole and keeps it round: one rectangle selection and one drag per ring
// all of these are affine, so the start keeps the Target's Layout and weights (ADR 0001)
export interface StartChange {
  squash: number;  // z is multiplied by this (z = 0 stays)
  stretchX: number; // x around the shape's middle
  stretchY: number; // y around the shape's middle
  lean: number;    // y moves by lean · z (z of the Target): the shape tips backwards
}

export function startChange(difficulty: number, revolution = false): StartChange {
  if (revolution) return { squash: 0.25, stretchX: 1, stretchY: 1, lean: difficulty >= 4 ? 0.35 : 0 };
  return {
    squash: 0.25,
    stretchX: difficulty >= 2 ? 1.3 : 1,
    stretchY: difficulty >= 5 ? 0.75 : 1,
    lean: difficulty >= 9 ? 0.35 : 0,
  };
}

export function changedStart(net: ControlNet, c: StartChange): ControlNet {
  const out = cloneNet(net);
  const count = out.countU * out.countV;
  const lo = [Infinity, Infinity], hi = [-Infinity, -Infinity];
  for (let k = 0; k < count; k++) {
    const [x, y] = getPoint(out, k);
    lo[0] = Math.min(lo[0], x); hi[0] = Math.max(hi[0], x);
    lo[1] = Math.min(lo[1], y); hi[1] = Math.max(hi[1], y);
  }
  const cx = (lo[0] + hi[0]) / 2, cy = (lo[1] + hi[1]) / 2;
  for (let k = 0; k < count; k++) {
    const [x, y, z] = getPoint(out, k);
    setPoint(out, k, [cx + c.stretchX * (x - cx), cy + c.stretchY * (y - cy) + c.lean * z, c.squash * z]);
  }
  return out;
}

const capitalized = (s: string) => s[0].toUpperCase() + s.slice(1);

// a shape of the library as a Task; its id stays lower case (URL, saved progress), its title is capitalized
export function shapeTask(shape: Shape): Task {
  return {
    id: shape.id,
    title: `Shape ${shape.difficulty} · ${capitalized(shape.id)}`,
    text: `${shape.order}. ${capitalized(shape.concept)}.`,
    target: shape.net,
    start: shape.start ?? changedStart(shape.net, startChange(shape.difficulty, shape.construction.startsWith('revolution'))),
    tools: ALL_TOOLS,
  };
}

// the Board's order: two Levels that teach one idea each, the library shapes from easy to hard, and
// superposition last (the hardest)
export function allTasks(): Task[] {
  return [level1(), weightsLevel(), ...allShapes().map(shapeTask), superpositionLevel()];
}

// copy of the control points, shifted along x
// moving all control points by the same vector moves the whole surface by it (affine invariance)
export function shiftedPoints(net: ControlNet, dx: number): Float32Array {
  const out = net.points.slice();
  for (let k = 0; k < out.length; k += 4) out[k] += dx * out[k + 3]; // stored as w * x
  return out;
}
