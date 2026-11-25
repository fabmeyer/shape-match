// a library of Targets for the Workshop: every shape is ONE NURBS surface with a known control net
// (multi-patch models are out of scope), built with three constructions:
//   revolution: a profile curve turned once around the z axis (exact circles through weights)
//   extrusion:  a closed profile pushed along y, both ends closed by collapsing a row into one point
//   loft:       closed cross sections along y, from section to section
// coordinates: z is up, the shapes stand on z = 0 and are about 2 wide, like the learner's start plane
import { clampedUniformKnots, type ControlNet } from './controlNet.ts';

export interface Shape {
  id: string;
  order: string;        // what the Customer asks for
  construction: string;
  concept: string;      // what the learner learns by matching it
  difficulty: number;   // 1 (easiest) ... 11, the suggested place in the sequence
  net: ControlNet;
  start?: ControlNet;   // the starting Learner surface, if the shape has a designed one (same Layout)
}

// a control point before it goes into the net: Cartesian position and weight
interface WPoint { x: number; y: number; z: number; w: number }

// builds the net from rows of points: rows[j][i] is control point (i, j), u runs along a row
function makeNet(rows: WPoint[][], p: number, q: number, knotsU: number[], knotsV: number[]): ControlNet {
  const countV = rows.length;
  const countU = rows[0].length;
  const points = new Float32Array(countU * countV * 4);
  rows.forEach((row, j) => row.forEach(({ x, y, z, w }, i) => {
    points.set([w * x, w * y, w * z, w], (j * countU + i) * 4);
  }));
  return { countU, countV, p, q, knotsU, knotsV, points };
}

// knots of a clamped curve whose interior knots are each repeated `mult` times:
// with mult = degree the curve has a corner (only C0) at every interior knot
export function repeatedKnots(spans: number, degree: number, mult: number): number[] {
  const knots = new Array<number>(degree + 1).fill(0);
  for (let s = 1; s < spans; s++) for (let m = 0; m < mult; m++) knots.push(s / spans);
  for (let m = 0; m <= degree; m++) knots.push(1);
  return knots;
}

// ---------------------------------------------------------
// The exact circle: degree 2, 9 points, weights 1 and √2/2 (The NURBS Book, Ex. 7.2)
// ---------------------------------------------------------

const S = Math.SQRT1_2;
const CIRCLE = [[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1], [1, 0]];
const CIRCLE_W = [1, S, 1, S, 1, S, 1, S, 1];
const CIRCLE_KNOTS = [0, 0, 0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1, 1, 1];

// ---------------------------------------------------------
// Constructions
// ---------------------------------------------------------

// profile point in the (r, z) half-plane, with its own weight
export interface ProfilePoint { r: number; z: number; w?: number }

// turns the profile once around the z axis: u goes around (the circle), v along the profile
// a profile that starts and ends on the axis (r = 0) gives a closed surface
export function revolve(profile: ProfilePoint[], q: number, knotsV = clampedUniformKnots(profile.length, q)): ControlNet {
  const rows = profile.map(({ r, z, w = 1 }) => CIRCLE.map(([cx, cy], i) => ({
    x: r * cx, y: r * cy, z, w: w * CIRCLE_W[i],
  })));
  return makeNet(rows, 2, q, CIRCLE_KNOTS, knotsV);
}

// pushes a closed profile in the xz-plane (first point = last point) along y from -depth/2 to +depth/2
// rows: back cap (all points at the centre), back profile, front profile, front cap. v has degree 1
function extrude(profile: [number, number][], p: number, knotsU: number[], depth: number): ControlNet {
  const n = profile.length - 1; // the last point repeats the first
  const cx = profile.slice(0, n).reduce((s, [x]) => s + x, 0) / n;
  const cz = profile.slice(0, n).reduce((s, [, z]) => s + z, 0) / n;
  const row = (y: number, cap: boolean): WPoint[] =>
    profile.map(([x, z]) => (cap ? { x: cx, y, z: cz, w: 1 } : { x, y, z, w: 1 }));
  const h = depth / 2;
  return makeNet([row(-h, true), row(-h, false), row(h, false), row(h, true)], p, 1, knotsU, [0, 0, 1 / 3, 2 / 3, 1, 1]);
}

// a loft through closed cross sections in the xz-plane, one per y: u goes around every section
// (degree 2, uniform knots), v from section to section (degree q).
// every section needs the same number of points, the first and last point of a section are the same,
// and a section whose points all coincide closes the body (nose, tail)
// a section can carry a weight for all of its points (default 1), and the v knots can be given
interface LoftSection { y: number; loop: [number, number][]; w?: number }
function loft(sections: LoftSection[], q: number, knotsV = clampedUniformKnots(sections.length, q)): ControlNet {
  const rows = sections.map(({ y, loop, w = 1 }) => loop.map(([x, z]) => ({ x, y, z, w })));
  return makeNet(rows, 2, q, clampedUniformKnots(rows[0].length, 2), knotsV);
}

// a section collapsed into one point: every point of the loop at the centre of the given loop
function collapsed(loop: [number, number][]): [number, number][] {
  const n = loop.length - 1;
  const cx = loop.slice(0, n).reduce((s, [x]) => s + x, 0) / n;
  const cz = loop.slice(0, n).reduce((s, [, z]) => s + z, 0) / n;
  return loop.map(() => [cx, cz]);
}

// ---------------------------------------------------------
// The shapes, from easy to hard
// ---------------------------------------------------------

// 1 · leaf: an open bicubic patch, the same kind of surface as the start plane
// kept simple for the first library Level: every control point on the edge lies in the plane z = 0,
// only the middle vein (the 4 inner points of column i = 2) is lowered, so the upper side of the leaf
// is a shallow groove along the vein, like a real leaf (and like the earlier leaf, whose edges were
// higher than its middle). The starting surface is the same leaf lying flat, so one rectangle
// selection of the vein and one drag down solves it
const LEAF_VEIN = -0.35;

function leafNet(vein: number): ControlNet {
  const countU = 5, countV = 6;
  const width = [0, 0.45, 0.6, 0.5, 0.3, 0]; // half-width along the leaf, 0 = stem and tip
  const rows: WPoint[][] = [];
  for (let j = 0; j < countV; j++) {
    const y = -1 + (2 * j) / (countV - 1);
    const row: WPoint[] = [];
    for (let i = 0; i < countU; i++) {
      const x = (i / (countU - 1) - 0.5) * 2 * width[j];
      const inner = i > 0 && i < countU - 1 && j > 0 && j < countV - 1;
      row.push({ x, y, z: inner && i === 2 ? vein : 0, w: 1 });
    }
    rows.push(row);
  }
  return makeNet(rows, 3, 3, clampedUniformKnots(countU, 3), clampedUniformKnots(countV, 3));
}

const leaf = () => leafNet(LEAF_VEIN);

// 2 · tent: a triangle pushed along y; degree 1 gives straight edges and sharp corners
function tent(): ControlNet {
  const profile: [number, number][] = [[-0.9, 0], [0.9, 0], [0, 1.1], [-0.9, 0]];
  return extrude(profile, 1, clampedUniformKnots(profile.length, 1), 1.6);
}

// 3 · star: a five-pointed star, still degree 1, but 10 corners
function star(): ControlNet {
  const profile: [number, number][] = [];
  for (let k = 0; k <= 10; k++) {
    const angle = Math.PI / 2 + (k * Math.PI) / 5; // first tip points up
    const r = k % 2 === 0 ? 1 : 0.42;
    profile.push([r * Math.cos(angle), 1 + r * Math.sin(angle)]);
  }
  return extrude(profile, 1, clampedUniformKnots(profile.length, 1), 0.35);
}

// 4 · ball: a half circle turned around the axis. Only the right weights make it exactly round
function ball(): ControlNet {
  const r = 0.9;
  return revolve([
    { r: 0, z: 0 }, { r, z: 0, w: S }, { r, z: r }, { r, z: 2 * r, w: S }, { r: 0, z: 2 * r },
  ], 2, [0, 0, 0, 0.5, 0.5, 1, 1, 1]);
}

// 5 · vase: a smooth cubic profile turned around the axis, open at the top
function vase(): ControlNet {
  return revolve([
    { r: 0, z: 0 }, { r: 0.45, z: 0 }, { r: 0.75, z: 0.35 }, { r: 0.75, z: 0.75 },
    { r: 0.3, z: 1.15 }, { r: 0.25, z: 1.45 }, { r: 0.42, z: 1.75 },
  ], 3);
}

// 6 · house: the outline of a house pushed along y. Degree 2 with every interior knot twice:
// multiplicity = degree turns the smooth curve into corners (the crease lesson)
function house(): ControlNet {
  const corners: [number, number][] = [[-0.8, 0], [0.8, 0], [0.8, 0.9], [0, 1.55], [-0.8, 0.9], [-0.8, 0]];
  // between two corners a straight degree-2 piece needs its midpoint as the middle control point
  const profile: [number, number][] = [corners[0]];
  for (let k = 1; k < corners.length; k++) {
    const [ax, az] = corners[k - 1];
    const [bx, bz] = corners[k];
    profile.push([(ax + bx) / 2, (az + bz) / 2], [bx, bz]);
  }
  return extrude(profile, 2, repeatedKnots(corners.length - 1, 2, 2), 1.4);
}

// 7 · tree: a fir with a trunk and two layers of branches, a degree-1 profile turned around the axis
function tree(): ControlNet {
  return revolve([
    { r: 0, z: 0 }, { r: 0.15, z: 0 }, { r: 0.15, z: 0.4 }, { r: 0.8, z: 0.4 },
    { r: 0.35, z: 0.95 }, { r: 0.6, z: 0.95 }, { r: 0, z: 1.8 },
  ], 1);
}

// 8 · cup: the profile goes up the outside, over the rim and back down the inside,
// so the cup has a wall thickness. Degree 2 rounds the rim and the bottom edge
function cup(): ControlNet {
  return revolve([
    { r: 0, z: 0 }, { r: 0.6, z: 0 }, { r: 0.66, z: 1.0 }, { r: 0.6, z: 1.08 },
    { r: 0.52, z: 0.98 }, { r: 0.48, z: 0.15 }, { r: 0, z: 0.15 },
  ], 2);
}

// 9 · mushroom: a stem with a cap that hangs over it
function mushroom(): ControlNet {
  return revolve([
    { r: 0, z: 0 }, { r: 0.22, z: 0 }, { r: 0.16, z: 0.6 }, { r: 0.8, z: 0.6 },
    { r: 0.9, z: 0.85 }, { r: 0.6, z: 1.25 }, { r: 0, z: 1.3 },
  ], 2);
}

// 10 · boat: a closed hull with a deck and a small cabin, lofted from bow to stern.
// Each section goes around the cabin, the deck and the hull (degree 2, so the bilge is round).
// At the bow the section is a vertical line (the stem), at the stern it closes to a point
function boat(): ControlNet {
  // half-width, deck height, keel height, cabin half-width, cabin top (= deck: no cabin)
  const section = (w: number, deck: number, keel: number, cw = 0, top = deck): [number, number][] => {
    const mid = (deck + keel) / 2;
    return [[0, top], [cw, top], [cw, deck], [w, deck], [w, mid], [0.55 * w, keel + 0.05], [0, keel],
      [-0.55 * w, keel + 0.05], [-w, mid], [-w, deck], [-cw, deck], [-cw, top], [0, top]];
  };
  const stern = section(0.55, 0.7, 0.2);
  return loft([
    { y: -1.3, loop: section(0, 0.9, 0.35) },  // stem: all points in x = 0
    { y: -1.0, loop: section(0.32, 0.8, 0.12) },
    { y: -0.55, loop: section(0.62, 0.72, 0.02) },
    { y: -0.15, loop: section(0.72, 0.68, 0, 0.42, 0.68) },
    { y: 0.0, loop: section(0.72, 0.68, 0, 0.42, 1.05) },  // front of the cabin
    { y: 0.55, loop: section(0.7, 0.68, 0, 0.42, 1.05) },  // back of the cabin
    { y: 0.7, loop: section(0.68, 0.68, 0.02, 0.42, 0.68) },
    { y: 1.15, loop: stern },                   // transom
    { y: 1.15, loop: collapsed(stern) },        // closes the transom
  ], 2);
}

// 11 · car (the final challenge), kept simple like a car drawn from basic shapes: a box for the body,
// a trapezoid for the cabin on top, and two round wheels. The outline seen from the side (yz-plane)
// is pushed across the car's width (along x), like the house. The outline is degree 2 with every
// interior knot twice (C0), so it is made of pieces with sharp corners between them:
//   a straight piece: its middle control point halfway between its ends
//   a wheel: two quarter circles under the body, their corner points with weight √2/2 -> exactly round
// the hood slopes more than the trunk; nose at -y, tail at +y. 14 pieces, 29 x 4 = 116 control points
export const CAR_WHEELS = { radius: 0.28, axle: 0.28, centres: [-0.7, 0.7] }; // the axle height = the sill

interface SidePoint { y: number; z: number; w: number }

// pushes a closed outline in the yz-plane (first point = last point) across x from -width/2 to +width/2
// rows: left cap (every point at the centre of the outline), left outline, right outline, right cap
function extrudeSide(outline: SidePoint[], knotsU: number[], width: number): ControlNet {
  const n = outline.length - 1;
  const cy = outline.slice(0, n).reduce((s, o) => s + o.y, 0) / n;
  const cz = outline.slice(0, n).reduce((s, o) => s + o.z, 0) / n;
  const h = width / 2;
  const row = (x: number, cap: boolean): WPoint[] =>
    outline.map(({ y, z, w }) => (cap ? { x, y: cy, z: cz, w } : { x, y, z, w }));
  return makeNet([row(-h, true), row(-h, false), row(h, false), row(h, true)], 2, 1, knotsU, [0, 0, 1 / 3, 2 / 3, 1, 1]);
}

function car(): ControlNet {
  const { radius: R, axle, centres: [front, rear] } = CAR_WHEELS;
  const outline: SidePoint[] = [{ y: -1.2, z: axle, w: 1 }]; // the nose, at the bottom
  const line = (y: number, z: number) => {
    const a = outline[outline.length - 1];
    outline.push({ y: (a.y + y) / 2, z: (a.z + z) / 2, w: 1 }, { y, z, w: 1 });
  };
  // a quarter circle: the corner of its control triangle has weight √2/2
  const quarter = (cornerY: number, cornerZ: number, y: number, z: number) => {
    outline.push({ y: cornerY, z: cornerZ, w: Math.SQRT1_2 }, { y, z, w: 1 });
  };
  // half a wheel below the body, from its back (yc + R) to its front (yc - R)
  const wheel = (yc: number) => {
    quarter(yc + R, axle - R, yc, axle - R);
    quarter(yc - R, axle - R, yc - R, axle);
  };

  line(-1.2, 0.55);   // front face
  line(-0.45, 0.7);   // hood, sloping up
  line(-0.2, 1.05);   // windscreen  } the cabin: a trapezoid on top of the body
  line(0.55, 1.05);   // roof        }
  line(0.85, 0.75);   // rear window }
  line(1.2, 0.7);     // trunk, almost level
  line(1.2, axle);    // rear face
  line(rear + R, axle);
  wheel(rear);
  line(front + R, axle); // the floor between the wheels
  wheel(front);
  line(-1.2, axle);   // back to the nose

  const pieces = (outline.length - 1) / 2;
  return extrudeSide(outline, repeatedKnots(pieces, 2, 2), 1.2);
}

export function allShapes(): Shape[] {
  return [
    { id: 'leaf', order: 'A leaf for a brooch', construction: 'open patch', concept: 'local support: lower the middle vein, the edge stays flat', difficulty: 1, net: leaf(), start: leafNet(0) },
    { id: 'tent', order: 'A tent for a toy camp', construction: 'extrusion, degree 1', concept: 'degree 1 = straight edges and sharp corners', difficulty: 2, net: tent() },
    { id: 'star', order: 'A star for the Christmas tree', construction: 'extrusion, degree 1', concept: 'many control points together (superposition)', difficulty: 3, net: star() },
    { id: 'ball', order: 'A perfectly round ball', construction: 'revolution', concept: 'weights: only √2/2 makes the circle exact', difficulty: 4, net: ball() },
    { id: 'vase', order: 'A vase for flowers', construction: 'revolution, cubic profile', concept: 'reshape the profile, the circle stays exact', difficulty: 5, net: vase() },
    { id: 'house', order: 'A house for a model railway', construction: 'extrusion, degree 2', concept: 'knot multiplicity = degree gives a corner', difficulty: 6, net: house() },
    { id: 'tree', order: 'A fir tree', construction: 'revolution, degree 1 profile', concept: 'sharp tiers on a round body', difficulty: 7, net: tree() },
    { id: 'cup', order: 'A cup', construction: 'revolution, degree 2 profile', concept: 'a profile that turns back on itself (wall thickness)', difficulty: 8, net: cup() },
    { id: 'mushroom', order: 'A mushroom', construction: 'revolution, degree 2 profile', concept: 'an overhang: the cap hangs over the stem', difficulty: 9, net: mushroom() },
    { id: 'boat', order: 'A boat', construction: 'loft, closed hull with deck', concept: 'a stem and a transom close the body, a cabin sits on the deck', difficulty: 10, net: boat() },
    { id: 'car', order: 'A toy car (final challenge)', construction: 'extrusion, degree 2, rational wheel arcs', concept: 'everything at once: straight panels, corners from double knots and wheels that are only round through weights', difficulty: 11, net: car() },
  ];
}
