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

// the control points of a closed polygon with rounded corners, for a uniform degree-2 curve:
// every corner C gets a point a little before it, C itself and a point a little after it,
// so the curve runs straight along the edges and turns in a small arc near each corner.
// the loop starts and ends in the middle of the edge from the last corner to the first one
function roundedPolygon(corners: [number, number][], round: number): [number, number][] {
  const n = corners.length;
  const toward = (c: [number, number], d: [number, number]): [number, number] => {
    const len = Math.hypot(d[0] - c[0], d[1] - c[1]);
    const t = len > 0 ? Math.min(round, len / 2) / len : 0;
    return [c[0] + t * (d[0] - c[0]), c[1] + t * (d[1] - c[1])];
  };
  const first = corners[0];
  const last = corners[n - 1];
  const start: [number, number] = [(first[0] + last[0]) / 2, (first[1] + last[1]) / 2];
  const loop: [number, number][] = [start];
  corners.forEach((c, k) => {
    loop.push(toward(c, corners[(k + n - 1) % n]), c, toward(c, corners[(k + 1) % n]));
  });
  loop.push(start);
  return loop;
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
// only the middle vein (the 4 inner points of column i = 2) is raised. The starting surface is the
// same leaf lying flat, so one rectangle selection of the vein and one drag up solves it
const LEAF_VEIN = 0.35;

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

// 11 · car (the final challenge): a box-like body with four half-cylinder wheels, still ONE surface.
// u goes around each section (degree 2): the body, the cabin and, at the bottom, the two wheel tracks.
// v runs from the nose to the tail (degree 2) in pieces, every interior knot twice (C0 between pieces):
//   a straight piece: the middle section is the average of its two ends -> flat panels, sharp edges
//   a wheel: two quarter circles in the yz-plane, the corner section has weight √2/2 -> exactly round.
// Over the wheels the body stays the same, so only the wheel tracks move with the circle
function car(): ControlNet {
  interface Body { w: number; sill: number; belt: number; roofW: number; roof: number }
  const TRACK = 0.2;  // width of a wheel
  const R = 0.27;     // wheel radius
  // the outline of one section; the wheel tracks hang `d` below the sill (d = 0: no wheel here)
  const section = ({ w, sill, belt, roofW, roof }: Body, d: number) => roundedPolygon([
    [roofW, roof], [w, belt], [w, sill],
    [w, sill - d], [w - TRACK, sill - d], [w - TRACK, sill],       // right wheel track
    [-(w - TRACK), sill], [-(w - TRACK), sill - d], [-w, sill - d], // left wheel track
    [-w, sill], [-w, belt], [-roofW, roof],
  ], 0.05);

  const bonnet: Body = { w: 0.68, sill: 0.3, belt: 0.72, roofW: 0.64, roof: 0.73 };
  const cabin: Body = { w: 0.7, sill: 0.3, belt: 0.76, roofW: 0.54, roof: 1.15 };
  const boot: Body = { w: 0.68, sill: 0.3, belt: 0.74, roofW: 0.64, roof: 0.75 };

  // the pieces from nose to tail, each adds its middle section and its end section
  const sections: LoftSection[] = [{ y: -1.3, loop: collapsed(section(bonnet, 0)) }]; // closes the nose
  const straight = (y: number, body: Body) => {
    const a = sections[sections.length - 1];
    const b = section(body, 0);
    sections.push({ y: (a.y + y) / 2, loop: a.loop.map(([x, z], i) => [(x + b[i][0]) / 2, (z + b[i][1]) / 2]) });
    sections.push({ y, loop: b });
  };
  const wheel = (yc: number, body: Body) => {
    straight(yc - R, body);                                                      // up to the front of the wheel
    sections.push({ y: yc - R, loop: section(body, R), w: Math.SQRT1_2 });       // corner of the first quarter
    sections.push({ y: yc, loop: section(body, R) });                            // lowest point of the wheel
    sections.push({ y: yc + R, loop: section(body, R), w: Math.SQRT1_2 });       // corner of the second quarter
    sections.push({ y: yc + R, loop: section(body, 0) });                        // back of the wheel
  };

  straight(-1.3, bonnet);  // front face
  wheel(-0.82, bonnet);    // front wheels
  straight(-0.4, bonnet);  // end of the bonnet
  straight(0.0, cabin);    // windscreen
  straight(0.45, cabin);   // roof
  wheel(0.85, boot);       // rear window, then the rear wheels
  straight(1.3, boot);     // boot
  const tail = sections[sections.length - 1];
  const last = { y: 1.3, loop: collapsed(tail.loop) };
  sections.push({ y: 1.3, loop: tail.loop.map((p, i) => [(p[0] + last.loop[i][0]) / 2, (p[1] + last.loop[i][1]) / 2]) });
  sections.push(last);     // closes the tail

  const pieces = (sections.length - 1) / 2;
  return loft(sections, 2, repeatedKnots(pieces, 2, 2));
}

export function allShapes(): Shape[] {
  return [
    { id: 'leaf', order: 'A leaf for a brooch', construction: 'open patch', concept: 'local support: raise the middle vein, the edge stays flat', difficulty: 1, net: leaf(), start: leafNet(0) },
    { id: 'tent', order: 'A tent for a toy camp', construction: 'extrusion, degree 1', concept: 'degree 1 = straight edges and sharp corners', difficulty: 2, net: tent() },
    { id: 'star', order: 'A star for the Christmas tree', construction: 'extrusion, degree 1', concept: 'many control points together (superposition)', difficulty: 3, net: star() },
    { id: 'ball', order: 'A perfectly round ball', construction: 'revolution', concept: 'weights: only √2/2 makes the circle exact', difficulty: 4, net: ball() },
    { id: 'vase', order: 'A vase for flowers', construction: 'revolution, cubic profile', concept: 'reshape the profile, the circle stays exact', difficulty: 5, net: vase() },
    { id: 'house', order: 'A house for a model railway', construction: 'extrusion, degree 2', concept: 'knot multiplicity = degree gives a corner', difficulty: 6, net: house() },
    { id: 'tree', order: 'A fir tree', construction: 'revolution, degree 1 profile', concept: 'sharp tiers on a round body', difficulty: 7, net: tree() },
    { id: 'cup', order: 'A cup', construction: 'revolution, degree 2 profile', concept: 'a profile that turns back on itself (wall thickness)', difficulty: 8, net: cup() },
    { id: 'mushroom', order: 'A mushroom', construction: 'revolution, degree 2 profile', concept: 'an overhang: the cap hangs over the stem', difficulty: 9, net: mushroom() },
    { id: 'boat', order: 'A boat', construction: 'loft, closed hull with deck', concept: 'a stem and a transom close the body, a cabin sits on the deck', difficulty: 10, net: boat() },
    { id: 'car', order: 'A toy car (final challenge)', construction: 'loft, rounded box sections, rational pieces in v', concept: 'everything at once: flat panels, sharp edges, double knots and wheels that are only round through weights', difficulty: 11, net: car() },
  ];
}
