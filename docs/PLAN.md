# Shape Match: plan after the proposal feedback

Status 25.11.2026, before progress check 2 (26.11). Terms are defined in [GLOSSARY.md](../GLOSSARY.md), and the two key decisions are in [docs/adr](adr/).

## 1. Where we stand

| Part | Status |
| --- | --- |
| Editor (Aim 1): GPU tessellation with exact normals, picking and dragging, weights, undo, parameter-space / partition-of-unity / formula panels, views | done |
| Influence view | open (cut first if time runs short) |
| Error pass, heatmap, Solved state (Aim 2) | not started |
| Levels 2–5, knot insertion, Hints (Aim 3) | only level 1 exists |
| Evaluation (correctness, performance, reliability of the measure) | planned below |
| Game layer | planned below |

Priorities until 17.12: (1) error pass + evaluation, (2) Levels with the Solved state, (3) the game layer. Hints and the influence view are cut first.

## 2. Shape distance and Tolerance

- **Shape distance**: from every sample of one surface to the nearest **triangle** of the other, in both directions. Samples are weighted by their area. The result is the RMS, divided by the diagonal of the Target's bounding box. ([ADR 0002](adr/0002-shape-distance-point-to-triangle.md))
- **Why triangles, not samples**: with point-to-point Chamfer, the same surface sampled twice gives a Noise floor that grows with the sample spacing h. With point-to-triangle it falls to about curvature × h². Area weighting removes the bias where weights or knots bunch samples together.
- **Solved** additionally needs the largest single distance (Hausdorff) under a looser bound, so a small wrong spot (for example the crease in level 4) can't hide in the average.
- **Tolerance τ, per Level and automatic**: when a Level loads, the GPU measures the Target against itself on a grid shifted by half a cell. That is the Noise floor. Then τ = max(3 × Noise floor, 0.2 % of the diagonal).
- **Stars**: 3★ ≤ τ, 2★ ≤ 3τ, 1★ ≤ 10τ. The factors live in a config and get tuned after a playtest.
- **Live grading, then Hand-in**: the Match and the Stars update live while the learner edits, so they see at once whether a move helped. When they are happy, they hand the surface in. The Hand-in runs the Shape match check: the error pass computes the Shape distance of the handed-in surface once more and gives Match, Stars and a verdict, Passed (at least 1★) or Not passed. A passing Hand-in Solves the Level, which unlocks the next one.
- **When the error pass runs**: only when the net changes, on its own grid (64² by default, adjustable), separate from the 128² render grid. The heatmap is interpolated onto the surface.
- **Parameter comparison** (control points, weights, knots) is used only for Hints and as a reference in the evaluation. It is off as soon as the learner's Layout differs from the Target's.

## 3. Evaluation

### 3.1 Correctness
- The worked example from the lecture: cubic, 7 points, U = [0,0,0,0, .25, .5, .75, 1,1,1,1], u = 0.42 → span 4, basis [0.0082, 0.3588, 0.5807, 0.0524]. `nurbs.ts` already matches it (checked 25.11).
- The course provides no test models, so we build our own **Reference models** with exact geometry (#7). **Not built yet** (planned for 10.12): an **open patch** (bicubic Bézier, against the Bernstein form), a **sphere** (every point at distance r from the centre) and a **torus** (every point fulfils (√(x² + y²) − R)² + z² = r²). Every sample can be checked, not just a few (u, v).
- Built so far: GPU tessellation is compared with the CPU `evaluate()` on the lecture net, Ball, Vase and House (`src/surfaceGrid.gpu.test.ts`), and the GPU Shape distance with the CPU reference (`src/errorPass.gpu.test.ts`).
- Once the Reference models exist, each check runs twice: on the CPU (`nurbs.ts`, vitest) and on the GPU (`tessellate.wgsl` with a readback), with a fixed f32 tolerance. The sphere and the torus also test rational and closed surfaces.

### 3.2 Performance
- Request `timestamp-query` as an optional feature. Without it, the UI shows "not available".
- Time the evaluate pass and the error pass separately at 32², 64², 128² and 256² samples. Report the median of 100 runs.
- CPU baseline: the same error pass in plain JavaScript at 32² and 64², to put a number on "slow in JavaScript".
- At least two GPUs: one integrated in a laptop and one dedicated card.

### 3.3 How reliable the Shape distance is
| | Experiment | Expected result |
| --- | --- | --- |
| E1 | Noise floor against sample count, point-to-point vs point-to-triangle | falls about like h for point-to-point and like h² for point-to-triangle |
| E2 | move one control point of the solution by a known δ | the distance grows steadily and smoothly with δ |
| E3 | the same shape in another form (knot insertion, all weights × c) | the distance stays at the Noise floor while the Parameter comparison differs |
| E4 | Levels with a fixed Layout: compare the rankings from the distance and from the Parameter comparison | the rankings agree |

## 4. Levels (Task sequence)

Each Level has its own Layout, shared by its Target and its starting Learner surface. ([ADR 0001](adr/0001-layout-per-level.md))

| Level | Concept | Layout | Learner can change | Order (story) |
| --- | --- | --- | --- | --- |
| 1 | Local support | 6×6 bicubic, clamped | control points | a dented sheet |
| 2 | Superposition | 6×6 bicubic, clamped | control points | a ridge and a valley |
| 3 | Weights | closed cylinder, degree 2×1, 9×2 points | **weights only**, starting at 1 | a pipe or barrel that is only round with the right weights |
| 4 | Knot multiplicity | bicubic | knot **insertion** (Parameter comparison off) | a folded crease |
| 5 | Combination | the vase from the shape library | everything | a vase |

### 4.1 Shape library

[src/shapes.ts](../src/shapes.ts) holds 11 Targets. Each is **one** NURBS surface with a known control net. `node scripts/shapes-preview.ts` checks them, writes `docs/shapes/obj/*.obj` (opens in Quick Look or Blender) and a numbered contact sheet, [docs/shapes/preview.png](shapes/preview.png).

| # | Shape | Construction | Points, degree | What it teaches |
| --- | --- | --- | --- | --- |
| 1 | leaf | open patch | 5×6, 3×3 | local support |
| 2 | tent | extrusion of a triangle | 4×4, 1×1 | degree 1 = straight edges, sharp corners |
| 3 | star | extrusion of a star polygon | 11×4, 1×1 | many points together |
| 4 | ball | revolution of a half circle | 9×5, 2×2 | weights: only √2/2 makes it round |
| 5 | vase | revolution, cubic profile | 9×7, 2×3 | reshape the profile, the circle stays exact |
| 6 | house | extrusion, degree 2, double knots | 11×4, 2×1 | knot multiplicity = degree gives a corner |
| 7 | tree | revolution, degree 1 profile | 9×7, 2×1 | sharp tiers on a round body |
| 8 | cup | revolution, profile turns back | 9×7, 2×2 | wall thickness |
| 9 | mushroom | revolution | 9×7, 2×2 | an overhang |
| 10 | boat | loft: closed hull with deck and cabin, stem and transom | 13×9, 2×2 | a body closed at both ends, a cabin on the deck |
| 11 | car (**final challenge**) | extrusion of the side outline: box body, trapezoid cabin, two wheel arcs | 29×4, 2×1 | everything at once: straight panels, double knots for corners, wheels that are only round through weights |

**The car in detail**: kept simple, like a car drawn from basic shapes: a box for the body, a trapezoid for the cabin, two round wheels. The outline seen from the side is pushed across the car's width, like the house. The outline is degree 2 with every interior knot twice, so it is 14 pieces with sharp corners between them. A straight piece has its middle control point halfway between its ends. A wheel is two quarter circles below the body whose corner points have weight √2/2. The hood slopes more than the trunk. The checks confirm that the wheels are round to 1e-5 (measured: 2.5e-8). It has 116 control points (the first car, a loft with four separate wheels, had 950, too many to edit by hand).

**Starting surfaces**: from the Tent on, a shape's start is wrong in more than one direction, harder step by step (see `startChange` in `src/target.ts`): flattened and wider in x (Tent, Star, Ball), also narrower in y (Vase, House, Tree, Cup), also leaning backwards (Mushroom, Boat, Car). All of these are affine, so the start keeps the Target's Layout and weights (ADR 0001).

The script checks every Layout (knot count, clamped, degree ≤ 3, weights > 0). For the closed shapes it checks that the seam closes, that the caps of the extrusions close, and that every revolution is round to 1e-5. That last check also tests `nurbs.ts` until the Reference models exist.

Not possible as one surface: a house with a separate roof, a tree with a separate trunk, a car with four separate wheels. Those would need several patches, which are out of scope.

**Finding for the evaluation**: on the house, a band of shading runs along the eaves, because the 40×40 grid doesn't put a sample exactly on the double knot. The same problem affects the Shape distance: a crease falls between samples unless the grid includes the knot values. Possible fix: always add the distinct knots to the sample positions.


## 5. Game

**Setting**: the Workshop, a craftsperson's workshop where Customers order objects. Every Level is an Order, and every Level Solved by a Hand-in puts the learner's own surface on the Shelf.

**Flow**: Title → Board → Order (the Customer explains the Task) → Workbench (play) → Hand-in → Result → Board.

**Hand-in and Result**: handing in is the reward moment. The Result screen shows the learner's score for this Hand-in (the Match in percent and the Stars) and their Best for the Level (the hi-score, nice to have), with a "New best!" marker when they beat it. When it Passed, the Level is Solved: the next Level unlocks, the object goes on the Shelf, and after a short countdown the game routes to the next Level automatically ("Next Level" skips the wait, "Stay and try for more Stars" cancels it). When it Not passed, they get an encouraging "Not passed yet" and go back to the Workbench with their surface unchanged.

**Rewards**: 1–3 Stars per Level, and 1★ unlocks the next space on the Board. The Board shows each Level's best Stars. Using a Hint caps that Level at 2★.

**Look**: a cozy illustrated frame in a warm palette (wood, brass, paper), rounded game UI, a title font with character, and Customer portraits with speech bubbles. The 3D view sits on a work table with a soft contact shadow, a background gradient and a rim light. Art: CC0 packs (for example Kenney.nl) plus our own SVG. No 3D workshop scene.

**Game feel**: on the Result screen the Match counts up and the Stars pop in one by one, confetti when a Hand-in Solves the Level, a "New best!" badge, Match shown as a filling meter that warms from cold to warm colours, smooth transitions between screens, CC0 sound effects that can be muted.

**How it's built**: the screens are 2D HTML/SVG in front of the WebGPU canvas. The Shelf uses canvas screenshots taken at a Hand-in that Solves the Level (replaced by a later, better one). Progress (the Best per Level) is kept in `localStorage` (no server). Animation is CSS only.

## 6. User interface (Workbench)

- **Task panel**: the Order text, the live Match meter, Stars, and a **Hand in** button. It is always enabled and looks "ready" once handing in would give at least 1★.
- **Heatmap** switch.
- **Hint** button: points at the point, weight or knot that is furthest off.
- **Performance** panel, collapsible: ms per pass (timestamp queries), FPS, sample counts of render and error pass, adapter name.
- **Weight slider** on an exponential scale (taken from Shene's tool).

## 7. Related work: positioning

Zhao, Lowther and Shene, *A Tool for Teaching Curve Design* (Michigan Tech, NSF DUE-9653244):
- **What it does**: Bézier, rational Bézier, B-spline and NURBS **curves**. Moving points, weights on an exponential slider, editing and repeating knots, closed curves, knot insertion, degree elevation, subdivision. It shows the partition of unity window, a marker that traces u, the convex hull and the de Casteljau and de Boor nets. Desktop, C / OpenGL / GLUT.
- **What it doesn't do**: no surfaces, and it is a free design environment that is explicitly "without constraints". It has no Targets, no tasks, no grading and no game frame, and the paper reports no evaluation.
- **What we take from it**: the partition of unity window, the marker that follows u, the exponential weight slider.

**Our niche**: NURBS **surfaces**, **goals with a Target**, **automatic geometric feedback computed on the GPU**, a **game frame** with progression and rewards, and an **evaluation** of correctness, performance and the measure itself. (This replaces the proposal's claim that other tools "stop at the viewing or changing level": Shene's tool already reaches constructing.)

## 8. Checkpoints

| Date | What we show |
| --- | --- |
| 26.11 (progress 2) | This plan: evaluation plan, measure design, Layout per Level, game concept (screen mock-ups if ready) |
| 10.12 (progress 3) | Error pass with τ calibration, heatmap, Solved + Stars, L1–L3, **skeleton of the game layer** (Board, Stars, Result), correctness tests, timing, E1–E2 |
| 17.12 (final) | L4–L5 with knot insertion, full art and game feel, E3–E4, related work |

## 9. Work split

| Owner | Part |
| --- | --- |
| Member 1 | Evaluate pass, correctness tests, timing and Performance panel, switching to a different Layout per Level |
| Member 2 | Error pass, τ calibration, heatmap, experiments E1–E4 |
| Member 3 | Levels and tools per Level, Hints, game layer and art |

## 10. Still open

- Nothing at the moment. (The course test models don't exist; we build our own Reference models instead, see §3.1. They are not built yet.)
