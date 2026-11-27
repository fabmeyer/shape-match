# Shape distance: parameter-wise comparison or point-to-triangle?

Status: research note, 26.11.2026. Terms as in [GLOSSARY.md](../../GLOSSARY.md). Nothing in the code or the ADRs was changed.

## Question

The teacher asked: why not evaluate the Target and the Learner surface on the same 128 × 128 grid of (u, v) values and compare the points pair by pair?

    d_ij = | S_T(u_i, v_j) − S_L(u_i, v_j) |,   grade = RMS and max of d_ij / Target diagonal

The current Shape distance ([ADR 0002](../adr/0002-shape-distance-point-to-triangle.md), [src/shapeDistance.ts](../../src/shapeDistance.ts)) instead searches, for every sample, the nearest triangle of the other surface, in both directions. Students find that hard to explain.

The idea is plausible because the Target and the start of a Level share one Layout ([ADR 0001](../adr/0001-layout-per-level.md)), and the learner can only move points and change weights ([src/target.ts](../../src/target.ts), `Tools`).

## Short answer

Keep the point-to-triangle Shape distance as the grade. Make it simpler to explain, and use the parameter-wise distance only where it is safe.

- The parameter-wise distance is never smaller than the geometric gap, so it never passes a wrong shape. But it fails some correct shapes. The worst case is realistic and invisible to the learner. In the Tent, the cap point can sit anywhere inside the triangular end face and the shape is the same. If the learner leaves it where the start had it, the Shape distance is 0 but the parameter-wise RMS is 1.95 % (10.4 % max): 0 Stars instead of 3. The House behaves the same (4.5 % RMS).
- Weights give a second case: a Möbius reparametrization of the circle gives the same pipe, but 0.89 % parameter-wise already at c = 1.1. That is 0 Stars in the Weights Level.
- The ADR 0002 arguments against "parameters" do not hit the teacher's proposal. Scaling all weights gives exactly 0, and knot insertion keeps the parametrization. The real arguments are the ones above.
- The current method has one part that does nothing at the moment: the Noise floor. In all 14 Levels, 3 × Noise floor is below 0.2 %, so τ is always the fixed minimum of 0.2 %. Dropping the Noise floor and using a fixed Tolerance τ = 0.2 % changes no grade, and it removes the hardest part of the explanation.

## How each method works (student explanation)

**Point-to-triangle (current).** We put a grid of about 64 × 64 dots on your surface and on the Target. For each dot on your surface we find the closest spot on the Target. For each dot on the Target we find the closest spot on your surface. Your Shape distance is the average of these gaps (as an RMS), compared with the size of the Target. The single largest gap must also be small, so one wrong corner cannot hide in the average. Because we look for the *closest spot*, it does not matter how your control points are spread over the surface, only what the surface looks like.

(Left out on purpose, because they are implementation details: "closest spot" is measured to the small triangles between the dots; each dot counts with the area it covers. Both are in ADR 0002.)

**Parameter-wise (proposed).** Both surfaces are drawn from the same 128 × 128 grid of (u, v) values. Dot (i, j) on your surface is compared with dot (i, j) on the Target. Your Shape distance is the average gap of these pairs, compared with the size of the Target, and the largest gap must also be small.

The parameter-wise version is one idea shorter: it pairs dots instead of searching. Its catch is hard to explain to a student who has made the right shape: "your surface is correct, but your points are spread differently".

## Where they disagree

### Why the parameter-wise distance can only be larger

S_L(u, v) is a point on the Learner surface. So the gap from S_T(u, v) to the closest spot on the Learner surface is at most |S_T(u, v) − S_L(u, v)|. The parameter-wise max is therefore an upper bound of the one-sided Hausdorff distance. It never says "close" when the shapes are far. Every disagreement is a correct shape graded too harshly. All rows in the table below agree with this.

The difference is the sliding *along* the surface. The parameter-wise distance counts it, the geometric one does not. Approximation methods know this well: a fit that measures point to point (the "point distance") is slowed down by tangential error, so they use parameter correction or the squared distance to the surface instead (Hoschek 1988; Wang, Pottmann & Liu 2006).

### When the same shape can have another parametrization in this game

1. **Uniform weight scaling.** Multiplying all weights by c does not change the surface or its parametrization. The rational basis functions R_i = N_i w_i / Σ N_j w_j do not change (The NURBS Book §4.2, homogeneous form). Measured: exactly 0 for both methods.
2. **Möbius reparametrization through weights.** For a rational Bézier segment, weights (w_0 c⁰, w_1 c¹, w_2 c², …) give the same curve under a rational linear change of parameter (Patterson 1985; Lee & Lucian 1991; The NURBS Book §6.4). The circle in [src/shapes.ts](../../src/shapes.ts) has double interior knots, so each quarter is its own quadratic rational Bézier. The weights `[1, √2/2·c, c², √2/2·c, 1, √2/2·c, c², √2/2·c, 1]` give the same circle for any c > 0. The learner can set these with the weight tool, so it affects the Weights Level (pipe), the Ball (both directions: circle and half-circle profile), and all revolutions. In practice a learner will rarely do this: the obvious solution only changes the corner weights. Still, a learner who changes the weights of the on-curve points would be graded down for a correct pipe.
3. **Collapsed rows (caps and poles).** `extrude` and `extrudeSide` close each end with a row of points that all sit at one point, the centre of the profile. In v the surface is degree 1, so the cap is a fan of straight lines from that point to the profile. For a convex end face (Tent) or one that is star-shaped from the point (House), any point inside the face gives the same cap. Moving it is invisible on screen. The start has it in a different place (squashed in z by `changedStart`), so a learner who fixes only the visible profile has the right shape and a large parameter-wise distance. The same happens at the poles of revolutions, but there the points sit on the axis and cannot move without changing the shape.
4. **The seam of a closed profile.** The first and last points are the same point. Shifting the whole profile by one vertex gives the same polygon with another start. Measured on the Tent: 39 % parameter-wise and 0 geometric. A learner will not do this by accident. It is listed because it shows how large the false error can get.
5. **Sliding points within a flat region.** In a flat part, inner points can move inside the plane without changing the shape (the boundary curves stay the same). This is the clearest case of tangential sliding. In the Levels, flat parts are bounded by bumps, so moving points there usually changes the shape too.
6. **Knot insertion (#19, planned).** Inserting a knot changes the control net but not the curve, "neither geometrically nor parametrically" (The NURBS Book §5.2; Boehm 1980). So S(u, v) stays the same function, and a parameter-wise comparison at the same (u, v) still works. One detail: [src/samples.ts](../../src/samples.ts) adds every interior knot to the grid. After insertion the two grids would differ, so the grid would have to use the union of both knot sets.

### Measured

Script: a scratch script (since deleted) that calls `shapeDistance` (CPU, 64 per direction plus knots, the game's grid) and a parameter-wise distance on the 128 grid from `samplePositions` (the render grid). Both are divided by the Target's bounding-box diagonal. The parameter-wise RMS is area-weighted with the Target's sample areas, like the Shape distance. Stars use the rules in [src/grading.ts](../../src/grading.ts) with the Level's overrides. The current τ comes from the Noise floor (it is 0.2 % in every Level). For the parameter-wise distance, τ is fixed at 0.2 % because Target against itself is exactly 0.

**Same shape, other control net** (both should be 0 / 3★):

| Level | Case | Shape distance RMS / max | ★ | Parameter-wise RMS / max | ★ |
|---|---|---|---|---|---|
| Weights (pipe) | all weights × 2 | 0 / 0 | 3 | 0 / 0 | 3 |
| Weights (pipe) | Möbius in u, c = 1.1 | 0.031 % / 0.041 % | 3 | 0.89 % / 1.23 % | 0 |
| Weights (pipe) | Möbius in u, c = 1.3 | 0.024 % / 0.054 % | 3 | 2.45 % / 3.39 % | 0 |
| Ball | Möbius in u (circle), c = 1.1 | 0.021 % / 0.038 % | 3 | 0.67 % / 1.14 % | 1 |
| Ball | Möbius in v (profile), c = 1.1 | 0.008 % / 0.010 % | 3 | 0.85 % / 1.14 % | 1 |
| Ball | Möbius in v (profile), c = 1.3 | 0.005 % / 0.012 % | 3 | 2.30 % / 3.13 % | 0 |
| Vase | Möbius in u, c = 1.3 | 0.014 % / 0.047 % | 3 | 1.62 % / 2.93 % | 0 |
| Tent | cap points at the start's place (inside the end face) | ~1e-14 | 3 | 1.95 % / 10.4 % | 0 |
| Tent | cap point at (0, 0.9), near the ridge | ~1e-15 | 3 | 3.78 % / 20.1 % | 0 |
| House | cap points at the start's place, in the end plane | ~1e-14 | 3 | 4.54 % / 19.1 % | 0 |
| Tent | seam shifted by one vertex | ~1e-17 | 3 | 39.4 % / 68.0 % | 0 |
| flat sheet (not a Level) | 16 inner points slid in the plane | ~1e-18 | 3 | 1.18 % / 1.77 % | 1 |

The small non-zero Shape distances for Möbius weights come from the grid. The dots land elsewhere on the circle, and the triangles between them cut the arc a little differently. They are far below τ.

Star, Car and Boat with the cap point at the start's place are not the same shape. There the start's point lies outside the region seen from it as a fan, so the fan folds over the outline. The Shape distance sees it (Star 1.2 %, Car 0.55 %, Boat 0.34 %). So a free cap point is only harmless where the end face is convex or star-shaped from it.

**Different shapes** (both should agree):

| Level | Target vs start: Shape distance RMS | Parameter-wise RMS | Ratio | Jitter 0.5 % of diagonal: Shape distance ★ | Parameter-wise ★ |
|---|---|---|---|---|---|
| Level 1 | 3.31 % | 3.81 % | 1.15 | 3 | 2 |
| Weights | 1.15 % | 1.85 % | 1.61 | 3 | 1 |
| Leaf | 3.62 % | 3.82 % | 1.05 | 3 | 2 |
| Tent | 9.81 % | 14.9 % | 1.52 | 3 | 2 |
| Star | 22.5 % | 31.3 % | 1.39 | 3 | 2 |
| Ball | 18.2 % | 25.5 % | 1.40 | 3 | 2 |
| Vase | 17.0 % | 24.4 % | 1.43 | 3 | 2 |
| House | 17.5 % | 23.9 % | 1.36 | 3 | 2 |
| Tree | 15.4 % | 21.9 % | 1.43 | 3 | 2 |
| Cup | 17.0 % | 23.5 % | 1.38 | 3 | 2 |
| Mushroom | 19.8 % | 27.0 % | 1.36 | 3 | 2 |
| Boat | 11.2 % | 16.6 % | 1.48 | 3 | 2 |
| Car | 12.3 % | 18.2 % | 1.48 | 3 | 2 |
| Superposition | 3.14 % | 3.34 % | 1.06 | 3 | 2 |

"Jitter" moves every control point of the Target by 0.5 % of the diagonal in a random direction (fixed seed). The geometric RMS is then 0.12–0.19 %, the parameter-wise RMS 0.21–0.34 %. Only the normal part of a random move changes the shape, so the parameter-wise distance is about 1.6–2.1 × larger. With the same τ it is stricter. It would need about τ ≈ 0.35 % to grade small errors like today.

**Weights Level: corner weight sweep** (Level grading: 2★ ≤ 1.5 τ, 1★ ≤ 3 τ):

| Corner weight | 0.60 | 0.65 | 0.70 | √2/2 | 0.75 | 0.80 | 0.90 | 1.00 |
|---|---|---|---|---|---|---|---|---|
| Shape distance RMS | 0.56 % | 0.29 % | 0.034 % | 0 | 0.20 % | 0.42 % | 0.81 % | 1.15 % |
| Stars | 1 | 2 | 3 | 3 | 3 | 1 | 0 | 0 |
| Parameter-wise RMS | 0.81 % | 0.42 % | 0.051 % | 0 | 0.30 % | 0.64 % | 1.27 % | 1.85 % |
| Stars (τ = 0.2 %) | 0 | 1 | 3 | 3 | 1 | 0 | 0 | 0 |

When only the corner weights change, both methods rank the same. The parameter-wise window is narrower.

## Cost

Counts of distance evaluations per measurement (both directions for point-to-triangle). Grid sizes are from `samplePositions` and include the knots.

| Grid | Level 1 (6 × 6, bicubic) | Car (29 × 4, most knots) |
|---|---|---|
| 64: point-to-triangle | 4 096 samples × 7 938 triangles × 2 = 65 M tests | 5 544 × 10 790 × 2 = 120 M |
| 128: point-to-triangle | 16 900 × 33 282 × 2 = 1.12 G tests | 20 328 × 40 086 × 2 = 1.63 G |
| 64: parameter-wise | 4 096 differences | 5 544 |
| 128: parameter-wise | 16 900 differences | 20 328 |

So the parameter-wise distance is O(N) instead of O(N²). At 128 it is about 60 000 × cheaper. The GPU runs point-to-triangle at 64 only when the net changes ([src/grader.ts](../../src/grader.ts)), so today the cost is not a problem. It would become one at 128.

What the GPU path would need for parameter-wise:

- No new evaluation. The render grid is already the 128 grid from `samplePositions` ([src/surfaceGpu.ts](../../src/surfaceGpu.ts), `RENDER_RES = 128`; [src/shaders/tessellate.wgsl](../../src/shaders/tessellate.wgsl)). The Target is drawn shifted by `TARGET_OFFSET` in x, so either subtract that or tessellate an unshifted copy.
- [src/shaders/distance.wgsl](../../src/shaders/distance.wgsl) becomes one line per thread: `distances[k] = length(from[k] − to[k])`. The area weights and [src/shaders/reduce.wgsl](../../src/shaders/reduce.wgsl) can stay. Only one direction is needed, because the pairs are symmetric.
- No Noise floor: the Target against itself on the same grid is exactly 0. τ must be a fixed share of the diagonal. The measurements suggest about 0.35 % to keep today's feel for small errors, and less for the Weights Level.

## Recommendation and what would change in the code

**Recommended: keep point-to-triangle for grading, simplify it, and use parameter-wise for hints.**

1. **Grade with the current Shape distance.** It is what the Order asks for (the shape), and the Glossary already defines the Shape distance as "independent of how either is parametrised". The parameter-wise distance fails correct Tents and Houses in a way the learner cannot see or fix without knowing about cap points. That is worse for learning than a harder explanation.
2. **Drop the Noise floor and use a fixed τ = 0.2 % of the diagonal.** Measured 3 × Noise floor per Level: 0 (Tent, Star, House: flat faces) up to 0.18 % (Car). So τ = 0.2 % already in all 14 Levels, and no grade changes. Files: [src/grading.ts](../../src/grading.ts) (`calibrate` without `noise`), [src/grader.ts](../../src/grader.ts) and [src/errorPass.ts](../../src/errorPass.ts) (`measureLevel` measures only the start), the Glossary entry for Noise floor and Tolerance, and a short ADR that replaces that part of ADR 0002. Keep the Noise floor as experiment E1 in [docs/PLAN.md](../PLAN.md), because it explains why triangles and not samples.
3. **Explain it with the short paragraph above.** Closest spot both ways, average and worst gap, relative to the Target's size. Triangles and area weights go in the report, not in the game.
4. **Use the parameter-wise gap for Hints and as a check.** While the Layout is the same, d_ij at the control point's peak tells the learner which part to move and in which direction. That fits the Parameter comparison already planned for Hints. Show it as a hint, never as the grade. It is also a cheap upper bound of the one-sided gaps (see above), so it could serve as a quick pre-check before the brute-force search. That is an optimisation, not needed today.

**If the parameter-wise distance should be the grade anyway** (for example for teaching reasons), it only becomes safe with these changes:

- Cap rows must not be free. Either they follow the profile's centre automatically (as `extrude` / `extrudeSide` build them), or they cannot be selected. Same for the Boat's transom point. Files: [src/pointDrag.ts](../../src/pointDrag.ts), [src/shapes.ts](../../src/shapes.ts) (mark the collapsed rows).
- Seam duplicates must move together (today they are separate points; nothing links them in `pointDrag.ts`).
- In Levels with weights, only the weights that the solution needs should be editable (the corner weights), so the learner cannot reach a Möbius-reparametrized solution. Or accept that a rare correct solution is graded down.
- τ fixed at about 0.35 % (stricter in the Weights Level). The Glossary definition of Shape distance must change, because it would no longer be independent of parametrization.
- For knot insertion (#19), sample the union of both knot sets.

Even then, the tangential sliding of free inner points in flat areas stays as a known failure case.

## Sources

- L. Piegl, W. Tiller, *The NURBS Book*, 2nd ed., Springer 1997. §4.2: NURBS curves and their weights (homogeneous form; scaling all weights changes nothing). §5.2: knot insertion leaves the curve unchanged geometrically and parametrically. §6.4: reparametrization, including the rational linear (Möbius) one that changes the weights. Ex. 7.2: the 9-point circle used in `shapes.ts`.
- W. Boehm, "Inserting new knots into B-spline curves", *Computer-Aided Design* 12(4), 199–201, 1980. https://doi.org/10.1016/0010-4485(80)90154-2
- R. R. Patterson, "Projective transformations of the parameter of a Bernstein–Bézier curve", *ACM Transactions on Graphics* 4(4), 276–290, 1985. https://doi.org/10.1145/6116.6118
- E. T. Y. Lee, M. L. Lucian, "Möbius reparametrizations of rational B-splines", *Computer Aided Geometric Design* 8(3), 213–215, 1991. https://doi.org/10.1016/0167-8396(91)90004-U
- P. Cignoni, C. Rocchini, R. Scopigno, "Metro: measuring error on simplified surfaces", *Computer Graphics Forum* 17(2), 167–174, 1998. https://doi.org/10.1111/1467-8659.00236 (sample one surface, distance to the nearest point of the other mesh, both directions, mean and Hausdorff: the same idea as ADR 0002).
- N. Aspert, D. Santa-Cruz, T. Ebrahimi, "MESH: Measuring errors between surfaces using the Hausdorff distance", *Proc. IEEE ICME* 2002, 705–708. https://doi.org/10.1109/ICME.2002.1035879
- G. Elber, T. Grandine, "Hausdorff and minimal distances between parametric freeforms in R² and R³", *Geometric Modeling and Processing (GMP 2008)*, LNCS 4975, 191–204. https://doi.org/10.1007/978-3-540-79246-8_15 (exact Hausdorff distance between parametric surfaces, not only meshes).
- J. Hoschek, "Intrinsic parametrization for approximation", *Computer Aided Geometric Design* 5(1), 27–31, 1988. https://doi.org/10.1016/0167-8396(88)90017-9 (a point-to-point parametric error overestimates the geometric error; correct the parameters).
- W. Wang, H. Pottmann, Y. Liu, "Fitting B-spline curves to point clouds by curvature-based squared distance minimization", *ACM Transactions on Graphics* 25(2), 214–238, 2006. https://doi.org/10.1145/1138450.1138453 (point distance vs. distance to the curve: the tangential part slows the fit down).
- In this repo: [ADR 0001](../adr/0001-layout-per-level.md), [ADR 0002](../adr/0002-shape-distance-point-to-triangle.md), [src/shapeDistance.ts](../../src/shapeDistance.ts), [src/samples.ts](../../src/samples.ts), [src/grading.ts](../../src/grading.ts), [src/shapes.ts](../../src/shapes.ts), [src/target.ts](../../src/target.ts).

### What ADR 0002 said, and whether it still holds

ADR 0002: "Comparing parameters breaks down as soon as the same shape has a different form: knot insertion adds control points, and multiplying all weights by the same factor changes nothing." Considered option: "Parameter comparison as the grade: exact and cheap, but only defined while the Layout stays the same."

That reasoning is about comparing *control points, weights and knots*. It does not apply directly to the teacher's proposal, which compares *surface points*. Uniform weight scaling gives exactly 0 (measured), and knot insertion keeps S(u, v) the same (§5.2). The proposal still fails for other reasons: collapsed cap rows, Möbius weights and points sliding along the surface (measured above). If the team decides, a new ADR should record these reasons, so the question does not come back.
