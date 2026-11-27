# How Shape Match grades your surface

This page explains the grader: how the game decides how close your surface is to the Target, and how that becomes Stars. Terms are as in [GLOSSARY.md](../GLOSSARY.md).

## In one paragraph

Only the **shape** counts. First we turn and move your surface so that it lies on the Target as well as possible (the **Best fit**). Then, for every point of your surface, we look for the nearest point on the Target, and the other way round. The average gap, compared with the size of the Target, is your **Shape distance**. A small Shape distance gives Stars. Two surfaces that look the same get the same grade, however different their control points, weights or knots are.

## Step by step

### 1. Both surfaces as grids of points

A NURBS surface is a function S(u, v): every pair of parameters (u, v) between 0 and 1 gives one point in 3D. We evaluate both surfaces on a grid of 64 × 64 values of (u, v). The knots are added to the grid too, so sharp edges always get points. Each point counts with the area of the surface it stands for, so a dense part of the grid doesn't count more than it should.

### 2. The Best fit: where it stands and how it is turned don't count

Before measuring, your surface is turned and moved onto the Target. A correct shape that stands in the wrong place, or is turned, still counts as correct. Its size still counts, because the Best fit does not scale.

**First guess: the same grid point.** In every Level your surface and the Target have the same **Layout** (the same number of control points, degrees and knots, [ADR 0001](adr/0001-layout-per-level.md)), so point (i, j) of your surface usually belongs to point (i, j) of the Target. For such pairs the best turn and move have an exact formula, no search and no machine learning:

1. **Centres.** The centre of each surface: the average of its grid points, weighted by area.
2. **Turn.** Horn's method (1987) puts the pairs into a 4 × 4 matrix. Its largest eigenvector is the best rotation, written as a quaternion.
3. **Move.** The centre of your surface goes onto the centre of the Target.

**Refinement: the nearest point (ICP).** The pairs of the first guess depend on how your control points are spread along the surface. If you slid a ring of the Vase a little along its profile, the surface looks the same, but the pairs no longer belong together, and the first guess is slightly off. So we repeat 8 times: pair every point of your surface with its *nearest* point on the Target, and compute the best fit of these pairs with the same formula. This is the Iterative Closest Point method (Besl and McKay, 1992). Each step removes about a third of what is left of the error, so after 8 steps the fit depends on the shape only.

The Hand-in animation shows this step.

### 3. The Shape distance

After the Best fit, every point of your surface gets its distance to the nearest point of the Target, and every point of the Target its distance to your surface. Both directions are needed: one direction alone would not notice a part of the Target you left out. We combine them into two numbers:

- **RMS:** the average gap, as a root mean square, each point weighted by its area.
- **Max:** the single largest gap (the Hausdorff distance). It catches one wrong corner that could hide in the average.

Both are divided by the diagonal of the Target's bounding box. So 1 % means "1 % of the Target's size", for small and large shapes alike.

### 4. From distance to Stars

| | Rule (τ = 0.2 % of the diagonal) |
|---|---|
| ★★★ | RMS ≤ τ |
| ★★ | RMS ≤ 3 τ = 0.6 % |
| ★ | RMS ≤ 10 τ = 2 % |
| Passed | at least ★, and Max ≤ 10 τ_H |

The **Tolerance τ** is 0.2 % of the Target's size, or three times the **Noise floor** if that is larger. The Noise floor is what the grid itself costs: the Target against itself on a grid shifted by half a cell. τ_H is the same for the Max. The Weights Level has stricter Stars (★★ ≤ 1.5 τ, ★ ≤ 3 τ), because there even the start is close to round.

The **Match** on the meter shows the same Shape distance as a percentage: 0 % is where you started, 100 % is within τ. In between it uses a log scale, so each Star is a visible step.

### 5. The Hand-in

Live grading runs the same steps every time you change the surface, so the meter and the Stars are always up to date. **Hand in** measures once more and shows it as an animation (about 3 seconds):

1. **Moving the shapes together:** your surface and the Target slide together and meet in the middle. The Target turns see-through.
2. **Finding the best fit:** your surface turns and moves into its Best fit.
3. **Match:** the result, then the Result screen with Passed or Not passed.

Only the drawing moves. Your control points stay as they are, and "Back to the Workbench" puts both surfaces back in their places.

## What runs on the GPU

Everything except the last few numbers. One submission to the GPU per measurement:

| Pass | Shader | What it does |
|---|---|---|
| 1 | `tessellate.wgsl` | Evaluates both surfaces on the grid (the Target only once per Level) |
| 2 | `paired.wgsl` | The area each grid point stands for |
| 3 | `align.wgsl` | First guess: centres, cross-covariance and Horn's rotation from the same grid points, in one workgroup of 256 threads |
| 4 | `transform.wgsl` | Moves your surface's grid points by the Best fit |
| 5 – 28 | `distance.wgsl`, `align.wgsl`, `transform.wgsl` | 8 refinements: nearest Target point of every moved point, the Best fit to those points, move again |
| 29 | `distance.wgsl` | Both directions of the Shape distance, nearest point on the triangles of the other surface |
| 30 | `reduce.wgsl` | Adds up the gaps: weighted squares, weights, largest gap |

Only 6 sums, the rotation and the two centres go back to the CPU. The same per-point gaps colour the heatmap on your surface. A measurement takes about 25 ms round trip. The animation also runs on the GPU, as one model matrix per surface in `surface.wgsl`.

## The simpler grader: the same grid point

Our teacher suggested a simpler idea, the **Paired distance**: compare point (i, j) of your surface directly with point (i, j) of the Target, on a 128 × 128 grid, with no search for the nearest point. It is easy to explain and very fast (about 1.5 ms). It is still used for the first guess of the Best fit, and it can be switched on as the grader in the Developer section.

As the grader it is too strict. It compares *where the grid points are*, not *what the surface looks like*. Two surfaces can look the same while their points are spread differently along them, and then it says "off":

- **A ring slid along the profile.** A Vase whose bulge ring sits 0.1 lower along the bulge's straight side looks almost the same, but the Paired distance gives it 1★ (67 % Match), against 2★ (85 %) with the nearest point.
- **Hidden points.** The Star, the House and the Car close their ends with a centre point that can sit anywhere inside the end face without changing the shape.
- **Unusual weights.** Some weight changes give the same circle with the points spread differently (a Möbius reparametrization).

The Paired distance never passes a wrong shape: the gap between two paired points is never smaller than the real gap between the surfaces. But a grade should not depend on how the control points are spread, so the game grades with the nearest point.

## Try it yourself

- **Developer → Grader** switches live between "Nearest point, lined up" (the default), "Nearest point", "Same grid point, lined up" and "Same grid point".
- `DEFAULT_METRIC` and `DEFAULT_ALIGN` in `src/grader.ts` set the grader for the whole app.
- **Developer → Copy debug report** includes the grader and the last measurement.

## Code

| File | |
|---|---|
| `src/shapeDistance.ts` | `shapeDistance` (with `align`: Best fit, refinements, nearest point) and `pairedDistance`: the graders on the CPU, the reference for the GPU |
| `src/alignment.ts` | The Best fit of a set of pairs on the CPU (Horn's method, Jacobi eigenvalues) |
| `src/errorPass.ts` | Runs the GPU passes and reads back the result |
| `src/grading.ts` | From distance to Stars, Passed and Match |
| `src/grader.ts` | Live grading and the Shape match check of a Hand-in |
| `src/matchShow.ts` | The Hand-in animation |

## Sources

- B. K. P. Horn, "Closed-form solution of absolute orientation using unit quaternions", *Journal of the Optical Society of America A* 4(4), 1987.
- P. J. Besl, N. D. McKay, "A method for registration of 3-D shapes", *IEEE Transactions on Pattern Analysis and Machine Intelligence* 14(2), 1992 (ICP).
- W. Kabsch, "A solution for the best rotation to relate two sets of vectors", *Acta Crystallographica A* 32, 1976.
- C. Ericson, *Real-Time Collision Detection*, Morgan Kaufmann 2005, §5.1.5 (nearest point on a triangle).
- L. Piegl, W. Tiller, *The NURBS Book*, 2nd ed., Springer 1997: §4.2 (rational surfaces), §5.2 (knot insertion keeps the parametrization), §6.4 (reparametrization through weights).
- Research notes behind this design: [docs/research/shape-distance-parameter-wise.md](research/shape-distance-parameter-wise.md) and [docs/research/shape-distance-alignment.md](research/shape-distance-alignment.md). [ADR 0002](adr/0002-shape-distance-point-to-triangle.md) explains the nearest-point distance.
