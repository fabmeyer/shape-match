# How Shape Match grades your surface

This page explains the grader: how the game decides how close your surface is to the Target, and how that becomes Stars. Terms are as in [GLOSSARY.md](../GLOSSARY.md).

## In one paragraph

We put the same grid of 128 × 128 points on your surface and on the Target. Point (i, j) of your surface belongs to point (i, j) of the Target. First we turn and move your surface so that it lies on the Target as well as possible (the **Best fit**). Then we measure how far apart each pair of points is. The average gap, compared with the size of the Target, is your **Shape distance**. A small Shape distance gives Stars.

## Step by step

### 1. The same grid on both surfaces

A NURBS surface is a function S(u, v): every pair of parameters (u, v) between 0 and 1 gives one point in 3D. We evaluate both surfaces at the same (u, v) values, 128 per direction. The knots are added to the grid too, so sharp edges always get points.

This works because in every Level your surface and the Target have the same **Layout**: the same number of control points, the same degrees and the same knots ([ADR 0001](adr/0001-layout-per-level.md)). So the same (u, v) means "the same place" on both surfaces.

```
d_ij = | S_Target(u_i, v_j) − S_Learner(u_i, v_j) |
```

This is the **Paired distance**, the idea our teacher suggested. It needs no search for the nearest point, so it is simple and fast.

### 2. The Best fit: only the shape counts

Before measuring, your surface is turned and moved so that it lies on the Target as well as possible. A correct shape that stands in the wrong place, or is turned, still counts as correct. Its size still counts, because the Best fit does not scale.

You might expect this to need a search or machine learning. It doesn't. Because we know which point belongs to which, the best turn and move have an exact formula:

1. **Centres.** Find the centre of each surface: the average of its grid points (each point weighted by the area it covers).
2. **Turn.** Find the best rotation around those centres. Horn's method (1987) puts the pairs into a 4 × 4 matrix. Its largest eigenvector is the best rotation, written as a quaternion.
3. **Move.** Put the centre of your surface onto the centre of the Target.

The Hand-in animation shows exactly this step.

### 3. The Shape distance

After the Best fit we measure all pairs again and combine them into two numbers:

- **RMS:** the average gap, as a root mean square. Each pair counts with the area its Target point covers, so a dense region of the grid doesn't count more than it should.
- **Max:** the single largest gap. It catches one wrong corner that could hide in the average.

Both are divided by the diagonal of the Target's bounding box. So 1 % means "1 % of the Target's size", for small and large shapes alike.

### 4. From distance to Stars

| | Rule (τ = 0.2 % of the diagonal) |
|---|---|
| ★★★ | RMS ≤ τ |
| ★★ | RMS ≤ 3 τ = 0.6 % |
| ★ | RMS ≤ 10 τ = 2 % |
| Passed | at least ★, and Max ≤ 10 τ = 2 % |

The **Tolerance τ** is 0.2 % of the Target's size. A surface compared with itself gives exactly 0, so no Noise floor is needed. The Weights Level has stricter Stars (★★ ≤ 1.5 τ, ★ ≤ 3 τ), because there even the start is close to round.

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
| 1 | `tessellate.wgsl` | Evaluates both surfaces on the 128 grid (the Target only once per Level) |
| 2 | `paired.wgsl` | The area each grid point covers |
| 3 | `align.wgsl` | Centres, cross-covariance and Horn's rotation, in one workgroup of 256 threads |
| 4 | `paired.wgsl` | The gap of every pair, with your surface turned and moved by the Best fit |
| 5 | `reduce.wgsl` | Adds up the gaps: weighted squares, weights, largest gap |

Only 6 sums, the rotation and the two centres go back to the CPU. The same per-point gaps colour the heatmap on your surface. A measurement takes about 1.5 ms round trip. The animation also runs on the GPU, as one model matrix per surface in `surface.wgsl`.

## What the grader cannot see

The Paired distance compares *where the grid points are*, not only *what the surface looks like*. Two surfaces can look the same while their grid points are spread differently along them. Then the grader says "off" although the shape is right:

- **Hidden points.** The Tent and the House close their ends with a centre point that can sit anywhere inside the end face without changing the shape. If you leave it where the start had it, the Tent counts as about 2 % off instead of 0.
- **Unusual weights.** Some weight changes give the same circle with the points spread differently (a Möbius reparametrization). A correct pipe can then count as off.

The opposite never happens: a wrong shape is never graded as right. The gap between two paired points is never smaller than the real gap between the surfaces. The Best fit doesn't help with these two cases: it can turn and move a surface, but it can't slide points along it.

The earlier grader, **Nearest point**, searches for the closest point on the other surface instead of pairing ([ADR 0002](adr/0002-shape-distance-point-to-triangle.md)). It doesn't have this weakness, but it is harder to explain and much slower: 65 to 120 million point-to-triangle tests per measurement on a 64 grid, against about 20,000 pairs here.

## Try it yourself

- **Developer → Grader** switches live between "Nearest point", "Same grid point" and "Same grid point, lined up" (the default).
- `DEFAULT_METRIC` and `DEFAULT_ALIGN` in `src/grader.ts` set the grader for the whole app.
- **Developer → Copy debug report** includes the grader and the last measurement.

## Code

| File | |
|---|---|
| `src/shapeDistance.ts` | `pairedDistance`: the grader on the CPU, the reference for the GPU |
| `src/alignment.ts` | The Best fit on the CPU (Horn's method, Jacobi eigenvalues) |
| `src/errorPass.ts` | Runs the GPU passes and reads back the result |
| `src/grading.ts` | From distance to Stars, Passed and Match |
| `src/grader.ts` | Live grading and the Shape match check of a Hand-in |
| `src/matchShow.ts` | The Hand-in animation |

## Sources

- B. K. P. Horn, "Closed-form solution of absolute orientation using unit quaternions", *Journal of the Optical Society of America A* 4(4), 1987.
- W. Kabsch, "A solution for the best rotation to relate two sets of vectors", *Acta Crystallographica A* 32, 1976.
- L. Piegl, W. Tiller, *The NURBS Book*, 2nd ed., Springer 1997: §4.2 (rational surfaces), §5.2 (knot insertion keeps the parametrization), §6.4 (reparametrization through weights).
- Research notes behind this design: [docs/research/shape-distance-parameter-wise.md](research/shape-distance-parameter-wise.md) and [docs/research/shape-distance-alignment.md](research/shape-distance-alignment.md).
