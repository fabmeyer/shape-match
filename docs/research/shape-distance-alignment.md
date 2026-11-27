# Shape distance: align the two surfaces first, then compare?

Status: research note, 27.11.2026. Terms as in [GLOSSARY.md](../../GLOSSARY.md). Nothing in the code or the ADRs was changed. Builds on [shape-distance-parameter-wise.md](shape-distance-parameter-wise.md) (the Paired distance); its findings are not repeated here.

## Question

The idea: "An ML model tries to match (move) the two 3D shapes over each other, finds the best position to superimpose them, then uses the NURBS parameters to calculate the similarity."

So two steps:

1. **Align.** Move the Learner surface onto the Target with a rigid transform (rotation + translation) or a similarity transform (+ one uniform scale factor).
2. **Compare.** Either control point k against control point k (the Layout is shared, [ADR 0001](../adr/0001-layout-per-level.md)), or sample (i, j) against sample (i, j) (the Paired distance, `pairedDistance` in [src/shapeDistance.ts](../../src/shapeDistance.ts)).

Today neither metric aligns. The Target is drawn `TARGET_OFFSET` to the right, but both are compared in the same unshifted coordinates ([src/target.ts](../../src/target.ts)). So the learner must build the shape at the right place, size and orientation.

## Short answer

- **No ML is needed.** Because the Layout is shared, we already know which point belongs to which. With known pairs, the best rotation, translation and scale have a closed-form solution: Kabsch (1976), Horn (1987), Umeyama (1991). It is a 3 × 3 matrix built in one pass over the points, plus one small eigen- or singular value problem. Learned registration (PointNetLK, Deep Closest Point) exists to *find* pairs when they are unknown. Deep Closest Point even ends with the same SVD step (Wang & Solomon 2019). We have the pairs, so a network adds training data, randomness and a black box to a grading tool, and gains nothing.
- **Alignment works as promised.** For all 14 Levels, the Target moved by 30° and shifted goes from 25–39 % Paired RMS to below 1e-8 after alignment. With an extra scale of 1.2, rigid alignment leaves 4.5–6.9 % (0★). Similarity alignment brings it below 1e-8 again (measured below).
- **It does not change any start's grade.** Every start stays at 0★ with every variant. Alignment shrinks the start's distance by up to 61 % (Mushroom: Paired 27.0 % → 10.5 %; Star 31.3 % → 13.1 %), mostly by lifting the flattened start up in z. A similarity cannot undo the starts' squash, stretch or lean, because these are non-uniform scales and a shear.
- **It does not fix the Paired distance's real problem.** The Tent with its cap points at the start's place is the same shape. Paired: 1.95 %, after alignment still 1.80 % (0★). Alignment moves the whole surface. The error there is sliding *along* the surface, and no rigid move removes that.
- **Control points are the wrong thing to compare.** In the Weights Level the start has the same control point positions as the Target. Only the weights differ. A control-point distance gives 0 (3★) to the unchanged start. Adding the weights (homogeneous coordinates) catches that start, but then a correct pipe with other weights gets 6.7 %, and all weights × 2 (same surface) gets 55 %.
- **The real question is about the game, not the maths.** With alignment, a learner who builds the right shape at the wrong place or size gets 3★. Decide first whether place and size are part of the Task. If they are not, use Umeyama in closed form on the paired surface samples. That costs about 0.2 ms on the CPU.

## How it would work (student explanation)

**Step 1: put your surface on top of the Target.** Every dot on your surface has a partner dot on the Target: the dot with the same (u, v). Picture each pair joined by a rubber band. We let your surface go and the bands pull it. It slides, turns and (if allowed) grows or shrinks until the bands are as relaxed as possible. "As relaxed as possible" means: the sum of the squared band lengths is smallest.

There is a formula for this end position. You do not need to try positions one by one. (1) Move both dot clouds so their centres sit at the origin. That is the translation. (2) Find the rotation that lines up the clouds best. That is one 3 × 3 matrix and its SVD, or one 4 × 4 matrix and its largest eigenvector. (3) If scale is allowed, it is the ratio of how far the clouds spread after the rotation.

**Step 2: measure.** Now compare dot (i, j) of your moved surface with dot (i, j) of the Target, as in the Paired distance. Average gap (RMS) and largest gap, relative to the Target's size.

What the learner should know: "We ignore *where* you built it and *how it is turned* (and *how big*, if scale is on). We only grade the form."

## Does it need ML?

### With known pairs: closed form

The problem is: find R (rotation), t (translation) and s (scale) that make the weighted sum Σ w_k |s R p_k + t − q_k|² smallest. Here p_k is a point of the Learner surface, q_k its partner on the Target, and w_k a weight per pair.

- **Kabsch (1976)** solved the rotation part for crystallography: rotate one set of vectors onto another, with weights. The solution comes from the eigenvectors of a 3 × 3 matrix built from the two point sets. The 1976 version can return a mirror image instead of a rotation. Kabsch (1978) adds the fix.
- **Horn (1987)** gives a closed form with unit quaternions. The best rotation is the eigenvector of the largest eigenvalue of a symmetric 4 × 4 matrix N, whose entries are sums of products of the centred coordinates. A unit quaternion is always a proper rotation, so mirror images cannot happen. Horn also derives translation (centroid difference) and scale. He notes that the simple scale formula is not symmetric (aligning p onto q is not the inverse of q onto p) and gives a symmetric version.
- **Arun, Huang & Blostein (1987)** do the same with the SVD of the 3 × 3 covariance matrix.
- **Umeyama (1991)** gives the complete least-squares solution for rotation, translation and scale, and shows how to correct the SVD so that it never returns a reflection.

These are the "orthogonal Procrustes" problem (Schönemann 1966). The cost is one pass over the N pairs to sum the centroids and the 3 × 3 covariance, then a fixed-size 3 × 3 or 4 × 4 problem. So it is O(N). It has one answer (except in symmetric cases, below), no starting guess and no randomness.

The script for this note implements Horn's method with a Jacobi eigen decomposition of N (about 40 lines, no dependency). Umeyama's scale is s = Σ w (q′ · R p′) / Σ w |p′|², where ′ means centred.

### Without pairs: ICP

If the two surfaces had different Layouts (for example after knot insertion, #19), the pairs would be unknown. The classic answer is the Iterative Closest Point method (Besl & McKay 1992; Chen & Medioni 1992). Pair each point with its nearest point on the other surface, solve the closed form above, and repeat. Besl & McKay prove it converges monotonically, but only to a *local* minimum. It needs a good starting position. In this game it would have one: the learner starts at the Target's place. The nearest-point search is the same brute force the current Shape distance already does ([ADR 0002](../adr/0002-shape-distance-point-to-triangle.md)). Chen & Medioni minimise the distance to the tangent plane, not to the point, which converges faster on smooth surfaces.

Note: knot insertion keeps S(u, v) the same (The NURBS Book §5.2), so even after #19 the pairs on a (u, v) grid stay known, as the earlier note explains. ICP is not needed in this game as it is.

### Learned registration

- **PointNetLK** (Aoki et al. 2019) runs the Lucas–Kanade image alignment idea on PointNet features of two point clouds. It is trained on ModelNet40 and iterates to a pose.
- **Deep Closest Point** (Wang & Solomon 2019) learns point features, uses attention to build *soft pairs*, and then solves the pose with the **SVD step of Kabsch/Umeyama**.

Both learn the part we already have: which point belongs to which. What a network would add here:

| Concern | Closed form (Umeyama/Horn) | Learned (PointNetLK, DCP) |
|---|---|---|
| Training data | none | a training set of surface pairs; ours would have to be made up |
| Result | the exact least-squares optimum | an estimate, depends on training; DCP's last step is the closed form anyway |
| Same input, same grade | yes | only with fixed weights and arithmetic; any retraining changes grades |
| Explaining a grade to a student | "the rubber band position", one formula | "the network said so" |
| Browser runtime | about 0.2 ms CPU, or one GPU reduction | needs a model download and an ML runtime (WebGPU or WASM) |

A grade must be repeatable and explainable. A network would make it neither, to solve a problem the Layout has already solved.

## What alignment changes for the game

### What it removes from the Task

The Glossary says a learner must "reproduce" the Target. Today that includes place, orientation and size. With rigid alignment, place and orientation no longer count. With similarity alignment, size no longer counts either.

In statistical shape analysis, this is exactly the definition of shape: "all the geometrical information that remains when location, scale and rotational effects are filtered out" (Kendall 1977, as quoted in Dryden & Mardia 2016). So a similarity-aligned distance is a "shape" distance in the strict sense. The game's current Shape distance is closer to a "form" or "fit" distance.

Whether that is good depends on what a Level teaches:

- **Against alignment.** The Orders are physical objects (a house for a model railway, a toy car). A house twice as big does not fit the railway. In Level 1 the lesson is local support: "which point, and how far?" With alignment, a bump that is a little too low can be partly hidden by lifting and tilting the whole sheet. Measured: Level 1's start goes from 3.81 % to 3.18 % just by lifting it 0.054 and tilting it 2.4°. The learner never moved a point.
- **For alignment.** NURBS lessons are about form: weights make circles, knots make creases, degree 1 makes straight edges. Moving all control points to place the object is tedious and teaches nothing new. Moving all points by one vector moves the surface by it (affine invariance), and that is already known after Level 1.
- **Middle way.** Only rigid alignment, so size still counts. Or align only in z, which is what alignment mostly does for the starts anyway (see the translations below).

### Does it change the grades of the starts?

No. Every start stays at 0★ with every variant (table below). What the alignment does to the starts (Horn, similarity, on the 128 paired samples):

- For the flattened shapes it mostly translates in z: Star +0.78, Ball +0.69, Tree +0.55, Vase +0.53. It also shrinks a little, s = 0.89–1.05. The squashed start is lifted to sit in the middle of the Target's height.
- For the leaning shapes it also rotates: Mushroom 9.9°, Car 6.5°, Boat 4.8°. A rotation partly undoes a lean, but a shear is not a rotation.
- Squash in z, stretch in x and narrowing in y are non-uniform scales. A similarity has only one scale factor, so it cannot undo them. The starts stay far off: 10.4–16.2 % similarity + Paired for the library shapes, 2.4–3.2 % for the open sheets (Level 1, Leaf, Superposition), 1.56 % for the pipe.

One side effect: the Match meter starts at the start's distance ([src/grading.ts](../../src/grading.ts), `matchPercent`). With alignment that distance is smaller, so the meter scale changes. That is automatic, because `calibrate` measures the start with the same metric.

### Degenerate and ambiguous alignments

- **Symmetric Targets (Ball, Pipe, Vase, revolutions).** Turning a ball about its axis gives the same surface. Rotation is then ambiguous for a geometric match. But with *paired* samples it is not: sample (i, j) has a fixed partner, so there is one best rotation. Measured: the Ball's Target turned 40° about z has Paired RMS 16.1 % (same surface!), and the alignment recovers exactly 40.0°, then 0. So for the Paired distance, alignment even removes a false error. A learner would rarely do this, though: they would have to move every point one step around the circle.
- **Nearly flat starts (Level 1, Superposition, Leaf).** A flat sheet is still a 2D point cloud, so the rotation is defined. Horn's 4 × 4 matrix has a unique largest eigenvalue as long as the points do not lie on a line. Measured on Level 1: a tilt of 2.4°, no jump. Trouble starts only when the learner surface collapses to a line or a point. Then the scale formula divides by zero (Σ |p′|² = 0). The code needs a guard (s = 1 if the spread is tiny).
- **Direction of the transform.** Always move the Learner surface onto the Target, never the other way. The distance is divided by the Target's diagonal. If the Target were scaled to a tiny Learner surface, the gaps would shrink with it and a tiny wrong surface would score well. Horn's symmetric scale (Horn 1987) avoids the asymmetry. It is not needed if the direction is fixed.

### A trap: paired alignment can make a correct surface look wrong

If the alignment is computed from paired samples but the grade uses the nearest-triangle Shape distance, a correct shape with a different net gets *worse*. Measured: the Tent with the cap at the start's place has Shape distance 0. After the paired alignment, the same surface has 0.51 % (2★). The House goes from 0 to 1.27 % (0★). The wrong pairs at the cap pull the whole surface away. So: pair the alignment with the metric it is computed for. For "nearest", alignment would have to be ICP-style (nearest-point pairs).

## Measured

Script: a scratch script (since deleted) that, for every Task in `allTasks()`, computes:

- **Nearest:** `shapeDistance` (point to triangle, both ways) at res 32 (the game uses 64; 32 for speed, so the tiny residues are larger than in the earlier note).
- **Paired:** `pairedDistance` at 128.
- **Rigid + Paired / Sim + Paired:** Horn's quaternion method on the 128 paired samples, weighted with the Target's sample areas, without and with Umeyama's scale; then the Paired RMS and max of the moved Learner surface.
- **Sim CP:** control points k ↔ k (Cartesian, equal weight per point), similarity-aligned among themselves, RMS.
- **CP homog.:** the same, but in homogeneous coordinates (w x, w y, w z, w), so weights count.

All divided by the Target's bounding-box diagonal. Stars use [src/grading.ts](../../src/grading.ts) with each Level's overrides and τ = 0.2 % (the value every Level has today, see the earlier note).

**Target vs start** (every cell 0★):

| Level | Nearest RMS | Paired RMS | Rigid + Paired | Sim + Paired (s) | Sim CP | CP homog. |
|---|---|---|---|---|---|---|
| Level 1 | 3.31 % | 3.81 % | 3.18 % | 3.18 % (1.00) | 6.88 % | 6.88 % |
| Weights | 1.15 % | 1.85 % | 1.85 % | 1.56 % (0.97) | **0** | 13.2 % |
| Leaf | 3.62 % | 3.81 % | 2.44 % | 2.44 % (1.00) | 5.19 % | 5.19 % |
| Tent | 9.85 % | 14.9 % | 11.3 % | 10.8 % (0.90) | 10.7 % | 10.7 % |
| Star | 22.6 % | 31.3 % | 13.3 % | 13.1 % (0.89) | 12.5 % | 12.5 % |
| Ball | 18.3 % | 25.5 % | 13.5 % | 13.3 % (0.93) | 20.2 % | 15.4 % |
| Vase | 17.1 % | 24.4 % | 14.6 % | 14.5 % (1.03) | 18.8 % | 16.4 % |
| House | 17.6 % | 23.9 % | 16.2 % | 16.2 % (0.97) | 13.8 % | 13.8 % |
| Tree | 15.4 % | 21.9 % | 10.9 % | 10.9 % (0.99) | 16.0 % | 14.0 % |
| Cup | 17.1 % | 23.5 % | 13.1 % | 13.0 % (0.97) | 18.4 % | 16.1 % |
| Mushroom | 19.8 % | 27.0 % | 10.7 % | 10.5 % (0.93) | 14.4 % | 12.5 % |
| Boat | 11.2 % | 16.6 % | 10.5 % | 10.4 % (1.05) | 10.7 % | 10.7 % |
| Car | 12.3 % | 18.2 % | 11.4 % | 11.4 % (1.04) | 9.94 % | 9.51 % |
| Superposition | 3.14 % | 3.34 % | 3.11 % | 3.11 % (1.00) | 7.05 % | 7.05 % |

Two things stand out. In the Weights Level, the control-point distance is 0 for a surface that is visibly not round. And the control-point distance does not rank the starts like the surface does: Level 1 is 2.1 × its Nearest value, Star only 0.56 ×.

**Target vs the Target moved** (same shape; should be 0 after alignment). Rotation 30° about (1, 2, 3), shift (0.5, −0.3, 0.2), and in the second row also scale 1.2. Ranges over all 14 Levels:

| Case | Nearest RMS | ★ | Paired RMS | ★ | Rigid + Paired | ★ | Sim + Paired | ★ | Sim CP |
|---|---|---|---|---|---|---|---|---|---|
| moved | 12.7–20.2 % | 0 | 25.2–38.5 % | 0 | < 1e-8 | 3 | < 1e-8 | 3 | < 2e-8 |
| moved + scale 1.2 | 13.5–22.5 % | 0 | 26.6–42.1 % | 0 | 4.5–6.9 % | 0 | < 1e-8 (s = 0.83 = 1/1.2) | 3 | < 2e-8 |

**Same shape, other control net** (should be 0 / 3★):

| Level | Case | Nearest RMS | ★ | Paired RMS | ★ | Sim + Paired | ★ | Sim + Paired, then Nearest | ★ | Sim CP | CP homog. |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Tent | cap points at the start's place | 4e-13 | 3 | 1.95 % | 0 | 1.80 % | 0 | 0.51 % | 2 | 5.09 % | 5.09 % |
| House | cap points at the start's place | 4e-13 | 3 | 4.54 % | 0 | 3.97 % | 0 | 1.27 % | 0 | 9.27 % | 9.27 % |
| Weights | Möbius weights in u, c = 1.1 | 0.078 % | 3 | 0.89 % | 0 | 0.89 % | 0 | 0.079 % | 3 | 2e-8 | 6.74 % |
| Ball | Möbius weights in u, c = 1.1 | 0.054 % | 3 | 0.67 % | 1 | 0.67 % | 1 | 0.054 % | 3 | 1e-8 | 5.35 % |
| Ball | Target turned 40° about z | 0.099 % | 3 | 16.1 % | 0 | 0 (40.0° found) | 3 | – | – | – | – |
| Weights, Ball, Car | all weights × 2 | 0 | 3 | 0 | 3 | 0 | 3 | 0 | 3 | 0 | 43–55 % |

Alignment helps only when the difference is a rigid move of the *whole* surface. Sliding along the surface (cap points, Möbius weights) stays.

**Runtime of the alignment step** (Node 26, f64, Apple Silicon, average of 2 000 runs). Horn's method on the paired samples, including centroids, covariance and the 4 × 4 Jacobi:

| | Samples / points | Alignment | For comparison: `pairedDistance` incl. evaluating both surfaces |
|---|---|---|---|
| Level 1, 128 grid | 16 900 | 0.18 ms | 16.9 ms |
| Car, 128 grid | 20 328 | 0.21 ms | 17.5 ms |
| Level 1, control points | 36 | 0.8 µs | – |
| Car, control points | 116 | 1.5 µs | – |

The alignment is about 1 % of the Paired distance's CPU time. The time goes into evaluating the surfaces, not into the alignment.

## Cost

- **Arithmetic.** One pass over N pairs: 17 sums (Σ w, Σ w p, Σ w q, Σ w p qᵀ (9 values), Σ w |p|²). Then a fixed 4 × 4 eigenproblem (Horn) or 3 × 3 SVD (Umeyama). Then one more pass to compute the distances of the moved points. O(N), like the Paired distance itself.
- **On the GPU.** The samples already live on the GPU ([src/shaders/paired.wgsl](../../src/shaders/paired.wgsl), [src/errorPass.ts](../../src/errorPass.ts)). Two options:
  1. Add a reduction pass in the style of [src/shaders/reduce.wgsl](../../src/shaders/reduce.wgsl) that sums the 17 values. Read them back (68 bytes), solve the 4 × 4 on the CPU, and upload R, s, t as a uniform for the paired pass. That adds one GPU → CPU → GPU round trip per measurement. Use the uncentred sums with S = Σ w p qᵀ − W p̄ q̄ᵀ. In f32, centre the coordinates on the Target's bounding-box centre first, to keep the cancellation small.
  2. Solve the 4 × 4 in WGSL in a single invocation (a few Jacobi sweeps) and keep everything on the GPU. No extra round trip, more shader code.
- **On the CPU.** Simpler and fast enough: 0.2 ms for the samples. But it needs the samples on the CPU. Evaluating them there costs about 17 ms (above), which is too slow for live grading every frame, but fine for the Hand-in's Shape match check. Aligning only the control points (1.5 µs) is cheap, but it is the wrong correspondence (see above).
- **Noise floor and Tolerance.** Aligned Paired distance of the Target against itself is exactly 0 (measured < 1e-8, rounding only). So, like the Paired distance, it has no Noise floor and needs a fixed τ. The aligned distance is never larger than the unaligned one, because "no move" is one of the candidates. So alignment can only raise Stars. τ = 0.2 % stays a sensible choice. The f32 rounding of the alignment on the GPU (about 1e-7 relative) is far below τ.
- **Heatmap.** The heatmap colours the learner's own surface ([src/grader.ts](../../src/grader.ts), `heatSource`). After alignment the gaps belong to the *moved* surface. Showing them on the unmoved one is confusing. Either draw a ghost of the aligned Learner surface over the Target, or say clearly "colours after alignment".

## Recommendation and code changes

**Recommended: no ML. Keep grading without alignment for now. Decide first whether place and size are part of the Task. If alignment is wanted, use Horn/Umeyama in closed form on the paired samples.**

1. **Decide the game rule.** This is a design choice, not a technical one. The Orders are physical objects, the start is already at the Target's place, and the Target is drawn next to it. So today "build it here, at this size" is a fair rule. Alignment would make Level 1 easier in the wrong way (lift instead of shaping), and it does not help with the Paired distance's real failures (cap points, Möbius weights). If the team agrees, nothing changes in the code, and this note documents why.
2. **If alignment is wanted: rigid or similarity as an option on the Paired metric.** Do not make it a third Metric. It is a step before the comparison, not another comparison. Files:
   - [src/shapeDistance.ts](../../src/shapeDistance.ts): `align(learnerPoints, targetPoints, weights, { scale })` (Horn with Jacobi, about 40 lines, guard s = 1 when the learner's spread is tiny). `pairedDistance(target, learner, res, { align: 'none' | 'rigid' | 'similarity' })` as the CPU reference. Tests in [src/shapeDistance.test.ts](../../src/shapeDistance.test.ts): Target moved/turned/scaled → 0; start unchanged ≤ unaligned.
   - [src/shaders/reduce.wgsl](../../src/shaders/reduce.wgsl) or a new small shader: the 17 sums. [src/shaders/paired.wgsl](../../src/shaders/paired.wgsl): apply R, s, t from a uniform before the difference. [src/errorPass.ts](../../src/errorPass.ts): the extra pass and the uniform.
   - [src/grader.ts](../../src/grader.ts): a switch next to `DEFAULT_METRIC` (for example `DEFAULT_ALIGN: 'none'`), or per Level in [src/target.ts](../../src/target.ts) (`Task.grading`), so a Level can say "size counts here".
   - Heatmap: ghost of the aligned surface, or a label.
   - [GLOSSARY.md](../../GLOSSARY.md): a term for it (for example **Alignment**: moving the Learner surface onto the Target before measuring), and the Shape distance entry must say whether place, rotation and size count. A short ADR records the decision.
3. **Do not compare control points.** They are not on the surface. Cartesian control points miss every weight error: the Weights Level's start scores 0, i.e. 3★. Homogeneous control points catch weights, but fail correct surfaces (Möbius 6.7 %, all weights × 2 → 43–55 %), mix weights with lengths, and fail the cap points too (Tent 5.1 %). Control points stay useful for Hints (the Parameter comparison), as the Glossary already says.
4. **Do not combine paired alignment with the nearest metric.** It turns correct Tents and Houses into 2★ and 0★ (measured above). If "nearest" ever needs alignment, use ICP (Besl & McKay 1992), starting from the identity, since the learner starts at the Target's place.

**Options at a glance:**

| Variant | Place counts | Rotation counts | Size counts | Extra cost | Effect on the measured cases |
|---|---|---|---|---|---|
| Today (no alignment) | yes | yes | yes | – | moved Target 0★ |
| Paired + translation only | no | yes | yes | centroids only | – (not measured; the starts' main move is a z shift) |
| Paired + rigid (Horn/Kabsch) | no | no | yes | 0.2 ms CPU or one reduction | moved Target 3★, scaled 0★ |
| Paired + similarity (Umeyama) | no | no | no | same | moved and scaled Target 3★ |
| ML registration | no | no | depends | model, runtime, training | not justified |

## Sources

- W. Kabsch, "A solution for the best rotation to relate two sets of vectors", *Acta Crystallographica* A32(5), 922–923, 1976. https://doi.org/10.1107/S0567739476001873
- W. Kabsch, "A discussion of the solution for the best rotation to relate two sets of vectors", *Acta Crystallographica* A34(5), 827–828, 1978. https://doi.org/10.1107/S0567739478001680 (the fix for reflections)
- B. K. P. Horn, "Closed-form solution of absolute orientation using unit quaternions", *Journal of the Optical Society of America A* 4(4), 629–642, 1987. https://doi.org/10.1364/JOSAA.4.000629 (rotation from the largest eigenvector of a 4 × 4 matrix; translation; scale and its symmetric form)
- K. S. Arun, T. S. Huang, S. D. Blostein, "Least-squares fitting of two 3-D point sets", *IEEE TPAMI* 9(5), 698–700, 1987. https://doi.org/10.1109/TPAMI.1987.4767965 (SVD version)
- S. Umeyama, "Least-squares estimation of transformation parameters between two point patterns", *IEEE TPAMI* 13(4), 376–380, 1991. https://doi.org/10.1109/34.88573 (rotation, translation and scale; no reflections)
- P. H. Schönemann, "A generalized solution of the orthogonal Procrustes problem", *Psychometrika* 31(1), 1–10, 1966. https://doi.org/10.1007/BF02289451
- P. J. Besl, N. D. McKay, "A method for registration of 3-D shapes", *IEEE TPAMI* 14(2), 239–256, 1992. https://doi.org/10.1109/34.121791 (ICP; converges to a local minimum)
- Y. Chen, G. Medioni, "Object modelling by registration of multiple range images", *Image and Vision Computing* 10(3), 145–155, 1992. https://doi.org/10.1016/0262-8856(92)90066-C (point-to-plane ICP)
- Y. Aoki, H. Goforth, R. A. Srivatsan, S. Lucey, "PointNetLK: Robust & efficient point cloud registration using PointNet", *CVPR* 2019. https://arxiv.org/abs/1903.05711
- Y. Wang, J. M. Solomon, "Deep Closest Point: Learning representations for point cloud registration", *ICCV* 2019. https://arxiv.org/abs/1905.03304 (learned soft pairs, then the SVD step)
- I. L. Dryden, K. V. Mardia, *Statistical Shape Analysis*, 2nd ed., Wiley 2016, ch. 1 (Kendall's definition of shape: what remains after removing location, scale and rotation; D. G. Kendall, "The diffusion of shape", *Advances in Applied Probability* 9(3), 428–430, 1977).
- L. Piegl, W. Tiller, *The NURBS Book*, 2nd ed., Springer 1997. §5.2 (knot insertion keeps S(u, v)); affine invariance of NURBS (moving the control points by a rigid or similarity transform moves the surface the same way), which the script uses to move whole Targets.
- In this repo: [shape-distance-parameter-wise.md](shape-distance-parameter-wise.md), [ADR 0001](../adr/0001-layout-per-level.md), [ADR 0002](../adr/0002-shape-distance-point-to-triangle.md), [src/shapeDistance.ts](../../src/shapeDistance.ts), [src/grader.ts](../../src/grader.ts), [src/grading.ts](../../src/grading.ts), [src/target.ts](../../src/target.ts), [src/shapes.ts](../../src/shapes.ts).
