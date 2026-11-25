# Shape distance: point-to-triangle in both directions, Parameter comparison only for Hints

A Level is graded by a geometric Shape distance, not by comparing control points, weights and knots. Comparing parameters breaks down as soon as the same shape has a different form: knot insertion adds control points, and multiplying all weights by the same factor changes nothing. We also measure from each sample to the nearest *triangle* of the other surface instead of the nearest *sample*. With nearest samples, the same surface sampled twice has a Noise floor that grows with the sample spacing h; with triangles it falls to about curvature × h², at 2–3× the cost per test. Samples are weighted by area, so places where weights or knots bunch samples together don't count more. The result is the RMS, normalised by the Target's bounding-box diagonal, and solving a Level also needs the largest single distance (Hausdorff) under a looser bound.

## Considered options

- Point-to-point Chamfer distance (the original proposal): cheaper, but its Noise floor depends strongly on resolution, so no fixed Tolerance works.
- Parameter comparison as the grade: exact and cheap, but only defined while the Layout stays the same. Kept for Hints and as a reference in the evaluation (E3, E4).
