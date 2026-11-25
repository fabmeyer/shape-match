# Course test models

The course provides three NURBS test models with reference values of S(u, v): an **open patch**,
a **vase** and a **torus** (issue #7). They are still missing. Probably in the exercise handout
on OLAT or in the course repo.

Put each model here as one JSON file. Both test runs pick them up automatically:

- `npm test`: the CPU evaluation (`nurbs.ts`) must hit every reference value
- `npm run test:gpu`: the tessellation compute pass must match the CPU on the model

```json
{
  "name": "torus",
  "source": "where the file and its values come from (handout, page, version)",
  "tolerance": 1e-6,
  "net": {
    "countU": 9, "countV": 9, "p": 2, "q": 2,
    "knotsU": [0, 0, 0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1, 1, 1],
    "knotsV": [0, 0, 0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1, 1, 1],
    "points": [[1, 0, 0, 1], [1, 1, 0, 0.7071]]
  },
  "references": [
    { "u": 0.42, "v": 0.5, "point": [0.1, 0.2, 0.3] }
  ]
}
```

- `points`: the control points P(i, j) as **Cartesian** x, y, z plus the weight w, row by row:
  index `j * countU + i`, u runs along a row. (Inside the app they are stored as w·x, w·y, w·z, w.)
- `references`: parameter values and the expected surface point S(u, v).
- `tolerance` (optional, default 1e-6): the largest allowed distance from each reference point.

If the course files come in another format, convert them to this one (and note how in `source`).
