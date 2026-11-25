# The Layout belongs to the Level, not to the whole game

The first version required every Target to use the learner's 6 × 6 bicubic net with clamped uniform knots, so the GPU could share the parameter and knot buffers and every Task had a known solution. That rule cannot express a closed cylinder that is only round through its weights (degree 2 × 1, 9 × 2 points, weights 1 and √2/2), nor the vase and torus test models from the course. We keep the rule only within a Level: the Target and the starting Learner surface of a Level share one Layout, so shared buffers and the known solution still hold, while each Level can bring its own Layout.

## Consequences

- The GPU buffers for parameters and knots are created per Level, not once at start-up.
- Inside a Level the learner's Layout can still change (knot insertion in level 4). The Parameter comparison is turned off as soon as it differs from the Target's Layout.
