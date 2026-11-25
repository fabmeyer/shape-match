# Shape Match

A learning game in which students reshape a NURBS surface until it matches a target surface, level by level.

## Shapes

**Target**:
The surface the learner has to reproduce in a Task. Its control net is known, so every Task has a known solution.
_Avoid_: Reference shape, goal shape

**Learner surface**:
The surface the learner edits during a Task.
_Avoid_: User shape, user surface

**Layout**:
The structure of a control net: the number of control points in u and v, the degrees and the knot vectors. The Target and the starting Learner surface of a Level share one Layout. Different Levels can have different Layouts.
_Avoid_: Net structure, topology

## Grading

**Shape distance**:
A non-negative number that says how far the Learner surface is from the Target as shapes, independent of how either is parametrised. It is measured relative to the Target's size. Lower is better, and 0 means identical.
_Avoid_: Similarity, error, score

**Match**:
The Shape distance shown to the learner as a percentage, where 100 % means identical. It is only a way of presenting the distance, not a second measure.
_Avoid_: Similarity, accuracy

**Parameter comparison**:
A comparison of the control points, weights and knots of the Learner surface and the Target, used for hints. It only makes sense while both still share the same Layout.
_Avoid_: Direct comparison, parameter distance

**Noise floor**:
The Shape distance between a Target and the same Target sampled on a shifted grid: the smallest distance the measure can tell apart from 0 at a given resolution.
_Avoid_: Baseline error, sampling error

**Tolerance**:
The Shape distance a Level accepts as an exact match, set per Level from its Noise floor. Written τ.
_Avoid_: Threshold, epsilon

**Hand-in**:
The learner's decision that their surface is finished. The Match and Stars at that moment are the result of the attempt. Grading is live before it, but only a Hand-in counts.
_Avoid_: Submit, check, finish

**Solved**:
A Level is solved once the learner has handed in a surface with at least one Star. Solving a Level unlocks the next one.
_Avoid_: Done, passed, completed

**Star**:
The reward for a Level, 0 to 3, depending on the Shape distance in relation to the Tolerance. It is shown live while editing; a Hand-in records it.
_Avoid_: Coin, point

**Best**:
The highest Stars and Match the learner has handed in for a Level, shown on the Result screen and the Board as their hi-score.
_Avoid_: Record, top score

**Hint**:
A pointer to the control point, weight or knot that differs most from the Target, found by Parameter comparison.
_Avoid_: Tip, help

## Progression

**Board**:
The path of all Levels in order, as the learner sees it between Tasks.
_Avoid_: Map, level select, world map

**Level**:
One space on the Board. It holds one Task and can be locked or unlocked.
_Avoid_: Stage, exercise

**Task**:
The content of a Level: a Target, the starting Learner surface and the instructions.
_Avoid_: Exercise, challenge, assignment

**Workshop**:
The story setting: a craftsperson's workshop where Customers order objects. It is the setting only, not a screen.
_Avoid_: World, story mode

**Customer**:
A character in the Workshop who places an Order.
_Avoid_: NPC, client

**Order**:
The story side of a Task: which Customer wants which object, and why.
_Avoid_: Quest, mission, request

**Workbench**:
The screen where the learner edits the Learner surface to solve a Task.
_Avoid_: Editor, play screen, Workshop

**Shelf**:
The collection of objects from all Solved Levels, each shown as a picture of the learner's own surface.
_Avoid_: Trophy room, inventory
