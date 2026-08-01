# Teacher Design V3

## Status

V3 is now the frozen privileged teacher reference for the project.

Frozen checkpoint:

- `/home/bhuvan/tools/IsaacLab/logs/rsl_rl/go2_privileged_teacher_rough_v3/2026-04-21_15-35-03/model_1999.pt`

Why it was frozen:

- strongest teacher across the `V0 -> V3` ladder
- best geometry and dynamics robustness profile
- best upper bound to carry into the adaptation phase

## Goal

Teacher V3 is the first teacher that combines:

- `z_terrain`: local terrain geometry privilege
- explicit hidden-dynamics privilege

The intent is to test whether terrain-only privilege plateaued because the
teacher still had to infer friction, mass, and actuator changes blindly.

## V3 Delta From V2

V2 already fixed the main terrain-only design choices:

- B2 warm-start
- anti-crouch reward carried from V1
- compressed terrain encoder

V3 changes only one conceptual axis:

- add a low-dimensional raw dynamics privilege branch

## Privileged Inputs

### Terrain branch

- same local height scan as V2
- encoded with the same compressed terrain branch:
  - `187 -> 64 -> 32 -> 8`

### Dynamics branch

Raw privileged dynamics observations:

- mean realized static friction
- mean realized dynamic friction
- realized base-mass ratio
- per-joint stiffness scale
- per-joint damping scale

This branch is kept raw rather than terrain-encoded.

## Architecture

Actor/critic consume:

- `policy` proprio
- `dynamics_privileged` raw dynamics vector
- `terrain_privileged` encoded through the terrain encoder

So V3 keeps the proven V2 terrain bottleneck but augments it with direct access
to hidden simulator dynamics.

## Why This Version Exists

Teacher V0/V1/V2 suggested:

- terrain privilege is somewhat useful
- especially on geometry-linked cases
- but not enough to deliver a decisive universal win over B2

V3 is the clean next test:

- if V3 improves geometry and dynamics OOD together, then hidden dynamics were
  likely the missing privileged ingredient
- if V3 still does not clearly outperform B2, then the project learns that even
  richer privilege is not obviously necessary under the current task setup

## Fairness Rule

V3 should inherit everything from V2 except the new dynamics privilege branch.

That means:

- same terrain family
- same rewards
- same anti-crouch term
- same warm-start
- same compressed terrain encoder

Only the hidden-dynamics information channel changes.
