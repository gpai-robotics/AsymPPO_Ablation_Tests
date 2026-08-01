# Teacher V4 Model 300 Card

This note is the canonical checkpoint card for the active teacher line.

## Identity

- task:
  `RMA-Go2-Privileged-Teacher-Rough-V4`
- checkpoint:
  `/home/bhuvan/tools/IsaacLab/logs/rsl_rl/go2_privileged_teacher_rough_v4_terrain_aux/2026-05-09_10-34-56/model_300.pt`
- current repo role:
  single canonical active teacher candidate

## Why This Checkpoint Matters

`Teacher V4` is the first post-`V3` branch that made the terrain path both:

- explicitly supervised
- behaviorally load-bearing on trusted general rough-terrain probes

`model_300` is the best-balanced checkpoint in that run:

- earlier checkpoints had terrain learning alive but not yet clearly useful
- later checkpoints stayed interesting but lost the cleaner all-around
  `boxes` story

## Architecture and Training Recipe

Relevant source files:

- `rma_go2_lab/models/teacher/ppo_v4_cfg.py`
- `rma_go2_lab/models/teacher/ppo_with_terrain_aux.py`
- `rma_go2_lab/models/teacher/terrain_targets.py`
- `rma_go2_lab/envs/teacher/rough_v3_cfg.py`
- `rma_go2_lab/envs/blind/rough_cfg.py`

### Inputs

Actor and critic both receive:

- `policy`
- `dynamics_privileged`
- `terrain_privileged`

### Terrain pathway

- terrain scan size:
  `187`
- terrain encoder:
  `187 -> 64 -> 32 -> 8`
- terrain target head:
  `8 -> 64 -> 13`

### PPO

- envs:
  `4096`
- steps per env:
  `32`
- max iterations:
  `2000`
- save interval:
  `20`
- learning rate:
  `1e-4`
- epochs:
  `5`
- mini-batches:
  `4`
- clip:
  `0.2`
- entropy coef:
  `0.002`
- gamma:
  `0.99`
- lambda:
  `0.95`
- desired KL:
  `0.01`
- max grad norm:
  `1.0`

### Warm start

- source:
  `rma_go2_lab/policies/blind_baseline2_warmstart_final.pt`

## What V4 Added Beyond V3

`V4` kept the stable `V3` environment and hidden-dynamics privilege, but added
explicit terrain-target supervision.

The terrain target is a deterministic 13D summary of the raw height scan:

- front / center / rear mean heights
- left / right mean heights
- front-center and center-rear differences
- left-right difference
- height std
- height range
- forward slope
- lateral slope
- contact roughness

This changed the teacher question from:

- terrain is available, maybe the actor will use it

to:

- the terrain branch must encode compact terrain structure explicitly

## Terrain Auxiliary Schedule

- coef `0.5` until iteration `300`
- coef `0.2` until iteration `800`
- coef `0.05` after that

This is part of why `model_300` matters: it sits at the end of the strongest
terrain-supervision phase.

## Training Evidence

Early training signals showed the new terrain objective was alive:

- iteration `0`
  - `terrain_target_regression = 0.0106`
  - `terrain_target_cosine = 0.8320`
  - reward `0.43`
  - episode length `16.18`

- iteration `56`
  - `terrain_target_regression = 0.0004`
  - `terrain_target_cosine = 0.9915`
  - reward `31.89`
  - episode length `858.80`

Meaning:

- terrain supervision activated immediately
- locomotion recovered quickly
- the auxiliary objective did not destabilize the teacher

## Ablation Results

Canonical audit artifacts:

- `artifacts/evaluations/teacher_dependency_watch/2026-05-09_10-34-56/teacher_v3_dependency_audit_random_rough_l5_model_100.json`
- `artifacts/evaluations/teacher_dependency_watch/2026-05-09_10-34-56/teacher_v3_dependency_audit_boxes_l5_model_100.json`
- `artifacts/evaluations/teacher_dependency_watch/2026-05-09_10-34-56/teacher_v3_dependency_audit_random_rough_l5_model_300.json`
- `artifacts/evaluations/teacher_dependency_watch/2026-05-09_10-34-56/teacher_v3_dependency_audit_boxes_l5_model_300.json`

### Model 100

`random_rough`

- `normal` reward:
  `0.040735`
- `zero_terrain` reward:
  `0.040742`
- `zero_dynamics` reward:
  `0.040634`
- `zero_terrain` action diff:
  `0.373933`
- `zero_dynamics` action diff:
  `0.629246`

`boxes`

- `normal` reward:
  `0.036927`
- `zero_terrain` reward:
  `0.037578`
- `zero_dynamics` reward:
  `0.036291`
- `zero_terrain` action diff:
  `0.458251`
- `zero_dynamics` action diff:
  `0.691287`

Interpretation:

- the terrain path was no longer a total no-op
- but terrain was not yet clearly helping behavior

### Model 300

`random_rough`

- `normal`
  - reward `0.041190`
  - vel err `0.056979`
- `zero_terrain`
  - reward `0.040861`
  - vel err `0.073126`
- `zero_dynamics`
  - reward `0.040900`
  - vel err `0.072366`
- `zero_both`
  - reward `0.040444`
  - vel err `0.092680`

Action diffs vs normal:

- `zero_terrain = 0.723248`
- `zero_dynamics = 0.588083`
- `zero_both = 0.826004`

`boxes`

- `normal`
  - reward `0.038115`
  - vel err `0.099648`
- `zero_terrain`
  - reward `0.037849`
  - vel err `0.114676`
- `zero_dynamics`
  - reward `0.037938`
  - vel err `0.114794`
- `zero_both`
  - reward `0.037483`
  - vel err `0.131336`

Action diffs vs normal:

- `zero_terrain = 0.364897`
- `zero_dynamics = 0.508035`
- `zero_both = 0.633641`

Interpretation:

- `normal > zero_terrain`
- `normal > zero_dynamics`
- `zero_both` is worst

This is the first checkpoint where terrain and dynamics are both clearly
behaviorally load-bearing on the trusted general rough probes.

## Why Model 300 Won Over Later Checkpoints

`model_500`

- stayed good on `random_rough`
- became mixed again on `boxes`
- `zero_terrain` slightly beat `normal` on boxes

`model_800`

- looked strong on `random_rough`
- again became mixed on `boxes`
- `zero_terrain` slightly beat `normal` on boxes

So later training did not produce a cleaner all-around checkpoint than
`model_300`.

## What Improved Over V3

Old `V3` truth:

- dynamics privilege clearly mattered
- terrain privilege was exposed but behaviorally unproven

`V4 model_300` truth:

- terrain branch is explicitly supervised
- terrain changes actions materially
- terrain ablation hurts behavior on trusted general rough probes
- dynamics ablation also hurts
- both together hurt most

## Known Limitations

This checkpoint is not a universal terrain-family winner.

What it does not prove:

- best behavior on every terrain family
- stair mastery
- final teacher recipe forever

Later terrain-family work showed that stairs needed separate treatment. That
produced archived `V4.1`, which helped stairs but violated the overall-teacher
contract by regressing the general rough-terrain story.

## Bottom Line

`Teacher V4 model_300` is the current canonical teacher because it best matches
the project contract:

- one overall rough-terrain teacher
- stable training
- explicit terrain supervision
- validated use of both terrain and dynamics on trusted general rough probes
- better all-around balance than `V3`
- better contract fit than later `V4` checkpoints or specialized follow-up
  branches
