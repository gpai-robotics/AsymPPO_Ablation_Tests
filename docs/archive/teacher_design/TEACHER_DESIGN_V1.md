# Teacher Design V1

Teacher V1 is the smallest follow-up to V0 that directly targets the observed
V0 failure mode.

## Motivation

Teacher V0 established that:

- terrain privilege is usable
- terrain encoder plus B2 warm-start is the right backbone
- but the learned controller tended toward a low-base crouched strategy during
  locomotion

The V1 goal is not to redesign the teacher stack. It is to test whether that
mixed V0 result was mainly caused by missing posture pressure during movement.

## What Stays Fixed

Relative to Teacher V0, keep fixed:

- task definition
- terrain privilege path
- terrain encoder architecture
- B2-aware warm-start
- PPO settings
- command distribution
- terrain family
- evaluation stack

## One Deliberate Change

Teacher V1 adds one reward term:

- motion-gated terrain-aware base-height penalty

Meaning:

- the penalty is active only when the commanded planar motion is non-trivial
- the target height is adjusted by the local terrain scan
- the purpose is to discourage the crouched low-base solution without
  penalizing quiet standstill posture

## Why This Is Fair

This is still a controlled teacher comparison because:

- the privilege path is unchanged from V0
- the optimizer recipe is unchanged from V0
- the new term is directly tied to the concrete V0 pathology
- the intervention is narrow and interpretable

This should be treated as:

- not a wholesale reward redesign
- but a targeted corrective term to test whether V0 underperformed because of
  posture collapse under motion

## What Would Count As Success

Teacher V1 should be considered successful only if it improves over both:

- Teacher V0
- frozen B2

especially on:

- forward planar speed tracking
- nominal moving base height
- nominal rough isolated suite behavior

while not obviously sacrificing:

- standstill stability
- general rough-terrain survivability

## Registered Task

Historical note:
- `RMA-Go2-Privileged-Teacher-Rough-V1` is archived and no longer kept in the
  active task registry.

- `RMA-Go2-Privileged-Teacher-Rough-V1`

Environment config:

- `rma_go2_lab/envs/teacher/rough_v1_cfg.py`

Runner config:

- `rma_go2_lab/models/teacher/ppo_v1_cfg.py`

## Naming Note

Evaluation artifact names may still include suffixes like:

- `blind_baseline_v1`
- `ood_geometry_v1`
- `ood_push_v1`

These refer to the version of the evaluation protocol, not the teacher
variant. So a file under `artifacts/.../teacher_v1/` with
`...blind_baseline_v1...` in its name means:

- teacher variant = `V1`
- evaluation suite protocol version = `v1`
