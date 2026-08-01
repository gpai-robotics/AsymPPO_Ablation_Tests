# Adapt-V3 Terrain-Lite Phase 2 Freeze

This file freezes the canonical terrain-lite blind-student `Phase 2` artifact
for the active `Adapt-V3` comparison.

It is the first successful terrain-aware blind student after replacing the
earlier raw-terrain privileged target with the compact
`terrain_lite_privileged` representation.

## Identity

- canonical name:
  - `adapt_v3_terrain_lite_phase2_stage_a_final`
- checkpoint:
  - `rma_go2_lab/policies/adapt_v3_terrain_lite_phase2_stage_a_final.pt`
- source run:
  - `/home/bhuvan/tools/IsaacLab/logs/rsl_rl/go2_adapt_v3_phase2_stage_a_terrain_lite/2026-04-30_09-55-10`
- selected source checkpoint:
  - `model_1999.pt`
- freeze date:
  - `2026-05-01`

## Purpose

This checkpoint freezes the terrain-aware blind student for the active
`Adapt-V3` head-to-head comparison.

Its job is to answer:

- can `phi(history)` recover a compact terrain-aware `mu(e_t)` well enough to
  produce a stable blind student without runtime privileged inputs?

The teacher/reference latent for this line includes:

- `terrain_lite_privileged`
- `dynamics_privileged`

The deployed student remains blind at inference:

- `policy + policy_history`
- `phi(history) -> z_hat`
- `pi(policy, z_hat) -> action`

## Training Definition

- task:
  - `RMA-Go2-Adapt-V3-TerrainLite-Phase2-StageA`
- env config:
  - `rma_go2_lab/envs/adaptation/rough_history_stage_a_terrain_lite_cfg.py`
- PPO config:
  - `rma_go2_lab/models/adaptation/adapt_v3_ppo_cfg.py`
- actor-critic:
  - `rma_go2_lab/models/adaptation/rma_v3_actor_critic.py`
- algorithm:
  - `rma_go2_lab/models/adaptation/ppo_rma_v3_phase2.py`

Key design choices:

- actor/critic trunk warm-started from:
  - `rma_go2_lab/policies/adapt_v3_terrain_lite_phase1_stage_a_final.pt`
- frozen Phase 1 terrain-lite checkpoint used as the privileged/reference path
- live Phase 2 training learns the blind adaptation module against the frozen
  teacher/reference policy
- temporary teacher-action imitation scaffold during bootstrap
- no within-episode adaptation switch during training

## Selection Rationale

`model_1999.pt` was selected because:

- the run completed the intended full `2000` iteration schedule
- imitation fully decayed to zero without training collapse
- the student latent remained stable and well aligned with the frozen
  terrain-lite teacher/reference latent
- rough-terrain locomotion stayed in a serious-performance band through the end
  of the run

## Final Behavior Summary

Final training snapshot at `Learning iteration 1999/2000`:

- reward:
  - `28.38`
- episode length:
  - `840.65`
- curriculum:
  - `terrain_levels = 4.3793`
- tracking error:
  - `error_vel_xy = 0.1602`
  - `error_vel_yaw = 0.2976`
- terminations:
  - `time_out = 0.7264`
  - `base_contact = 0.0000`
  - `base_orientation = 0.0171`
  - `base_height = 0.2535`
  - `low_progress = 0.0037`

Latent/student-health snapshot:

- `latent_regression = 1.1293`
- `latent_cosine = 0.8122`
- `latent_active_frac = 0.8696`
- imitation fully off:
  - `teacher_imitation_coef = 0.0000`
  - `teacher_imitation = 0.0000`

## Interpretation

This run provides strong evidence that the terrain-lite blind-student path is
real and trainable.

What it means:

- `phi(history)` can recover a compact terrain-aware + dynamics latent without
  runtime privileged observations
- the earlier terrain-student failure mode was not fundamental to all
  terrain-aware privilege
- the repo now has a legitimate terrain-aware blind-student candidate for final
  comparison

What it does not prove:

- that this student is the final winner over the canonical dyn-only student
- that the current `terrain_lite` summary is already the best terrain target we
  could use

Important caveat:

- the final student tends to settle into a conservative, slightly lower-base
  posture than desired
- `base_height` remained elevated through the end of training
- tracking quality stayed somewhat softer than the dyn-only line

So this is a qualified success:

- successful blind terrain-aware student
- not yet an obvious knockout over dyn-only

## Branch Meaning

This checkpoint is the canonical terrain-aware blind-student artifact for the
active `terrain-lite` branch.

It is suitable as:

- the terrain-aware candidate in the final dyn-only versus terrain-aware eval
  comparison
- the correct terrain-aware artifact to send through the same evaluation battery
  as the dyn-only candidate
- the current upper-confidence answer to whether compact terrain privilege can
  survive all the way to a blind student

## Freeze Statement

`adapt_v3_terrain_lite_phase2_stage_a_final` is now frozen.

Do not silently replace this checkpoint. Any later stronger terrain-aware blind
student should be added as an explicitly versioned continuation or documented
successor.
