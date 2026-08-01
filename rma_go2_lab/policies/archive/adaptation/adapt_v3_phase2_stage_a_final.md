# Adapt-V3 Phase 2 Stage A Freeze

This file freezes the canonical `Phase 2 Stage A` bootstrap for the `Adapt-V3`
branch.

`Phase 2 Stage A` is now frozen. Do not retune or overwrite it in place. Any
future changes should create a new explicitly versioned variant or continue
into `Phase 2 Mixed` / later switched training from this frozen base.

## Identity

- canonical name:
  - `adapt_v3_phase2_stage_a_final`
- checkpoint:
  - `rma_go2_lab/policies/adapt_v3_phase2_stage_a_final.pt`
- source run:
  - `/home/bhuvan/tools/IsaacLab/logs/rsl_rl/go2_adapt_v3_phase2_stage_a/2026-04-27_19-37-16`
- selected source checkpoint:
  - `model_2098.pt`
- freeze date:
  - `2026-04-28`

## Purpose

`Phase 2 Stage A` is the history-path bootstrap phase of true `Adapt-V3`.

It represents:

- deployable actor path:
  - `phi(history) -> z_hat`
  - `pi(obs, z_hat) -> action`
- frozen `Stage A` privileged base as the latent target source
- stationary randomized episodes without within-episode switches
- temporary teacher-action scaffold during early `phi` learning
- the first `Phase 2` recipe in the repo that preserved locomotion instead of
  collapsing behavior while training the history pathway

It is used to answer:

- whether `phi(history)` can be trained without destroying the locomotion base
- whether the repo needs explicit `Phase 2` staging just like `Phase 1`
- which frozen base should seed later `Phase 2 Mixed` continuation

## Training Definition

- task:
  - `RMA-Go2-Adapt-V3-Phase2-StageA`
- env config:
  - `rma_go2_lab/envs/adaptation/rough_history_stage_a_cfg.py`
- PPO config:
  - `rma_go2_lab/models/adaptation/adapt_v3_ppo_cfg.py`
- actor-critic:
  - `rma_go2_lab/models/adaptation/rma_v3_actor_critic.py`
- algorithm:
  - `rma_go2_lab/models/adaptation/ppo_rma_v3_phase2.py`

Key retained design choices:

- frozen Phase 1 base:
  - `rma_go2_lab/policies/adapt_v3_phase1_stage_a_final.pt`
- frozen actor `pi`
- frozen extrinsics encoder `mu`
- trainable adaptation module `phi`
- trainable critic/value side
- temporary teacher-action imitation scaffold used only during early bootstrap
- within-episode adaptation switch disabled for `Phase 2 Stage A`

## Selection Rationale

`model_2098.pt` was selected because:

- it completed the full planned `Phase 2 Stage A` bootstrap run
- locomotion remained strong after teacher imitation decayed to zero
- terrain curriculum remained high instead of collapsing
- the run stayed behaviorally healthy even though latent fit plateaued
- it is the cleanest frozen continuation base for later `Phase 2 Mixed`

## Final Behavior Summary

Late-run training snapshot near freeze:

- reward:
  - `29.81`
- episode length:
  - `881.66`
- curriculum:
  - `terrain_levels = 4.9151`
- tracking error:
  - `error_vel_xy = 0.1709`
  - `error_vel_yaw = 0.2733`
- terminations:
  - `time_out = 0.7642`
  - `base_contact = 0.0000`
  - `base_orientation = 0.0499`
  - `base_height = 0.1793`
  - `low_progress = 0.0071`

History-bootstrap snapshot near freeze:

- `latent_regression = 1.0333`
- `latent_active_frac = 0.8796`
- `latent_cosine = 0.3640`
- teacher imitation fully off:
  - `teacher_imitation_coef = 0.0000`

## Interpretation

This freeze should be read as:

- a successful behavior-preserving `Phase 2` bootstrap
- not as final `Adapt-V3` completion

What it proved:

- `phi(history)` can be trained without collapsing the gait recovered in
  `Phase 1 Stage A`
- the repo likely needs staged optimization in `Phase 2`, not just in
  `Phase 1`

What it did not yet prove:

- that the history pathway is already strong under real switched adaptation
  pressure
- that `Adapt-V3` is ready to freeze as a final deployable student

## Branch Meaning

This freeze is not the final deployable `Adapt-V3` policy.

It is the frozen `Phase 2` bootstrap base that should support:

- later `Phase 2 Mixed` continuation
- possible later full switched continuation if the mixed regime succeeds

So `Phase 2 Stage A` should be read as:

- a necessary repo-specific optimization stage
- not the terminal adaptive result

## Freeze Statement

`adapt_v3_phase2_stage_a_final` is now frozen.

Do not change:

- its checkpoint identity
- its source run lineage
- its task/config association
- its role as the canonical `Phase 2` history-bootstrap base for `Adapt-V3`

Any stronger `V3` result must be introduced as:

- `Phase 2 Mixed`
- later switched continuation
- or an explicitly new `V3` variant

not by silently replacing this frozen bootstrap stage.
