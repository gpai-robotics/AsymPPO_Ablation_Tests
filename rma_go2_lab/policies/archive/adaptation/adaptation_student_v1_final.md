# Adaptation Student V1 Freeze

This file freezes the final definition of the first explicit-latent adaptation
student in the repo.

`studentAdapt-V1` is now frozen. Do not retune or overwrite it in place. Any
future changes should create a new explicitly versioned variant.

## Identity

- canonical name: `adaptation_student_v1_final`
- checkpoint:
  - `rma_go2_lab/policies/adaptation_student_v1_final.pt`
- source run:
  - `/home/bhuvan/projects/rma/rma_go2_lab/logs/rsl_rl/go2_adaptation_student_history_v1/2026-04-23_12-31-29`
- selected source checkpoint:
  - `model_1999.pt`
- freeze date:
  - `2026-04-23`

## Purpose

`studentAdapt-V1` is the first completed explicit-latent adaptation result in
the repo.

It represents:

- deployable student observations at deployment
- current proprio plus history window
- explicit latent prediction from history
- frozen privileged `V3` teacher latent target during training
- hidden mid-episode dynamics switch regime
- latent-regression adaptation route rather than pure teacher action imitation

It is used to answer:

- whether an explicit latent-prediction route can become a strong adaptation
  student under the same switched task
- whether the repo can move toward a cleaner RMA-like formulation without
  losing adaptation performance

## Training Definition

- task:
  - `RMA-Go2-Adaptation-Student-Rough-History-V1`
- env config:
  - `rma_go2_lab/envs/adaptation/rough_history_cfg.py`
- PPO config:
  - `rma_go2_lab/models/adaptation/adapt_v1_ppo_cfg.py`
- algorithm:
  - `rma_go2_lab/models/adaptation/ppo_with_v3_latent.py`

## Final Training Snapshot

Final training snapshot at iteration `1999/2000`:

- reward:
  - `30.03`
- episode length:
  - `845.84`
- timeout:
  - `0.7857`
- terrain levels:
  - `3.5941`
- error_vel_xy:
  - `0.2398`
- error_vel_yaw:
  - `0.1645`
- base_height termination:
  - `0.1793`
- base_orientation termination:
  - `0.0129`
- feet_slide:
  - `-0.0386`
- switch_reached_frac:
  - `0.8287`
- latent_regression:
  - `294.2954`
- latent_cosine:
  - `0.7325`

## Current Interpretation

`studentAdapt-V1` is the first completed explicit-latent adaptation candidate.

What it already shows:

- the latent-prediction path trains to completion
- latent supervision remains active through the final run
- the branch produces a strong and mature adaptation policy

The strongest visible characteristics at freeze time are:

- robust survival
- good contact quality
- strong timeout rate
- explicit latent alignment that remains non-degenerate

The main visible weakness at freeze time is:

- translational tracking remained less sharp than the strongest imitation-based
  path

## Canonical Evaluation Status

`studentAdapt-V1` now has a completed canonical post-fix evaluation matrix.

Canonical eval artifacts live in:

- `artifacts/evaluations/adaptation_student_v1/`
- `artifacts/ood_evaluations/adaptation_student_v1/`

The frozen eval lineage covers:

- gait
- blind suite
- OOD geometry
- OOD dynamics
- OOD push
- OOD switch

## Freeze Statement

`studentAdapt-V1` is now frozen.

Do not change:

- its checkpoint identity
- its source run lineage
- its task/config association

The comparison status against `NA`, `V0`, and `V2` should now be read together
with:

- `docs/ADAPTATION_PHASE_SYNTHESIS.md`
