# Adaptation Student NA Freeze

This file freezes the final definition of the no-adaptation student used in the
adaptation phase comparison.

`studentNA` is now frozen. Do not retune or overwrite it in place. Any future
changes should create a new explicitly versioned variant.

## Identity

- canonical name: `adaptation_student_na_final`
- checkpoint:
  - `rma_go2_lab/policies/adaptation_student_na_final.pt`
- source run:
  - `/home/bhuvan/projects/rma/rma_go2_lab/logs/rsl_rl/go2_adaptation_student_no_adapt_v0/2026-04-22_15-22-13`
- selected source checkpoint:
  - `model_1999.pt`
- freeze date:
  - `2026-04-23`

## Purpose

`studentNA` is the deployable no-adaptation baseline for the hidden-switch
phase.

It represents:

- deployable proprio-only control
- no privileged observations at deployment
- no adaptation latent
- no history encoder
- one hidden mid-episode dynamics switch during training

It is used to answer:

- how strong a deployable non-adaptive student can become under within-episode
  hidden dynamics changes
- what performance tax remains without an adaptation mechanism

## Training Definition

- task:
  - `RMA-Go2-Adaptation-Student-Rough-NoAdapt`
- env config:
  - `rma_go2_lab/envs/adaptation/rough_cfg.py`
- PPO config:
  - `rma_go2_lab/models/adaptation/adapt_ppo_cfg.py`

## Final Training Snapshot

Final training snapshot at iteration `1999/2000`:

- reward:
  - `30.85`
- episode length:
  - `843.98`
- timeout:
  - `0.8000`
- terrain levels:
  - `3.6164`
- error_vel_xy:
  - `0.2424`
- error_vel_yaw:
  - `0.1833`
- base_height termination:
  - `0.1778`
- base_orientation termination:
  - `0.0016`
- switch_reached_frac:
  - `0.8816`

## Final Evaluation Artifacts

Canonical methods reference:

- `docs/EVALUATION_METHODS.md`

Canonical freeze/synthesis note:

- `docs/ADAPTATION_PHASE_SYNTHESIS.md`

Controller-quality checks:

- `artifacts/evaluations/adaptation_student_na/gait_student_na_model1999_standstill.json`
- `artifacts/evaluations/adaptation_student_na/gait_student_na_model1999_forward.json`

Canonical isolated suite:

- `artifacts/evaluations/adaptation_student_na/isolated_suite_model_1999_blind_baseline_v1_random_rough_levelspread_normal_seed999.json`
- `artifacts/evaluations/adaptation_student_na/isolated_suite_model_1999_blind_baseline_v1_random_rough_levelspread_normal_seed999.csv`

Canonical OOD suites:

- `artifacts/ood_evaluations/adaptation_student_na/ood_suite_model_1999_ood_geometry_v1_normal_seed999.json`
- `artifacts/ood_evaluations/adaptation_student_na/ood_suite_model_1999_ood_dynamics_v1_normal_seed999.json`
- `artifacts/ood_evaluations/adaptation_student_na/ood_suite_model_1999_ood_push_v1_normal_seed999.json`
- `artifacts/ood_evaluations/adaptation_student_na/ood_suite_model_1999_ood_switch_v1_normal_seed999.json`

## Final Evaluation Summary

Suite averages:

- blind baseline suite average score:
  - `0.3646`
- OOD geometry suite average score:
  - `-5.5027`
- OOD dynamics suite average score:
  - `-87.2026`
- OOD push suite average score:
  - `-9.9514`
- OOD switch suite average score:
  - `-55.2426`

Interpretation:

- `studentNA` is a real and strong deployable baseline
- it remains viable under hidden switches
- but it is materially weaker than `studentAdapt-V0` under the harsher OOD
  dynamics, push, and switch evaluations

## Freeze Statement

`studentNA` is now frozen.

Do not change:

- its checkpoint identity
- its source run lineage
- its task/config association
- its evaluation artifact lineage

If a stronger no-adaptation baseline is trained later, it must be created as a
new explicitly versioned policy rather than replacing this file in place.
