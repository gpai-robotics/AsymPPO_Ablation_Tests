# Adaptation Student V0 Freeze

This file freezes the final definition of the first completed adaptation
student.

`studentAdapt-V0` is now frozen. Do not retune or overwrite it in place. Any
future changes should create a new explicitly versioned variant.

## Identity

- canonical name: `adaptation_student_v0_final`
- checkpoint:
  - `rma_go2_lab/policies/adaptation_student_v0_final.pt`
- source run:
  - `/home/bhuvan/projects/rma/rma_go2_lab/logs/rsl_rl/go2_adaptation_student_history_v0/2026-04-22_15-22-16`
- selected source checkpoint:
  - `model_1999.pt`
- freeze date:
  - `2026-04-23`

## Purpose

`studentAdapt-V0` is the first completed positive adaptation result in the
repo.

It represents:

- deployable student observations at deployment
- current proprio plus history window
- history encoder inside the student actor-critic
- frozen privileged `V3` teacher guidance during training
- hidden mid-episode dynamics switch regime
- imitation-based adaptation route, not explicit latent regression

It is used to answer:

- whether history-based adaptation plus frozen-teacher guidance can beat the
  no-adaptation student on the same switched task

## Training Definition

- task:
  - `RMA-Go2-Adaptation-Student-Rough-History`
- env config:
  - `rma_go2_lab/envs/adaptation/rough_history_cfg.py`
- PPO config:
  - `rma_go2_lab/models/adaptation/adapt_ppo_cfg.py`
- algorithm:
  - `rma_go2_lab/models/adaptation/ppo_with_v3_expert.py`

## Final Training Snapshot

Final training snapshot at iteration `1999/2000`:

- reward:
  - `31.72`
- episode length:
  - `872.87`
- timeout:
  - `0.8081`
- terrain levels:
  - `3.5847`
- error_vel_xy:
  - `0.1940`
- error_vel_yaw:
  - `0.1778`
- base_height termination:
  - `0.1679`
- base_orientation termination:
  - `0.0020`
- switch_reached_frac:
  - `0.8316`

## Final Evaluation Artifacts

Canonical methods reference:

- `docs/EVALUATION_METHODS.md`

Canonical freeze/synthesis note:

- `docs/ADAPTATION_PHASE_SYNTHESIS.md`

Controller-quality checks:

- `artifacts/evaluations/adaptation_student_v0/gait_student_adapt_v0_model1999_standstill.json`
- `artifacts/evaluations/adaptation_student_v0/gait_student_adapt_v0_model1999_forward.json`

Canonical isolated suite:

- `artifacts/evaluations/adaptation_student_v0/isolated_suite_model_1999_blind_baseline_v1_random_rough_levelspread_normal_seed999.json`
- `artifacts/evaluations/adaptation_student_v0/isolated_suite_model_1999_blind_baseline_v1_random_rough_levelspread_normal_seed999.csv`

Canonical OOD suites:

- `artifacts/ood_evaluations/adaptation_student_v0/ood_suite_model_1999_ood_geometry_v1_normal_seed999.json`
- `artifacts/ood_evaluations/adaptation_student_v0/ood_suite_model_1999_ood_dynamics_v1_normal_seed999.json`
- `artifacts/ood_evaluations/adaptation_student_v0/ood_suite_model_1999_ood_push_v1_normal_seed999.json`
- `artifacts/ood_evaluations/adaptation_student_v0/ood_suite_model_1999_ood_switch_v1_normal_seed999.json`

## Final Evaluation Summary

Suite averages:

- blind baseline suite average score:
  - `5.3160`
- OOD geometry suite average score:
  - `0.8138`
- OOD dynamics suite average score:
  - `0.0560`
- OOD push suite average score:
  - `1.5515`
- OOD switch suite average score:
  - `-3.6059`

Representative wins over `studentNA`:

- `fric_min_random_rough_l5`:
  - `+22.496`
- `ood_very_heavy_random_rough_l5`:
  - `+38.180`
- `ood_very_weak_motor_random_rough_l5`:
  - `+293.568`
- `ood_switch_very_weak_motor_random_rough_l5`:
  - `+198.089`

Interpretation:

- `studentAdapt-V0` is the first completed positive adaptation result
- it beats `studentNA` on the completed evaluation ladder
- the largest gains appear in the hardest dynamics and switched stress cases

## Freeze Statement

`studentAdapt-V0` is now frozen.

Do not change:

- its checkpoint identity
- its source run lineage
- its task/config association
- its evaluation artifact lineage

If a stronger adaptation student is trained later, it must be created as a new
explicitly versioned policy rather than replacing this file in place.
