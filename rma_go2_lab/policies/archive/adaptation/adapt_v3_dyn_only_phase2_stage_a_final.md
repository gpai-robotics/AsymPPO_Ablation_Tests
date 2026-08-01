# Adapt-V3 Dynamics-Only Phase 2 Stage A Freeze

This file freezes the canonical dynamics-only `Adapt-V3 Phase 2 Stage A`
student checkpoint.

It is the first successful deployable-path `phi(history) -> z_hat -> pi`
checkpoint in the active dynamics-only reboot.

## Identity

- canonical name:
  - `adapt_v3_dyn_only_phase2_stage_a_final`
- checkpoint:
  - `rma_go2_lab/policies/adapt_v3_dyn_only_phase2_stage_a_final.pt`
- source run:
  - `/home/bhuvan/tools/IsaacLab/logs/rsl_rl/go2_adapt_v3_phase2_stage_a_dyn_only/2026-04-28_15-58-38`
- selected source checkpoint:
  - `model_1999.pt`
- freeze date:
  - `2026-04-29`

## Purpose

This checkpoint freezes the dynamics-only RMA student after the privileged
base policy has been transferred to the deployable history path.

Important wording note:

- this file describes the deployed architecture truthfully
- it should not be read as proof that the frozen Stage A winner demonstrates
  strong online-changing latent behavior under hidden-dynamics switches
- that later question is addressed in:
  - `docs/ADAPTATION_PROBE_NOTES.md`
  - `docs/ADAPT_V3_POISONING_AUDIT.md`

The active actor path is:

- `policy`
- `policy_history`
- `phi(history) -> z_hat`
- `pi(policy, z_hat) -> action`

The privileged target used for supervision is dynamics-only:

- friction
- base mass / payload
- motor strength / joint scaling

No terrain privilege is part of this active dynamics-only student.

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
- frozen Phase 1 reference:
  - `rma_go2_lab/policies/adapt_v3_dyn_only_phase1_stage_a_final.pt`

Key retained design choices:

- frozen `mu`
- frozen `pi`
- trainable `phi`
- trainable critic
- stationary Stage A regime
- no within-episode adaptation switch
- action imitation scaffold decayed to zero before freeze

## Final Behavior Summary

Final training snapshot at `Learning iteration 1999/2000`:

- reward:
  - `30.83`
- episode length:
  - `897.71`
- curriculum:
  - `terrain_levels = 4.8929`
- tracking error:
  - `error_vel_xy = 0.1643`
  - `error_vel_yaw = 0.2976`
- terminations:
  - `time_out = 0.7951`
  - `base_contact = 0.0000`
  - `base_orientation = 0.0290`
  - `base_height = 0.1703`
  - `low_progress = 0.0063`

Latent/student-health snapshot:

- `latent_regression = 0.8718`
- `latent_cosine = 0.7996`
- `latent_active_frac = 0.8734`
- imitation fully off:
  - `teacher_imitation_coef = 0.0000`
  - `teacher_imitation = 0.0000`

## Selection Rationale

`model_1999.pt` was selected because:

- behavior remained strong after imitation fully decayed to zero
- terrain curriculum did not collapse
- low-progress termination stayed near zero
- the history latent remained aligned with the frozen dynamics latent
- the run completed the intended full `2000` iteration Stage A schedule

## Debug Validation Summary

Canonical debug artifact:

- `artifacts/debug/adapt_v3_dyn_only_phase2_stage_a_final_debug.json`

What it confirmed after freeze:

- active actor path is the deployable Phase 2 path:
  - `policy_obs_groups = ["policy", "policy_history"]`
  - `phase_mode = "phase2_history_actor"`
- terrain privilege is absent:
  - `terrain_group_name = null`
  - `terrain_privileged_stats = null`
- actor-path contract is exact:
  - `inference_vs_pi_x_phi_history = 0.0`
- frozen dynamics `mu` is present and live inside the checkpoint:
  - `extrinsics_latent std = 1.5617`
- learned history latent tracks the frozen dynamics latent:
  - `masked_history_vs_extrinsics_latent_mse = 0.9024`
  - `masked_history_vs_extrinsics_latent_cosine = 0.8019`
- zeroing the latent changes action materially:
  - `masked_inference_vs_pi_x_zero_latent = 0.0766`
- history-policy action is close to privileged `mu` action:
  - `masked_pi_x_phi_history_vs_pi_x_mu_e = 0.0150`

Freeze repair note:

- the raw Phase 2 training checkpoint stored a zeroed `mu` because training
  used a separate frozen Phase 1 reference for latent labels while initializing
  the live Phase 2 policy through `actor_init_path`
- before freezing, the Phase 1 dynamics-only `mu` and dynamics decoder weights
  were copied into this final checkpoint
- this preserves the trained Phase 2 `phi` while making the checkpoint
  self-contained as `mu + pi + phi`

## Evaluation Summary

Canonical evaluation artifacts:

- gait:
  - `artifacts/evaluations/adapt_v3_dyn_only/gait_adapt_v3_dyn_only_phase2_stage_a_final_standstill.json`
  - `artifacts/evaluations/adapt_v3_dyn_only/gait_adapt_v3_dyn_only_phase2_stage_a_final_forward.json`
- blind suite:
  - `artifacts/evaluations/adapt_v3_dyn_only/isolated_suite_adapt_v3_dyn_only_phase2_stage_a_final_blind_baseline_v1_random_rough_levelspread_normal_seed999.json`
  - `artifacts/evaluations/adapt_v3_dyn_only/isolated_suite_adapt_v3_dyn_only_phase2_stage_a_final_blind_baseline_v1_random_rough_levelspread_normal_seed999.csv`
- OOD suites:
  - `artifacts/ood_evaluations/adapt_v3_dyn_only/ood_suite_adapt_v3_dyn_only_phase2_stage_a_final_ood_geometry_v1_normal_seed999.json`
  - `artifacts/ood_evaluations/adapt_v3_dyn_only/ood_suite_adapt_v3_dyn_only_phase2_stage_a_final_ood_dynamics_v1_normal_seed999.json`
  - `artifacts/ood_evaluations/adapt_v3_dyn_only/ood_suite_adapt_v3_dyn_only_phase2_stage_a_final_ood_push_v1_normal_seed999.json`
  - `artifacts/ood_evaluations/adapt_v3_dyn_only/ood_suite_adapt_v3_dyn_only_phase2_stage_a_final_ood_switch_v1_normal_seed999.json`

What this evaluation established:

- the checkpoint completed the full blind-suite battery:
  - `9 / 9` scenarios
- all exploratory OOD batteries completed:
  - geometry `4 / 4`
  - dynamics `4 / 4`
  - push `4 / 4`
  - switch `4 / 4`
- base-contact failure stayed effectively absent across the suite outputs:
  - `base_contact_events_per_env = 0.0` in the consolidated blind/OOD CSVs

Blind-suite summary:

- mean suite score:
  - `8.36`
- strongest blind-suite case:
  - `motor_max_random_rough_l5 = 10.43`
- nominal random rough terrain remains healthy:
  - `nominal_random_rough_l5 = 8.02`
- weakest blind-suite case:
  - `motor_min_random_rough_l5 = 7.13`

OOD summary:

- geometry suite mean score:
  - `9.66`
- strongest geometry case:
  - `ood_random_rough_l9 = 14.40`
- weakest geometry case:
  - `ood_stairs_up_l5 = 7.27`
- dynamics suite mean score:
  - `7.46`
- push suite mean score:
  - `7.45`
- switch suite mean score:
  - `7.45`

Gait caveat:

- standstill is clean:
  - zero terminal failures in the standstill gait eval
- forward gait remains the main weakness:
  - `gait_interpretation = "non_trot_or_serial"`
  - `diagonal_trot_score = 0.1119`
  - `forward_lateral_drift_per_meter_mean = 1.5535`
  - forward gait eval recorded:
    - `terminal_base_orientation = 11`
    - `terminal_base_height = 1`

Evaluation verdict:

- this is a serious finalized comparison candidate
- it is robust enough to serve as the canonical dynamics-only `Adapt-V3`
  blind-student baseline
- it is not yet the unquestioned final hardware winner because forward gait
  quality and drift/heading composure remain weaker than ideal

## Branch Meaning

This checkpoint is the canonical dynamics-only `Adapt-V3` Stage A student.

It is suitable as:

- the deployable-path baseline for package/deployment checks
- the parent for a later dynamics-only mixed/switch continuation
- the baseline comparator before terrain-lite is introduced
- a strong stationary blind-student winner

It should not be casually described as:

- a proven online-adaptive final deployment artifact

It is not a terrain-aware student and should not be described as solving
terrain geometry adaptation.

## Freeze Statement

`adapt_v3_dyn_only_phase2_stage_a_final` is now frozen.

Do not silently replace this checkpoint. Any stronger result should be added as
a versioned continuation or explicitly documented successor.
