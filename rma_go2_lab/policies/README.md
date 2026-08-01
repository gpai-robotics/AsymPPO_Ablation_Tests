# RMA-Go2 Policy Archive

This directory is now frozen-policy-first and intentionally small:

- current blind-baseline procedure docs live here
- only the `.pt` files needed for current frozen comparisons are kept here
- the active archive now includes the completed adaptation freeze, not just the
  blind baselines

Read these first:

1. `docs/PROJECT_GUIDE.md`
2. `rma_go2_lab/policies/blind_baseline_protocol.md`
3. `docs/PROJECT_PROGRESS_TIMELINE.md`

## Active Meaning

Right now, the live project focus is:

1. flat prior sanity
2. blind baseline training
3. privileged teacher upper bound
4. adaptation phase comparison

So this directory should be read as the canonical archive of frozen checkpoints
that currently matter for the project story. Future branches should add new
versioned files rather than replacing these in place.

## Canonical `.pt` Files

These are the only policy checkpoints currently kept here on purpose:

- `rma_go2_lab/policies/flat1499.pt`
  - selected flat locomotion prior
  - used to warm-start the blind warm-start baseline
  - corrected to the non-normalized `2026-04-17_14-14-36/model_1499.pt`
    lineage after restored-run audit
- `rma_go2_lab/policies/blind_baseline1_scratch_final.pt`
  - frozen final Baseline 1 checkpoint
  - selected from `model_1999.pt`
- `rma_go2_lab/policies/blind_baseline2_warmstart_final.pt`
  - frozen final Baseline 2 checkpoint
  - selected from `model_1500.pt`
- `rma_go2_lab/policies/blind_baseline3_warmstart_imitation_final.pt`
  - frozen final Baseline 3 checkpoint
  - selected from `model_560.pt` after an intra-run checkpoint sweep
- `rma_go2_lab/policies/adaptation_student_na_final.pt`
  - frozen final no-adaptation student
  - selected from `model_1999.pt`
- `rma_go2_lab/policies/adaptation_student_v0_final.pt`
  - frozen final imitation-based adaptation student
  - selected from `model_1999.pt`
- `rma_go2_lab/policies/adaptation_student_v1_final.pt`
  - frozen final explicit-latent adaptation student
  - selected from `model_1999.pt`
- `rma_go2_lab/policies/adaptation_student_v2_final.pt`
  - frozen final modular RMA-like adaptation student
  - selected from `model_1999.pt`
- `rma_go2_lab/policies/adapt_v3_phase1_stage_a_final.pt`
  - frozen privileged `Adapt-V3` Stage A base
  - selected from `model_1500.pt`
- `rma_go2_lab/policies/adapt_v3_dyn_only_phase1_stage_a_final.pt`
  - frozen active dynamics-only `Adapt-V3` Stage A base
  - selected from `model_1220.pt`
- `rma_go2_lab/policies/adapt_v3_phase2_stage_a_final.pt`
  - frozen `Adapt-V3` Phase 2 Stage A bootstrap
  - selected from `model_2098.pt`
- `rma_go2_lab/policies/adapt_v3_dyn_only_phase2_stage_a_final.pt`
  - frozen active dynamics-only `Adapt-V3` Phase 2 Stage A student
  - selected from `model_1999.pt`
  - repaired into a self-contained `mu + pi + phi` checkpoint by copying the
    Phase 1 dynamics-only `mu` and dynamics decoder weights back into the
    final artifact
  - canonical eval verdict:
    - strong blind/OOD robustness candidate
    - forward gait quality still weaker than ideal
- `rma_go2_lab/policies/adapt_v3_terrain_lite_phase1_stage_a_final.pt`
  - frozen terrain-aware `terrain-lite` `Adapt-V3` Phase 1 base
  - selected from `model_1999.pt`
  - canonical terrain-aware privileged/base reference for the next Phase 2
    student run
- `rma_go2_lab/policies/adapt_v3_terrain_lite_phase2_stage_a_final.pt`
  - frozen terrain-aware `terrain-lite` `Adapt-V3` Phase 2 blind student
  - selected from `model_1999.pt`
  - canonical terrain-aware blind-student comparison candidate
  - qualified final verdict:
    - real blind terrain-aware student
    - somewhat conservative lower-base posture persists
    - final winner still to be decided by eval
- `rma_go2_lab/policies/adapt_v3_dyn_only_phase2_recovery_low_switch_final.pt`
  - frozen low-switch recovery `Adapt-V3` checkpoint
  - selected from `model_1200.pt` after a recovery-branch checkpoint sweep
  - canonical recovery artifact for the first branch that restored non-collapsed
    online latent behavior under active switch pressure
  - should be read as the recovery anchor, not as a silent replacement for the
    stationary Stage A winner
- `rma_go2_lab/policies/adapt_v3_dyn_only_phase2_recovery_low_switch_latent_reg_final.pt`
  - frozen bounded-latent continuation of the low-switch recovery branch
  - selected from `model_220.pt` after an early-mid checkpoint sweep
  - canonical bounded-latent recovery challenger for the first training-side
    fix that materially improved unclamped MuJoCo behavior
  - should be read as the first Sim2Sim-oriented recovery refinement base, not
    as a silent replacement for either the stationary Stage A winner or the
    earlier recovery anchor
- `rma_go2_lab/policies/adapt_v3_dyn_only_structured_z27_phase2_phi_supervised_v1_candidate.pt`
  - frozen first viable structured Phase 2 offline `phi` candidate
  - selected from `artifacts/models/structured_z27_phase2_phi_supervised_v1/best.pt`
  - canonical working artifact for the first RMA-style structured C2 line that
    preserved healthy locomotion without online PPO drift
  - should be read as the working structured C2 pipeline winner-for-now
  - later `v2`, `v3`, bottleneck, and residual refresh rounds were informative
    alternates, but none replaced this checkpoint overall

Baseline 1 freeze note:

- `rma_go2_lab/policies/blind_baseline1_scratch_final.md`

Baseline 2 freeze note:

- `rma_go2_lab/policies/blind_baseline2_warmstart_final.md`

Baseline 3 freeze note:

- `rma_go2_lab/policies/blind_baseline3_warmstart_imitation_final.md`

Adaptation student NA freeze note:

- `rma_go2_lab/policies/adaptation_student_na_final.md`

Adaptation student V0 freeze note:

- `rma_go2_lab/policies/adaptation_student_v0_final.md`

Adaptation student V1 freeze note:

- `rma_go2_lab/policies/adaptation_student_v1_final.md`

Adaptation student V2 freeze note:

- `rma_go2_lab/policies/adaptation_student_v2_final.md`

Adapt-V3 Stage A freeze note:

- `rma_go2_lab/policies/adapt_v3_phase1_stage_a_final.md`

Adapt-V3 dynamics-only Stage A freeze note:

- `rma_go2_lab/policies/adapt_v3_dyn_only_phase1_stage_a_final.md`

Adapt-V3 Phase 2 Stage A freeze note:

- `rma_go2_lab/policies/adapt_v3_phase2_stage_a_final.md`

Adapt-V3 dynamics-only Phase 2 Stage A freeze note:

- `rma_go2_lab/policies/adapt_v3_dyn_only_phase2_stage_a_final.md`

Adapt-V3 terrain-lite Phase 1 freeze note:

- `rma_go2_lab/policies/adapt_v3_terrain_lite_phase1_stage_a_final.md`

Adapt-V3 terrain-lite Phase 2 freeze note:

- `rma_go2_lab/policies/adapt_v3_terrain_lite_phase2_stage_a_final.md`

Adapt-V3 recovery low-switch freeze note:

- `rma_go2_lab/policies/adapt_v3_dyn_only_phase2_recovery_low_switch_final.md`

Adapt-V3 recovery low-switch latent-reg freeze note:

- `rma_go2_lab/policies/adapt_v3_dyn_only_phase2_recovery_low_switch_latent_reg_final.md`

Structured Phase 2 offline `phi` candidate freeze note:

- `rma_go2_lab/policies/adapt_v3_dyn_only_structured_z27_phase2_phi_supervised_v1_candidate.md`

Structured offline final-baseline hardening note:

- `docs/C2_STRUCTURED_OFFLINE_FINAL_BASELINE_CARD.md`

Flat prior freeze note:

- `rma_go2_lab/policies/flat_prior_final.md`

## Current Evaluation Artifacts

Use `artifacts/evaluations/` for active outputs, organized by artifact family.

The flat-prior sanity artifacts currently worth keeping are:

- `artifacts/evaluations/flat_prior/gait_flat_prior_model1499_standstill.json`
- `artifacts/evaluations/flat_prior/gait_flat_prior_model1499_forward.json`

The current blind baseline artifacts also live there as they are produced, for example:

- `artifacts/evaluations/baseline1/gait_blind_scratch_model1999_standstill.json`
- `artifacts/evaluations/baseline1/gait_blind_scratch_model1999_forward.json`
- `artifacts/evaluations/baseline1/isolated_suite_model_1999_blind_baseline_v1_random_rough_levelspread_normal_seed999.json`
- `artifacts/evaluations/baseline1/isolated_suite_model_1999_blind_baseline_v1_random_rough_levelspread_normal_seed999.csv`
- `artifacts/evaluations/baseline2/gait_blind_warmstart_model1500_standstill.json`
- `artifacts/evaluations/baseline2/gait_blind_warmstart_model1500_forward.json`
- `artifacts/evaluations/baseline2/isolated_suite_model_1500_blind_baseline_v1_random_rough_levelspread_normal_seed999.json`
- `artifacts/evaluations/baseline2/isolated_suite_model_1500_blind_baseline_v1_random_rough_levelspread_normal_seed999.csv`
- `artifacts/evaluations/baseline3/gait_blind_warmstart_imitation_model560_standstill.json`
- `artifacts/evaluations/baseline3/gait_blind_warmstart_imitation_model560_forward.json`
- `artifacts/evaluations/baseline3/isolated_suite_model_560_blind_baseline_v1_random_rough_levelspread_normal_seed999.json`
- `artifacts/evaluations/baseline3/isolated_suite_model_560_blind_baseline_v1_random_rough_levelspread_normal_seed999.csv`

The completed adaptation freeze artifacts also live there:

- `artifacts/evaluations/adaptation_student_na/gait_student_na_model1999_standstill.json`
- `artifacts/evaluations/adaptation_student_na/gait_student_na_model1999_forward.json`
- `artifacts/evaluations/adaptation_student_na/isolated_suite_model_1999_blind_baseline_v1_random_rough_levelspread_normal_seed999.json`
- `artifacts/evaluations/adaptation_student_v0/gait_student_adapt_v0_model1999_standstill.json`
- `artifacts/evaluations/adaptation_student_v0/gait_student_adapt_v0_model1999_forward.json`
- `artifacts/evaluations/adaptation_student_v0/isolated_suite_model_1999_blind_baseline_v1_random_rough_levelspread_normal_seed999.json`
- `artifacts/ood_evaluations/adaptation_student_na/ood_suite_model_1999_ood_geometry_v1_normal_seed999.json`
- `artifacts/ood_evaluations/adaptation_student_v0/ood_suite_model_1999_ood_geometry_v1_normal_seed999.json`
- `artifacts/ood_evaluations/adaptation_student_na/ood_suite_model_1999_ood_dynamics_v1_normal_seed999.json`
- `artifacts/ood_evaluations/adaptation_student_v0/ood_suite_model_1999_ood_dynamics_v1_normal_seed999.json`
- `artifacts/ood_evaluations/adaptation_student_na/ood_suite_model_1999_ood_push_v1_normal_seed999.json`
- `artifacts/ood_evaluations/adaptation_student_v0/ood_suite_model_1999_ood_push_v1_normal_seed999.json`
- `artifacts/ood_evaluations/adaptation_student_na/ood_suite_model_1999_ood_switch_v1_normal_seed999.json`
- `artifacts/ood_evaluations/adaptation_student_v0/ood_suite_model_1999_ood_switch_v1_normal_seed999.json`

The canonical dynamics-only `Adapt-V3` evaluation artifacts now also live there:

- `artifacts/evaluations/adapt_v3_dyn_only/gait_adapt_v3_dyn_only_phase2_stage_a_final_standstill.json`
- `artifacts/evaluations/adapt_v3_dyn_only/gait_adapt_v3_dyn_only_phase2_stage_a_final_forward.json`
- `artifacts/evaluations/adapt_v3_dyn_only/isolated_suite_adapt_v3_dyn_only_phase2_stage_a_final_blind_baseline_v1_random_rough_levelspread_normal_seed999.json`
- `artifacts/evaluations/adapt_v3_dyn_only/isolated_suite_adapt_v3_dyn_only_phase2_stage_a_final_blind_baseline_v1_random_rough_levelspread_normal_seed999.csv`
- `artifacts/ood_evaluations/adapt_v3_dyn_only/ood_suite_adapt_v3_dyn_only_phase2_stage_a_final_ood_geometry_v1_normal_seed999.json`
- `artifacts/ood_evaluations/adapt_v3_dyn_only/ood_suite_adapt_v3_dyn_only_phase2_stage_a_final_ood_dynamics_v1_normal_seed999.json`
- `artifacts/ood_evaluations/adapt_v3_dyn_only/ood_suite_adapt_v3_dyn_only_phase2_stage_a_final_ood_push_v1_normal_seed999.json`
- `artifacts/ood_evaluations/adapt_v3_dyn_only/ood_suite_adapt_v3_dyn_only_phase2_stage_a_final_ood_switch_v1_normal_seed999.json`
