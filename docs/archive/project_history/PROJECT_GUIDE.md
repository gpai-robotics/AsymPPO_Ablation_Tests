# RMA-Go2 Project Guide

This is the top-level canonical doc map for the repo.

Use this file first. Then jump to the linked topic docs only when needed.

## Source Of Truth

Use these as the canonical final artifact cards:

- teacher:
  `docs/TEACHER_V4_MODEL300_CARD.md`
- Candidate 1:
  `docs/C1_STAGEA_MODEL400_DEPLOY_CARD.md`
- Candidate 2:
  `docs/C2_ADAPT_V3_BASELINE_CANDIDATE_CARD.md`

Live task ids come from:

- `rma_go2_lab/__init__.py`

Reading rule:

- treat this file as the single entry point
- treat the linked topic docs as the working source of truth for that topic
- treat older `PLAN`, `START`, and branch-specific notes as historical lineage
  unless this file explicitly names them as canonical

## Current Canonical Artifacts

- teacher checkpoint:
  `/home/bhuvan/tools/IsaacLab/logs/rsl_rl/go2_privileged_teacher_rough_v4_terrain_aux/2026-05-09_10-34-56/model_300.pt`
- Candidate 1 source checkpoint:
  `/home/bhuvan/tools/IsaacLab/logs/rsl_rl/go2_c1_ethlike_v3_v4teacher300/2026-05-11_13-10-12/model_400.pt`
- Candidate 1 exported bundle:
  `rma_go2_lab/policies/exported/c1_ethlike_v3_model_400_candidate/`
- Candidate 2 active task:
  `RMA-Go2-Adapt-V3-DynOnly-Phase2-Recovery-LowSwitch-LatentReg`
- Candidate 2 active baseline artifact:
  `rma_go2_lab/policies/adapt_v3_dyn_only_phase2_recovery_low_switch_latent_reg_final.pt`

Companion history docs:

- `docs/PROJECT_PROGRESS_TIMELINE.md`
- `docs/EXPLORATION_LEDGER.md`
- `docs/archive/guides/PHASED_DOCS_GUIDE.md`

## Project Goal

The active project goal is now broader than the original blind ladder:

1. flat prior
2. blind rough scratch baseline
3. blind rough warm-start baseline
4. blind rough warm-start + imitation baseline
5. privileged teacher phase
6. no-adaptation switched student
7. adaptation student

The core scientific question for the active repo state is:

> how much do privilege and history-based adaptation improve a deployable
> rough-terrain quadruped controller under hidden mismatch and rough-terrain
> geometry variation while remaining blind at inference?

## Canonical Doc Map

Use these as the primary working docs by topic.

### Repo orientation

- `docs/PROJECT_GUIDE.md`
  - single entry point and doc map
- `docs/REPO_MENTAL_MODEL.md`
  - fastest way for a new engineer to understand the repo lanes, artifact flow,
    and active vs frozen vs archive structure
- `docs/PROJECT_PROGRESS_TIMELINE.md`
  - milestone-oriented project history
- `docs/EXPLORATION_LEDGER.md`
  - active, bridge, planned, and rejected branch ledger

### Blind baselines

- `docs/FROZEN_BASELINE_SYNTHESIS.md`
  - canonical baseline story
- `docs/FROZEN_BASELINE_RESULTS_AT_A_GLANCE.md`
  - compact baseline summary
- `docs/BASELINE_COMPARISON_FINAL.md`
  - detailed frozen comparison

### Privileged teacher

- `docs/TEACHER_V4_MODEL300_CARD.md`
  - canonical teacher card
- `docs/TEACHER_PHASE_SYNTHESIS.md`
  - supporting teacher lineage and historical synthesis

### Candidate 1: blind reactive

- `docs/C1_STAGEA_MODEL400_DEPLOY_CARD.md`
  - canonical Candidate 1 card
- `docs/CANDIDATE1_BLIND_REACTIVE_PLAN.md`
  - supporting historical working note
- `docs/REFERENCE_ECOSYSTEM_MAP.md`
  - how C1 and C2 relate to the external reference repos

Current frozen C1 finalist:

- `rma_go2_lab/policies/exported/c1_ethlike_v3_model_400_candidate/`

### Candidate 2: RMA-style adaptive

- `docs/C2_ADAPT_V3_BASELINE_CANDIDATE_CARD.md`
  - canonical Candidate 2 card
- `docs/ADAPT_V3_ACTIVE_ROADMAP.md`
  - supporting active branch roadmap
- `docs/ADAPT_V3_EXECUTION_SPEC.md`
  - architecture and implementation contract
- `docs/OG_RMA_VS_REPO_DIVERGENCE.md`
  - clean explanation of RMA faithfulness vs repo divergence

### Evaluation

- `docs/EVALUATION_METHODS.md`
  - canonical repo-wide evaluation method doc
- `docs/ADAPTIVE_POLICY_EVAL_PROTOCOL.md`
  - adaptive-branch-specific evaluation gate

### Deployment

- `docs/DEPLOYMENT_PLAN.md`
  - canonical deployment and Sim2Sim doc
- `docs/DEPLOYMENT_AUDIT_ADAPT_V3.md`
  - current deployment-surface audit
- `docs/SIM2REAL_C1_BRINGUP_PLAN.md`
  - first repo-native C1 hardware bring-up plan
- `docs/SIM2REAL_REFERENCE_MAP.md`
  - which local reference repos are actually useful for sim2real and why

### Historical reading

- `docs/archive/guides/PHASED_DOCS_GUIDE.md`
  - phase-oriented reading map kept mainly for historical navigation
- `docs/archive/`
  - archived design lineage

## Canonical Experiment Ladder

### Flat prior

- Task: `RMA-Go2-Flat`
- Purpose: train a clean locomotion prior used to initialize later runs
- Status: validated checkpoint already selected

Selected flat prior:

- `rma_go2_lab/policies/flat1499.pt`
  - canonical lineage corrected to the non-normalized
    `2026-04-17_14-14-36/model_1499.pt` run
- sanity reports:
  - `artifacts/evaluations/flat_prior/gait_flat_prior_model1499_standstill.json`
  - `artifacts/evaluations/flat_prior/gait_flat_prior_model1499_forward.json`

### Blind baselines

1. `RMA-Go2-Blind-Baseline-Rough`
   - rough blind scratch baseline
2. `RMA-Go2-Blind-Baseline-Rough-WarmStart`
   - same rough blind baseline, actor warm-started from the flat prior
3. `RMA-Go2-Blind-Baseline-Rough-WarmStart-Imitation`
   - same rough blind baseline, warm-start plus temporary imitation prior

Blind baselines are comparison baselines, not moving targets.

Current frozen Baseline 1 checkpoint:

- `rma_go2_lab/policies/blind_baseline1_scratch_final.pt`

Current frozen Baseline 2 checkpoint:

- `rma_go2_lab/policies/blind_baseline2_warmstart_final.pt`

Current frozen Baseline 3 checkpoint:

- `rma_go2_lab/policies/blind_baseline3_warmstart_imitation_final.pt`

### Teacher phase

Privileged teacher work is now frozen through the `V3` phase.

Canonical teacher card:

- `docs/TEACHER_V4_MODEL300_CARD.md`

Supporting synthesis:

- `docs/TEACHER_PHASE_SYNTHESIS.md`

Historical teacher-design notes:

- `docs/archive/teacher_design/`

### Adaptation phase

Current frozen no-adaptation student:

- `rma_go2_lab/policies/adaptation_student_na_final.pt`

Current frozen adaptation student:

- `rma_go2_lab/policies/adaptation_student_v0_final.pt`
- `rma_go2_lab/policies/adaptation_student_v1_final.pt`
- `rma_go2_lab/policies/adaptation_student_v2_final.pt`

Historical frozen `Adapt-V3` privileged base:

- `rma_go2_lab/policies/adapt_v3_phase1_stage_a_final.pt`

Active frozen dynamics-only `Adapt-V3` privileged base:

- `rma_go2_lab/policies/adapt_v3_dyn_only_phase1_stage_a_final.pt`

Current frozen terrain-lite `Adapt-V3` privileged base:

- `rma_go2_lab/policies/adapt_v3_terrain_lite_phase1_stage_a_final.pt`

Historical frozen `Adapt-V3` Phase 2 bootstrap:

- `rma_go2_lab/policies/adapt_v3_phase2_stage_a_final.pt`

Active frozen dynamics-only `Adapt-V3` Phase 2 bootstrap:

- `rma_go2_lab/policies/adapt_v3_dyn_only_phase2_stage_a_final.pt`

Canonical low-switch recovery `Adapt-V3` checkpoint:

- `rma_go2_lab/policies/adapt_v3_dyn_only_phase2_recovery_low_switch_final.pt`

Canonical bounded-latent low-switch recovery `Adapt-V3` checkpoint:

- `rma_go2_lab/policies/adapt_v3_dyn_only_phase2_recovery_low_switch_latent_reg_final.pt`

Canonical adaptation references:

- `docs/ADAPTATION_PHASE_SYNTHESIS.md`
- `docs/ADAPT_V3_ACTIVE_ROADMAP.md`
- `docs/ADAPT_V3_EXECUTION_SPEC.md`
- `docs/OG_RMA_VS_REPO_DIVERGENCE.md`
- `docs/ADAPTIVE_POLICY_EVAL_PROTOCOL.md`
- `docs/DEPLOYMENT_AUDIT_ADAPT_V3.md`

Supporting history:

- `docs/ADAPTATION_IMPLEMENTATION_V0.md`
- `docs/archive/adaptation/ADAPTATION_V2_PLAN.md`
- `docs/archive/adapt_v3/ADAPT_V3_TRUE_RMA_PLAN.md`
- `docs/archive/adapt_v3/KNOWN_LIMITATIONS_AND_BRANCH_FOLLOWUPS.md`
- `docs/V1_V2_CLOSEOUT_CHECKLIST.md`
- `docs/FINAL_CANDIDATE_COMPARISON_RUBRIC.md`

Important active `V3` note:

- the earlier terrain-plus-dynamics `V3` line produced valuable frozen
  historical artifacts, but it is no longer the active implementation path
- the active reboot is now a dynamics-only `Stage A` line meant to test whether
  a blind history student can reliably infer hidden dynamics before terrain
  geometry is asked of it
- an attempted mixed mid-episode-switch continuation was tried and retired as
  failed exploration
- the current forward path is:
  - canonical dyn-only student:
    `adapt_v3_dyn_only_phase2_stage_a_final.pt`
  - canonical recovery checkpoint:
    `adapt_v3_dyn_only_phase2_recovery_low_switch_final.pt`
  - canonical bounded-latent recovery challenger:
    `adapt_v3_dyn_only_phase2_recovery_low_switch_latent_reg_final.pt`
  - terrain-lite Phase 2 challenger:
    `adapt_v3_terrain_lite_phase2_stage_a_final.pt`
  - current winner of the first clean head-to-head:
    `adapt_v3_dyn_only_phase2_stage_a_final.pt`
- interpretation:
  - Stage A dyn-only remains the current deployment-side winner
  - low-switch recovery is now the canonical adaptation-recovery anchor
  - bounded-latent low-switch recovery is now the canonical Sim2Sim-oriented
    recovery refinement base
- see `docs/ADAPT_V3_ACTIVE_ROADMAP.md`
- detailed comparison verdict:
  - `docs/ADAPT_V3_FINAL_CANDIDATE_COMPARISON.md`
- adaptation-claim audit for the dyn-only Stage A winner:
  - `docs/ADAPTATION_PROBE_NOTES.md`
  - `docs/ADAPT_V3_POISONING_AUDIT.md`

Historical drafts and superseded notes:

- `docs/archive/`
- `public/archive/`

## Governing Principle For Blind Baselines

Blind baselines should not be trained to become unbeatable obstacle
specialists.

They should be trained as competent fixed controllers, then evaluated under
controlled hidden mismatch:

- friction
- mass
- motor strength
- terrain geometry
- held-out switch stress only as evaluation, not as the active training
  contract

Canonical SOP:

- `rma_go2_lab/policies/blind_baseline_protocol.md`

## Where Things Live

### Environments

- `rma_go2_lab/envs/priors/`
  - shared flat-prior envs
- `rma_go2_lab/envs/blind/`
  - blind baseline envs
- `rma_go2_lab/envs/teacher/`
  - privileged teacher envs
- `rma_go2_lab/envs/adaptation/`
  - history-student and `Adapt-V3` envs

### Models

- `rma_go2_lab/models/priors/`
  - shared flat-prior PPO configs
- `rma_go2_lab/models/blind/`
  - blind PPO configs, warm-start actor-critic, imitation PPO
- `rma_go2_lab/models/teacher/`
  - privileged teacher PPO configs and teacher actor-critic variants
- `rma_go2_lab/models/adaptation/`
  - no-adaptation student, history student, frozen teacher wrappers, and
    adaptation PPO variants

### Evaluators

- `scripts/eval/gait.py`
  - gait, standstill, step response, forward drift checks
- `scripts/eval/isolated.py`
  - single controlled evaluation scenario
- `scripts/eval/run_isolated_suite.py`
  - suite runner across many controlled scenarios
- `scripts/eval/blind_baseline_diagnostics.py`
  - blind baseline health diagnostics

### Export / evaluation artifacts

- `scripts/deploy/`
- `docs/DEPLOYMENT_AUDIT_ADAPT_V3.md`
- `artifacts/evaluations/`
- `artifacts/ood_evaluations/`
- `rma_go2_lab/policies/exported/`

Note:

- `logs/` at the repo root is a local symlink to the IsaacLab run directory.
  It is generated convenience state, not source code.
- `docs/archive/` stores historical design notes that are no longer canonical
  reading.
- `public/archive/` stores public-facing README drafts and rewrite proposals
  after a newer public README becomes the active source.
- `scripts/deploy/` is the isolated deployment surface for export, packaging,
  deployable-I/O validation, and sim-side deployment rehearsal.

## Active Files That Matter Most

If you only open a few files, open these:

- `rma_go2_lab/__init__.py`
  - active registered tasks only
- `rma_go2_lab/envs/priors/flat_forward_prior_cfg.py`
  - current flat-prior environment
- `rma_go2_lab/models/priors/flat_prior_runner_cfg.py`
  - current flat-prior PPO recipe
- `rma_go2_lab/envs/blind/blind_rough_forward_cfg.py`
  - current shared rough environment for Baseline 1, Baseline 2, and Baseline 3
- `rma_go2_lab/models/blind/blind_rough_runner_cfg.py`
  - blind PPO ladder and warm-start checkpoint wiring
- `rma_go2_lab/models/blind/ppo_with_flat_expert.py`
  - imitation variant
- `rma_go2_lab/envs/teacher/rough_v3_cfg.py`
  - final privileged teacher env
- `rma_go2_lab/envs/adaptation/rough_cfg.py`
  - no-adaptation switched student env
- `rma_go2_lab/envs/adaptation/rough_history_cfg.py`
  - history-based adaptation student env
- `rma_go2_lab/models/adaptation/ppo_with_v3_expert.py`
  - completed `Adapt-V0` training path
- `rma_go2_lab/models/adaptation/ppo_with_v3_latent.py`
  - `Adapt-V1` latent-regression path
- `rma_go2_lab/models/adaptation/modular_actor_critic.py`
  - `Adapt-V2` modular RMA-like scaffold
- `rma_go2_lab/models/adaptation/rma_v3_actor_critic.py`
  - active `Adapt-V3` actor-critic with explicit `mu / pi / phi`

## How To Read A Training Run

Look at these first:

- `track_lin_vel_xy_exp`
- `Metrics/base_velocity/error_vel_xy`
- `Episode_Termination/time_out`
- `Episode_Termination/base_height`
- `Episode_Termination/base_orientation`
- `Episode_Termination/low_progress`
- `Curriculum/terrain_levels`

Interpretation:

- high tracking + high timeout + low failure terms:
  healthy run
- low progress high:
  stuck policy
- base height high:
  terrain-clearance / body-clearance problem
- base orientation high:
  tipping / posture instability problem

## How To Judge A Baseline

Do not judge a baseline mainly by reward.

Use:

- tracking error
- time-to-failure
- failure-cause distribution
- recovery under mismatch
- slip / drift
- action effort

Reward is for training. Degradation under mismatch is the research result.

## Recommended Reading Order

Notes:

- synthesis and freeze docs should be treated as canonical truth first
- older `*PLAN*` and `*START*` notes are preserved mainly for phase history and
  decision lineage
  - check their status banners before treating them as current guidance

1. `docs/PROJECT_GUIDE.md`
2. `docs/archive/guides/PHASED_DOCS_GUIDE.md`
2. `rma_go2_lab/policies/blind_baseline_protocol.md`
3. `rma_go2_lab/policies/README.md`
4. `artifacts/evaluations/README.md`
5. `docs/BASELINE_COMPARISON_FINAL.md`
6. `docs/FROZEN_BASELINE_SYNTHESIS.md`
7. `docs/FROZEN_BASELINE_RESULTS_AT_A_GLANCE.md`
8. `docs/BASELINE_REGIME_CLOSED.md`
9. `docs/OOD_PROBE_PROTOCOL.md`
10. `docs/OOD_FINDINGS_B1_B2_B3.md`
11. `docs/archive/teacher/PRIVILEGED_TEACHER_START.md`
12. `docs/TEACHER_V4_MODEL300_CARD.md`
13. `docs/TEACHER_PHASE_SYNTHESIS.md`
14. `docs/archive/adaptation/ADAPTATION_PHASE_PLAN.md`
15. `docs/ADAPTATION_IMPLEMENTATION_V0.md`
16. `docs/EVALUATION_METHODS.md`
17. `docs/ADAPTATION_PHASE_SYNTHESIS.md`
18. `docs/archive/adaptation/ADAPTATION_V2_PLAN.md`
19. task-specific env / PPO config files

## What Not To Do

- do not keep reinventing blind-baseline reward design every few days
- do not merge every reward idea from every reference repo
- do not broaden command distributions casually
- do not use raw reward as proof that RMA is unnecessary

## One-Line Mental Model

This repo is organized around one clean story:

> establish fixed blind baselines, measure their degradation under hidden
> mismatch, justify privileged experts, then test deployable adaptation on the
> same switched tasks.
