# Adapt-V3 Dyn-Only Phase 2 Recovery Low-Switch Freeze

This file freezes the first successful recovery checkpoint that restored
non-collapsed online latent behavior under a low but nonzero mid-episode
switch-training contract.

It should be read as the canonical checkpoint for the recovery branch, not as a
replacement for the original stationary Stage A winner.

## Identity

- canonical name:
  - `adapt_v3_dyn_only_phase2_recovery_low_switch_final`
- checkpoint:
  - `rma_go2_lab/policies/adapt_v3_dyn_only_phase2_recovery_low_switch_final.pt`
- source run:
  - `/home/bhuvan/tools/IsaacLab/logs/rsl_rl/go2_adapt_v3_phase2_recovery_low_switch_dyn_only/2026-05-03_18-41-30`
- selected source checkpoint:
  - `model_1200.pt`
- freeze date:
  - `2026-05-04`

## Purpose

This checkpoint exists because the original dyn-only Phase 2 Stage A winner:

- was a strong blind deployable student
- but did not demonstrate real online-changing latent behavior under the
  deployment-side adaptation probe

The recovery branch reintroduced low-probability within-episode hidden-dynamics
switches and explicit latent-health diagnostics in order to answer a different
question:

- can the student recover real online latent adaptation pressure without
  immediately collapsing locomotion?

This checkpoint is the first branch artifact that answered that question
positively enough to freeze.

## Training Definition

- task:
  - `RMA-Go2-Adapt-V3-DynOnly-Phase2-Recovery-LowSwitch`
- env config:
  - `rma_go2_lab/envs/adaptation/rough_history_switch_recovery_dyn_only_cfg.py`
- PPO config:
  - `rma_go2_lab/models/adaptation/adapt_v3_ppo_cfg.py`
- actor-critic:
  - `rma_go2_lab/models/adaptation/rma_v3_actor_critic.py`
- algorithm:
  - `rma_go2_lab/models/adaptation/ppo_rma_v3_phase2.py`

Key recovery design choices:

- reuse the strong Stage A actor/critic trunk
- do not inherit the collapsed Stage A `phi`
- initialize `phi` from a small random start rather than zero
- keep `mu` initialization fixed/quiet
- low but nonzero switch regime
- temporary small teacher-imitation scaffold at the start
- explicit latent-health logging during training

## Why This Branch Worked

The branch only became viable after several recovery-specific fixes:

- actor-only warm start stopped inheriting the collapsed full Phase 2 student
- `phi` stopped starting from a dead-zero initialization
- `phi` initialization was reduced from full Xavier to a smaller-gain
  initialization (`small_xavier`)
- the switch regime was kept low-probability rather than aggressively mixed

This created the first training lane in which:

- `student_latent_batch_std` stayed nonzero
- `latent_cosine` recovered strongly
- locomotion remained good enough to survive into evaluation

## Selection Rationale

This run did **not** select the last checkpoint by default.

Observed run story:

- early recovery:
  - `phi` came back to life
  - latent-health metrics improved sharply
- late recovery:
  - latent alignment kept improving
  - final `model_1999.pt` showed very strong adaptation metrics
  - but long-horizon locomotion quality degraded badly by the end of training

Because of that, a checkpoint sweep was run.

Shortlisted sweep checkpoints:

- `model_220.pt`
- `model_1200.pt`
- `model_1999.pt`

Selection logic:

1. screen by gait diagnostics
2. compare shortlisted checkpoints on the blind suite
3. break the tie using `ood_switch_v1`

Outcome:

- `model_1200.pt` won the blind suite
- `model_1200.pt` also won the `ood_switch_v1` tie-break against
  `model_1999.pt`

So `model_1200.pt` is the best recovered compromise point:

- real adaptation restored
- stronger overall evaluation quality than both the earlier and later
  shortlisted checkpoints

## Training Snapshot Context

The final training endpoint (`model_1999.pt`) is important negative evidence:

- adaptation metrics became excellent there
- but posture and body-height terminations worsened too much

That final endpoint should therefore be read as:

- proof that adaptation can be revived
- not proof that the last checkpoint is the best overall artifact

This checkpoint freeze preserves the stronger compromise discovered by the
checkpoint sweep.

## Evaluation Summary

Evaluation directory:

- `artifacts/evaluations/adapt_v3_recovery_low_switch_ckpt_sweep`

Canonical selected checkpoint artifacts:

- blind suite:
  - `artifacts/evaluations/adapt_v3_recovery_low_switch_ckpt_sweep/model_1200/isolated_suite_model_1200_blind_baseline_v1_random_rough_levelspread_normal_seed999.json`
  - `artifacts/evaluations/adapt_v3_recovery_low_switch_ckpt_sweep/model_1200/isolated_suite_model_1200_blind_baseline_v1_random_rough_levelspread_normal_seed999.csv`
- switch OOD tie-break:
  - `artifacts/evaluations/adapt_v3_recovery_low_switch_ckpt_sweep/model_1200/ood_suite_model_1200_ood_switch_v1_normal_seed999.json`
  - `artifacts/evaluations/adapt_v3_recovery_low_switch_ckpt_sweep/model_1200/ood_suite_model_1200_ood_switch_v1_normal_seed999.csv`

Key comparison result:

- blind suite mean score:
  - `model_1200 = 8.3897`
  - `model_1999 = 8.3758`
  - `model_220 = 8.2214`
- switch OOD mean score:
  - `model_1200 = 7.2702`
  - `model_1999 = 7.1681`

Interpretation:

- `model_1200` is the best current recovery checkpoint by the same style of
  metric used elsewhere in the repo
- `model_1999` remained a credible challenger, but did not beat `1200` once
  both broad blind robustness and switched hidden-dynamics robustness were
  considered together

## Project Meaning

This checkpoint is the first canonical artifact in the repo that supports all
of the following at once:

- blind student deployment contract
- low-probability real switch exposure during training
- non-collapsed online latent behavior during training
- checkpoint-swept selection instead of naive last-checkpoint freezing

It does **not** yet prove that the recovery branch is the final best policy
family for deployment.

It does prove that the project can recover genuine adaptation pressure without
immediately losing the entire locomotion stack.

That makes this checkpoint the new canonical recovery anchor for future work.
