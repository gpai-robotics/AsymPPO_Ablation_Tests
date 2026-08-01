# Adapt-V3 Dyn-Only Phase 2 Recovery Low-Switch Latent-Reg Freeze

This file freezes the first bounded-latent continuation of the low-switch
recovery branch.

It should be read as the canonical bounded-latent recovery challenger, not as a
silent replacement for either:

- the stationary dyn-only Stage A deployment winner, or
- the earlier low-switch recovery anchor that first restored non-collapsed
  adaptation pressure.

## Identity

- canonical name:
  - `adapt_v3_dyn_only_phase2_recovery_low_switch_latent_reg_final`
- checkpoint:
  - `rma_go2_lab/policies/adapt_v3_dyn_only_phase2_recovery_low_switch_latent_reg_final.pt`
- source run:
  - `/home/bhuvan/tools/IsaacLab/logs/rsl_rl/go2_adapt_v3_phase2_recovery_low_switch_dyn_only_latent_reg/2026-05-04_15-25-43`
- selected source checkpoint:
  - `model_220.pt`
- freeze date:
  - `2026-05-05`

## Purpose

The earlier recovery artifact proved that the modern `Adapt-V3` stack could
recover real online adaptation pressure under active low-probability within-run
switches.

However, MuJoCo Sim2Sim debugging then exposed a new failure mode:

- `phi(history)` could produce a runaway latent under cross-engine history
  mismatch
- the latent explosion drove actor input norm blow-up
- the resulting action surge destabilized the robot in MuJoCo

This bounded-latent continuation exists to answer the next question:

- can we preserve real low-switch adaptation while making the student latent
  less brittle in MuJoCo without relying entirely on a deploy-side clamp?

This checkpoint is the first branch artifact that answers that question
positively enough to freeze.

## Training Definition

- task:
  - `RMA-Go2-Adapt-V3-DynOnly-Phase2-Recovery-LowSwitch-LatentReg`
- PPO config:
  - `rma_go2_lab/models/adaptation/adapt_v3_ppo_cfg.py`
- algorithm:
  - `rma_go2_lab/models/adaptation/ppo_rma_v3_phase2.py`

Key bounded-latent additions relative to the earlier low-switch recovery run:

- explicit student latent magnitude logging:
  - `student_latent_l2`
  - `student_latent_max_abs`
- small student latent L2 penalty during training
- checkpoint search focused on the early-mid training window instead of assuming
  the final checkpoint would be best

## Why This Branch Matters

This continuation preserved the things that made the recovery lane scientifically
useful:

- active low-probability switch exposure during training
- non-collapsed student latent behavior
- strong teacher-student latent alignment

While adding the first direct training-side response to the MuJoCo failure:

- keep latent magnitude in a tighter regime
- reduce the chance of catastrophic latent blow-up under cross-engine history
  mismatch

The important change is not that late training suddenly became perfect.

The important change is that the student latent stopped living in a regime that
made MuJoCo failure immediate and catastrophic.

## Selection Rationale

This run was treated explicitly as a checkpoint-search branch.

Early-mid shortlisted checkpoints:

- `model_100.pt`
- `model_160.pt`
- `model_220.pt`
- `model_300.pt`

Selection logic:

1. gait screen on the early-mid window
2. blind suite on the strongest gait candidates
3. `ood_switch_v1` tie-break on the top two
4. MuJoCo Sim2Sim check on the leading checkpoint

Outcome:

- gait screen leader:
  - `model_160.pt`
- blind-suite leader:
  - `model_220.pt`
- switch OOD leader:
  - `model_220.pt`

So `model_220.pt` became the first bounded-latent checkpoint to carry forward
into MuJoCo.

## Evaluation Summary

Evaluation directories:

- `artifacts/evaluations/adapt_v3_recovery_low_switch_latent_reg_ckpt_sweep/`
- `artifacts/debug/`

Canonical selected checkpoint artifacts:

- blind suite:
  - `artifacts/evaluations/adapt_v3_recovery_low_switch_latent_reg_ckpt_sweep/model_220/isolated_suite_model_220_blind_baseline_v1_random_rough_levelspread_normal_seed999.json`
  - `artifacts/evaluations/adapt_v3_recovery_low_switch_latent_reg_ckpt_sweep/model_220/isolated_suite_model_220_blind_baseline_v1_random_rough_levelspread_normal_seed999.csv`
- switch OOD tie-break:
  - `artifacts/evaluations/adapt_v3_recovery_low_switch_latent_reg_ckpt_sweep/model_220/ood_suite_model_220_ood_switch_v1_normal_seed999.json`
  - `artifacts/evaluations/adapt_v3_recovery_low_switch_latent_reg_ckpt_sweep/model_220/ood_suite_model_220_ood_switch_v1_normal_seed999.csv`

Key comparison result:

- blind suite mean score:
  - `model_220 = 8.5199`
  - `model_160 = 8.4304`
  - `model_100 = 8.2995`
- switch OOD mean score:
  - `model_220 = 7.5180`
  - `model_160 = 7.5017`

Interpretation:

- `model_220` is the strongest bounded-latent checkpoint by the same
  checkpoint-sweep discipline used elsewhere in the repo
- `model_160` remained a credible conservative challenger, especially in the
  `very_heavy` switch case, but did not win the aggregate tie-break

## MuJoCo Meaning

The main reason this checkpoint matters is what happened in Sim2Sim.

Compared to the earlier recovery artifact:

- unclamped MuJoCo reward proxy improved substantially
- velocity and yaw error fell sharply
- mean base height improved materially
- mean tilt improved materially
- catastrophic action explosion disappeared

Key debug artifacts:

- earlier recovery unclamped:
  - `artifacts/debug/adapt_v3_dyn_only_phase2_recovery_low_switch_sim2sim.json`
- earlier recovery clamp reference:
  - `artifacts/debug/adapt_v3_dyn_only_phase2_recovery_low_switch_sim2sim_clamp5.json`
- bounded-latent `model_220` unclamped:
  - `artifacts/debug/adapt_v3_dyn_only_phase2_recovery_low_switch_latent_reg_model220_sim2sim.json`
- bounded-latent `model_220` clamp reference:
  - `artifacts/debug/adapt_v3_dyn_only_phase2_recovery_low_switch_latent_reg_model220_sim2sim_clamp5.json`

High-level interpretation:

- the earlier recovery artifact failed because the latent exploded almost
  immediately
- the bounded-latent continuation reduced that failure from “catastrophic
  runaway” to “residual latent drift”
- deploy-side clamping still helps, but the training-side fix already changed
  the MuJoCo behavior class in a meaningful way

## Project Meaning

This checkpoint is the first canonical artifact in the repo that supports all
of the following at once:

- real low-switch adaptation pressure during training
- bounded-latent training instrumentation
- a training-side fix for MuJoCo latent brittleness
- materially improved unclamped MuJoCo behavior relative to the first recovery
  artifact

It does **not** yet prove that the adaptive deployment story is finished.

It does prove that the project’s next barrier is now narrower:

- preserve the adaptation gains
- keep locomotion strong
- and further reduce latent drift under deployment-side history mismatch

That makes this checkpoint the canonical bounded-latent recovery challenger and
the right refinement base for the next iteration.

## Later Branch Outcome

A later refinement branch added a stronger coordinate-wise max-abs penalty on
top of this bounded-latent line:

- `RMA-Go2-Adapt-V3-DynOnly-Phase2-Recovery-LowSwitch-LatentReg-MaxAbs`

That branch selected `model_300.pt` as its best Isaac checkpoint after gait,
blind-suite, and switch-OOD selection, but it did not hold up in MuJoCo.

Compared with this bounded-latent checkpoint, the later max-abs branch was:

- worse unclamped in MuJoCo
- more brittle in latent magnitude under deployment-side history shift
- not strong enough even after clamp-5 to justify replacing this artifact as
  the active adaptive challenger

So this freeze remains the canonical adaptive MuJoCo challenger after the
max-abs branch outcome was recorded.

A later refinement branch then added a weak temporal latent-delta penalty on
top of the same bounded-latent line:

- `RMA-Go2-Adapt-V3-DynOnly-Phase2-Recovery-LowSwitch-LatentReg-Smooth`

That branch selected `model_100.pt` as its best Isaac checkpoint after gait,
blind-suite, and switch-OOD selection, but it also did not hold up in MuJoCo.

Compared with this bounded-latent checkpoint, the later smooth branch was:

- substantially worse unclamped in MuJoCo
- better than the max-abs branch under clamp-5
- still not strong enough to justify replacing this artifact as the active
  adaptive challenger

So this freeze remains the canonical adaptive MuJoCo challenger after both the
max-abs and temporal-smoothness branch outcomes were recorded.
