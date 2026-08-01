# Adaptation Student V2 Freeze

This file freezes the final definition of the first explicitly modular
RMA-like adaptation student in the repo.

`studentAdapt-V2` is now frozen. Do not retune or overwrite it in place. Any
future changes should create a new explicitly versioned variant.

## Identity

- canonical name: `adaptation_student_v2_final`
- checkpoint:
  - `rma_go2_lab/policies/adaptation_student_v2_final.pt`
- source run:
  - `/home/bhuvan/projects/rma/rma_go2_lab/logs/rsl_rl/go2_adaptation_student_history_v2/2026-04-23_16-00-02`
- selected source checkpoint:
  - `model_1999.pt`
- freeze date:
  - `2026-04-24`

## Purpose

`studentAdapt-V2` is the first completed modular adaptation result in the repo.

It represents:

- deployable student observations at deployment
- current proprio plus history window
- explicit modular split:
  - `phi(history) -> z_hat`
  - `pi(current_obs, z_hat) -> action`
- frozen privileged `V3` teacher latent target during training
- hidden mid-episode dynamics switch regime
- the first repo branch that is structurally RMA-like rather than only
  RMA-inspired

It is used to answer:

- whether the repo can train a modular adaptation student to completion under
  the same switched task
- whether the explicit `phi` / `pi` split is viable before the future
  full-latent-contract `Adapt-V3` branch

## Training Definition

- task:
  - `RMA-Go2-Adaptation-Student-Rough-History-V2`
- env config:
  - `rma_go2_lab/envs/adaptation/rough_history_cfg.py`
- PPO config:
  - `rma_go2_lab/models/adaptation/adapt_v2_ppo_cfg.py`
- actor-critic:
  - `rma_go2_lab/models/adaptation/modular_actor_critic.py`
- algorithm:
  - `rma_go2_lab/models/adaptation/ppo_with_v3_latent.py`

## Final Training Snapshot Status

The final `1999/2000` terminal scalar block was not preserved as a trusted
artifact at freeze time.

What is known and frozen as truth:

- the `V2` run completed through `model_1999.pt`
- the final checkpoint exists and was archived from the canonical `V2` run
- the run lineage is separate from `V1`
- `V2` had already shown healthy mid-run learning and a valid modular
  adaptation path before completion

Because the final terminal block is unavailable, final standing should be
determined from the canonical post-fix eval matrix rather than from remembered
training scalars.

## Current Interpretation

`studentAdapt-V2` is the first completed modular RMA-like adaptation checkpoint
in the repo.

What it already proves:

- the explicit modular split is trainable end to end
- the repo can carry a `phi(history) -> z_hat`, `pi(obs, z_hat) -> action`
  branch through a full run
- modularization is now a completed result rather than just a scaffold

What it does not yet prove at freeze time:

- that `V2` is stronger than `V1`
- that `V2` is the final deployment winner
- that the modular split alone resolves the remaining body-stability issues

## Canonical Evaluation Status

`studentAdapt-V2` now has a completed canonical post-fix evaluation matrix.

Canonical eval artifacts live in:

- `artifacts/evaluations/adaptation_student_v2/`
- `artifacts/ood_evaluations/adaptation_student_v2/`

The frozen eval lineage covers:

- gait
- blind suite
- OOD geometry
- OOD dynamics
- OOD push
- OOD switch

Most important canonical finding:

- `V2` does not produce a distinct empirical result relative to `V1`
- across the canonical post-fix eval matrix, `V2` and `V1` produce identical
  evaluation outputs
- the shared actor/critic weights in the archived `V1` and `V2` checkpoints are
  identical
- the only checkpoint parameter-name difference is the expected architectural
  rename:
  - `history_encoder.*` in `V1`
  - `adaptation_module.*` in `V2`

So `V2` should be treated as:

- a real completed modular architectural milestone
- not a new performance improvement over `V1`

## Freeze Statement

`studentAdapt-V2` is now frozen.

Do not change:

- its checkpoint identity
- its source run lineage
- its task/config association

The comparison status against `NA`, `V0`, `V1`, and future `Adapt-V3` should now
be read together with:

- `docs/ADAPTATION_PHASE_SYNTHESIS.md`
