# C2 Supervised Restart Plan

## Purpose

This note defines the next serious C2 direction after the failed long-horizon
Phase 2 branches:

- dyn-only PPO Phase 2 variants
- dyn-only TCN variants
- dyn-only DAgger-style Phase 2
- terrain-lite DAgger-style reopening

The common pattern was:

- `phi(history)` improved against the teacher latent
- hidden-switch adaptation became real
- locomotion quality still degraded over long horizons
- terrain-aware variants also reintroduced the old crouch / posture fragility

So the problem is no longer "can `phi` learn anything?".

The problem is:

**how do we keep adaptation load-bearing without letting the control policy fall
into a degraded gait family?**

## Core Lesson From Reference Repos

Two reference repos point to the same structural answer:

- `rl_locomotion`
  - freezes the expert policy and student controller MLP
  - trains only the history encoder against expert latent targets
- `vision_locomotion`
  - treats latent production as a separate predictor problem
  - keeps the controller fixed
  - swaps / falls back between latent sources at runtime

The shared lesson is:

**do not keep relearning locomotion while learning the latent supplier.**

## New C2 Decomposition

The next C2 path should be split into explicit stages.

### Stage S0: Fixed Controller Root

Choose a controller root that is already behaviorally acceptable.

For the dyn-only restart, that is the current C2 baseline root:

- `rma_go2_lab/policies/adapt_v3_dyn_only_phase2_recovery_low_switch_latent_reg_final.pt`

For a future terrain-aware restart, do not reuse the old terrain-lite branch
root blindly. Build a stronger root from the modern trusted teacher lineage.

### Stage S1: Teacher-Target Dataset Collection

Collect rollout data from the chosen task regime while logging:

- `policy`
- `policy_history`
- privileged extrinsics groups
  - `dynamics_privileged`
  - optionally `terrain_privileged`
- `teacher_latent = mu(e_t)`
- `teacher_action = pi(x_t, mu(e_t))`
- rollout action actually applied
- switch metadata / active mask

This dataset is meant to define the teacher latent contract offline.

### Stage S2: Supervised Latent-Estimator Training

Train `phi(history)` offline on the collected dataset.

Primary objective:

- regress `phi(history)` to `teacher_latent`

Optional secondary objective:

- reproduce `teacher_action` through a frozen controller head

Important constraint:

- the controller stays fixed
- no PPO policy improvement is active in this stage

This is the key shift from the failed C2 Phase 2 branches.

### Stage S3: Runtime Substitution Evaluation

Evaluate the frozen controller with:

- teacher latent path
  - `pi(x_t, mu(e_t))`
- deployable history latent path
  - `pi(x_t, phi(history_t))`

Then compare:

- gait quality
- hidden-switch recovery
- long-horizon stability
- low-friction weakness

### Stage S4: Optional Fine-Tuning Only If Needed

If supervised `phi` is close but not good enough, allow only a very bounded
follow-up stage. This must not become another full long-horizon PPO branch by
default.

Any follow-up fine-tune should preserve:

- fixed controller trunk
- explicit early-stop criteria
- mandatory gait inspection

## Why This Is Better Than Current Phase 2

Current Phase 2, even in DAgger-like form, still lives inside the RL rollout
loop. That has kept latent learning and behavior drift too entangled.

The supervised restart separates:

- behavior generation
- teacher target definition
- deployable latent estimation

This makes failures easier to interpret:

- if supervised `phi` fails, the latent estimator is the problem
- if supervised `phi` succeeds but runtime gait is still bad, the root
  controller / latent contract is the problem

## Immediate Implementation Pieces

The first concrete tooling for this restart is:

- `scripts/adaptation/collect_adapt_v3_teacher_dataset.py`

This collector produces teacher-supervision datasets from real IsaacLab
rollouts using:

- a rollout policy checkpoint
- a frozen Phase 1 teacher reference

It logs the deployable history inputs and the privileged teacher targets in the
same samples.

## Success Criteria

The supervised restart is only worth continuing if it shows all of these:

1. `phi(history)` reaches strong latent alignment offline.
2. Runtime gait remains at least as clean as the controller root.
3. Hidden-switch recovery remains real under deployable history-only inference.
4. Terrain-aware variants do not reintroduce crouch / support-heavy posture by
   default.

## Non-Goals

This restart is not meant to:

- prove that more PPO pressure fixes C2
- reopen the old terrain-lite branch directly
- reward-hack around bad gait while latent metrics improve

The purpose is to make C2 structurally cleaner and easier to debug.
