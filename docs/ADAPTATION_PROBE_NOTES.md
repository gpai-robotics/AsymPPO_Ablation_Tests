# Adaptation Probe Notes

This note captures the current deployment-side adaptation probe result for the
active dyn-only `Adapt-V3` winner.

## Candidate

- Policy:
  [adapt_v3_dyn_only_phase2_stage_a_final.pt](/home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/policies/adapt_v3_dyn_only_phase2_stage_a_final.pt)
- Bundle:
  [adapt_v3_dyn_only_phase2_stage_a_final](/home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/policies/exported/adapt_v3_dyn_only_phase2_stage_a_final)
- Probe entrypoint:
  [play_deploy_policy.py](/home/bhuvan/projects/rma/rma_go2_lab/scripts/deploy/play_deploy_policy.py)

## Probe Intent

The question was not "does the student perform well overall?".

That was already answered by the evaluation and deployment-rehearsal work.

The question here was narrower:

- does the hidden dynamics actually change mid-episode?
- does `policy_history` change after that switch?
- does `phi(history)` produce a different latent?
- does the actor actually care about the latent it receives?

## Probe Setup

The probe reuses the existing Stage-A hidden-dynamics switch machinery from the
adaptation environment rather than inventing a separate perturbation path.

Test conditions:

- one environment
- fixed command:
  - `vx = 0.5`
  - `vy = 0.0`
  - `yaw = 0.0`
- one forced mid-rollout switch at step `400`
- probe windows:
  - `100` steps before switch
  - `100` steps after switch

Tested hidden-factor switches:

- friction drop
- added base mass
- weakened motor gains

## What Was Verified

The hidden factor change is real.

Examples:

- friction probe:
  - static friction changed from about `1.2235` to `0.1000`
  - dynamic friction changed from about `0.9645` to `0.1000`
- mass probe:
  - base-mass ratio changed from about `1.4156` to `1.5497`
- motor probe:
  - joint stiffness scale changed from about `0.9749` to `0.7000`
  - joint damping scale changed from about `1.0347` to `0.7000`

Behavior also changes after the switch, especially in the friction case:

- `vel_err_step_mean` increases
- `yaw_err_step_mean` increases
- `base_height_mean` decreases

So the probe is not fake or inert.

## Main Finding

The history stream changes, but the inferred latent does not.

Under the friction probe:

- pre-switch:
  - `history_delta_from_prev_mean = 36.60`
  - `history_delta_from_initial_mean = 44.17`
- post-switch:
  - `history_delta_from_prev_mean = 44.31`
  - `history_delta_from_initial_mean = 54.96`

But:

- `latent_delta_from_prev_mean = 0.0`
- `latent_delta_from_initial_mean = 0.0`
- `latent_cosine_to_initial_mean = 1.0`

Detailed latent snapshots around the switch also show the same result:

- latent mean is unchanged step to step
- latent norm is unchanged
- cosine to initial latent stays exactly `1.0`

This means the current deployed dyn-only student does **not** show evidence of
online latent adaptation in the strong sense.

## Latent Usage Ablation

The actor still cares about the latent it is given.

Full-rollout friction-probe ablation:

- `live_vs_frozen_latent_action_abs_diff_mean_full_rollout = 0.0`
- `live_vs_zero_latent_action_abs_diff_mean_full_rollout = 0.2084`
- `frozen_vs_zero_latent_action_abs_diff_mean_full_rollout = 0.2084`

Interpretation:

- replacing the live latent with the frozen initial latent does nothing
- replacing the latent with zeros changes the action noticeably

So the actor uses the latent channel, but `phi(history)` appears to have
collapsed to a fixed code under this probe.

## Current Interpretation

The best current description of the dyn-only deployed student is:

- a strong blind proprioceptive student
- with a latent-conditioned actor
- where the latent behaves like a fixed learned bias rather than a visibly
  changing online estimate under the tested hidden-dynamics switch

This is an important project truth.

It does **not** invalidate:

- the final dyn-only candidate quality
- the deployment packaging work
- the Sim2Sim bridge work

But it does change what we should honestly claim.

## What We Can Honestly Claim Now

Reasonable:

- the dyn-only final student is strong and deployable
- the student consumes `policy_history`
- the actor uses a non-zero latent channel
- the deployment bridge is working

Not yet justified:

- that the final dyn-only deployment artifact demonstrates clear online latent
  adaptation to hidden dynamics changes

## Recommended Follow-Up

If we want stronger adaptation behavior in a future branch, investigate:

- whether Phase 2 training lets `phi(history)` collapse too easily
- stronger temporal/variation pressure on the latent
- stronger switch-conditioned adaptation objectives
- training-time diagnostics that log latent motion before freezing

For now, this note should be treated as the source of truth for the current
dyn-only adaptation claim.
