# GO2 Deployment Patch Log

Date: 2026-05-28

This note records the deployment-side patches and experiments we introduced while stabilizing Go2 hardware deployment in `/home/bhuvan/projects/rma/rma_go2_lab`.

The goal is twofold:

1. keep a reproducible record of what we changed at deployment time
2. later decide which of these are really sim-to-real training gaps versus deploy-only necessities

Related:

- [GO2_OLD_ROBOT_FROZEN_BASELINE_20260528.md](/home/bhuvan/projects/rma/rma_go2_lab/docs/GO2_OLD_ROBOT_FROZEN_BASELINE_20260528.md)
- [UNITREE_RL_LAB_REPRO_AUDIT_20260528.md](/home/bhuvan/projects/rma/rma_go2_lab/docs/UNITREE_RL_LAB_REPRO_AUDIT_20260528.md)
- [deploy_frozen_old_robot_20260528.yaml](/home/bhuvan/projects/rma/rma_go2_lab/reference_repos/unitree_rl_lab/deploy/robots/go2/config/policy/velocity/c1_blind_rough_omni_usable_v1_final/params/deploy_frozen_old_robot_20260528.yaml)

## Scope

This log focuses on deployment/runtime patches for the old robot stack, not the separate new-robot diagnosis work.

## Patch Record

### 1. Pin old robot to the correct policy bundle

Problem:
- the old robot was accidentally loading the new-robot bundle because `policy_dir` pointed at a parent directory and the runtime auto-selected a child bundle

Patch:
- pin `policy_dir` to the explicit old bundle path

Files:
- [config.yaml](/home/bhuvan/projects/rma/rma_go2_lab/reference_repos/unitree_rl_lab/deploy/robots/go2/config/config.yaml)
- [param.h](/home/bhuvan/projects/rma/rma_go2_lab/reference_repos/unitree_rl_lab/deploy/include/param.h)

Why this matters:
- this is a deployment/reproducibility issue, not a training issue

Training candidate:
- `No`

### 2. Restore `base_lin_vel` observation usage

Problem:
- old-robot deployment had `base_lin_vel` effectively zeroed, which caused posture-only behavior and poor locomotion response

Patch:
- restore `observations.policy_obs.base_lin_vel.scale`
- restore `observations.policy_history.base_lin_vel.scale`

Files:
- [deploy.yaml](/home/bhuvan/projects/rma/rma_go2_lab/reference_repos/unitree_rl_lab/deploy/robots/go2/config/policy/velocity/c1_blind_rough_omni_usable_v1_final/params/deploy.yaml)

Why this matters:
- this was a deployment config regression, not a policy weakness

Training candidate:
- `No`

### 3. Remove new-robot-only compatibility layers from old robot baseline

Problem:
- `pose_compatibility` and `neutral_action_compensation` were introduced during new-robot debugging and contaminated the old-robot stack

Patch:
- disable both in the old-robot active config

Files:
- [deploy.yaml](/home/bhuvan/projects/rma/rma_go2_lab/reference_repos/unitree_rl_lab/deploy/robots/go2/config/policy/velocity/c1_blind_rough_omni_usable_v1_final/params/deploy.yaml)

Why this matters:
- this is mainly stack hygiene and old/new robot separation

Training candidate:
- `No`

### 4. Add faster release decay for velocity commands

Problem:
- after joystick release, `target` dropped to zero but `filtered` command decayed too slowly

Patch:
- add optional `release_slew_rate` support
- use faster rates on release than on command ramp-up

Files:
- [observations.h](/home/bhuvan/projects/rma/rma_go2_lab/reference_repos/unitree_rl_lab/deploy/include/isaaclab/envs/mdp/observations/observations.h)
- [deploy.yaml](/home/bhuvan/projects/rma/rma_go2_lab/reference_repos/unitree_rl_lab/deploy/robots/go2/config/policy/velocity/c1_blind_rough_omni_usable_v1_final/params/deploy.yaml)

Why this matters:
- this is a real deployment interface improvement
- however, needing it strongly may indicate training did not sufficiently emphasize stop-on-release behavior

Training candidate:
- `Partial`

Notes:
- likely still useful as a deployment control primitive even if retraining improves stop behavior

### 5. Add zero-command stop latch

Problem:
- even with faster release decay, command release still felt sticky

Patch:
- track neutral-command hold time
- after a short hold window, hard-zero target and filtered commands

Files:
- [manager_based_rl_env.h](/home/bhuvan/projects/rma/rma_go2_lab/reference_repos/unitree_rl_lab/deploy/include/isaaclab/envs/manager_based_rl_env.h)
- [observations.h](/home/bhuvan/projects/rma/rma_go2_lab/reference_repos/unitree_rl_lab/deploy/include/isaaclab/envs/mdp/observations/observations.h)
- [deploy.yaml](/home/bhuvan/projects/rma/rma_go2_lab/reference_repos/unitree_rl_lab/deploy/robots/go2/config/policy/velocity/c1_blind_rough_omni_usable_v1_final/params/deploy.yaml)

Why this matters:
- this is a deployment-side operator safety/control patch
- but the fact that we need it is a strong signal that training under-penalized residual motion at zero command

Training candidate:
- `Yes`

Training interpretation:
- stronger reward/penalty for zero-command standstill
- more command pulse/release curriculum

### 6. Tune command envelopes by axis

Problem:
- shared omni command limits did not fit all axes equally
- forward was fragile at magnitudes that lateral/yaw could tolerate

Patch:
- explore axis-specific command ranges
- settle around a more conservative forward range and separate lateral/yaw caps

Files:
- [deploy.yaml](/home/bhuvan/projects/rma/rma_go2_lab/reference_repos/unitree_rl_lab/deploy/robots/go2/config/policy/velocity/c1_blind_rough_omni_usable_v1_final/params/deploy.yaml)

Why this matters:
- some of this is normal deployment tuning
- but strong asymmetry can also indicate training did not produce sufficiently decoupled axis behavior on real hardware

Training candidate:
- `Partial`

Training interpretation:
- increase isolated-axis command coverage
- penalize off-axis motion during single-axis commands

### 7. Reduce yaw command cap from frozen baseline

Problem:
- frozen baseline still allowed `ang_vel_z` up to `±1.0`, which produced extremely large yaw commands in practice

Patch:
- reduce yaw range to a saner deploy value

Files:
- [deploy.yaml](/home/bhuvan/projects/rma/rma_go2_lab/reference_repos/unitree_rl_lab/deploy/robots/go2/config/policy/velocity/c1_blind_rough_omni_usable_v1_final/params/deploy.yaml)
- baseline reference:
  - [deploy_frozen_old_robot_20260528.yaml](/home/bhuvan/projects/rma/rma_go2_lab/reference_repos/unitree_rl_lab/deploy/robots/go2/config/policy/velocity/c1_blind_rough_omni_usable_v1_final/params/deploy_frozen_old_robot_20260528.yaml)

Why this matters:
- mostly deployment safety/range selection

Training candidate:
- `Partial`

### 8. Reduce `JointPositionAction.scale`

Problem:
- policy output felt too aggressive
- robot showed overshoot, continued motion after release, and cross-axis coupling

Patch:
- reduce action scale from `0.25` down through `0.20`, `0.18`, and then compromise around `0.19`

Files:
- [deploy.yaml](/home/bhuvan/projects/rma/rma_go2_lab/reference_repos/unitree_rl_lab/deploy/robots/go2/config/policy/velocity/c1_blind_rough_omni_usable_v1_final/params/deploy.yaml)

Why this matters:
- this is one of the clearest deployment compensations for a real sim-to-real behavior gap

Training candidate:
- `Yes`

Training interpretation:
- stronger action-rate regularization
- smoother policy outputs
- more realistic actuation/response in sim

### 9. Add joint target slew limit after policy output

Problem:
- other Go2 sim-to-real deployments stabilize hardware by limiting how much target joint positions can change per policy step
- our runtime did not have this protection

Patch:
- add optional `slew_limit_rad`
- apply it in joint action processing
- set `slew_limit_rad: 0.10`

Files:
- [joint_actions.h](/home/bhuvan/projects/rma/rma_go2_lab/reference_repos/unitree_rl_lab/deploy/include/isaaclab/envs/mdp/actions/joint_actions.h)
- [deploy.yaml](/home/bhuvan/projects/rma/rma_go2_lab/reference_repos/unitree_rl_lab/deploy/robots/go2/config/policy/velocity/c1_blind_rough_omni_usable_v1_final/params/deploy.yaml)

Why this matters:
- this is a strong deployment-side stabilizer
- but if the policy only works with heavy post-filtering, training likely still needs work

Training candidate:
- `Yes`

Training interpretation:
- include comparable action/target slew limitation during training
- or increase action smoothness penalties so the policy naturally stays inside this envelope

### 10. Old-robot stack hardening and tooling

Problem:
- deployment workflow had become fragmented and hard to reproduce

Patch:
- add canonical wrappers
- add summaries
- add frozen baseline and repo snapshots

Files:
- [run_go2_old_robot_stack.sh](/home/bhuvan/projects/rma/rma_go2_lab/scripts/deploy/run_go2_old_robot_stack.sh)
- [summarize_go2_ctrl_log.py](/home/bhuvan/projects/rma/rma_go2_lab/scripts/deploy/summarize_go2_ctrl_log.py)
- [GO2_OLD_ROBOT_FROZEN_BASELINE_20260528.md](/home/bhuvan/projects/rma/rma_go2_lab/docs/GO2_OLD_ROBOT_FROZEN_BASELINE_20260528.md)
- [UNITREE_RL_LAB_REPRO_AUDIT_20260528.md](/home/bhuvan/projects/rma/rma_go2_lab/docs/UNITREE_RL_LAB_REPRO_AUDIT_20260528.md)
- [snapshot_unitree_rl_lab_state.sh](/home/bhuvan/projects/rma/rma_go2_lab/scripts/deploy/snapshot_unitree_rl_lab_state.sh)

Why this matters:
- reproducibility and operator workflow

Training candidate:
- `No`

## Preliminary Training Migration Candidates

These patches look like the strongest signs of training-side gaps rather than permanent deployment hacks:

- zero-command stop latch
- action scale reduction
- joint target slew limit
- strong axis-specific command fragility and cross-axis coupling

These are likely training themes:

- better stop-on-release behavior
- more decoupled `vx` / `vy` / `wz` tracking
- stronger penalties on off-axis motion
- stronger penalties on residual motion at zero command
- smoother action sequences
- training with action/target slew constraints that match deployment

## Likely Permanent Deployment Concerns

These should probably stay as deployment/runtime concerns even after retraining:

- explicit bundle pinning
- frozen baseline snapshots
- old/new robot stack separation
- command range safety bounds
- logging/summaries and reproducibility tooling

## Next Use

When we reopen training, use this note as a checklist:

1. mark which deploy patches still seem necessary after retraining
2. remove the ones that were only compensating for training shortcomings
3. keep the ones that are really operator-safety or reproducibility features
