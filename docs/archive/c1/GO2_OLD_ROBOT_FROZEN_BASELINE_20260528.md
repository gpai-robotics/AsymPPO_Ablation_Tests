# Go2 Old Robot Frozen Baseline

Date frozen: `2026-05-28`

This document freezes the currently accepted "working, kinda stable"
old-robot deployment baseline for the Go2 controller path.

Use this as the rollback point for old-robot deployment work. Do not casually
retune the live old-robot deploy config without preserving a new dated snapshot.

## Frozen Config Snapshot

Exact frozen file:

- [deploy_frozen_old_robot_20260528.yaml](/home/bhuvan/projects/rma/rma_go2_lab/reference_repos/unitree_rl_lab/deploy/robots/go2/config/policy/velocity/c1_blind_rough_omni_usable_v1_final/params/deploy_frozen_old_robot_20260528.yaml)

Active live file that should match the frozen snapshot right now:

- [deploy.yaml](/home/bhuvan/projects/rma/rma_go2_lab/reference_repos/unitree_rl_lab/deploy/robots/go2/config/policy/velocity/c1_blind_rough_omni_usable_v1_final/params/deploy.yaml)

Controller config pinned to the old bundle:

- [config.yaml](/home/bhuvan/projects/rma/rma_go2_lab/reference_repos/unitree_rl_lab/deploy/robots/go2/config/config.yaml)

## Baseline Intent

This baseline is for the old robot only.

Important choices retained in this freeze:

- old robot explicitly pinned to the old policy bundle
- `pose_compatibility` disabled
- `neutral_action_compensation` disabled
- `base_lin_vel` source set to `odometry`
- conservative omnidirectional command envelope

The exact deploy file is the source of truth. At freeze time, the accepted feel
was:

- forward modest and usable
- lateral usable
- yaw not exercised in the accepted run
- overall "working, kinda stable" rather than polished/final

## Accepted Run

Accepted summary source:

- [go2_ctrl_20260528_160113.log](/home/bhuvan/projects/rma/rma_go2_lab/logs/go2_ctrl/go2_ctrl_20260528_160113.log)

Why this run was accepted:

- clean FSM lifecycle
- command path behaved as expected
- old robot felt good enough to keep as a safe baseline
- better than the more aggressive or over-constrained variants tested earlier

## Restore Procedure

If the live old-robot config gets changed and you want to restore this exact
baseline:

```bash
cd /home/bhuvan/projects/rma/rma_go2_lab
cp \
  reference_repos/unitree_rl_lab/deploy/robots/go2/config/policy/velocity/c1_blind_rough_omni_usable_v1_final/params/deploy_frozen_old_robot_20260528.yaml \
  reference_repos/unitree_rl_lab/deploy/robots/go2/config/policy/velocity/c1_blind_rough_omni_usable_v1_final/params/deploy.yaml
```

Then use the canonical old-robot stack:

```bash
cd /home/bhuvan/projects/rma/rma_go2_lab
scripts/deploy/run_go2_old_robot_stack.sh odom enp0s31f6
scripts/deploy/run_go2_old_robot_stack.sh bridge enp0s31f6
scripts/deploy/run_go2_old_robot_stack.sh ctrl enp0s31f6
scripts/deploy/run_go2_old_robot_stack.sh summary latest
```

## Change Discipline

If we tune the old robot again later:

1. copy `deploy.yaml` to a new dated frozen filename first
2. document the accepted log for that new version
3. do not overwrite this `20260528` baseline snapshot
