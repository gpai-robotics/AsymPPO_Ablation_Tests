# Unitree RL Lab Repro Audit

Date: `2026-05-28`

This note exists because `reference_repos/unitree_rl_lab` has been used as a
live deployment workbench during Go2 hardware bring-up. That was useful for
debugging, but it means reproducibility now depends on explicitly freezing the
state instead of assuming the reference repo stayed pristine.

## Why This Matters

The repo at:

- [reference_repos/unitree_rl_lab](/home/bhuvan/projects/rma/rma_go2_lab/reference_repos/unitree_rl_lab)

is no longer just a clean upstream checkout. It now contains:

- tracked code changes in the deploy runtime
- old-robot deployment config edits
- new-robot experimental runtime/config additions
- extra binaries / entrypoints / helpers used during hardware debugging

So future deployment work should treat this repo as a locally evolved runtime,
not as an untouched reference.

## Current Base Commit

Current `unitree_rl_lab` HEAD at audit time:

- `4960b84`

## Current Dirty State Summary

Tracked modified files included at audit time:

- `deploy/include/FSM/FSMState.h`
- `deploy/include/FSM/State_FixStand.h`
- `deploy/include/FSM/State_RLBase.h`
- `deploy/include/isaaclab/algorithms/algorithms.h`
- `deploy/include/isaaclab/assets/articulation/articulation.h`
- `deploy/include/isaaclab/envs/manager_based_rl_env.h`
- `deploy/include/isaaclab/envs/mdp/actions/joint_actions.h`
- `deploy/include/isaaclab/envs/mdp/observations/observations.h`
- `deploy/include/isaaclab/manager/manager_term_cfg.h`
- `deploy/include/isaaclab/manager/observation_manager.h`
- `deploy/include/unitree_articulation.h`
- `deploy/robots/go2/CMakeLists.txt`
- `deploy/robots/go2/config/config.yaml`
- `deploy/robots/go2/include/Types.h`
- `deploy/robots/go2/main.cpp`
- `deploy/robots/go2/src/State_RLBase.cpp`

Untracked additions included at audit time:

- `deploy/robots/go2/config/config_new_robot.yaml`
- `deploy/robots/go2/config/policy/`
- `deploy/robots/go2/include/OdometryState.h`
- `deploy/robots/go2/recover_main.cpp`
- `deploy/robots/go2/smoke_main.cpp`
- `deploy/robots/go2_new_robot_runtime/`

## Important Practical Split

At this point we should think of the local `unitree_rl_lab` tree as containing
two layers:

1. old-robot baseline path
2. new-robot experimental path

Those should not be mixed casually.

The currently frozen old-robot deployment baseline is documented separately in:

- [GO2_OLD_ROBOT_FROZEN_BASELINE_20260528.md](/home/bhuvan/projects/rma/rma_go2_lab/docs/GO2_OLD_ROBOT_FROZEN_BASELINE_20260528.md)

## Reproducibility Guardrail

Use this script to snapshot the current local `unitree_rl_lab` state before any
future large deployment changes:

- [snapshot_unitree_rl_lab_state.sh](/home/bhuvan/projects/rma/rma_go2_lab/scripts/deploy/snapshot_unitree_rl_lab_state.sh)

Example:

```bash
cd /home/bhuvan/projects/rma/rma_go2_lab
scripts/deploy/snapshot_unitree_rl_lab_state.sh
```

That writes a dated snapshot under:

- `artifacts/unitree_rl_lab_snapshots/<timestamp>/`

including:

- base commit
- tracked diff patch
- diff stat
- git status
- untracked file list
- copies of untracked files/directories

## Recommended Discipline From Here

1. Freeze old-robot working configs before tuning further.
2. Snapshot `unitree_rl_lab` before major deploy-runtime edits.
3. Keep new-robot experiments isolated from old-robot baseline work.
4. Treat this local repo as a maintained fork until we intentionally cleanly
   split, patch-stack, or upstream the changes.
