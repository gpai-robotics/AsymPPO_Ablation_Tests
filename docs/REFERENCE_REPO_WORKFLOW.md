# Reference Repo Workflow

Date: 2026-06-02

## Purpose

Keep the reference repositories in this project reproducible and easy to reason
about.

## Rule

Treat:
- [reference_repos/unitree_rl_lab](/home/bhuvan/projects/rma/rma_go2_lab/reference_repos/unitree_rl_lab)

as:
- upstream baseline
- read-only reference
- not a place for custom scripts, policy bundles, runtime patches, or local experiments

Treat:
- [reference_repos/unitree_rl_lab_go2_old_robot_experiments](/home/bhuvan/projects/rma/rma_go2_lab/reference_repos/unitree_rl_lab_go2_old_robot_experiments)

as:
- the writable runtime sandbox for old-robot Go2 deployment work
- the place for deploy/runtime modifications
- the place for policy bundles and experiment-only files

## Where custom work should go

### Runtime / deploy experiments

Put them in:
- [reference_repos/unitree_rl_lab_go2_old_robot_experiments](/home/bhuvan/projects/rma/rma_go2_lab/reference_repos/unitree_rl_lab_go2_old_robot_experiments)

### Project-level helper scripts

Put them in:
- [scripts/deploy](/home/bhuvan/projects/rma/rma_go2_lab/scripts/deploy)
- [scripts/eval](/home/bhuvan/projects/rma/rma_go2_lab/scripts/eval)

### Logs, monitor captures, and diagnostics

Put them in:
- [artifacts](/home/bhuvan/projects/rma/rma_go2_lab/artifacts)

## Why this matters

If the baseline repo gets modified directly, we lose:
- a trustworthy upstream reference
- reproducibility
- clear diffs between original code and our experiment changes

## Practical workflow

1. Read from the baseline repo when we want the original implementation.
2. Make changes only in the experiment clone.
3. Keep project-specific tooling outside both reference repos when possible.
4. If we need another experiment line, clone again rather than polluting the baseline.

## Current status

As of 2026-06-02:
- the baseline repo has been restored to clean state
- the experiment clone is the active writable runtime repo
