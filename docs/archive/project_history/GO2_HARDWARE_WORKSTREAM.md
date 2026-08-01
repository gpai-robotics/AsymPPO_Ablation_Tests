# Go2 Hardware Workstream

Date: 2026-06-02

## Purpose

This file is the entry point for the current Go2 hardware / deployment thread.

It exists because the top-level `docs/` folder contains a large amount of
historical project material, and the hardware investigation should be readable
without guessing which note is current.

## Start here

- [artifacts/diagnostics/20260602_go2_old_robot_master_summary.md](/home/bhuvan/projects/rma/rma_go2_lab/artifacts/diagnostics/20260602_go2_old_robot_master_summary.md)
- [artifacts/diagnostics/20260601_deployment_vs_training_recommendation.md](/home/bhuvan/projects/rma/rma_go2_lab/artifacts/diagnostics/20260601_deployment_vs_training_recommendation.md)
- [artifacts/diagnostics/20260602_mjlab_port_shortlist.md](/home/bhuvan/projects/rma/rma_go2_lab/artifacts/diagnostics/20260602_mjlab_port_shortlist.md)

## Runtime repo split

Read:
- [REFERENCE_REPO_WORKFLOW.md](/home/bhuvan/projects/rma/rma_go2_lab/docs/REFERENCE_REPO_WORKFLOW.md)

Current rule:
- [reference_repos/unitree_rl_lab](/home/bhuvan/projects/rma/rma_go2_lab/reference_repos/unitree_rl_lab) is upstream baseline and read-only
- [reference_repos/unitree_rl_lab_go2_old_robot_experiments](/home/bhuvan/projects/rma/rma_go2_lab/reference_repos/unitree_rl_lab_go2_old_robot_experiments) is the active writable runtime repo

## Current active hardware notes

- [GO2_OLD_ROBOT_FROZEN_BASELINE_20260528.md](/home/bhuvan/projects/rma/rma_go2_lab/docs/GO2_OLD_ROBOT_FROZEN_BASELINE_20260528.md)
- [GO2_DEPLOYMENT_PATCH_LOG_20260528.md](/home/bhuvan/projects/rma/rma_go2_lab/docs/GO2_DEPLOYMENT_PATCH_LOG_20260528.md)
- [GO2_OLD_ROBOT_BASELINE_AND_EXPERIMENT_SPLIT_20260529.md](/home/bhuvan/projects/rma/rma_go2_lab/docs/GO2_OLD_ROBOT_BASELINE_AND_EXPERIMENT_SPLIT_20260529.md)
- [GO2_READONLY_COMPAT_CHECK.md](/home/bhuvan/projects/rma/rma_go2_lab/docs/GO2_READONLY_COMPAT_CHECK.md)
- [DEPLOYMENT_VALIDATION_PROTOCOL.md](/home/bhuvan/projects/rma/rma_go2_lab/docs/DEPLOYMENT_VALIDATION_PROTOCOL.md)
- [DEPLOY_RUNTIME_OBSERVATION_PIPELINE.md](/home/bhuvan/projects/rma/rma_go2_lab/docs/DEPLOY_RUNTIME_OBSERVATION_PIPELINE.md)
- [MUJOCO_SIM2SIM_VALIDATION.md](/home/bhuvan/projects/rma/rma_go2_lab/docs/MUJOCO_SIM2SIM_VALIDATION.md)

## Current diagnostic notes

- [GO2_BLIND_STUDENT_ACTUATION_DIAGNOSIS_20260529.md](/home/bhuvan/projects/rma/rma_go2_lab/docs/GO2_BLIND_STUDENT_ACTUATION_DIAGNOSIS_20260529.md)
- [GO2_BLIND_STUDENT_TRAINING_FOLLOWUPS_20260529.md](/home/bhuvan/projects/rma/rma_go2_lab/docs/GO2_BLIND_STUDENT_TRAINING_FOLLOWUPS_20260529.md)
- [WHAT_THE_CURRENT_HARDWARE_GAP_SUGGESTS_FOR_ISAACSIM_20260529.md](/home/bhuvan/projects/rma/rma_go2_lab/docs/WHAT_THE_CURRENT_HARDWARE_GAP_SUGGESTS_FOR_ISAACSIM_20260529.md)
- [WHY_MUJOCO_IF_WE_ALREADY_HAVE_ISAACSIM_20260529.md](/home/bhuvan/projects/rma/rma_go2_lab/docs/WHY_MUJOCO_IF_WE_ALREADY_HAVE_ISAACSIM_20260529.md)

## Historical but still useful

- [UNITREE_RL_LAB_REPRO_AUDIT_20260528.md](/home/bhuvan/projects/rma/rma_go2_lab/docs/UNITREE_RL_LAB_REPRO_AUDIT_20260528.md)
- [UNITREE_RL_LAB_GO2_PATH.md](/home/bhuvan/projects/rma/rma_go2_lab/docs/UNITREE_RL_LAB_GO2_PATH.md)

## Monitor artifacts

See:
- [artifacts/go2_realtime_monitor/README.md](/home/bhuvan/projects/rma/rma_go2_lab/artifacts/go2_realtime_monitor/README.md)

## Practical reading order

1. Read the master summary.
2. Read the deployment-vs-training recommendation.
3. Read the mjlab port shortlist.
4. Use the notes in this file only when you need supporting detail.

## Scope boundary

This file is only for the current Go2 hardware thread.

It does not replace:
- [PROJECT_GUIDE.md](/home/bhuvan/projects/rma/rma_go2_lab/docs/PROJECT_GUIDE.md)
- [REPO_MENTAL_MODEL.md](/home/bhuvan/projects/rma/rma_go2_lab/docs/REPO_MENTAL_MODEL.md)

