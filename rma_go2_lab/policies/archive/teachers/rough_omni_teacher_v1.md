# Rough Omni Teacher Freeze

This file freezes the selected privileged rough-terrain omni teacher for the
current project phase.

The teacher is now frozen. Do not silently replace it in place. If a later
rough omni teacher is trained, it should be versioned explicitly.

## Identity

- canonical name: `rough_omni_teacher_v1`
- checkpoint:
  - `rma_go2_lab/policies/rough_omni_teacher_v1.pt`
- source run:
  - `/home/bhuvan/tools/IsaacLab/logs/rsl_rl/go2_privileged_teacher_rough_omni_v1/2026-05-20_17-29-55`
- selected source checkpoint:
  - `model_1999.pt`

## Purpose

This teacher is the canonical upstream supervisor for the deployable blind rough
omni student line.

It represents:

- full omnidirectional rough-terrain command coverage
- explicit terrain + dynamics privilege
- the first coherent omni teacher in the repo’s cleaned `flat omni -> omni
  teacher -> deployable omni student` ladder

It is not itself the deployment artifact.

## Training Definition

- task:
  - `RMA-Go2-Privileged-Teacher-Rough-Omni-V1`
- env config:
  - `rma_go2_lab/envs/teacher/rough_omni_v1_cfg.py`
- PPO config:
  - `rma_go2_lab/models/teacher/ppo_omni_v1_cfg.py`

Frozen characteristics retained in this version:

- omni flat warm start from:
  - `rma_go2_lab/policies/flat_omni_v1.pt`
- terrain privileged height-scan branch
- dynamics privileged branch:
  - static friction
  - dynamic friction
  - base mass ratio
  - joint stiffness scale
  - joint damping scale
- command curriculum widened to:
  - `lin_vel_x = ±1.0`
  - `lin_vel_y = ±0.4`
  - `ang_vel_z = ±1.0`

## Selection Rationale

`model_1999.pt` from the `2026-05-20_17-29-55` run was selected because:

- full omni command curriculum was reached and held
- tracking remained strong at the full command envelope
- rough-terrain robustness stayed excellent
- gait quality and body behavior remained clean
- visual playback confirmed that this is the first omni teacher branch that
  feels coherent end-to-end

This freeze also closes the earlier philosophical mismatch where omni students
were being trained against a forward-only rough teacher. That earlier setup was
useful as a feasibility probe, but it is no longer the clean canonical omni
training story.

## Final Behavior Summary

Training-side final regime near freeze:

- tracking reward:
  - `track_lin_vel_xy_exp = 1.4401`
  - `track_ang_vel_z_exp = 0.7000`
- tracking error:
  - `error_vel_xy = 0.1390`
  - `error_vel_yaw = 0.1964`
- terminations:
  - `time_out = 0.9802`
  - `base_contact = 0.0000`
  - `base_orientation = 0.0000`
  - `base_height = 0.0112`
  - `low_progress = 0.0086`
- gait/body cleanliness:
  - `feet_slide = -0.0114`
  - `flat_orientation_l2 = -0.0037`

## Pipeline Role

This teacher is the frozen omni supervision source for:

- the active deployable blind rough omni student:
  - `RMA-Go2-C1-Omni-Usable-V1-StageA`

It should replace the earlier forward-only teacher for omni-student work.

## Freeze Statement

The rough omni teacher is now frozen.

Do not change:

- its checkpoint identity
- its training recipe
- its role as the canonical omni teacher

If a stronger omni teacher is trained later, it must be created as a new,
explicitly versioned artifact rather than replacing this one silently.
