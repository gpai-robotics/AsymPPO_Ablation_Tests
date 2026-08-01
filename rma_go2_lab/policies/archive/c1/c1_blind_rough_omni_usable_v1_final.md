# C1 Blind Rough Omni Usable Freeze

This file freezes the selected deployable blind rough-terrain omni student for
the current project phase.

The student is now the canonical deployment candidate for rough omnidirectional
locomotion. Do not silently replace it in place. If a later deployable omni
student is trained, it should be versioned explicitly.

## Identity

- canonical name: `c1_blind_rough_omni_usable_v1_final`
- checkpoint:
  - `rma_go2_lab/policies/c1_blind_rough_omni_usable_v1_final.pt`
- source run:
  - `/home/bhuvan/tools/IsaacLab/logs/rsl_rl/go2_c1_blind_rough_omni_usable_v1/2026-05-21_12-32-46`
- selected source checkpoint:
  - `model_2000.pt`

## Purpose

This student is the canonical deployable rough-terrain omni policy for the
current repo phase.

It represents:

- proprioceptive plus history-only deployment observations
- a practical usable omnidirectional command envelope rather than a max-range
  paper envelope
- supervision from the frozen omni rough teacher
- the first cleaned end-to-end `flat omni -> omni teacher -> deployable omni
  student` ladder that remained stable late in training

It is the artifact we should export and use for deployment work and sim2real gap
study.

## Training Definition

- task:
  - `RMA-Go2-C1-Omni-Usable-V1-StageA`
- env config:
  - `rma_go2_lab/envs/blind/c1_blind_rough_omni_usable_cfg.py`
- PPO config:
  - `rma_go2_lab/models/blind/blind_rough_runner_cfg.py`

Frozen characteristics retained in this version:

- omni flat warm start from:
  - `rma_go2_lab/policies/flat_omni_v1.pt`
- privileged teacher supervision from:
  - `rma_go2_lab/policies/rough_omni_teacher_v1.pt`
- deployable observation contract:
  - `policy`
  - `policy_history`
- usable command envelope:
  - `lin_vel_x = ±0.8`
  - `lin_vel_y = ±0.3`
  - `ang_vel_z = ±0.6`

## Selection Rationale

`model_2000.pt` from the `2026-05-21_12-32-46` run was selected because:

- the student remained strong after teacher imitation tapered fully to zero
- tracking stayed clean at the full usable command envelope
- rough-terrain robustness remained strong with very low collapse and
  low-progress rates
- gait and body behavior stayed clean in both metrics and playback
- the branch provided the clearest deployment-oriented omni result seen in the
  repo so far

Note:

- this run descended from a warm-restart continuation after an earlier resume
  attempt reset curriculum and loss schedule state
- we freeze this artifact honestly as the winning deployment candidate from that
  cleaned continuation, not as a byte-for-byte uninterrupted run lineage

## Final Behavior Summary

Training-side late regime near freeze:

- tracking reward:
  - `track_lin_vel_xy_exp = 1.4436`
  - `track_ang_vel_z_exp = 0.7028`
- tracking error:
  - `error_vel_xy = 0.1294`
  - `error_vel_yaw = 0.1842`
- terminations:
  - `time_out = 0.9790`
  - `base_contact = 0.0000`
  - `base_orientation = 0.0002`
  - `base_height = 0.0127`
  - `low_progress = 0.0093`
- gait/body cleanliness:
  - `feet_slide = -0.0099`
  - `flat_orientation_l2 = -0.0036`

## Pipeline Role

This student is now the active downstream deployment artifact for:

- policy export
- deployment bundle validation
- sim-side deployment rehearsal
- sim2real gap study

It should be treated as the current C1 omni deployment candidate rather than as
an open-ended training branch.

## Deployment Audit Summary

Deployment-side export, Isaac rehearsal, and a nominal flat-surface MuJoCo
runtime sanity check were validated for this artifact.

Settled deployment truths:

- bundled TorchScript actions match the frozen source checkpoint to numerical
  parity
- deployable observation contract remains:
  - `policy`
  - `policy_history`
- fixed-command Isaac deployment rehearsal remained stable without rollout
  failures
- nominal flat-surface MuJoCo Sim2Sim runtime executed successfully and showed
  sensible commanded behavior without obvious deploy-contract pathologies

The first useful hidden-dynamics failure study is also now complete.

Observed deploy-time probe result:

- low friction is the primary uncovered robustness gap
- moderate added mass did not materially destabilize tracking in the tested
  range
- moderate motor weakening did not materially destabilize tracking in the
  tested range

Important interpretation note:

- Isaac is the primary source of controlled hidden-dynamics robustness evidence
  for this baseline
- MuJoCo is currently used mainly as a flat-surface deployment-contract and
  behavior sanity check
- exploratory MuJoCo corridor scenarios were informative but are not treated as
  decisive baseline evidence because the corridor environment is not yet trusted
  enough to define the failure story on its own

In other words:

- this is a real deployment baseline, not just a training artifact
- it is strong enough to ship into careful rehearsal
- it is not a solved low-traction controller

## Canonical Failure Case

The first deployment-facing failure case for this baseline is low friction.

Under a live deploy-time friction switch:

- static friction moved from about `1.09` to `0.10`
- dynamic friction moved from about `0.92` to `0.10`
- post-switch planar tracking error rose sharply
- post-switch yaw tracking error also rose sharply
- rollout failures appeared, unlike the mass and motor probes

Why this matters:

- it gives the repo an honest reason to add future robustness modules
- it keeps the baseline useful as a comparison anchor
- it prevents improvement work from becoming speculative architecture churn

Future improvement work should therefore justify itself primarily against this
low-friction failure mode, not against vague dissatisfaction with the baseline.

## MuJoCo Usage Boundary

For this baseline, MuJoCo should currently be read as answering:

- do the joints behave sensibly after export?
- does commanded forward/lateral/yaw behavior look qualitatively correct?
- are there obvious oscillations, drift, or actuator/runtime pathologies?

It should not yet be treated as the final arbiter of rough-terrain or corridor
robustness for this student.

## Freeze Statement

The deployable blind rough omni student is now frozen as a deployment
candidate.

Do not change:

- its checkpoint identity
- its upstream flat prior / omni teacher pairing
- its role as the canonical rough omni deployable artifact

If a stronger deployable omni student is trained later, it must be created as a
new, explicitly versioned artifact rather than replacing this one silently.
