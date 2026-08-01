# C2 Wrong-Task Launch Archive

This note records a mistaken launch that was initially interpreted as an
anchored C2 training run.

## What happened

The run at:

- `/home/bhuvan/tools/IsaacLab/logs/rsl_rl/go2_adaptation_student_no_adapt_v0/2026-05-14_10-53-43`

was inspected after training and turned out to be:

- `experiment_name: go2_adaptation_student_no_adapt_v0`
- `policy.class_name: WarmStartActorCritic`
- `obs_groups.policy: [policy]`

So it was a real `no_adapt` baseline run, not:

- `RMA-Go2-Adapt-V3-DynOnly-Phase2-Recovery-LowSwitch-LatentReg-TCN-Anchored`

## Why it was rejected

- wrong task family
- visually unacceptable tucked-leg gait across checkpoints
- not relevant evidence for the active anchored C2 line

## Meaning

This was not an architecture failure of the anchored C2 branch.

It was a launch mismatch and should be treated as:

- a discarded mistaken run
- not part of the active adaptive branch record
