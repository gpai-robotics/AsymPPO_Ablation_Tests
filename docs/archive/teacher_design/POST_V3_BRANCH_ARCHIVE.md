# Post-V3 Teacher Branch Archive

This note records the post-`V3` teacher branches that were explored after the
frozen `Teacher V3 final` diagnosis was revised.

## Contract Rule

The project contract is:

- keep one canonical active teacher line for overall rough-terrain locomotion
- archive specialized or unstable follow-up branches unless they become the
  best overall candidate

## Canonical Active Line

- `Teacher V4 model_300`
  - checkpoint:
    `/home/bhuvan/tools/IsaacLab/logs/rsl_rl/go2_privileged_teacher_rough_v4_terrain_aux/2026-05-09_10-34-56/model_300.pt`
  - why it stays active:
    current best overall teacher candidate

## Archived Follow-Ups

- `Teacher V4.1 model_1999`
  - why archived:
    improved stair-family behavior, but not the overall rough-terrain story

- `Teacher V5`
  - why archived:
    terrain-family diagnostics were useful, but the branch was unstable and
    not suitable as a canonical teacher line

- `Teacher V6`
  - why archived:
    useful as an IsaacLab-style direct-input idea, but not adopted as the
    canonical line
