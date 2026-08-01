# mjlab Blind Student Branch Blueprint

Date: 2026-06-02
Branch: `blind-student-mjlab-sim2real`

## Purpose

Define the new training and deployment line inspired by:
- `reference_repos/mjlab`
- `reference_repos/unitree_rl_mjlab`

This branch is a fresh line, not an in-place continuation of the current
blind student deployment bundle.

## Scope

The branch goal is:
- replace the current actor/deploy contract with a cleaner hardware-honest one
- keep the current deployment debugging tooling
- rebuild the flat -> asymmetric-PPO rough actor path under the new contract

## Core contract change

### Current actor/deploy contract

The current deployed blind student bundle expects:
- `base_lin_vel`
- `base_ang_vel`
- `projected_gravity`
- `velocity_commands`
- `joint_pos_rel`
- `joint_vel_rel`
- `last_action`

### New actor/deploy contract

The new branch will target:
- `base_ang_vel`
- `projected_gravity`
- `velocity_commands`
- `joint_pos_rel`
- `joint_vel_rel`
- `last_action`

So:
- actor-side `base_lin_vel` is removed
- deploy runtime should not need a makeshift odometry path for actor inputs

Optional actor ablation:
- `gait_phase`

Branch default:
- branch v1 uses the core no-`base_lin_vel` actor contract without phase
- `gait_phase` remains available as an explicit ablation, not a locked design

## Training-stage structure

This branch should be treated as a 3-stage pipeline:

1. flat prior / expert
2. asymmetric-PPO blind rough actor
3. optional blind student only if we later add true distillation

Do not call the rough actor an RMA teacher unless its actor receives privileged
inputs or trains a student through an explicit distillation objective.

## New config namespace

Use new config files instead of editing existing ones in place.

Suggested file family:

### Environment configs

- `rma_go2_lab/envs/priors/flat_mjlab_prior_cfg.py`
- `rma_go2_lab/envs/teacher/blind_rough_mjlab_asymppo_cfg.py`
- `rma_go2_lab/envs/blind/blind_rough_mjlab_student_cfg.py`

### Model / runner configs

- `rma_go2_lab/models/priors/flat_mjlab_prior_runner_cfg.py`
- `rma_go2_lab/models/teacher/ppo_mjlab_asymppo_cfg.py`
- `rma_go2_lab/models/blind/blind_mjlab_student_runner_cfg.py`

### Policy cards

- `rma_go2_lab/policies/flat_mjlab_prior_v1.md`
- `rma_go2_lab/policies/blind_rough_mjlab_asymppo_v1.md`
- `rma_go2_lab/policies/blind_mjlab_student_v1.md`

## Sim2real features and ablation order

The first rough asymmetric-PPO comparison must not change the robot/actuator plant
relative to the frozen flat prior. The frozen flat baseline was trained with
the original IsaacLab Go2 plant, so introducing the `mjlab` split actuator
model in the rough actor creates a mid-pipeline plant mismatch.

Branch v1 comparison order:

1. keep the C1 rough omni environment recipe
2. keep the original IsaacLab robot/actuator model used by the frozen flat
3. apply the `mjlab` actor/deploy contract change:
   - actor does not consume `base_lin_vel`
   - critic can still consume privileged `base_lin_vel`
4. only after this works, test `mjlab` actuator priors, encoder bias, and
   observation delay as explicit ablations

From the `mjlab` / `unitree_rl_mjlab` review, the most relevant additions are:

1. actor/critic split that keeps `base_lin_vel` off the actor
2. optional `gait_phase` actor prior as an ablation
3. encoder bias randomization
4. observation delay modeling
5. structured robot-parameter randomization:
   - foot friction
   - base COM offset
   - pushes
6. Go2 joint-type actuator priors:
   - hip/thigh `20/1`
   - calf `40/2`

Important: item 6 is not part of the current rough actor baseline unless the
flat prior is retrained with the same actuator model.

## Deployment/output separation

Do not overwrite the current old-line deploy bundle.

Create a new deploy bundle root under the experiment runtime repo, for example:

- `reference_repos/unitree_rl_lab_go2_old_robot_experiments/deploy/robots/go2/config/policy/velocity/c1_blind_rough_omni_usable_mjlab_v1`

The new bundle must contain:
- a new `deploy.yaml`
- a matching `deploy_config.json`
- a matching ONNX export

## Logging / artifact namespace

Keep logs separate from the old line.

Suggested log roots:
- `logs/rsl_rl/flat_mjlab_prior/...`
- `logs/rsl_rl/go2_blind_rough_asymppo_mjlab_v1/...`
- `logs/rsl_rl/blind_mjlab_student/...`

## What this branch keeps from the current line

Keep:
- current deployment probing/monitoring tooling
- current runtime diagnostic scripts
- current stock-vs-policy evaluation mindset

The branch replaces the actor/deploy contract and training assumptions, not the
entire deployment-debugging surface.

## Initial execution order

1. Scaffold the new config namespace.
2. Define the new actor observation contract in training configs.
3. Add the `mjlab`-style randomization package.
4. Train flat prior.
5. Train asymmetric-PPO blind rough actor.
6. Train blind student only if we reintroduce true distillation.
7. Export a new deploy bundle.
8. Test only that new bundle in the experiment runtime repo.

## Immediate next step

Start by scaffolding the new config family and wiring the new actor observation
contract into those configs.

## Frozen Flat Baseline

The current flat prior is frozen as the stage-1 baseline for this branch:

- run: `logs/rsl_rl/go2_flat_mjlab_prior_v1/2026-06-02_14-30-48`
- checkpoint: `model_1499.pt`

Do not modify the flat branch unless a later rough actor result shows a
regression that clearly originates in stage-1.

## Frozen Rough Asymmetric-PPO Baseline

The current rough asymmetric-PPO actor is frozen as the stage-2 baseline for this branch:

- source run: `logs/rsl_rl/go2_rough_mjlab_teacher_v1/2026-06-03_10-02-12`
- source checkpoint: `model_1000.pt`
- materialized policy: `rma_go2_lab/policies/rough_mjlab_teacher_v1.pt`

This actor uses the C1 rough omni recipe and history architecture with the
actor-side `base_lin_vel` removed from both `policy` and `policy_history`.

## Successful Final Candidate

The final validated and hardware-tested candidate supersedes the earlier
`model_1000.pt` diagnostic freeze:

- run:
  `logs/rsl_rl/go2_blind_rough_asymppo_mjlab_v1/2026-06-04_10-31-03`
- checkpoint: `model_1999.pt`
- bundle:
  `rma_go2_lab/policies/exported/go2_blind_rough_asymppo_mjlab_v1_candidate`
- policy card:
  `rma_go2_lab/policies/go2_blind_rough_asymppo_mjlab_v1_candidate.md`
- retrospective:
  `docs/ASYMPPO_SIM2REAL_SUCCESS_RETROSPECTIVE_20260612.md`

Important correction to earlier planning notes:

- the successful final candidate retains `history_length=100`
- encoder bias and observation delay remain disabled in training
- no gait phase is used
- no distilled student is required for deployment
