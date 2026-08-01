# MJLAB Branch Delta Spec

Date: 2026-06-02
Branch: `blind-student-mjlab-sim2real`

## Purpose

Define exactly:
- what the current line is
- what the new `mjlab`-style branch changes
- what stays unchanged
- why each change is necessary

This is the contract document for the new branch. If a future config change does
not match this file, it should be treated as a deliberate deviation and
documented.

## 1. Current Line: What We Actually Have

The current training/deployment line is not a generic blind locomotion stack.
It is a specific 3-part structure:

1. flat prior / expert
2. privileged rough teacher
3. blind history student

Relevant files:
- flat prior:
  - [flat_forward_prior_cfg.py](/home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/envs/priors/flat_forward_prior_cfg.py)
  - [flat_omni_prior_cfg.py](/home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/envs/priors/flat_omni_prior_cfg.py)
- privileged teacher:
  - [rough_cfg.py](/home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/envs/teacher/rough_cfg.py)
  - [rough_v3_cfg.py](/home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/envs/teacher/rough_v3_cfg.py)
  - [rough_omni_v1_cfg.py](/home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/envs/teacher/rough_omni_v1_cfg.py)
- blind student:
  - [blind_rough_forward_cfg.py](/home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/envs/blind/blind_rough_forward_cfg.py)
  - [blind_rough_forward_history_cfg.py](/home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/envs/blind/blind_rough_forward_history_cfg.py)
  - [c1_blind_rough_teacher_history_cfg.py](/home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/envs/blind/c1_blind_rough_teacher_history_cfg.py)
  - [c1_blind_rough_omni_usable_cfg.py](/home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/envs/blind/c1_blind_rough_omni_usable_cfg.py)

### 1.1 Current flat prior

The flat prior is:
- flat terrain
- no terrain scanner
- no privileged group
- standard velocity-task proprioceptive policy

Key facts:
- terrain scanner disabled in
  [flat_forward_prior_cfg.py](/home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/envs/priors/flat_forward_prior_cfg.py)
- command space later widened for omni in
  [flat_omni_prior_cfg.py](/home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/envs/priors/flat_omni_prior_cfg.py)

### 1.2 Current teacher

The teacher is not just “same policy with more obs”.

Teacher V0 adds:
- terrain height scan as a separate privileged group

Teacher V3 adds:
- terrain privilege
- dynamics privilege:
  - tracked static friction
  - tracked dynamic friction
  - tracked base mass ratio
  - joint stiffness scale
  - joint damping scale

This is implemented in:
- [rough_cfg.py](/home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/envs/teacher/rough_cfg.py)
- [rough_v3_cfg.py](/home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/envs/teacher/rough_v3_cfg.py)

### 1.3 Current blind student

The current blind student is not a plain feedforward blind actor.

It is:
- a blind proprioceptive policy
- with a duplicated `policy_history` group
- flattened temporal history
- a temporal encoder over that history
- imitation/warmstart from prior or teacher checkpoints

This is implemented in:
- history env:
  [blind_rough_forward_history_cfg.py](/home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/envs/blind/blind_rough_forward_history_cfg.py)
- teacher-history env wrapper:
  [c1_blind_rough_teacher_history_cfg.py](/home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/envs/blind/c1_blind_rough_teacher_history_cfg.py)
- blind temporal actor:
  [history_actor_critic.py](/home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/models/asymppo/history_actor_critic.py)
- blind runner config:
  [blind_rough_runner_cfg.py](/home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/models/blind/blind_rough_runner_cfg.py)

### 1.4 Current deploy contract

The current deployed old-line bundle expects actor inputs that include
`base_lin_vel`.

The live active old-line deploy bundle is under:
- [c1_blind_rough_omni_usable_v1_final](/home/bhuvan/projects/rma/rma_go2_lab/reference_repos/unitree_rl_lab_go2_old_robot_experiments/deploy/robots/go2/config/policy/velocity/c1_blind_rough_omni_usable_v1_final)

This matters because real hardware does not naturally expose a clean
`base_lin_vel` observation for the actor. That mismatch is the main contract
problem in the current line.

## 2. Current Line: What Is Good And Should Stay

These parts are worth preserving.

### 2.1 The staged training structure stays

We keep:
1. flat prior
2. rough teacher
3. blind student

Reason:
- changing the actor contract and sim2real package is already a large change
- collapsing the ladder at the same time would make failures hard to attribute

### 2.2 The deployment debugging surface stays

We keep:
- monitor tooling
- post-run analyzers
- mirrored-leg diagnostics
- stock-vs-policy comparison workflow

Reason:
- this is one of the strongest parts of the current repo
- `mjlab` has a cleaner contract, but not a better deployment-debug workflow

### 2.3 Rough omni task scope stays

We keep:
- rough-terrain target
- omnidirectional velocity control
- deploy focus on Go2

Reason:
- that is the real task we care about
- the new branch is not a flat-only branch

## 3. Current Line: What Must Change

### 3.1 Remove actor-side `base_lin_vel`

This is the most important change.

Current problem:
- the deployed actor contract expects `base_lin_vel`
- real hardware does not expose it cleanly
- that forces runtime estimation/patching

New rule:
- actor does not consume `base_lin_vel`
- critic may still consume privileged linear velocity during training

Why:
- this removes the biggest sim/deploy mismatch in the current line
- this matches the `unitree_rl_mjlab` design

Reference:
- actor/critic split in
  [velocity_env_cfg.py](/home/bhuvan/projects/rma/rma_go2_lab/reference_repos/unitree_rl_mjlab/src/tasks/velocity/velocity_env_cfg.py)

### 3.2 Treat `gait_phase` as an ablation, not a fixed branch requirement

Current line:
- relies on proprioception + action history
- no explicit phase signal in the deployed actor contract

New branch:
- actor may get `gait_phase`
- branch v1 defaults to no phase

Why:
- `mjlab` uses phase explicitly in the actor
- it gives the actor a clean rhythmic anchor without needing fake velocity
- but for robust rough-terrain omni locomotion, a fixed global clock is not an
  obviously dominant choice, so it should be tested rather than assumed

Reference:
- phase term in
  [velocity_env_cfg.py](/home/bhuvan/projects/rma/rma_go2_lab/reference_repos/unitree_rl_mjlab/src/tasks/velocity/velocity_env_cfg.py)
- deploy contract in
  [deploy.yaml](/home/bhuvan/projects/rma/rma_go2_lab/reference_repos/unitree_rl_mjlab/deploy/robots/go2/config/policy/velocity/v0/params/deploy.yaml)

### 3.3 Keep C1 history for the first fair rough-teacher comparison

Current line:
- blind student explicitly duplicates full policy obs into `policy_history`
- current temporal actor encodes long flattened history
- this is a core part of
  [history_actor_critic.py](/home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/models/asymppo/history_actor_critic.py)

New branch:
- rough teacher and student should keep the C1 history pathway for the first
  fair comparison
- the `mjlab` contract change is applied inside both `policy` and
  `policy_history`
- flat prior can remain no-history because it is only the stage-1 backbone

Why:
- the old C1 line already relies on history for rough recovery
- removing history at the same time as changing the actor contract made the
  rough-teacher comparison too different from C1
- the current branch goal is to isolate whether the deploy-honest contract
  helps sim2real, not to redesign the teacher architecture

Important nuance:
- `mjlab` framework supports history strongly
- but Go2 velocity deploy uses `history_length: 1`, which is effectively no
  stacked actor history
- that remains a later ablation, not the current rough-teacher baseline

### 3.4 Move from “teacher with terrain+dynamics privilege feeding blind history student” to “critic privilege + staged branch training”

Current line:
- teacher is its own privileged model family
- student explicitly imitates teacher/prior

New branch:
- keep staged flat/teacher/student training
- keep the C1 teacher/student training recipe for the first comparison
- apply the `mjlab` actor contract by moving `base_lin_vel` out of actor-facing
  groups and into a critic-only privileged group

Why:
- the failed teacher attempts showed that changing plant, history, terrain
  pressure, and sensing assumptions together is not attributable
- first comparison should answer one question:
  - does removing actor-side `base_lin_vel` improve the deploy story while
    preserving the C1 training recipe?

This is the one area where we are not simply copying `mjlab`.
- `mjlab` does not provide our exact flat -> teacher -> blind student ladder
- we are keeping our ladder, but putting it on a cleaner contract

## 4. New MJLAB Branch: Exact Target Contract

### 4.1 Actor observation contract

The new actor contract should be:
- `base_ang_vel`
- `projected_gravity`
- `velocity_commands`
- `joint_pos_rel`
- `joint_vel_rel`
- `last_action`

This is the contract the new line should train and deploy around.

Optional actor ablation:
- `gait_phase`

For C1-style history models, the same contract applies to each frame inside
`policy_history`; `base_lin_vel` must be removed from both current policy obs
and history obs.

### 4.2 Critic privilege

The new critic can additionally consume:
- `base_lin_vel`
- terrain privilege if needed

The current rough-teacher baseline consumes:
- `policy`
- `policy_history`
- `critic_privileged` with `base_lin_vel`
- `terrain_privileged`
- `dynamics_privileged`

### 4.3 Robot and actuator model

Do not introduce the `mjlab` split actuator model in the rough teacher unless
the flat prior was trained with that same actuator model.

Current frozen flat:
- original IsaacLab Go2 robot/actuation
- no actor `base_lin_vel`

Therefore current rough teacher should use:
- original IsaacLab Go2 robot/actuation
- C1 rough omni training recipe
- mjlab actor contract

`mjlab` actuator priors:
- hip/thigh `20/1`
- calf `40/2`

These are a valid later ablation, but they require retraining flat, teacher,
and student under the same plant assumptions.
- dynamics privilege if needed

Reason:
- critic privilege is fine during training
- actor/deploy mismatch is the actual problem, not critic privilege

### 4.3 Terrain handling

Current line:
- teacher privilege exposes terrain scans
- blind actor is fully blind

New branch:
- keep the blind actor blind to terrain scans in deployment
- keep terrain privilege available on the training side where needed

Reason:
- this preserves deploy realism
- this keeps rough-terrain competence trainable

## 5. New MJLAB Branch: Sim2Real Package To Add

These are the additions we want from `mjlab`.

### 5.1 Observation delay modeling

Add:
- delayed observations in training

Why:
- real deployment has latency
- `mjlab` supports this directly in the observation pipeline

Reference:
- [observation_manager.py](/home/bhuvan/projects/rma/rma_go2_lab/reference_repos/mjlab/src/mjlab/managers/observation_manager.py)

### 5.2 Encoder bias randomization

Add:
- encoder/joint observation bias randomization

Why:
- this is a realistic sim2real error source
- it is more relevant than only broad symmetric gain randomization

### 5.3 Structured robot-parameter randomization

Add:
- foot friction randomization
- base COM offset randomization
- pushes

Why:
- this matches the stronger `mjlab` sim2real package
- it is a better first upgrade than ad hoc deployment knob tuning

### 5.4 Go2 actuator priors

Use:
- hip/thigh `20 / 1`
- calf `40 / 2`

Reason:
- these are more hardware-shaped than the flat old deployment gains
- they should be part of the training assumptions, not only a deploy patch

## 6. What The New Branch Should Not Copy Blindly

### 6.1 Do not copy `mjlab` as if it already has our teacher-student ladder

It does not.

What it has:
- actor/critic split
- deploy-honest actor observations
- history/delay support in the framework

What it does not clearly provide for Go2 velocity:
- our flat prior -> privileged teacher -> blind student pipeline

So:
- keep our ladder
- replace our actor/deploy contract
- import their robustness machinery

### 6.2 Do not carry the current long-history student forward by default

The current history student was built to solve the old contract family.

For branch v1:
- first establish a clean no-`base_lin_vel`, phase-aware actor
- only add larger history later if it proves necessary

### 6.3 Do not overwrite the current deploy bundle

The new branch must export to a separate path.

Reason:
- the new actor contract is different
- mixing bundles would destroy reproducibility

## 7. Concrete Stay / Change / Why Table

### Stays

- rough-terrain target
  - because that is the real task
- omnidirectional velocity control
  - because the new branch is still an omni branch
- flat -> teacher -> blind student stage order
  - because it keeps the training logic attributable
- deployment monitoring and analyzers
  - because they are already strong and useful
- experiment clone as the only writable runtime repo
  - because reproducibility depends on it

### Changes

- remove actor-side `base_lin_vel`
  - because deploy should not depend on makeshift odometry
- add `gait_phase`
  - because it gives the actor a clean locomotion rhythm signal
- reduce/remove large stacked actor history for branch v1
  - because we want a clean contract reset first
- add delay modeling
  - because latency matters in sim2real
- add encoder bias randomization
  - because it is realistic and relevant
- add structured friction/COM/push randomization
  - because `mjlab` is stronger there than our current line
- bake Go2 actuator priors into training assumptions
  - because deployment-only gain changes are not enough

### Changes later, not on day 1

- larger temporal history in the actor
  - only if the lean contract proves insufficient
- per-leg adaptive gain mechanisms
  - only after we establish the simpler branch baseline
- asymmetry-targeted randomization
  - useful, but separate from the core `mjlab` contract reset

## 8. Immediate Implementation Policy

Before training anything in the new branch:

1. Scaffold new config files instead of editing current ones in place.
2. Define the actor contract first.
3. Define the critic privilege second.
4. Add sim2real randomization package third.
5. Only then wire flat prior, teacher, and blind student.

If we do not follow that order, the branch will drift back into the same
contract confusion as the old line.
