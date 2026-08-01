# MJLAB Branch V1 Randomization Spec

Date: 2026-06-02
Branch: `blind-student-mjlab-sim2real`

## Goal

Define the branch-v1 training randomization policy for:
- flat prior
- rough teacher
- blind student

This spec is intentionally staged. The first branch-v1 rough comparison keeps
the old C1 training/randomization recipe and changes the actor/deploy contract.
Actuator-model changes, encoder bias, and observation delay are separate
ablations because they change the plant or sensing problem enough to obscure
the core comparison.

## 1. Current Branch-V1 Baseline

Current branch-v1 baseline:

1. C1 rough omni teacher/student training recipe
2. original IsaacLab Go2 robot/actuator model
3. C1-style `policy_history`
4. actor-side `base_lin_vel` removed from both `policy` and `policy_history`
5. critic-only `base_lin_vel` in `critic_privileged`

This keeps the comparison narrow:
- old line: C1 recipe + actor `base_lin_vel`
- new line: C1 recipe + no actor `base_lin_vel`

Do not introduce the `mjlab` split actuator model in this baseline unless the
flat prior is retrained with the same actuator assumptions.

## 2. Branch-V1 Core Randomization Package

### 2.1 Friction randomization

Keep friction randomization, but make it structured and realistic.

Branch-v1 rule:
- keep startup friction randomization
- use it in both rough teacher and blind student
- preserve tracking of realized friction where critic privilege uses it

Direction:
- moderate range
- not extreme low-friction stress as the default branch-v1 center

Reason:
- friction variation is one of the highest-value terrain transfer randomizers
- we already rely on it in the old line
- `mjlab` also treats foot friction as first-class

### 2.2 Base mass and COM randomization

Keep:
- base mass randomization
- base COM randomization

Branch-v1 rule:
- both should be active in rough teacher and blind student
- flat prior can use reduced ranges if needed to keep initial gait learning
  stable

Reason:
- these are realistic and load-bearing sim2real factors
- they matter directly for posture, recovery, and turning behavior

### 2.3 Push disturbances

Keep pushes, but do not start with extreme push-heavy training.

Branch-v1 rule:
- rough teacher: enabled
- blind student: enabled
- flat prior: optional or reduced

Reason:
- pushes improve recovery robustness
- but too much push emphasis too early can distort initial locomotion learning

### 2.4 Actuator gain randomization

For the current branch-v1 baseline, keep the C1 actuator randomization recipe.
The goal is a fair C1-vs-mjlab-contract comparison, not actuator-model
research.

The `mjlab` actuator prior is a later ablation.

Mjlab nominal center:
- hips/thighs: `20 / 1`
- calves: `40 / 2`

If tested later:
- multiplicative stiffness scale
- multiplicative damping scale
- bounded around 1.0
- retrain flat, teacher, and student with this actuator model

Reason:
- changing actuator priors mid-pipeline already caused rough-teacher failures
- plant assumptions must be consistent across stages

### 2.5 Observation delay

Observation delay is not part of the current branch-v1 baseline.

Later ablation rule:
- introduce short bounded delay on actor observations
- do not start with large or highly erratic latency

Reason:
- delay is one of the clearest sim/deploy mismatches
- `mjlab` treats delay as a first-class part of the observation pipeline
- but it should be tested after the C1-style no-`base_lin_vel` baseline works

Previous experimental defaults:
- flat prior:
  - actor sensor delay `0..1` control steps
  - hold probability `0.85`
  - lag update period `4` steps
- rough teacher:
  - actor sensor delay `0..2` control steps
  - hold probability `0.85`
  - lag update period `4` steps
- blind student:
  - actor sensor delay `0..2` control steps
  - hold probability `0.85`
  - lag update period `4` steps

Applied only to actor sensor-like terms:
- `base_ang_vel`
- `projected_gravity`
- `joint_pos_rel`
- `joint_vel_rel`

Not delayed in branch v1:
- commands
- `last_action`
- critic privileged groups

### 2.6 Encoder bias

Encoder bias is not part of the current branch-v1 baseline.

Later ablation rule:
- bias joint position observations
- compensate action application consistently so physical targets remain coherent
- do not bias joint velocity directly

Reason:
- this is realistic
- it matches the `mjlab` encoder-bias design
- it is more meaningful than simply adding generic observation noise
- but it should be tested after the C1-style no-`base_lin_vel` baseline works

Previous experimental defaults:
- flat prior:
  - encoder bias range `(-0.01, 0.01)` rad
- rough teacher:
  - encoder bias range `(-0.015, 0.015)` rad
- blind student:
  - encoder bias range `(-0.015, 0.015)` rad

Implementation rule:
- actor `joint_pos_rel` is biased
- actor `joint_vel_rel` is not biased
- joint-position actions subtract the stored bias before writing physical targets

## 3. What To Keep From The Old Line

### 3.1 Terrain family

Keep:
- mixed rough terrain
- not flat-only
- not stair-specialist only

Reason:
- branch target is robust rough omni locomotion

### 3.2 Curriculum

Keep command curriculum.

Reason:
- rough omni skill still needs staged command expansion
- removing curriculum while changing the actor contract would be unnecessary
  chaos

### 3.3 Reward structure

Keep the general reward philosophy:
- tracking-dominant
- modest regularization
- no over-prescriptive gait template

Reason:
- branch-v1 should change contract/randomization first
- reward redesign is a separate axis

## 4. What To Modify Carefully

### 4.1 Global PD widening

Do not use very broad symmetric global PD widening as the main answer.

Instead:
- center on real Go2 gains
- use bounded multiplicative ranges
- preserve joint-type structure

Reason:
- the deployment issues did not look like uniform global gain mismatch only
- broad widening is too blunt

### 4.2 Large actor history

The successful final branch-v1 candidate retains `history_length=100`.

Reason:
- the two-second proprioceptive window is part of the validated deployment
  contract
- it allows the actor to infer motion and actuator response without
  actor-side `base_lin_vel`
- shorter history is now an ablation, not the branch-v1 default

### 4.3 Phase prior

Treat `gait_phase` as optional.

Reason:
- useful as an ablation
- not obviously required for best rough-terrain omni robustness

## 5. What To Defer To Branch V2+

These are valid ideas, but not branch-v1 defaults.

### 5.1 Strong asymmetric impairment randomization

Examples:
- one-leg weakness
- diagonal imbalance
- joint-specific lag

Defer because:
- we first want to know how much the cleaner contract + core sim2real package
  already helps

### 5.2 Per-leg adaptive PD policy outputs

Defer because:
- this adds another layer of policy complexity
- it will muddy attribution in the first branch

### 5.3 Very aggressive domain randomization

Defer because:
- branch-v1 should aim for robust learning, not immediate worst-case stress

## 6. Stage-Specific Guidance

### 6.1 Flat prior

Use:
- flat terrain
- reduced randomization set
- enough actuator realism to match the new branch assumptions

Keep moderate:
- friction
- actuator scaling

Reduce or disable initially:
- large pushes
- very wide mass/COM variation

Reason:
- flat prior should still learn a clean locomotion backbone first

### 6.2 Rough teacher

Use:
- full branch-v1 rough randomization package
- critic privilege for terrain and dynamics

Reason:
- this is the main privileged upper-bound for the branch

### 6.3 Blind student

No separately distilled blind student was required for the successful final
candidate. The asymmetric-PPO actor is directly deployable.

Reason:
- its actor is already blind and uses only deployable proprioception
- privileged information is confined to the critic during training
- a student stage should only return if a future teacher has actor-side
  privilege that must be distilled away

## 7. Branch-V1 Implementation Status

Successful final candidate:

- friction randomization
- base mass randomization
- base COM randomization
- push disturbances
- global actuator gain scaling
- 100-step proprioceptive history
- actor-side `base_lin_vel` removed
- critic-only terrain, dynamics, and base-velocity privilege

Intentionally disabled in the successful training run:

- encoder bias randomization
- observation delay modeling
- action delay modeling
- command delay modeling
- gait phase
- joint-type-specific MJLAB nominal actuator groups

These are future controlled ablations. They are not missing prerequisites for
the frozen successful baseline.

## 8. Post-Success Rule

The branch-v1 contract is now frozen. Future work must:

1. preserve the successful checkpoint and bundle
2. change one named factor at a time
3. use the same cross-simulator validation profile
4. compare repeated logged hardware runs
5. avoid describing untested additions as causes of the current success
