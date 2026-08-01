# Architecture Flow: Flat Expert To Adapt-V3

This note gives the cleanest end-to-end architecture view of the project from
the earliest flat prior through the current adaptive `Adapt-V3` refinement
line.

Use this file when the question is:

- what did we train first?
- how did the architecture evolve from stage to stage?
- what is frozen at each phase?
- what is the current adaptive stack, and how did we reach it?

This note is intentionally synthetic.

It does not replace the deeper branch-specific docs such as:

- `docs/ADAPTATION_PHASE_SYNTHESIS.md`
- `docs/archive/adapt_v3/ADAPT_V3_TRUE_RMA_PLAN.md`
- `docs/ADAPT_V3_EXECUTION_SPEC.md`
- `docs/ADAPT_V3_ACTIVE_ROADMAP.md`

It exists because those notes are individually useful, but the full project
flow is otherwise spread across multiple places.

## One-Line Project Arc

The architecture progression is:

```text
Flat expert
-> blind rough locomotion baselines
-> early adaptation students (NA / V0 / V1 / V2)
-> Adapt-V3 Phase 1 privileged latent base policy
-> Adapt-V3 Phase 2 Stage A blind adaptive student
-> low-switch recovery branch
-> bounded-latent recovery branch
-> max-abs bounded-latent refinement
```

## Phase 0: Flat Expert

Purpose:

- build the first stable locomotion prior in a simpler setting

Primary artifact:

- `rma_go2_lab/policies/flat1499.pt`

Role in the ladder:

- provides the earliest locomotion competence prior
- forms the foundation from which blind rough locomotion can later be
  warm-started

Conceptual architecture:

```text
flat locomotion observations
-> locomotion policy
-> action
```

At this point there is:

- no rough-terrain branch yet
- no adaptation
- no explicit latent

## Phase 1: Blind Rough Locomotion Baselines

Purpose:

- establish a strong deployable blind locomotion baseline on rough terrain

Primary frozen artifacts:

- `rma_go2_lab/policies/blind_baseline1_scratch_final.pt`
- `rma_go2_lab/policies/blind_baseline2_warmstart_final.pt`
- `rma_go2_lab/policies/blind_baseline3_warmstart_imitation_final.pt`

Role in the ladder:

- prove rough locomotion works without privileged runtime inputs
- create the practical blind locomotion anchor used by later adaptation work

Conceptual architecture:

```text
policy observation
-> blind actor
-> action
```

Main differences across B1/B2/B3:

- scratch vs warm-start
- with or without imitation support

Important project meaning:

- these runs solve the “can the repo walk well blindly?” question before the
  adaptation problem is layered on top

## Phase 2: Early Adaptation Student Line

Purpose:

- introduce deployable history-based adaptation

Primary frozen artifacts:

- `rma_go2_lab/policies/adaptation_student_na_final.pt`
- `rma_go2_lab/policies/adaptation_student_v0_final.pt`
- `rma_go2_lab/policies/adaptation_student_v1_final.pt`
- `rma_go2_lab/policies/adaptation_student_v2_final.pt`

Canonical synthesis doc:

- `docs/ADAPTATION_PHASE_SYNTHESIS.md`

### studentNA

Purpose:

- no explicit adaptation pathway
- establish how far a deployable proprio-only student can go under hidden
  randomization without online adaptation

Conceptual architecture:

```text
policy observation
-> blind student actor
-> action
```

### studentAdapt-V0

Purpose:

- first history-based adaptation result

Conceptual architecture:

```text
current observation + history
-> history-conditioned student
-> action
```

Meaning:

- proves history-conditioned adaptation is useful in this repo

### studentAdapt-V1

Purpose:

- introduce an explicit latent target

Conceptual architecture:

```text
history
-> latent predictor
-> z_hat

current observation + z_hat
-> actor
-> action
```

Meaning:

- first explicit-latent adaptation milestone

### studentAdapt-V2

Purpose:

- modularize the adaptation pathway clearly

Conceptual architecture:

```text
phi(history) -> z_hat
pi(current_obs, z_hat) -> action
```

Meaning:

- first explicit modular split
- important architecture milestone even though its empirical result did not
  exceed `V1`

## Phase 3: Adapt-V3 Motivation

Purpose:

- move from “useful history adaptation” to a true original-RMA-style teacher /
  student latent contract

Canonical design docs:

- `docs/archive/adapt_v3/ADAPT_V3_TRUE_RMA_PLAN.md`
- `docs/ADAPT_V3_EXECUTION_SPEC.md`

Why this phase exists:

- `V0/V1/V2` proved adaptation ideas are workable
- but they do not yet implement the full explicit RMA contract in the cleanest
  sense

The intended RMA-style decomposition becomes:

```text
teacher side:
e_t -> mu -> z_t
x_t, z_t -> pi -> a_t

student / deployable side:
history_t -> phi -> z_hat_t
x_t, z_hat_t -> pi -> a_t
```

That introduces three explicit modules:

- `mu`: privileged extrinsics encoder
- `phi`: history-to-latent adaptation module
- `pi`: latent-conditioned base actor

## Phase 4: Adapt-V3 Phase 1

Purpose:

- train the privileged teacher/base policy side first

Primary frozen artifacts:

- `rma_go2_lab/policies/adapt_v3_phase1_stage_a_final.pt`
- `rma_go2_lab/policies/adapt_v3_dyn_only_phase1_stage_a_final.pt`
- `rma_go2_lab/policies/adapt_v3_terrain_lite_phase1_stage_a_final.pt`

The current active reboot is the dyn-only line.

### Phase 1 structure

Conceptual architecture:

```text
privileged extrinsics e_t
-> mu(e_t)
-> z_t

policy observation x_t (+ previous action semantics inside the actor path)
+ z_t
-> pi(x_t, z_t)
-> action
```

In this repo, the dyn-only reboot uses privileged dynamics factors such as:

- friction
- base mass ratio
- joint stiffness scale
- joint damping scale

Meaning:

- Phase 1 creates the teacher latent target and proves the actor can use that
  latent meaningfully

Important repo lesson from this phase:

- a structurally correct latent path is not enough
- `z` has to be genuinely load-bearing, not bypassed by the actor

## Phase 5: Adapt-V3 Phase 2 Stage A

Purpose:

- train the deployable blind adaptive student under a stable stationary regime

Primary frozen artifact:

- `rma_go2_lab/policies/adapt_v3_dyn_only_phase2_stage_a_final.pt`

This is the current stationary deployment-side winner.

### Phase 2 structure

Teacher/reference side:

```text
e_t -> mu -> z_t
```

Student/deployable side:

```text
history_t
-> phi(history_t)
-> z_hat_t

current policy observation x_t
+ z_hat_t
-> pi(x_t, z_hat_t)
-> action
```

Phase 2 training behavior:

- `mu` and the privileged base side are treated as reference
- `phi` is trained to recover the latent from deployable history
- PPO still trains the student behavior in the live rollout regime

This is the first successful deployable-path:

```text
phi(history) -> z_hat -> pi
```

artifact in the repo.

## Phase 6: Low-Switch Recovery Branch

Purpose:

- restore real online adaptation after the stationary Stage A artifact was
  found not to express the adaptation story strongly enough

Primary frozen artifact:

- `rma_go2_lab/policies/adapt_v3_dyn_only_phase2_recovery_low_switch_final.pt`

Canonical audit doc:

- `docs/ADAPT_V3_POISONING_AUDIT.md`

What changed:

- same core architecture
- different training regime with low-probability within-episode hidden-dynamics
  switches

Architecture remains:

```text
teacher:
e_t -> mu -> z_t

student:
history -> phi -> z_hat
policy obs + z_hat -> pi -> action
```

Meaning:

- this branch restored real adaptation pressure
- it became the canonical adaptation-recovery anchor
- but it remained too fragile in MuJoCo

## Phase 7: Bounded-Latent Recovery Branch

Purpose:

- reduce MuJoCo latent blow-up without abandoning the recovered adaptive line

Primary frozen artifact:

- `rma_go2_lab/policies/adapt_v3_dyn_only_phase2_recovery_low_switch_latent_reg_final.pt`

What changed:

- same architecture
- same low-switch recovery task family
- added training-side latent magnitude control

Architecture is still:

```text
history -> phi -> z_hat
policy obs + z_hat -> pi -> action
```

Training change:

- add student latent L2 regularization

New monitored quantities:

- `student_latent_l2`
- `student_latent_max_abs`

Meaning:

- first training-side Sim2Sim robustness repair
- materially improved unclamped MuJoCo behavior relative to the earlier
  recovery artifact

## Phase 8: Max-Abs Bounded-Latent Refinement

Purpose:

- continue the bounded-latent line with a more targeted control of coordinate
  spikes

This was a later explored refinement branch in the same adaptive family. It is
now retained as a documented non-winning result rather than an active
implementation path.

What changed:

- same architecture
- keep latent L2 penalty
- add thresholded coordinate-wise max-abs control

Conceptual training change:

```text
base Phase 2 losses
+ latent regression to teacher z
+ latent L2 control
+ thresholded max-abs excess penalty on z_hat
```

New monitored quantity:

- `student_latent_max_abs_excess`

Meaning:

- this is still the same adaptive architecture family
- we are refining the latent behavior, not replacing the architecture

Follow-on note:

- a temporal-smoothness refinement branch was also explored later in the same
  family
- both the max-abs and temporal-smoothness branches are now historical
  refinements, not active code paths

## Current Live Architecture

For the active dyn-only adaptive line, the deployed controller is:

```text
policy_history (960)
-> phi
-> z_hat (32)

policy observation (48)
+ z_hat (32)
-> pi
-> action (12)
```

Teacher/reference path in training:

```text
dynamics_privileged
-> mu
-> z
```

So the full live contract is:

```text
teacher during training:
e_t -> mu -> z_t

student during training and deployment:
history_t -> phi -> z_hat_t
x_t, z_hat_t -> pi -> a_t
```

## Flowchart Version

If you want one single text flowchart, use this:

```text
Flat Expert
  -> first locomotion competence prior

Blind Rough Baselines
  -> B1 scratch
  -> B2 warm-start
  -> B3 warm-start + imitation
  -> strong blind rough locomotion anchor

Early Adaptation Line
  -> studentNA
  -> studentAdapt-V0
  -> studentAdapt-V1
  -> studentAdapt-V2
  -> deployable history adaptation established

Adapt-V3 Phase 1
  privileged extrinsics e_t
    -> mu
    -> z_t
  policy obs + z_t
    -> pi
    -> action
  -> privileged latent base policy established

Adapt-V3 Phase 2 Stage A
  history
    -> phi
    -> z_hat
  policy obs + z_hat
    -> pi
    -> action
  -> first successful deployable Adapt-V3 student

Low-Switch Recovery
  same architecture
  + switch pressure
  -> real online adaptation restored

Bounded-Latent Recovery
  same architecture
  + latent L2 control
  -> first MuJoCo-oriented adaptive repair

Max-Abs Refinement
  same architecture
  + thresholded coordinate spike control
  -> current adaptive refinement branch
```

## How To Read The Project Now

The most important distinction today is:

- the architecture from `Adapt-V3` onward has stayed mostly stable
- what changed most recently is:
  - training regime
  - latent regularization
  - checkpoint selection discipline
  - Sim2Sim evaluation rigor

So the current project is not repeatedly replacing the whole architecture.

It is:

- building a stable blind-adaptive RMA-style stack
- then making that stack scientifically honest
- then making it less brittle under deployment-like simulator shift
