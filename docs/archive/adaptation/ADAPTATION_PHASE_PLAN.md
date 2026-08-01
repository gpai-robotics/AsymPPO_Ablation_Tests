# Adaptation Phase Plan

Status:

- historical planning note
- kept for branch-history context
- superseded by:
  - `docs/ADAPTATION_PHASE_SYNTHESIS.md`
  - `docs/archive/adapt_v3/ADAPT_V3_TRUE_RMA_PLAN.md`
  - `docs/ADAPT_V3_EXECUTION_SPEC.md`

This note defines the planned post-teacher branch after `V3`.

The goal is not to keep extending teacher-only PPO variants. The goal is to
move toward an RMA-style argument where the latent/adaptation mechanism becomes
structurally important rather than merely available.

## Why This Phase Exists

The teacher phase taught us:

- terrain privilege is somewhat useful
- terrain + dynamics privilege is more promising than terrain-only privilege
- plain PPO can use privilege to some extent
- but teacher-only results do not yet produce an original-RMA-strength case

What still appears missing is not simply "more privilege". It is a setup where:

- hidden factors change enough to matter
- a deployable blind policy cannot fully average them away
- an adaptation module has to infer a compact latent online

That is the purpose of this phase.

## Main Question

The main scientific question becomes:

> when hidden terrain/dynamics factors vary within an episode, does an
> adaptation latent materially outperform a blind robust policy while
> approaching the privileged teacher upper bound?

## Target Comparison Ladder

The intended ladder for this phase is:

1. Blind robust baseline
   - deployable policy
   - no latent
   - no adaptation module

2. Student without adaptation
   - same student policy family
   - no latent inference module
   - used to isolate the value of adaptation itself

3. Student with adaptation
   - proprio/history encoder predicts latent `z_hat_t`
   - deployable policy consumes `z_hat_t`

4. Privileged expert / teacher upper bound
   - receives privileged terrain/dynamics information during training/eval
   - defines the reachable upper bound

Optional later baselines:

- explicit system-ID latent baseline
- offline latent optimization baseline

## What Must Change Relative To Teacher Phase

Teacher-only PPO was useful, but it still allowed a robust controller to solve
too much of the problem directly.

To make the adaptation case stronger, the new phase should:

- make hidden factors vary within an episode
- keep the deployment policy blind to those factors
- provide a privileged teacher/expert during training
- make the adaptation latent the only mechanism that can explain those changes

## Environment Requirements

### 1. Within-episode hidden variation

The training environment should no longer rely only on startup randomization.

It should include controlled mid-episode changes such as:

- friction switches
- payload / mass switches
- motor strength switches
- possibly terrain difficulty transitions if handled cleanly

These changes should be frequent enough that a "single robust compromise gait"
is not obviously sufficient.

### 2. Same benchmark philosophy

We should keep the existing benchmark discipline where possible:

- same rough terrain family
- same command family
- same core reward structure
- same termination family

This keeps adaptation claims comparable to the blind and teacher phases.

### 3. Explicit deployability

The student policy should remain deployable:

- no height scan at deployment
- no direct friction/mass/gain values at deployment
- only proprioception, action history, and the learned adaptation latent

## Planned Model Split

### Expert / privileged teacher

Use the strongest available privileged teacher recipe as the expert reference.

Locked choice:

- `V3 final`
- checkpoint:
  `/home/bhuvan/tools/IsaacLab/logs/rsl_rl/go2_privileged_teacher_rough_v3/2026-04-21_15-35-03/model_1999.pt`

### Student policy

Student policy should consume:

- current proprio observation
- latent `z_hat_t`

It should **not** directly consume privileged channels.

### Adaptation module

The adaptation module should consume:

- short proprio history
- recent action history

and output:

- `z_hat_t`

This should be closer in spirit to original RMA / `rl_locomotion` than to
concatenating extra privileged observations into PPO.

## Training Strategy

### Phase A: privileged expert fixed

- freeze final privileged teacher choice
- do not keep changing the expert while student/adaptation training begins

### Phase B: student without adaptation

Train/evaluate a student policy without a latent adapter.

Purpose:

- establish the no-adaptation gap explicitly

### Phase C: student with adaptation

Train the adaptation module to infer `z_hat_t` online from history.

The exact supervision strategy remains open, but likely candidates are:

- behavior cloning / imitation toward expert actions
- latent regression toward teacher latent
- hybrid PPO + imitation

We do not need to commit to the final method before Phase B is complete, but
the ladder itself should stay fixed.

## Metrics To Carry Forward

Use the same metrics we already trust:

- nominal forward speed / drift
- standstill quality
- isolated suite scores
- geometry OOD
- dynamics OOD
- push OOD
- switch OOD

Additional adaptation-specific metrics to add:

- recovery delay after switch
- latent convergence speed after switch
- performance drop immediately after switch
- performance recovered after switch

These are the metrics most likely to make an adaptation case stronger than the
teacher-only phase.

## What Counts As A Stronger-Than-Teacher Result

To approach an original-RMA-strength story, we want something like:

- student with adaptation clearly beats blind robust / no-adaptation student
- student with adaptation approaches the privileged teacher
- improvement is especially visible under within-episode hidden changes
- latent behavior changes coherently during disturbances or switches

That is stronger than simply saying:

- privilege helped PPO a bit

## What We Should Not Do

After `V3`, avoid:

- more terrain-only teacher micro-variants
- more tiny reward tweaks to force a story
- more latent-size sweeps without changing the formulation
- claiming an RMA-style result from teacher-only comparisons alone

Those would likely add noise without clarifying the adaptation question.

## Concrete Implementation Order

1. Close `V3`
   - finish training
   - run full eval ladder
   - freeze teacher conclusion

2. Lock expert choice
   - `V3 final`

3. Define adaptation env branch
   - preserve current rough task family
   - add controlled within-episode switches during training

4. Implement student-without-adaptation baseline
   - same deployment observation family as future student
   - no latent module

5. Implement adaptation module
   - history encoder
   - latent output `z_hat_t`

6. Implement student-with-adaptation policy
   - fuse proprio + `z_hat_t`

7. Add adaptation-specific diagnostics
   - switch recovery metrics
   - latent evolution logging

8. Run comparison ladder
   - blind robust
   - student without adaptation
   - student with adaptation
   - privileged expert

## Immediate Next Repo Tasks

After `V3` closure, the first code tasks should be:

- add an adaptation planning doc checkpoint to the synthesis
- create adaptation env/config scaffolds
- define the deployable student observation interface
- define the history window and latent size candidates

## Bottom Line

The post-`V3` branch should not be "Teacher V4".

It should be:

- a controlled adaptation phase
- with a fixed privileged expert
- a no-adaptation student baseline
- and a history-based adaptation module

That is the most direct path to making a stronger case like original RMA.
