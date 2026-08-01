# Deployment Plan

This document turns the current repo state into a concrete real-robot
deployment roadmap.

The goal is not to jump directly from sim success to uncontrolled outdoor
trials. The goal is to:

- choose the right deployment candidates
- harden the software interface
- validate transfer through Sim2Sim before robot use
- compare against the default Unitree controller fairly
- run structured failure-case testing
- graduate through increasingly difficult testbeds

## Deployment Workspace

This file is now the canonical deployment doc.

Deployment work should live in an isolated repo surface rather than being
spread across training and evaluation helpers.

Canonical deployment surface:

- `docs/DEPLOYMENT_PLAN.md`
  - policy order, workspace contract, Sim2Sim gate, and acceptance criteria
- `scripts/deploy/`
  - deployment-facing scripts only
- `rma_go2_lab/policies/exported/`
  - frozen exported candidate bundles only

Do not treat:

- `scripts/eval/`
- `rma_go2_lab/tools/`
- ad hoc IsaacLab training entrypoints

as the canonical deployment surface.

### Ownership Rule

Without an isolated deployment section, the repo drifts toward:

- training scripts being reused as deployment tools
- evaluation helpers being mistaken for robot runtime utilities
- exported policy artifacts accumulating without clear ownership
- hardware readiness becoming a scattered set of notes rather than a pipeline

This section exists to prevent that.

### Deployment Surface Layout

Planning and policy selection:

- `docs/DEPLOYMENT_PLAN.md`

Deployment scripts:

- `scripts/deploy/`

This folder owns:

- export
- bundle packaging
- deployable-I/O validation
- simulation-side deployment rehearsal
- Sim2Sim transfer rehearsal
- deployment log inspection

Frozen exported artifacts:

- `rma_go2_lab/policies/exported/`

This folder should contain only versioned exported artifacts that correspond to
frozen deployment candidates.

Ad hoc simulation tools:

- `rma_go2_lab/tools/`

This folder is not the canonical deployment surface.

## Deployment Candidates

The current policy ladder should be interpreted as:

### Reference / not a deployment target

- `V3` privileged teacher
  - training-time oracle benchmark
  - privileged reference policy
  - not a deployable hardware target

### Real deployment candidates

1. `B2`
   - strongest clean blind deployable reference from the blind phase
   - simplest strong policy to validate the software stack on hardware
2. `c1_ethlike_v3_model_400_candidate`
   - current Candidate 1 blind-history deployment winner
   - teacher root repaired through `Teacher V4 model_300`
   - exported bundle validated through Isaac deploy rehearsal and MuJoCo OOD
     suites
3. `studentNA`
   - deployable baseline trained under hidden within-episode switches
   - measures what fixed non-adaptive robustness can do
4. `studentAdapt-V0`
   - first completed positive adaptation result
   - current best adaptation deployment candidate
5. `adapt_v3_dyn_only_phase2_stage_a_final`
   - active adaptive deployment-path reference
6. terrain-aware `Adapt-V3`
   - frozen challenger artifact exists
   - not the current deployment winner

## Deployment Roles

It is important to separate:

- research winner
- deployment winner

The research winner is the strongest scientifically justified policy under the
eval ladder.

The deployment winner is the best policy to put on hardware first, balancing:

- performance
- simplicity
- safety
- observability
- debugging difficulty

These may be the same policy, but they do not have to be.

## Current Recommendation

Recommended real-robot order:

1. Unitree default controller
2. `B2`
3. `c1_ethlike_v3_model_400_candidate`
4. `studentNA`
5. `studentAdapt-V0`
6. `adapt_v3_dyn_only_phase2_stage_a_final` once the adaptive line clears the
   same deployment-validation honesty bar
7. terrain-aware `Adapt-V3` only if a later refinement beats dyn-only

Reasoning:

- the default controller is the operational baseline the lab already trusts
- `B2` is the cleanest learned deployable reference
- `c1_ethlike_v3_model_400_candidate` is now the best validated blind-history
  deployment artifact in the repo
- `studentNA` lets us test the switched-task baseline without adaptation
- `studentAdapt-V0` is the first adaptation policy that already won in sim
- `adapt_v3_dyn_only_phase2_stage_a_final` remains an adaptive deployment-path
  reference rather than the practical blind-history winner
- terrain-aware `Adapt-V3` should only move ahead if its Phase 2 student
  finishes and wins the final comparison

## Pre-Deployment Readiness Checklist

Before any learned policy is put on the robot, verify:

### Sim2Sim gate

- frozen export exists for the exact candidate being considered
- deployable-I/O rehearsal passes in the source stack
- MuJoCo Sim2Sim rehearsal passes with the deployable policy
- the Sim2Sim run is logged and attached to the candidate record
- any sim-to-sim mismatch is explained before hardware use

### Interface correctness

- joint ordering matches the real robot interface exactly
- action units match the deployment controller assumptions
- action scaling is documented and fixed
- base-frame conventions match sim assumptions
- IMU and joint-state signal conventions are verified

### Timing and control

- deployment control rate is defined
- policy inference rate is defined
- history update rate is defined for adaptation policies
- any asynchronous latent update logic is documented
- total end-to-end latency is measured

### Safety

- joint position clamps are enforced
- torque / action saturation behavior is understood
- emergency stop path is working
- watchdog behavior is defined
- startup stance / transition logic is safe
- a human spotter protocol is defined

### Logging

- commanded velocity is logged
- estimated base velocity is logged
- joint positions / velocities are logged
- actions are logged
- safety triggers / abort causes are logged
- rollout video is recorded when possible

### Policy integrity

- policy checkpoint path is frozen and documented
- deployable observations are the only observations used at runtime
- no privileged inputs leak into deployment

## Deployment Pipeline Contract

Every deployment candidate should move through the same repo-level pipeline:

1. freeze a simulation checkpoint
2. write or update the freeze note
3. export the checkpoint into deployment-ready formats
4. package the deployment bundle with metadata
5. validate deployable observations and action scaling
6. rehearse in sim using only the deployment I/O contract
7. run a Sim2Sim transfer rehearsal in MuJoCo with the deployable policy
8. graduate into hardware testbeds only after the deployment checklist passes

Current first-entered bundle:

- `rma_go2_lab/policies/exported/adapt_v3_dyn_only_phase2_stage_a_final/`

## Sim2Sim Contract

Sim2Sim is the formal phase between IsaacLab-side validation and real Go2
deployment.

Its job is not to replace hardware testing. Its job is to catch:

- observation-contract mistakes
- action-scaling mistakes
- timing/control-rate mismatches
- history-update mistakes for adaptive students
- gross behavior regressions when the policy leaves the original simulator

For this repo, Sim2Sim means:

- the frozen exported policy is run in MuJoCo
- only deployable observations are used
- the control/update contract matches the intended robot runtime as closely as
  possible
- command tracking, posture, and failure modes are logged before hardware use

The intended order is:

1. exact deployable-I/O rehearsal
2. MuJoCo Sim2Sim rehearsal
3. hardware ladder

### Primary Go2 MuJoCo Reference

For the current deployment path, treat:

- `reference_repos/mujoco_menagerie/unitree_go2/scene.xml`

as the canonical Go2 MuJoCo model entrypoint.

Use:

- `reference_repos/unitree_mujoco/`

as the runtime/integration reference rather than the primary model source.

## Required Bundle Metadata

Every frozen deployment bundle should record:

- policy name
- source checkpoint path
- task name
- training phase
- whether the policy is:
  - blind fixed policy
  - blind history policy
  - blind adaptive student
  - privileged base only
- deployable observation groups
- control rate
- latent update semantics if adaptive
- Sim2Sim status and reference run path once available
- export format paths
- freeze note path

## Deployment Evaluation Goals

The real deployment program should answer four questions:

1. Can the learned policy operate safely on hardware at all?
2. How does it compare to the default Unitree controller?
3. Which failure modes are handled by fixed robustness alone?
4. Which failure modes benefit from adaptation?

## Comparison With Default Unitree Controller

The default Unitree controller should be treated as a first-class baseline, not
just a convenience fallback.

### Why compare against it

- it is the real operational baseline users already have
- it provides a strong standard for basic reliability
- it exposes whether the learned policy is actually useful in practice

### Comparison protocol

For each testbed and task, compare:

- Unitree default controller
- `B2`
- `studentNA`
- `studentAdapt-V0`
- `studentAdapt-V1` if available and frozen

### Common metrics

- success / failure
- time to failure
- distance traveled
- tracking quality
- visible slip
- base sag / posture collapse
- recovery after disturbance
- operator interventions

### Reporting rule

Never report only “policy X worked.”

Always report relative to:

- Unitree default controller
- strongest relevant learned baseline

## Failure-Case Testing

Failure-case testing should be deliberate, not accidental.

The point is to find out:

- how the policy fails
- how fast it fails
- whether failure is recoverable
- whether adaptation helps

### Primary failure families

1. Friction mismatch
   - likely the most posture-destabilizing hidden factor
   - watch for slip, roll/pitch instability, and base-height loss
2. Added payload / mass shift
   - watch for sag, sluggish correction, and reduced acceleration authority
3. Weak actuation / low motor authority
   - watch for poor command tracking and inability to recover stance
4. External pushes
   - watch for recovery latency and post-push drift
5. Geometry mismatch
   - watch for foothold errors, stumbling, and clearance failures

### Failure-case protocol

For each failure family:

- start at nominal intensity
- increase one factor at a time
- keep command profile fixed
- repeat enough times to observe consistency
- log the first failure mode, not just binary failure

### Failure labels to record

- slip-induced instability
- base-height collapse
- pitch/roll loss
- low-progress stall
- unsafe oscillation
- recovery succeeded after transient

## Testbed Ladder

The robot should move through testbeds in a controlled progression.

### Testbed 0: Bench / hanger / no-ground sanity

Purpose:

- verify observation correctness
- verify joint ordering
- verify action scaling
- verify policy timing and startup behavior

Checks:

- stance posture
- small action responses
- no exploding outputs

### Testbed 1: Flat indoor clean floor

Purpose:

- first locomotion sanity
- compare against Unitree default controller in the simplest setting

Checks:

- standing stability
- forward walking
- turning
- stop-and-go transitions

### Testbed 2: Controlled indoor perturbation floor

Examples:

- rubber mat
- tarp / low-slip patch
- foam patch

Purpose:

- controlled mismatch without outdoor chaos

Checks:

- slip handling
- posture recovery
- repeated traversal consistency

### Testbed 3: Payload and actuation stress

Examples:

- removable payload packs
- battery/load variants if relevant

Purpose:

- stress mass and authority robustness

Checks:

- sag
- tracking degradation
- ability to recover after stops and turns

### Testbed 4: Mild outdoor rough terrain

Examples:

- compact dirt
- grass
- uneven but non-extreme trail

Purpose:

- validate that nominal sim robustness transfers outside the lab

Checks:

- sustained locomotion
- drift
- stumble frequency
- operator intervention rate

### Testbed 5: Challenge testbeds

Examples:

- slippery patches
- loose gravel / pebbles
- steeper slopes
- step-downs or stairs if explicitly approved

Purpose:

- targeted challenge demonstrations after basic safety is established

Checks:

- repeated success rate
- failure type distribution
- whether adaptation policies separate from blind baselines

## Recommended Hardware Sequence

For each policy candidate:

1. exact deployable-I/O rehearsal in sim
2. MuJoCo Sim2Sim rehearsal
3. bench sanity
4. flat indoor nominal
5. flat indoor forward / turn / stop
6. controlled perturbation patch
7. payload stress
8. mild outdoor rough terrain
9. challenge testbeds

Do not advance to the next stage unless:

- safety is acceptable
- behavior is repeatable
- logs are interpretable

## Suggested First Hardware Trial Matrix

### Unitree default controller

- flat indoor nominal
- mild slope
- one controlled slippery patch

### `B2`

- flat indoor nominal
- mild slope
- one controlled payload condition

### `studentNA`

- flat indoor nominal
- one friction stress test
- one payload stress test

### `studentAdapt-V0`

- same as `studentNA`
- plus one repeated hidden-mismatch-like stress sequence

This creates a fair early comparison without jumping into dangerous terrain too
soon.

## Success Criteria

Early deployment success should mean:

- safe bring-up
- repeatable flat-ground walking
- no privileged-input leakage
- no immediate instability under mild perturbation

Strong deployment success should mean:

- consistent superiority over at least one meaningful baseline
- either:
  - better robustness than the Unitree controller in target cases
  - or better robustness than the no-adaptation student in target cases

## What Would Count As A Real Adaptation Win On Hardware

The cleanest hardware win would be:

- `studentAdapt-V0` or `studentAdapt-V1`
- outperforming `studentNA`
- on friction / payload / switched-like mismatch stress
- while preserving acceptable nominal locomotion

The strongest practical win would be:

- outperforming the default Unitree controller
- on one or more targeted challenging testbeds
- without giving up basic safety and repeatability

## Immediate Next Actions

1. freeze deployment candidate order
2. create the deployment bundle / manifest workflow in `scripts/deploy/`
3. define the MuJoCo Sim2Sim rehearsal entrypoint and logging contract
4. define deployment logging bundle
5. define first hardware safety checklist
6. prepare the first indoor testbeds
7. compare Unitree default controller and `B2` first
8. graduate to `studentNA` and `studentAdapt-V0`

## Recommended References For This Phase

- `docs/archive/deployment/DEPLOYMENT_WORKSPACE_SPEC.md`
- `docs/ADAPTATION_PHASE_SYNTHESIS.md`
- `docs/EVALUATION_METHODS.md`
- `docs/TEACHER_V4_MODEL300_CARD.md`
- `rma_go2_lab/policies/adaptation_student_na_final.md`
- `rma_go2_lab/policies/adaptation_student_v0_final.md`
