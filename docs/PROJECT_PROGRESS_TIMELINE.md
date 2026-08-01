# Project Progress Timeline

This note summarizes what the project has accomplished from the start of the
repo lineage up to the current `Adapt-V3` recovery and Sim2Sim debugging work.

The goal is not to list every run. The goal is to record the major
project-level risk reductions, the frozen artifacts that support them, and the
remaining frontier that is still being worked on.

Use this file when the question is:

- what did we actually finish so far?
- which milestones are backed by frozen artifacts?
- where did the project story change?
- what is still in-flight rather than proven?

Companion canonical docs:

- `docs/PROJECT_GUIDE.md`
- `rma_go2_lab/policies/README.md`
- `docs/ADAPT_V3_ACTIVE_ROADMAP.md`
- `docs/ADAPT_V3_POISONING_AUDIT.md`
- `docs/DEPLOYMENT_AUDIT_ADAPT_V3.md`

## Reading Rule

This file is intentionally milestone-oriented.

- frozen checkpoints and freeze notes are treated as the canonical evidence
- major eval artifacts and audits are cited where they materially support a
  milestone
- current in-flight branches are described separately from frozen claims

## Milestone 1: Flat Prior Backbone

The first completed milestone was a stable flat-ground locomotion prior that
later branches could reuse instead of learning rough locomotion from scratch.

Canonical frozen artifact:

- `rma_go2_lab/policies/flat1499.pt`

Freeze note:

- `rma_go2_lab/policies/flat_prior_final.md`

Supporting sanity artifacts:

- `artifacts/evaluations/flat_prior/gait_flat_prior_model1499_standstill.json`
- `artifacts/evaluations/flat_prior/gait_flat_prior_model1499_forward.json`

What this milestone proved:

- the repo can train a usable locomotion prior
- later warm-start experiments do not need to begin from a random actor
- the corrected flat-prior lineage is stable enough to serve as a project root

## Milestone 2: Blind Rough Baseline Ladder

The next completed milestone was the blind rough-terrain baseline ladder. This
established the fixed-policy comparison backbone used throughout the rest of
the project.

Canonical frozen artifacts:

- `rma_go2_lab/policies/blind_baseline1_scratch_final.pt`
- `rma_go2_lab/policies/blind_baseline2_warmstart_final.pt`
- `rma_go2_lab/policies/blind_baseline3_warmstart_imitation_final.pt`

Freeze notes:

- `rma_go2_lab/policies/blind_baseline1_scratch_final.md`
- `rma_go2_lab/policies/blind_baseline2_warmstart_final.md`
- `rma_go2_lab/policies/blind_baseline3_warmstart_imitation_final.md`

Supporting synthesis:

- `docs/BASELINE_COMPARISON_FINAL.md`
- `docs/FROZEN_BASELINE_SYNTHESIS.md`
- `docs/FROZEN_BASELINE_RESULTS_AT_A_GLANCE.md`
- `docs/OOD_FINDINGS_B1_B2_B3.md`

Representative eval artifacts:

- `artifacts/evaluations/baseline1/`
- `artifacts/evaluations/baseline2/`
- `artifacts/evaluations/baseline3/`
- `artifacts/ood_evaluations/baseline2/`

What this milestone proved:

- blind rough locomotion works in the repo without adaptation
- warm-starting from the flat prior is useful
- the blind baseline regime can be frozen and audited instead of constantly
  moving

## Milestone 3: Privileged Teacher Lineage

The project then built and froze the privileged teacher line. This is the upper
bound and supervision source for the later adaptation work.

Canonical teacher card:

- `docs/TEACHER_V4_MODEL300_CARD.md`

Supporting synthesis:

- `docs/TEACHER_PHASE_SYNTHESIS.md`

Canonical phase notes:

- `docs/archive/teacher_design/`

Key effect on the project:

- privileged teacher training became a stable ingredient rather than an
  unresolved exploratory subsystem
- later adaptation and `Adapt-V3` work could be framed around specific teacher
  contracts instead of generic “teacher helps student” intuition

Important later qualification:

- later dependency audits showed that the frozen `Teacher V3` checkpoint is
  materially using `dynamics_privileged`
- the same checkpoint appears to ignore `terrain_privileged` on the audited
  forward probes
- so Milestone 3 should now be read as “teacher lineage frozen and auditable,”
  not “teacher fully validated as a terrain+dynamics user”

Post-freeze recovery update:

- later teacher recovery work produced one clear overall active candidate and
  several archived follow-up branches
- `Teacher V4 model_300`
  - current canonical overall teacher candidate
  - validated as using both terrain and dynamics privilege on the audited
    `random_rough` and `boxes` probes
- `Teacher V4.1 model_1999`
  - archived stair-specialized diagnostic branch
  - validated as using both terrain and dynamics privilege on the audited
    `pyramid_stairs` and `pyramid_stairs_inv` probes
- `Teacher V5` and `Teacher V6`
  - archived exploratory branches
  - not part of the active teacher surface

Current limitation:

- the repo still does not have one single teacher checkpoint that is the best
  validated terrain+dynamics user across every terrain family
- to stay honest to the project contract, the repo now keeps only the best
  overall candidate active and archives specialized detours

## Milestone 4: First Adaptation Lineage

Before `Adapt-V3`, the repo completed a full earlier adaptation lineage. This
is important because it proves that the repo already knew how to train real
switched-task adaptation before the later `Adapt-V3` confusion.

Canonical frozen artifacts:

- `rma_go2_lab/policies/adaptation_student_na_final.pt`
- `rma_go2_lab/policies/adaptation_student_v0_final.pt`
- `rma_go2_lab/policies/adaptation_student_v1_final.pt`
- `rma_go2_lab/policies/adaptation_student_v2_final.pt`

Freeze notes:

- `rma_go2_lab/policies/adaptation_student_na_final.md`
- `rma_go2_lab/policies/adaptation_student_v0_final.md`
- `rma_go2_lab/policies/adaptation_student_v1_final.md`
- `rma_go2_lab/policies/adaptation_student_v2_final.md`

Supporting references:

- `docs/ADAPTATION_IMPLEMENTATION_V0.md`
- `docs/ADAPTATION_PHASE_SYNTHESIS.md`
- `docs/V1_V2_CLOSEOUT_CHECKLIST.md`

Representative eval artifacts:

- `artifacts/evaluations/adaptation_student_na/`
- `artifacts/evaluations/adaptation_student_v0/`
- `artifacts/evaluations/adaptation_student_v1/`
- `artifacts/evaluations/adaptation_student_v2/`
- `artifacts/ood_evaluations/adaptation_student_na/`
- `artifacts/ood_evaluations/adaptation_student_v0/`

What this milestone proved:

- switched-task adaptation is real in this repo
- `history -> latent -> actor` style ideas are not purely aspirational
- there is a completed pre-`Adapt-V3` adaptation archive that remains valid

## Milestone 5: Modern `Adapt-V3` Architecture

The repo then built the newer `Adapt-V3` architecture family, whose intended
contract is:

- privileged extrinsics encoder `mu(e)`
- blind history adaptation module `phi(history)`
- actor `pi(obs, z_hat)`

Historical and active frozen bases:

- `rma_go2_lab/policies/adapt_v3_phase1_stage_a_final.pt`
- `rma_go2_lab/policies/adapt_v3_dyn_only_phase1_stage_a_final.pt`
- `rma_go2_lab/policies/adapt_v3_phase2_stage_a_final.pt`

Freeze notes:

- `rma_go2_lab/policies/adapt_v3_phase1_stage_a_final.md`
- `rma_go2_lab/policies/adapt_v3_dyn_only_phase1_stage_a_final.md`
- `rma_go2_lab/policies/adapt_v3_phase2_stage_a_final.md`

What this milestone proved:

- the modern architecture was implemented end-to-end
- Stage A training could produce a strong blind student candidate
- the project now had a modern privileged-latent deployment-side stack rather
  than only the earlier adaptation family

## Milestone 6: Frozen `Adapt-V3` Finalists

The first major `Adapt-V3` finishing step was freezing the main Phase 2
blind-student candidates.

Canonical frozen finalists:

- `rma_go2_lab/policies/adapt_v3_dyn_only_phase2_stage_a_final.pt`
- `rma_go2_lab/policies/adapt_v3_terrain_lite_phase2_stage_a_final.pt`

Supporting parent artifact:

- `rma_go2_lab/policies/adapt_v3_terrain_lite_phase1_stage_a_final.pt`

Freeze notes:

- `rma_go2_lab/policies/adapt_v3_dyn_only_phase2_stage_a_final.md`
- `rma_go2_lab/policies/adapt_v3_terrain_lite_phase1_stage_a_final.md`
- `rma_go2_lab/policies/adapt_v3_terrain_lite_phase2_stage_a_final.md`

Supporting comparison docs:

- `docs/ADAPT_V3_FINAL_CANDIDATE_COMPARISON.md`
- `docs/FINAL_CANDIDATE_COMPARISON_RUBRIC.md`
- `docs/ADAPT_V3_ACTIVE_ROADMAP.md`

Representative eval artifacts:

- `artifacts/evaluations/adapt_v3_dyn_only/`
- `artifacts/evaluations/adapt_v3_terrain_lite/`
- `artifacts/ood_evaluations/adapt_v3_dyn_only/`

What this milestone proved:

- the repo could produce competitive modern blind-student finalists
- the dyn-only and terrain-lite paths were both real enough to freeze and
  compare
- the current deployment-side winner could be chosen from a documented
  head-to-head process

Current official interpretation:

- `adapt_v3_dyn_only_phase2_stage_a_final.pt` is the current deployment-side
  winner
- `adapt_v3_terrain_lite_phase2_stage_a_final.pt` is the retained terrain-aware
  challenger

## Milestone 7: Adaptation Claim Audit

One of the most important project events was not a training win but a
scientific audit.

The deployment-side probe showed that the frozen dyn-only `Adapt-V3` winner did
not actually demonstrate active online-changing latent behavior under the
tested hidden-dynamics switch.

Canonical audit docs:

- `docs/ADAPTATION_PROBE_NOTES.md`
- `docs/ADAPT_V3_POISONING_AUDIT.md`

What this milestone proved:

- the repo’s strongest deployment-side artifact was not automatically the
  strongest adaptation-story artifact
- the final dyn-only Stage A training contract had `adaptation_switch_episode_prob = 0.0`
- without the audit, the project could easily have overclaimed modern online
  adaptation

Why this matters:

- it prevented a polished but wrong project story
- it separated “strong blind student” from “proven online-adaptive final
  deployment policy”
- it created the need for a repair branch instead of letting the repo drift
  into narrative debt

## Milestone 8: Recovery Branch Restored Real Adaptation

After the audit, the repo started a low-switch recovery line whose purpose was
to restore real non-collapsed online latent behavior in the modern `Adapt-V3`
stack.

Canonical frozen recovery artifact:

- `rma_go2_lab/policies/adapt_v3_dyn_only_phase2_recovery_low_switch_final.pt`

Freeze note:

- `rma_go2_lab/policies/adapt_v3_dyn_only_phase2_recovery_low_switch_final.md`

Supporting status docs:

- `docs/ADAPT_V3_ACTIVE_ROADMAP.md`
- `docs/PROJECT_GUIDE.md`
- `docs/DEPLOYMENT_AUDIT_ADAPT_V3.md`

Recovery sweep provenance:

- `artifacts/evaluations/adapt_v3_recovery_low_switch_ckpt_sweep/`

What this milestone proved:

- the modern `Adapt-V3` architecture can again show real switch-aware latent
  behavior
- the recovery branch is scientifically stronger than the stationary Stage A
  winner with respect to adaptation truth
- checkpoint selection matters; the canonical recovery artifact was selected
  from `model_1200.pt` rather than from the final checkpoint

Current official interpretation:

- `adapt_v3_dyn_only_phase2_recovery_low_switch_final.pt` is the canonical
  adaptation-recovery anchor
- it does not silently replace the Stage A dyn-only winner as the default
  deployment artifact

## Milestone 9: Deployment Surface Became Real

The repo then converted deployment from a vague promise into an explicit
artifact, packaging, export, and Sim2Sim workflow.

Canonical deployment docs:

- `docs/DEPLOYMENT_AUDIT_ADAPT_V3.md`
- `docs/archive/deployment/DEPLOYMENT_WORKSPACE_SPEC.md`
- `docs/SIM2SIM_DEBUGGING_NOTES.md`
- `docs/DEPLOYMENT_PLAN.md`

Canonical export surface:

- `scripts/deploy/package_candidate.py`
- `scripts/deploy/export_policy.py`
- `scripts/deploy/run_sim2sim.py`
- `scripts/deploy/mujoco_runtime.py`

Canonical exported bundle for the stationary winner:

- `rma_go2_lab/policies/exported/adapt_v3_dyn_only_phase2_stage_a_final/`

What this milestone proved:

- export artifacts can be generated reproducibly
- deployment/runtime contract is explicit rather than implicit
- Sim2Sim can expose subtle semantic failures that would be easy to miss in a
  casual runtime demo

## Milestone 10: Recovery Sim2Sim Failure Was Isolated

The recovery checkpoint was then pushed through the same deploy/export/Sim2Sim
surface.

Key debug artifacts:

- `artifacts/debug/adapt_v3_dyn_only_phase2_recovery_low_switch_sim2sim.json`
- `artifacts/debug/adapt_v3_dyn_only_phase2_recovery_low_switch_sim2sim_clamp10.json`
- `artifacts/debug/adapt_v3_dyn_only_phase2_recovery_low_switch_sim2sim_clamp5.json`

What the debug work showed:

- export, bundle, and runtime preflight were not the main problem
- the MuJoCo history stream pushed `phi(history)` into a bad regime
- the student latent blew up before the posture collapsed
- deploy-side latent clamping suppressed the blow-up and materially improved
  MuJoCo behavior

Why this matters:

- the new main bottleneck is now known precisely
- the recovery branch’s deployment weakness is not a vague “MuJoCo is bad”
  issue
- the problem is specifically latent robustness under cross-engine history
  mismatch

## Milestone 11: First Training-Side Fix For Sim2Sim Latent Brittleness

This milestone introduced the first successful model-side attempt to repair the
MuJoCo latent failure without relying on a deploy-only clamp. It remains the
active adaptive refinement baseline after later max-abs and temporal-smoothness
follow-up branches did not replace it.

Canonical frozen bounded-latent artifact:

- `rma_go2_lab/policies/adapt_v3_dyn_only_phase2_recovery_low_switch_latent_reg_final.pt`

Freeze note:

- `rma_go2_lab/policies/adapt_v3_dyn_only_phase2_recovery_low_switch_latent_reg_final.md`

Key code changes:

- `rma_go2_lab/models/adaptation/ppo_rma_v3_phase2.py`
- `rma_go2_lab/models/adaptation/adapt_v3_ppo_cfg.py`
- `rma_go2_lab/__init__.py`

Current non-frozen task id:

- `RMA-Go2-Adapt-V3-DynOnly-Phase2-Recovery-LowSwitch-LatentReg`

Checkpoint sweep provenance:

- `artifacts/evaluations/adapt_v3_recovery_low_switch_latent_reg_ckpt_sweep/`

What is different in this branch:

- student latent magnitude is now logged explicitly:
  - `student_latent_l2`
  - `student_latent_max_abs`
- a small latent-L2 regularizer is applied during training
- the run is being treated as a checkpoint-search branch rather than as a
  “final checkpoint wins automatically” branch

What is currently supported by the repo evidence:

- the branch remains locomotion-healthy through the early and mid stages
- adaptation alignment remains strong
- latent magnitude stays much closer to the MuJoCo-safe regime discovered by
  the clamp experiments

What this milestone proved:

- a small training-side latent regularizer can preserve the low-switch
  adaptation lane while keeping latent magnitude in a much tighter regime
- the best bounded-latent checkpoint can be selected from the early-mid window
  rather than assumed to be the last checkpoint
- the resulting artifact materially improves unclamped MuJoCo behavior relative
  to the first recovery artifact

## Current Canonical Role Split

The repo now has multiple canonical artifacts with different roles.

Use this split rather than pretending there is one universal winner.

### Flat/source anchor

- `rma_go2_lab/policies/flat1499.pt`

### Baseline warm-start anchor

- `rma_go2_lab/policies/blind_baseline2_warmstart_final.pt`

### Earlier switched-task adaptation anchor

- `rma_go2_lab/policies/adaptation_student_v2_final.pt`

### Current deployment-side `Adapt-V3` winner

- `rma_go2_lab/policies/adapt_v3_dyn_only_phase2_stage_a_final.pt`

### Current adaptation-recovery `Adapt-V3` anchor

- `rma_go2_lab/policies/adapt_v3_dyn_only_phase2_recovery_low_switch_final.pt`

### Current bounded-latent recovery challenger

- `rma_go2_lab/policies/adapt_v3_dyn_only_phase2_recovery_low_switch_latent_reg_final.pt`

### Current terrain-aware challenger

- `rma_go2_lab/policies/adapt_v3_terrain_lite_phase2_stage_a_final.pt`

## What The Project Has Finished

The repo can now honestly claim all of the following:

- stable locomotion prior exists
- blind rough baseline ladder exists and is frozen
- teacher lineage exists and is frozen
- earlier adaptation lineage exists and is frozen
- modern `Adapt-V3` finalists exist and are frozen
- the dyn-only Stage A winner has been audited honestly
- a recovery branch restored real online adaptation pressure in the modern
  `Adapt-V3` family
- deployment/export/Sim2Sim tooling is real enough to expose latent failure
  modes

## What The Project Has Not Finished Yet

The repo should **not** yet claim that:

- the modern recovered adaptive branch is deployment-ready
- the best adaptation-side artifact is already the best Sim2Sim artifact
- modern `Adapt-V3` online adaptation is solved all the way through MuJoCo and
  hardware
- the bounded-latent recovery story is already finished end-to-end through
  deployment-quality adaptive behavior

## Current Frontier

The current project frontier is now much narrower and more useful than before:

- preserve real online adaptation
- keep locomotion strong
- and make the blind history latent robust enough to survive cross-engine
  deployment-side history mismatch

That is the next real barrier, and the repo is now in a position to attack it
directly rather than guessing where the failure is.
