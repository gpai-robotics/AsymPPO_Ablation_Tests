# Repo Mental Model

This document is the fast orientation layer for engineers who are new to this
repo.

Read this before diving into branch-specific plans or historical notes.

## One-Sentence Summary

This repo studies how to get from a simple flat locomotion prior to a robust
rough-terrain quadruped controller through three major lanes:

- flat priors
- blind rough controllers
- adaptive / privileged branches

## What This Repo Is Really Organizing

Most of the clutter comes from one repeated pattern:

1. define an environment contract
2. define a runner / training recipe
3. train a checkpoint
4. evaluate it
5. either promote it, freeze it as an ablation, or archive it

So the repo is not just “lots of models.”
It is a layered experiment system with a few important artifact families.

## The Main Lanes

### 1. Flat Prior Lane

Purpose:

- learn stable locomotion on flat terrain
- provide warm-start checkpoints for later rough-terrain work

Active files:

- [flat_forward_prior_cfg.py](/home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/envs/priors/flat_forward_prior_cfg.py)
- [flat_omni_prior_cfg.py](/home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/envs/priors/flat_omni_prior_cfg.py)
- [flat_prior_runner_cfg.py](/home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/models/priors/flat_prior_runner_cfg.py)

Mental model:

- `flat_forward_prior` is the old forward-only root
- `flat_omni_prior` is the current preferred omni flat warm start

### 2. Blind Rough Lane

Purpose:

- learn deployable rough-terrain policies that do not need privileged inputs at
  inference

Active files:

- [blind_rough_forward_cfg.py](/home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/envs/blind/blind_rough_forward_cfg.py)
- [blind_rough_forward_history_cfg.py](/home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/envs/blind/blind_rough_forward_history_cfg.py)
- [c1_blind_rough_teacher_history_cfg.py](/home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/envs/blind/c1_blind_rough_teacher_history_cfg.py)
- [c1_blind_rough_omni_usable_cfg.py](/home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/envs/blind/c1_blind_rough_omni_usable_cfg.py)
- [blind_rough_runner_cfg.py](/home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/models/blind/blind_rough_runner_cfg.py)

Mental model:

- `blind_rough_forward` is the clean rough baseline
- `blind_rough_forward_history` adds deployable proprioceptive history
- `c1_blind_rough_teacher_history` is the C1 blind-history student with
  teacher-only privileged groups
- `c1_blind_rough_omni_usable` is the active deployable rough omni target

### 3. Teacher Lane

Purpose:

- train privileged policies that are not deployment targets themselves, but
  supervise or benchmark the blind lanes

Key area:

- `rma_go2_lab/envs/teacher/`
- `rma_go2_lab/models/teacher/`

Mental model:

- teacher work provides stronger learning signals and comparison anchors
- most engineers should treat teacher artifacts as upstream support, not the
  main deployment artifact

### 4. Adaptation Lane

Purpose:

- study history-based or latent-based adaptation under hidden dynamics and
  terrain mismatch

Key area:

- `rma_go2_lab/envs/adaptation/`
- `rma_go2_lab/models/adaptation/`

Mental model:

- this is the most research-heavy lane
- it has more branch history and more architecture churn
- new engineers should not start here unless they already understand the flat
  and blind rough lanes

## How Artifacts Flow

The typical dependency flow is:

1. flat prior
2. blind rough warm-start baseline
3. teacher-trained rough student branches
4. adaptation branches

For the current omni work, the relevant flow is:

1. `flat_omni_prior`
2. `privileged rough omni teacher`
3. `c1_blind_rough_omni_usable`

That is why the flat omni prior matters so much: it shapes the starting point
for the rough omni policy.

## Active Vs Frozen Vs Archived

This repo uses three practical states:

### Active

Meaning:

- part of the current working surface
- should stay easy to discover
- should be named clearly

Typical locations:

- `rma_go2_lab/envs/priors/`
- `rma_go2_lab/envs/blind/`
- `rma_go2_lab/models/priors/`
- `rma_go2_lab/models/blind/`

### Frozen

Meaning:

- result is accepted as canonical or as an important comparison artifact
- should be documented
- should not keep spawning new sibling files in the active dirs unless there is
  a new hypothesis

Typical indicators:

- final policy files in `rma_go2_lab/policies/`
- artifact cards in `docs/`

### Archived

Meaning:

- useful for lineage
- not part of the normal active navigation surface

Typical locations:

- `docs/archive/`
- `rma_go2_lab/archive/`

If an engineer is asking “why does this old file exist?”, the answer is usually:

- it was once a real branch
- it lost or was superseded
- we kept it for traceability, but removed it from the active lane

## What Is Canonical Right Now

For orientation, the important current truths are:

### Flat omni prior

- preferred active flat omni prior:
  - `rma_go2_lab/policies/flat_omni_v1.pt`

### Omni rough teacher

- preferred active privileged omni teacher:
  - `rma_go2_lab/policies/rough_omni_teacher_v1.pt`

### Deployable rough omni student

- preferred active deployable rough omni student:
  - `rma_go2_lab/policies/c1_blind_rough_omni_usable_v1_final.pt`
- practical deployment truth:
  - export and Isaac deploy rehearsal are validated
  - nominal flat-surface MuJoCo Sim2Sim replay is validated as a behavior sanity check
  - low friction is the first clear deployment-facing failure mode
- framework direction:
  - this repo owns the deployment bundle and contract
  - `unitree_rl_lab` deploy is the long-term runtime architecture reference
  - `sim2real_unitree_sdk2py` is the safety shell reference
  - `mujoco_playground` is a replay/runtime utility reference, not the
    hardware framework

### Observation ablations

- contact-only and contact+phase flat omni branches were both explored
- both are frozen as valid ablations
- neither replaced `flat_omni_v1`

### Rough omni

- the old omni student branches should be treated as provisional probes
- the active student target is now the usable blind omni branch trained
  against the frozen omni teacher

That is an important nuance for new engineers:

- flat omni is relatively settled
- rough omni teacher is now settled enough to be canonical
- deployable rough omni student is now frozen as the active deployment
  candidate

## Where New Engineers Should Start

If someone has one hour:

1. read [PROJECT_GUIDE.md](/home/bhuvan/projects/rma/rma_go2_lab/docs/PROJECT_GUIDE.md)
2. read this file
3. open [__init__.py](/home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/__init__.py)
4. open:
   - [flat_omni_prior_cfg.py](/home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/envs/priors/flat_omni_prior_cfg.py)
   - [rough_omni_v1_cfg.py](/home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/envs/teacher/rough_omni_v1_cfg.py)
   - [c1_blind_rough_omni_usable_cfg.py](/home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/envs/blind/c1_blind_rough_omni_usable_cfg.py)
   - [flat_prior_runner_cfg.py](/home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/models/priors/flat_prior_runner_cfg.py)
   - [blind_rough_runner_cfg.py](/home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/models/blind/blind_rough_runner_cfg.py)

If someone has one day:

1. read the canonical artifact cards
2. read [C1_OMNIDIRECTIONAL_COMMAND_PLAN.md](/home/bhuvan/projects/rma/rma_go2_lab/docs/C1_OMNIDIRECTIONAL_COMMAND_PLAN.md)
3. inspect the latest active evaluation artifacts
4. only then branch into adaptation docs if needed

## How To Not Get Lost

When touching this repo, always ask:

1. which lane am I in?
2. is this active, frozen, or archive?
3. is this a new hypothesis or just another naming variant of an old one?
4. does this belong in the active surface, or should it live in archive?

That habit is what keeps the repo understandable.

## Related Docs

- [PROJECT_GUIDE.md](/home/bhuvan/projects/rma/rma_go2_lab/docs/PROJECT_GUIDE.md)
- [EXPLORATION_LEDGER.md](/home/bhuvan/projects/rma/rma_go2_lab/docs/EXPLORATION_LEDGER.md)
- [C1_OMNIDIRECTIONAL_COMMAND_PLAN.md](/home/bhuvan/projects/rma/rma_go2_lab/docs/C1_OMNIDIRECTIONAL_COMMAND_PLAN.md)
- [C1_OMNI_DEPLOYMENT_CHECKLIST.md](/home/bhuvan/projects/rma/rma_go2_lab/docs/C1_OMNI_DEPLOYMENT_CHECKLIST.md)
- [rough_omni_teacher_v1.md](/home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/policies/rough_omni_teacher_v1.md)
