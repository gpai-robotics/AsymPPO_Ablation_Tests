# Public Repo Blueprint

This note defines the recommended public-facing side repo that should be built
from this private root codebase.

The goal is not to mirror the full research history.

The goal is to publish:

- the working modules
- the selected finalists
- the clean evaluation path
- and the core project idea

without carrying over the branch archaeology, dead ends, or internal clutter
that only matter inside the private lab repo.

## Core decision

Do **not** replace this repo with the public repo.

Instead:

- keep this repo as the private master lab repo
- build a separate public distilled repo from the successful paths

Recommended role split:

- private repo:
  - full research history
  - all failed branches
  - all audits
  - all candidate comparisons
  - messy truth

- public repo:
  - clean story
  - minimal code surface
  - selected frozen artifacts
  - reproducible working path

## Public repo narrative

The public repo should answer one clear question:

- how do we train a deployable quadruped locomotion policy that works from
  blind deployable observations, while preserving a simple and understandable
  training/evaluation stack?

That means the public repo should be:

- direct
- reproducible
- easy to explain
- selective

It should not try to tell the whole internal C1/C2 research history.

## Main character

The public repo should be **C1-first**.

Why:

- C1 is the cleaner deployment-facing story
- C1 has the simpler runtime contract
- C1 has stronger source/export/runtime validation
- C1 is easier to understand without losing the point of the project

That means:

- the main published policy should be the current C1 blind-history finalist
- C2 should not be the headline path

## Optional secondary character

C2 can appear in the public repo only as a **small research extension**.

If included at all, it should be framed as:

- an explicit adaptive latent branch
- dynamics-only teacher-derived
- a research extension beyond the main blind-history path

It should **not** dominate the public repo unless its adaptation story becomes
much cleaner than it is now.

## Recommended public repo scope

### Include

- one main locomotion line
  - current recommendation:
    - `C1`

- one teacher path
  - only the teacher actually needed to explain the successful student
  - likely the current `Teacher V4 model_300` lineage in simplified form

- one clean deploy/export path

- one clean evaluation stack
  - nominal eval
  - OOD / disturbance eval
  - export parity check

- one architecture explainer

- one results summary

### Exclude

- failed branches
- near-duplicate alternates
- retired recovery attempts
- internal archive notes
- experimental rescue branches
- branch-specific special-case tooling that only exists to debug dead ends

## Recommended public repo structure

```text
public_repo/
  README.md
  LICENSE
  pyproject.toml
  requirements/
  configs/
    teacher/
    student/
    eval/
  models/
    teacher/
    blind_history/
    optional_adaptation/
  envs/
    rough/
    deployment_eval/
  scripts/
    train_teacher.py
    train_c1.py
    export_policy.py
    eval_isaac.py
    eval_deploy.py
    compare_parity.py
  checkpoints/
    README.md
  docs/
    architecture.md
    training.md
    evaluation.md
    results.md
    limitations.md
```

## Recommended first release contents

### Required first release

- teacher of record:
  - simplified documented `Teacher V4` path

- student of record:
  - C1 blind-history finalist path

- deploy/runtime path:
  - export script
  - parity check
  - nominal runtime eval

- eval story:
  - at least one disturbance / OOD battery
  - clear explanation of what the student sees at deployment

### Optional first release

- C2 appendix or extension folder
- one carefully selected adaptation note

If this makes the repo harder to understand, omit it from `v1`.

## Recommended public positioning

The public repo should say something close to:

- this project studies deployable quadruped locomotion from blind observations
- the primary published result is a history-conditioned blind controller
- privileged teacher signals are used during training only
- explicit online adaptation remains an active research extension, not the main
  published claim

That keeps the story honest and strong.

## Suggested internal mapping from this repo

### Public mainline likely comes from

- [C1_STAGEA_MODEL400_DEPLOY_CARD.md](/home/bhuvan/projects/rma/rma_go2_lab/docs/C1_STAGEA_MODEL400_DEPLOY_CARD.md)
- current C1 exported artifact lineage
- current C1 teacher lineage

### Public comparison context may cite

- [C1_C2_STATUS_COMPARISON.md](/home/bhuvan/projects/rma/rma_go2_lab/docs/C1_C2_STATUS_COMPARISON.md)

### Public architecture explainer can be distilled from

- [NN_ARCHITECTURE_REPORT.md](/home/bhuvan/projects/rma/rma_go2_lab/docs/NN_ARCHITECTURE_REPORT.md)

### Public evaluation explainer can be distilled from

- [ADAPTIVE_POLICY_EVAL_PROTOCOL.md](/home/bhuvan/projects/rma/rma_go2_lab/docs/ADAPTIVE_POLICY_EVAL_PROTOCOL.md)

## What should stay private

These should stay in the root/private repo:

- archive notes
- failed candidate cards
- dead branch runbooks
- experimental repair attempts
- internal branch ranking debates
- partial or superseded checkpoints

Reason:

- they are valuable as research truth
- but they weaken clarity in a public repo

## Public repo success criteria

The public repo is good if a new reader can understand:

1. what the robot sees at deployment
2. what the teacher sees during training
3. what model is actually deployed
4. how to train the main successful line
5. how to evaluate it without hidden branch knowledge

If the public repo requires the reader to understand the entire C1/C2 branch
history, it is too complicated.

## Phased extraction plan

### Phase 1: design freeze

- freeze the public narrative
- decide that C1 is the mainline
- decide whether C2 appears at all in `v1`

### Phase 2: code selection

- list the exact modules needed for:
  - teacher
  - C1 student
  - export
  - eval
- reject everything else from the first public cut

### Phase 3: config cleanup

- rename configs to public-friendly names
- remove branch-era naming noise where possible
- keep one canonical training path

### Phase 4: doc rewrite

- write fresh docs for public readers
- do not copy internal candidate cards directly as the main docs

### Phase 5: artifact and parity pass

- verify the selected public checkpoints/export path still work cleanly
- generate one canonical parity report

## Recommendation on C2 in the public repo

Current recommendation:

- do **not** make C2 central in the first public release

If included, keep it to:

- one short section
- one optional module family
- one honest limitation note

Reason:

- C2 is still the right private research line
- but its story is currently more complex than its practical deployment
  advantage

## Immediate next private-repo step

Before copying code into a new repo, produce one extraction memo that lists:

- exact files to copy
- exact files to rewrite
- exact files to omit

That should be the next concrete planning artifact after this blueprint.
