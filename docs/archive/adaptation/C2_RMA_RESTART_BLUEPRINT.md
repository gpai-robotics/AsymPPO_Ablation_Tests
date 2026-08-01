# C2 RMA Restart Blueprint

This note records how we would restart Candidate 2 if we were beginning again
with everything we now know.

Use this file when the question is:

- if we started a new branch, what should we keep
- what should we avoid repeating from this repo
- what is the clean intended architecture from day one
- what are the non-negotiable gates before promotion

This repo should now be treated as:

- the original experimentation-first branch
- the place where the adaptation space was explored and de-risked
- the place that taught us what to avoid

The next branch should not repeat that exploratory shape blindly.

It should start from this blueprint instead.

## Status Assumption

What is already proven in this repo:

- blind rough locomotion is real
- privileged Phase 1 teacher/root training is real
- explicit `mu / phi / pi` adaptation contracts are real
- online PPO Phase 2 is the wrong default for this line
- offline supervised `phi` training is the correct Phase 2 default
- the structured offline dyn-only `v1` artifact is the first working C2
  pipeline baseline

What is not worth rediscovering:

- whether the pipeline should be online PPO in Phase 2
- whether unconstrained branch churn will magically produce a clean winner
- whether the runtime student can rely on privilege at deployment

## The Restart Goal

The restart branch should aim for:

- a clean RMA-style adaptive line from day one
- a deployable runtime contract that is fixed early
- a teacher/root that is explicitly trained to make the deployable latent
  matter
- a compact but not overly lossy deployable latent
- offline Phase 2 adaptation training only
- early export/runtime validation rather than late hardening

## Core Restart Principles

### 1. Design around the final student contract first

Do not start by asking:

- what privileged signals can we stuff into a big latent

Start by asking:

- what deployable latent should the student predict at runtime
- what observable history will be available to it
- what actor contract will consume that latent

The final runtime contract should be fixed before the first serious Phase 1
training run.

### 2. Separate teacher factors from deployable factors

Do not assume:

- the teacher’s richest hidden-factor vector should be the student target

Instead:

- let the teacher/root consume richer privileged information if needed
- learn a smaller action-relevant extrinsics bottleneck for deployment
- train the student only against that deployable bottleneck

This is the biggest structural lesson from the current repo.

### 3. Keep Phase 2 strictly offline

The restart branch should treat this as a rule, not an experiment:

- freeze `pi`
- freeze `mu`
- collect on-policy rollouts
- train only `phi(history) -> z_hat`

No online PPO actor updates in Phase 2 by default.

### 4. Stage the ambition

Do not reopen full terrain+dynamics adaptation immediately.

The staged restart order should be:

1. dynamics-only adaptive line
2. make it exportable and stable
3. only then reopen terrain-aware adaptation if needed

This keeps the first restarted line narrow enough to finish cleanly.

## Recommended Architecture

The intended clean contract is:

```text
teacher/root training:
privileged_env_factors e_t
  -> mu
  -> z_t

deployable runtime:
observable history h_t
  -> phi
  -> z_hat_t

current deployable observation x_t + z_hat_t
  -> pi
  -> action
```

Recommended policy story:

- `pi` should be trained from the beginning as a policy that genuinely depends
  on `z`
- `z` should be compact enough to be inferable
- `z` should not simply be a raw hidden-factor dump

## Restart Scope Recommendation

### First restart branch

The first clean restart branch should be:

- `dyn-only`
- rough-terrain trained
- explicit deployable student history
- compact deployable latent
- offline Phase 2 only

### What to defer

Defer these until the first restart branch is clean:

- terrain-aware latent reopening
- residual/bottleneck correction variants
- switch-heavy training contracts
- factor-specific weighting sweeps
- MuJoCo-specific latent polishing

Those are second-order improvements, not restart foundations.

## Latent Design Recommendation

The restart branch should not begin from the full structured `z27` as the
deployable target.

It also should not begin from an aggressively tiny latent like `8` without the
teacher/root having been designed for that contract.

Recommended approach:

- keep a rich privileged factor view internally
- compress it to a moderate deployable extrinsics latent during Phase 1
- likely start in the `8–16` range
- pick the final size by early runtime gates, not by abstract neatness

Important:

- the compact latent should be a Phase 1 design choice
- not a Phase 2 after-the-fact compression experiment

## Training Plan

### Phase 1: privileged root

Train a privileged root that:

- consumes the chosen privileged factors
- emits a deployable extrinsics latent through `mu`
- trains `pi(obs, z)` under the final actor contract

Success requirement:

- the actor must actually depend on `z`
- not merely tolerate it

### Phase 2: offline student

Freeze the Phase 1 root and run:

1. on-policy rollout collection with imperfect `phi`
2. supervised `phi(history) -> z_teacher`
3. canonical runtime evaluation

Do not treat Phase 2 as a PPO continuation.

### Switch training

Switch difficulty should be staged:

1. static hidden mismatch first
2. low-probability within-episode switches second
3. harsher switch regimes only after the static path is stable

This avoids asking the first serious branch to solve the hardest version of the
problem immediately.

## Evaluation Contract

The restart branch should define promotion gates before training begins.

Required gates:

1. gait health
2. blind nominal suite
3. OOD dynamics suite
4. OOD switch suite
5. export/load/runtime validation

No branch should be promoted on:

- loss alone
- latent cosine alone
- one cherry-picked scenario

## Export And Deployment Discipline

The restart branch should treat deployment hardening as part of the core path.

That means:

- bundle manifest path exists early
- export path exists early
- source/export/runtime contract is validated early
- deployment-side validation is not postponed until after branch selection

This repo learned that late hardening creates ambiguity about what the “real”
artifact actually is.

## What To Avoid Repeating

Do not repeat these patterns in the new branch:

- online PPO Phase 2 as the default adaptation recipe
- late discovery of the deployable latent contract
- treating the teacher hidden-factor vector itself as the student target by
  default
- reopening terrain+dynamics too early
- unconstrained branch-family proliferation
- using training loss improvement as a proxy for runtime usefulness

## Branch Discipline

The restart branch should keep a much narrower branch ladder:

1. one canonical Phase 1 root
2. one canonical offline Phase 2 baseline
3. at most one compact-latent follow-up if needed
4. at most one targeted refinement if clearly justified

If a follow-up does not beat the baseline on the defined gates, archive it and
move on.

## How To Read The Current Repo

This repo should be read as:

- the source of the blind locomotion ladder
- the source of the teacher/adaptation evolution
- the place where the online PPO failure mode was discovered
- the place where the structured offline dyn-only `v1` baseline was finally
  proven

It should not be copied mechanically as the shape of the next branch.

The next branch should copy:

- the proven contracts
- the working offline Phase 2 pattern
- the evaluation discipline

and should avoid copying:

- the full exploratory chronology
- the dead-end branch families
- the accidental architecture drift

## Practical Restart Recommendation

If we actually restarted now, the first branch should be:

- dynamics-only
- compact deployable extrinsics latent designed from Phase 1
- rough-terrain trained
- offline Phase 2 only
- early export/runtime validation included

That is the highest-confidence path to a cleaner “true RMA-style” branch than
the original exploratory line in this repo.

## Companion Reading

Read these around this blueprint:

1. [C2_ADAPT_V3_BASELINE_CANDIDATE_CARD.md](/home/bhuvan/projects/rma/rma_go2_lab/docs/C2_ADAPT_V3_BASELINE_CANDIDATE_CARD.md)
2. [C2_STRUCTURED_OFFLINE_PIPELINE_RUNBOOK.md](/home/bhuvan/projects/rma/rma_go2_lab/docs/C2_STRUCTURED_OFFLINE_PIPELINE_RUNBOOK.md)
3. [OG_RMA_VS_REPO_DIVERGENCE.md](/home/bhuvan/projects/rma/rma_go2_lab/docs/OG_RMA_VS_REPO_DIVERGENCE.md)
4. [ARCHITECTURE_FLOW_FROM_FLAT_TO_ADAPTV3.md](/home/bhuvan/projects/rma/rma_go2_lab/docs/ARCHITECTURE_FLOW_FROM_FLAT_TO_ADAPTV3.md)
