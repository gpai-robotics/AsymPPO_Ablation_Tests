# V1/V2 Closeout Checklist

This is the final-stage operational checklist for the remaining adaptation
branches:

- `Adapt-V1`
- `Adapt-V2`

The goal is to make freeze, evaluation, comparison, and artifact handling
mechanical rather than improvised.

## Scope

Use this checklist only after:

- the run is near completion or clearly plateaued
- no structural bug is suspected
- the branch has already passed startup/smoke validation

This checklist is for:

- final checkpoint selection
- canonical evaluation
- clip generation
- policy archival
- synthesis updates

It is not for:

- mid-training debugging
- reward redesign
- architecture changes

## Candidate Identity

Before freezing anything, write down the exact identity of the run:

- task id
- runner config
- experiment name
- log directory
- checkpoint path
- freeze date

Target identities:

- `V1`
  - task: `RMA-Go2-Adaptation-Student-Rough-History-V1`
  - config: `rma_go2_lab.models.adaptation.adapt_v1_ppo_cfg:Go2AdaptationStudentV1PPORunnerCfg`
- `V2`
  - task: `RMA-Go2-Adaptation-Student-Rough-History-V2`
  - config: `rma_go2_lab.models.adaptation.adapt_v2_ppo_cfg:Go2AdaptationStudentV2PPORunnerCfg`

## Freeze Gate

Freeze only if the run looks like a serious candidate.

Minimum gate:

- reward has stabilized into the mature-policy regime
- episode length is consistently high
- `time_out` is substantial
- `adaptation_switch_reached_frac` is substantial
- no obvious late-run collapse
- no NaNs or exploding losses

Adaptation-specific gate:

- `latent_regression` is nonzero and behaving stably
- `latent_active_frac` is meaningfully above zero
- `latent_cosine` is finite and not collapsing

For `V2`, add one more gate:

- modular path has already been smoke-tested
- no suspicion that `phi(history) -> z_hat` and `pi(x_t, z_hat)` drifted into a broken implementation

## Checkpoint Selection

Default freeze target:

- final checkpoint from the selected run

If the last checkpoint is visibly worse than an earlier late checkpoint:

- freeze the best late checkpoint
- document why it was chosen over the final one

Do not silently choose a non-final checkpoint without documenting it.

## Policy Archive Naming

When the candidate earns a canonical slot, copy it into
`rma_go2_lab/policies/` using stable names:

- `adaptation_student_v1_final.pt`
- `adaptation_student_v2_final.pt`

Companion notes:

- `adaptation_student_v1_final.md`
- `adaptation_student_v2_final.md`

Each note should include:

- source run path
- source checkpoint path
- task/config identity
- freeze date
- short training summary
- canonical eval artifact paths

## Canonical Evaluation Rule

`V1` and `V2` must get the same eval coverage as the other frozen candidates.

No candidate should be missing a canonical section that another frozen
candidate has.

Required sections:

- `gait`
- `blind_suite`
- `ood_geometry`
- `ood_dynamics`
- `ood_push`
- `ood_switch`

Use the same evaluator versions and the same post-fix semantics already
established in:

- `docs/EVALUATION_METHODS.md`

## Eval Execution

Preferred path:

- extend `scripts/eval/run_frozen_eval_matrix.py` once `V1` and/or `V2` are ready
- run them through the same matrix process as the other frozen candidates

Do not use ad hoc partial evals as canonical freeze evidence.

If a quick sanity eval is done before the full matrix:

- treat it as provisional
- do not cite it as final comparison evidence

## Eval Completion Gate

The freeze is not complete until:

- every required section finished successfully
- canonical JSON outputs exist
- canonical CSV outputs exist where expected
- no section is missing for the candidate

If matrix validation fails:

- fix the missing section
- rerun that section
- do not proceed to policy archival yet

## Clip Set

Each frozen adaptation candidate should eventually receive the same clip
matrix.

Minimum recommended clip set:

- nominal forward
- ultra-low friction
- very weak motor
- switched very weak motor

Output layout:

- `artifacts/evaluations/clips/adaptation_student_v1/`
- `artifacts/evaluations/clips/adaptation_student_v2/`

Each clip directory should contain:

- one `.mp4`
- one `metadata.json`

Do not record extra hero clips for one candidate while leaving another frozen
candidate without the matching core clip set.

## Comparison Set

Once frozen, compare against the current adaptation ladder:

- `studentNA`
- `studentAdapt-V0`
- `studentAdapt-V1`
- `studentAdapt-V2` if frozen

Privileged reference:

- `V3`

Use two comparison views:

- research comparison
- deployment comparison

Research comparison asks:

- which policy is strongest under the canonical eval suites?

Deployment comparison asks:

- which policy gives the best balance of performance, simplicity, and expected hardware safety?

These two winners may differ.

## Synthesis Update Tasks

After a candidate is frozen, update:

- `docs/ADAPTATION_PHASE_SYNTHESIS.md`
- `docs/ADAPTATION_IMPLEMENTATION_V0.md`
- `docs/PROJECT_GUIDE.md`
- `artifacts/evaluations/README.md`

Add:

- final training snapshot
- canonical checkpoint path
- canonical eval artifact paths
- short honest interpretation

## Claim Discipline

When writing the freeze conclusion, keep the claim scope honest.

Safe claim types:

- `V1` improved the explicit latent path relative to `V0`
- `V2` is the first modular RMA-like implementation in the repo
- the candidate is stronger than `NA` on the canonical suite

Avoid claiming without evidence:

- sim-to-real success
- final hardware superiority
- body-stability resolution
- exact reproduction of original RMA unless the architecture and evidence truly support it

## Final Sign-Off

A branch is ready to be called "closed out" only when all of the following are true:

- canonical checkpoint selected
- policy copied into `rma_go2_lab/policies/`
- companion freeze note written
- full canonical eval matrix completed
- required clip set planned or recorded
- synthesis docs updated
- final comparison position understood

If one of those is missing, the branch is still "in progress", even if training has ended.
