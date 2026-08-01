# Adapt-V3 Poisoning Audit

This note traces the current `Adapt-V3` dyn-only adaptation-claim issue and
records which parts of the repo are affected, which parts remain valid, and
what should be phrased more carefully going forward.

## Why This Note Exists

A deployment-side probe showed that the frozen dyn-only `Adapt-V3` winner does
not currently demonstrate online-changing latent behavior under the tested
hidden-dynamics switch.

That finding forced a backtrack:

- was the deployment probe wrong?
- was `phi(history)` broken?
- or was the final training contract itself not actually requiring online
  adaptation?

The answer is now clear enough to document.

## Root Cause

The canonical dyn-only final student:

- [adapt_v3_dyn_only_phase2_stage_a_final.pt](/home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/policies/adapt_v3_dyn_only_phase2_stage_a_final.pt)

was trained under the stationary Stage A regime:

- [rough_history_stage_a_cfg.py](/home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/envs/adaptation/rough_history_stage_a_cfg.py)

and that env explicitly sets:

- `adaptation_switch_episode_prob = 0.0`

So the winning dyn-only artifact did **not** train under active mid-episode
hidden-dynamics switching.

This means the final training contract did not strongly require within-episode
latent motion from `phi(history)`.

## What Is Still True

These statements remain valid:

- the dyn-only final student is strong
- the dyn-only final student is deployable through `policy + policy_history`
- the actor uses a latent channel
- the export path is real
- the Sim2Sim bridge is real
- the eval comparison between dyn-only and terrain-lite is still valid
- the earlier adaptation-phase results (`V0`, `V1`, `V2`) remain real positive
  adaptation results under their switched task lineage
- a later low-switch recovery branch can restore real adaptation pressure again;
  see:
  - `rma_go2_lab/policies/adapt_v3_dyn_only_phase2_recovery_low_switch_final.pt`

## What Was Overread

These stronger readings are not justified for the dyn-only final Stage A
artifact:

- that it is a proven online-adaptive final deployment policy
- that `phi(history)` visibly changes in response to hidden-dynamics switches
- that the dyn-only head-to-head win over terrain-lite reflects stronger online
  adaptation behavior

## Scope Of The Problem

This issue is **not** a repo-wide collapse of the scientific story.

It is a narrower mismatch between:

- the training contract of the final dyn-only `Adapt-V3` winner
- and the stronger adaptation claim someone might casually infer from the
  architecture name alone

In other words:

- architecture name:
  `phi(history) -> z_hat -> pi`
- actual winning training regime:
  stationary Stage A bootstrap with no active within-episode switch

## High-Risk Files

These are the places where careless readers could most easily overread the
claim if we do not phrase things carefully:

- [adapt_v3_dyn_only_phase2_stage_a_final.md](/home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/policies/adapt_v3_dyn_only_phase2_stage_a_final.md)
- [README.md](/home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/policies/exported/adapt_v3_dyn_only_phase2_stage_a_final/README.md)
- [PROJECT_PUBLIC_SUMMARY.md](/home/bhuvan/projects/rma/rma_go2_lab/public/PROJECT_PUBLIC_SUMMARY.md)
- [DEPLOYMENT_AUDIT_ADAPT_V3.md](/home/bhuvan/projects/rma/rma_go2_lab/docs/DEPLOYMENT_AUDIT_ADAPT_V3.md)
- [RESULTS_SHOWCASE_PLAN.md](/home/bhuvan/projects/rma/rma_go2_lab/public/RESULTS_SHOWCASE_PLAN.md)

## Low-Risk / Still Valid Areas

These areas remain fundamentally sound:

- baseline ladder documentation
- switched-task adaptation results for earlier branches
- teacher lineage
- eval infrastructure
- export / bundle / Sim2Sim bridge work

## Terrain-Lite Check

The terrain-lite final student does **not** resolve this specific discrepancy.

Its Phase 2 Stage A env also sets:

- `adaptation_switch_episode_prob = 0.0`

So the dyn-only vs terrain-lite comparison was not biased by one finalist
having active switch training and the other not. On this axis, both finalists
were matched.

## Operational Rule Going Forward

For the dyn-only final Stage A artifact, say:

- strong blind deployable student
- latent-conditioned actor
- stationary Stage A bootstrap winner

Do **not** casually say:

- online-adaptive deployed student

unless a future continuation actually demonstrates that with a training
contract and probe result that support the claim.

Current recovery update:

- the stationary Stage A caveat remains true
- but the repo now also contains a distinct frozen recovery artifact that was
  trained under active low-probability switch pressure and selected by a
  checkpoint sweep:
  - `rma_go2_lab/policies/adapt_v3_dyn_only_phase2_recovery_low_switch_final.pt`
- this does not retroactively repair the old Stage A claim
- it does establish a new canonical branch anchor for future stronger
  adaptation claims
