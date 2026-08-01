# Two Final Candidates Roadmap

This file is now a compact map, not a second full narrative for each
candidate.

Use the candidate cards as the single source of truth:

- Candidate 1:
  [C1_STAGEA_MODEL400_DEPLOY_CARD.md](/home/bhuvan/projects/rma/rma_go2_lab/docs/C1_STAGEA_MODEL400_DEPLOY_CARD.md)
- Candidate 2:
  [C2_ADAPT_V3_BASELINE_CANDIDATE_CARD.md](/home/bhuvan/projects/rma/rma_go2_lab/docs/C2_ADAPT_V3_BASELINE_CANDIDATE_CARD.md)

## Repo Split

The repo intentionally keeps two final candidate lines:

1. Candidate 1
   - blind-history deployable policy
   - robustness-first deployment line
2. Candidate 2
   - explicit RMA-style adaptive policy
   - `mu / pi / phi` deployment line

## Current Canonical Artifacts

Candidate 1:

- exported bundle:
  `rma_go2_lab/policies/exported/c1_ethlike_v3_model_400_candidate/`

Candidate 2:

- active baseline artifact:
  `rma_go2_lab/policies/adapt_v3_dyn_only_phase2_recovery_low_switch_latent_reg_final.pt`

## Reading Rule

If the question is about a candidate itself:

- read its candidate card first

If the question is about branch history, implementation detail, or evaluation
procedure:

- then follow the linked supporting docs from that card

## Supporting Docs

Candidate 1 support:

- [CANDIDATE1_BLIND_REACTIVE_PLAN.md](/home/bhuvan/projects/rma/rma_go2_lab/docs/CANDIDATE1_BLIND_REACTIVE_PLAN.md)

Candidate 2 support:

- [ADAPT_V3_ACTIVE_ROADMAP.md](/home/bhuvan/projects/rma/rma_go2_lab/docs/ADAPT_V3_ACTIVE_ROADMAP.md)
- [ADAPT_V3_EXECUTION_SPEC.md](/home/bhuvan/projects/rma/rma_go2_lab/docs/ADAPT_V3_EXECUTION_SPEC.md)
- [ADAPTIVE_POLICY_EVAL_PROTOCOL.md](/home/bhuvan/projects/rma/rma_go2_lab/docs/ADAPTIVE_POLICY_EVAL_PROTOCOL.md)

## Policy Rule

We no longer want multiple docs independently re-defining the same candidate.

The intended structure is:

- one canonical card per candidate
- support docs for branch history and procedures
- archive notes for failed or retired variants
