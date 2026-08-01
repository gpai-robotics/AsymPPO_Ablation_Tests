# C2 Structured Offline Final Baseline Card

This is the compact operational card for the current structured Candidate 2
baseline.

Use this file when the question is:

- what is the active structured C2 artifact right now
- how do we validate it reproducibly
- what export/bundle path belongs to it

## Active Baseline

- checkpoint:
  `rma_go2_lab/policies/adapt_v3_dyn_only_structured_z27_phase2_phi_supervised_v1_candidate.pt`
- task:
  `RMA-Go2-Adapt-V3-DynOnly-StructuredZ27-Phase2-Recovery-LowSwitch`
- policy kind:
  `blind_adaptive_student`
- repo role:
  canonical working structured offline C2 baseline

## Why This Is The Baseline

This artifact is the first structured Candidate 2 line that:

- kept the structured Phase 1 root intact
- trained `phi(history)` offline instead of continuing online PPO drift
- preserved healthy locomotion through Phase 2
- produced a stable end-to-end collect/train/eval pipeline

Later refreshes and bottleneck/residual branches were informative, but none
replaced this checkpoint overall.

## Validation Surface

The baseline hardening path validates five things:

1. bundle manifest packaging
2. deployable export generation
3. structural bundle validation
4. canonical Isaac-side gait and blind nominal gate
5. canonical Isaac-side OOD dynamics and OOD switch gates

## One-Command Validation

Generate the validation script:

```bash
python /home/bhuvan/projects/rma/rma_go2_lab/scripts/adaptation/prepare_structured_phase2_final_baseline_validation.py
```

That writes:

- `artifacts/pipeline_runs/structured_z27_phase2_phi_supervised_v1_final_baseline_validation.sh`

Then run:

```bash
bash /home/bhuvan/projects/rma/rma_go2_lab/artifacts/pipeline_runs/structured_z27_phase2_phi_supervised_v1_final_baseline_validation.sh
```

## Expected Outputs

Bundle:

- `rma_go2_lab/policies/exported/adapt_v3_dyn_only_structured_z27_phase2_phi_supervised_v1_candidate/`

Validation eval outputs:

- `artifacts/final_baseline_validation/structured_z27_phase2_phi_supervised_v1/evaluations/`
- `artifacts/final_baseline_validation/structured_z27_phase2_phi_supervised_v1/ood_evaluations/`

## Reading Order

Read these in order:

1. [C2_ADAPT_V3_BASELINE_CANDIDATE_CARD.md](/home/bhuvan/projects/rma/rma_go2_lab/docs/C2_ADAPT_V3_BASELINE_CANDIDATE_CARD.md)
2. [C2_STRUCTURED_OFFLINE_PIPELINE_RUNBOOK.md](/home/bhuvan/projects/rma/rma_go2_lab/docs/C2_STRUCTURED_OFFLINE_PIPELINE_RUNBOOK.md)
3. this file
4. [C1_C2_TRANSFER_EXECUTION_PLAN.md](/home/bhuvan/projects/rma/rma_go2_lab/docs/C1_C2_TRANSFER_EXECUTION_PLAN.md)

## What This Does Not Mean

This file does not claim that Candidate 2 is the final research endpoint for
all adaptive work.

It does mean:

- the mission-critical structured offline pipeline exists
- this is the frozen artifact we should treat as the current operational C2
  baseline
- future C2 optimization should beat this checkpoint explicitly, not bypass it
