# C2 Structured Z27 Rebuild Plan

This note records the structured Candidate 2 rebuild that has now been
completed.

## Why We Are Rebuilding

The repo already established three things:

- `policy_history` contains real hidden-dynamics information
- the controller does use latent at runtime
- improving `phi -> teacher_latent` fit did not translate into better runtime behavior

That points at the **root controller / latent contract** as the most likely C2
bottleneck.

## New Hypothesis

The old free-form `32`-D latent is too unconstrained for deployable adaptation.

So the next branch rebuilds the C2 root around a **structured dyn-only latent**
with:

- `latent_dim = 27`
- one latent coordinate per privileged hidden-dynamics degree of freedom

For the current dyn-only setting, those hidden factors are:

- static friction
- dynamic friction
- base mass ratio
- joint stiffness scale (`12`)
- joint damping scale (`12`)

Total:

- `27`

## What Changes

Instead of:

```text
e_t -> mu -> z_32
history -> phi -> zhat_32
```

we now train:

```text
e_t -> mu -> z_27
history -> phi -> zhat_27
```

The actor is retrained from scratch around that smaller contract, rather than
trying to retrofit deployable adaptation onto the older `32`-D geometry.

## New Task Surfaces

Phase 1 root:

- `RMA-Go2-Adapt-V3-DynOnly-StructuredZ27-Phase1-StageA`

Phase 2 recovery:

- `RMA-Go2-Adapt-V3-DynOnly-StructuredZ27-Phase2-Recovery-LowSwitch`

## Training Order

1. Train the structured Phase 1 Stage A root.
2. Evaluate whether the new root preserves healthy locomotion.
3. Freeze the selected Phase 1 artifact.
4. Train the structured Phase 2 low-switch recovery branch.
5. Compare the resulting adaptive branch against the canonical bounded-latent
   dyn-only baseline.

## Success Criteria

This rebuild is only interesting if it improves at least one of:

- cleaner gait family than the current adaptive baseline
- better low-friction or low-friction-switch behavior
- less late-training decay during Phase 2
- stronger runtime behavior without losing real switch adaptation

If it fails those, the repo should treat the latent contract itself as largely
disconfirmed and consider a broader C2 redesign.

## Current Status

The structured Phase 1 root has now been frozen as:

- `rma_go2_lab/policies/adapt_v3_dyn_only_structured_z27_phase1_stage_a_final.pt`

Freeze note:

- [adapt_v3_dyn_only_structured_z27_phase1_stage_a_final.md](/home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/policies/adapt_v3_dyn_only_structured_z27_phase1_stage_a_final.md)

Validation summary before promotion:

- strong nominal gait and blind-suite behavior
- nominal latent ablations show partial dependence:
  - `zero` latent is survivable
  - `shuffled` latent is more harmful
- targeted OOD ablations show stronger latent value under:
  - heavy mass shift
  - weak motors
  - ultra-low friction remains weak for all modes, with `shuffled` worst

The structured offline Phase 2 pipeline has now also produced a canonical
working artifact:

- `rma_go2_lab/policies/adapt_v3_dyn_only_structured_z27_phase2_phi_supervised_v1_candidate.pt`

Operational runbook:

- [C2_STRUCTURED_OFFLINE_PIPELINE_RUNBOOK.md](/home/bhuvan/projects/rma/rma_go2_lab/docs/C2_STRUCTURED_OFFLINE_PIPELINE_RUNBOOK.md)

That line established the mission-critical result for this rebuild:

- the repo now has a working structured C2 pipeline with
  - frozen structured Phase 1 root
  - on-policy history collection
  - offline supervised `phi` training
  - runtime evaluation without the old online PPO collapse

Important caveat carried through the rebuild:

- the latent is real and OOD-useful, but the policy still retains a meaningful
  nominal fallback when latent is zeroed

## Outcome

The rebuild should now be read as:

- successful at the pipeline level
- not yet fully optimized at the branch-selection level

Later structured offline refresh rounds, bottleneck replacements, and residual
correction follow-ups explored tradeoffs, but none replaced the canonical `v1`
working artifact overall.
