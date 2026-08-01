# C2 Structured Offline V3 Low-Friction Alternate Result

## Identity

- branch:
  `structured_z27_phase2_phi_supervised_v3_lowfric_balanced`
- date:
  `2026-05-18`
- parent:
  `adapt_v3_dyn_only_structured_z27_phase2_phi_supervised_v1_candidate.pt`

## Why This Branch Existed

This was a targeted weighted offline follow-up intended to improve the main
structured `v1` weakness:

- low-friction robustness

The trainer was modified to upweight:

- low-friction samples
- switched samples
- very-heavy samples
- weak-motor samples

## Outcome

This branch also did not replace `v1`.

What improved:

- best gait quality of the three structured offline variants
- OOD dynamics average:
  `10.3283 -> 10.3340`
- ultra-low-friction OOD dynamics case improved slightly
- very-heavy OOD dynamics case improved strongly

What regressed:

- blind nominal average:
  `12.5054 -> 12.4338`
- nominal low-friction case did not improve
- OOD switch average fell below `v1`
- weak-motor dynamics case regressed further

## Verdict

Useful heavy-dynamics / gait-leaning alternate, but not the canonical
structured C2 winner.

Best reading:

- evidence that weighted offline training can move the tradeoff surface
- not a clean global improvement over `v1`
