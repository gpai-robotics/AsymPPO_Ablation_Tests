# C2 Structured Offline V2 Alternate Result

## Identity

- branch:
  `structured_z27_phase2_phi_supervised_v2`
- date:
  `2026-05-18`
- parent:
  `adapt_v3_dyn_only_structured_z27_phase2_phi_supervised_v1_candidate.pt`

## Why This Branch Existed

This was the first on-policy refresh round on top of the working structured
offline `v1` pipeline.

Goal:

- improve low-friction behavior without losing the healthy structured offline
  runtime story

## Outcome

This branch did not replace `v1`.

What improved:

- nominal blind average:
  `12.5054 -> 12.6224`
- nominal low-friction case:
  `7.6510 -> 7.9025`
- gait remained healthy

What regressed:

- OOD dynamics average:
  `10.3283 -> 10.1257`
- OOD switch average:
  `9.1835 -> 9.0629`
- weak-motor dynamics case regressed materially

## Verdict

Useful alternate result, but not the canonical structured C2 winner.

Best reading:

- nominal low-friction leaning alternate
- evidence that the offline pipeline is tunable
- not a full replacement for `v1`
