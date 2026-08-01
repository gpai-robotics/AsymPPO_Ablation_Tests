# C2 Structured Offline `resb12_weakmotor_switch` Alternate

This note freezes the final constrained residual follow-up:

- artifact family:
  `structured_z27_phase2_phi_supervised_resb12_weakmotor_switch`
- parent baseline:
  `adapt_v3_dyn_only_structured_z27_phase2_phi_supervised_v1_candidate.pt`

## Why It Was Tried

`resb12` was the first residual branch that beat `v1` on blind nominal and OOD
dynamics, but it still lagged on switch robustness.

This follow-up kept the same residual `12`-D correction shape and added only
two targeted training nudges:

- modest switch upweighting
- stronger weak-motor upweighting

## Outcome

This branch improved the residual line in the intended direction:

- switch average recovered and beat `v1`
- switch very-heavy and switch very-weak-motor cases improved relative to the
  untuned residual branch
- blind nominal also improved slightly

But it still did not replace `v1` overall because:

- OOD dynamics dropped back below `v1`
- very-weak-motor static OOD remained weaker than the canonical baseline

## Final Reading

Treat `resb12_weakmotor_switch` as:

- the best switch-leaning residual structured offline alternate
- the last targeted optimization pass before stopping structured C2 tuning
- not the canonical final structured C2 artifact
