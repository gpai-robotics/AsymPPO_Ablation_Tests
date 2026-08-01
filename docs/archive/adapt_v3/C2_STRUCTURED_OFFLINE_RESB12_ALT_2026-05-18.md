# C2 Structured Offline `resb12` Alternate

This note freezes the first residual-bottleneck structured C2 follow-up:

- artifact family:
  `structured_z27_phase2_phi_supervised_resb12`
- parent baseline:
  `adapt_v3_dyn_only_structured_z27_phase2_phi_supervised_v1_candidate.pt`

## Why It Was Tried

Pure bottleneck replacement (`b8`, `b12`) made the main tradeoff obvious:

- nominal and low-friction behavior could improve
- but hard adaptation cases lost too much capacity

So `resb12` changed the shape of the adaptation contract:

- keep the strong full-latent `v1` history adapter as a frozen base
- learn only a compact `12`-D residual correction

## Outcome

`resb12` was the first architecture follow-up that improved the right things in
a meaningful way:

- blind nominal average improved over `v1`
- OOD dynamics average also improved over `v1`
- gait quality stayed close to `v1`

But it did not become the final promoted artifact because:

- OOD switch average still remained below `v1`
- weak-motor cases still lagged the canonical structured baseline

## Final Reading

Treat `resb12` as:

- the best residual dynamics-leaning structured offline alternate
- a useful proof that residual correction is a better direction than pure
  bottleneck replacement
- not the canonical final structured C2 artifact
