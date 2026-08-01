# Adapt-V3 Dyn-Only TCN Branch Archive

This note archives the first unanchored temporal-convolutional Candidate 2
branch after it was retired from the active repo path.

## Identity

- task:
  - `RMA-Go2-Adapt-V3-DynOnly-Phase2-Recovery-LowSwitch-LatentReg-TCN`
- experiment name:
  - `go2_adapt_v3_phase2_recovery_low_switch_dyn_only_latent_reg_tcn`
- archived log root:
  - `artifacts/evaluations/archive/go2_adapt_v3_phase2_recovery_low_switch_dyn_only_latent_reg_tcn/2026-05-13_16-34-12`

## Why It Existed

This branch was the first serious stronger-history-encoder follow-up after the
bounded-latent MLP recovery baseline.

It kept the explicit `mu / pi / phi` adaptive contract, but replaced the
flattened-history MLP preprocessing in `phi(history)` with a small TCN-style
temporal encoder.

## Outcome

The branch was informative but non-winning.

What improved:

- latent alignment became very strong by late training
- latent regression also improved substantially

What failed:

- locomotion quality collapsed late in training
- episode length fell sharply
- base-height terminations dominated the final regime
- switch exposure became effectively irrelevant because episodes became too short

The final behavior conclusion was:

- the branch learned to match the privileged latent well
- but did not preserve a healthy adaptive locomotion solution

## Why It Was Retired

The active repo path should point only at branches we still want people to run.

This branch is retired in favor of:

- `RMA-Go2-Adapt-V3-DynOnly-Phase2-Recovery-LowSwitch-LatentReg-TCN-Anchored`

That follow-up keeps the TCN idea, but changes the optimization context:

- warm-start from the stronger recovery trunk
- lower pure latent-matching pressure
- keep teacher-action anchoring alive much longer

## Archive Meaning

This branch should be remembered as:

- a useful architectural probe
- evidence that better `phi(history)` alignment alone is not enough
- the direct reason the anchored TCN continuation exists
