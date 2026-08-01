# C1 Omni Deployment Checklist

This is the practical rollout checklist for the frozen deployable blind rough
omni student:

- policy:
  - `rma_go2_lab/policies/c1_blind_rough_omni_usable_v1_final.pt`
- bundle:
  - `rma_go2_lab/policies/exported/c1_blind_rough_omni_usable_v1_final/`

The goal is not to invent more training work.
The goal is to move this baseline safely through deployment rehearsal while
keeping the first real sim2real gap explicit.

## What Is Already Proven

- export parity is clean
- deployable observation contract is clean:
  - `policy`
  - `policy_history`
- Isaac deployment rehearsal is stable in nominal settings
- fixed-command deploy rehearsal is stable
- nominal flat-surface MuJoCo Sim2Sim runtime is stable enough to trust as a
  deployment-contract sanity check
- hidden deploy-time probes now work for:
  - friction
  - mass
  - motor

## Current Sim2Real Readout

Trusted evidence split:

- Isaac is the primary controlled robustness surface
- MuJoCo is currently a flat-surface command/joint/behavior sanity surface
- MuJoCo corridor scenarios are exploratory and should not be treated as the
  final robustness judge yet

Current gap ranking:

1. low friction
2. added mass
3. weaker motors

Interpretation:

- low friction is the main deployment risk
- mass shift is tolerated reasonably well in the tested range
- moderate motor weakening is tolerated reasonably well in the tested range

## Immediate Rollout Order

1. keep this policy as the deployment baseline
2. start real rehearsal only on traction-friendly surfaces
3. avoid slippery surfaces during first hardware-facing trials
4. use low-friction behavior as the first failure watchpoint
5. only propose new improvement modules after confirming the same weakness in
   hardware or higher-fidelity deployment rehearsal

## Hardware-Rehearsal Priorities

- start on high-traction flat ground
- prefer conservative command magnitudes first
- verify clean standstill, forward, lateral, and gentle yaw before stress cases
- only then move to rougher geometry
- delay intentionally slippery surfaces until nominal behavior is trusted

## What Counts As A Successful Baseline Rollout

- no immediate deployment-surface mismatch
- no export/runtime parity surprises
- stable nominal omnidirectional command tracking
- sensible joint behavior and posture under flat-surface MuJoCo replay
- predictable rather than mysterious failure under low traction

## What Would Justify The Next Improvement Module

Future improvement work should be justified by a concrete statement like:

- this baseline is deployment-credible, but low-friction degradation is the
  dominant uncovered weakness

Not by:

- generalized desire for more complexity
- new architecture ideas without a measured baseline failure

## Canonical Next Step

The next step after this checklist is:

- deployment rehearsal first
- rerun and visually inspect the trusted flat-surface MuJoCo sanity check
- low-friction failure confirmation second
- targeted robustness module design only after that
