# Exploration Ledger

This file is the lightweight running ledger for branch exploration.

Its purpose is simple:

- keep the repo from turning into an ocean of disconnected experiments
- record what was tried, why, and what happened
- make later branch decisions traceable

Use this file as the first stop when the question is:

- have we already tried this?
- which branches are still active?
- which ones were negative results?
- what is the current bridge versus the intended final target?

## Status Labels

- `active`
  currently relevant working line
- `active reference`
  comparison anchor that remains intentionally in scope
- `active baseline`
  currently leading baseline for a candidate line
- `active deployment winner reference`
  strongest practical artifact used as a deployment-side anchor
- `bridge`
  useful intermediate candidate, not necessarily the intended final form
- `frozen`
  accepted canonical artifact
- `implemented skeleton`
  codepath exists and compiles, but has not yet cleared training smoke/eval
- `rejected`
  evaluated and not promoted
- `planned`
  explicitly intended next branch, not yet implemented

## Current Ledger

| Date | Line | Status | Identity | Why it exists | Outcome / meaning |
| --- | --- | --- | --- | --- | --- |
| 2026-05-18 | C2 | frozen | `adapt_v3_dyn_only_structured_z27_phase2_phi_supervised_v1_candidate.pt` | First structured Phase 2 candidate trained with on-policy collection plus offline `phi` supervision from the frozen structured root | Canonical working structured C2 pipeline artifact; avoids the old online PPO collapse and remains the best overall structured offline candidate after `v2` and `v3` follow-up refreshes |
| 2026-05-18 | C2 | rejected | `structured_z27_phase2_phi_supervised_v2` | First on-policy refresh on top of the structured offline `v1` pipeline | Improved nominal low-friction behavior, but regressed OOD dynamics/switch robustness and did not replace `v1` |
| 2026-05-18 | C2 | rejected | `structured_z27_phase2_phi_supervised_v3_lowfric_balanced` | Weighted offline follow-up emphasizing low-friction and switch-heavy samples | Improved gait and some OOD dynamics cases, but still did not beat `v1` overall and remained weaker on several switch / weak-motor cases |
| 2026-05-18 | C2 | rejected | `structured_z27_phase2_phi_supervised_b8` | First compact bottleneck replacement test using an `8`-D deployable latent | Too compressed for this setup; nominal low-friction improved a bit, but gait and hard adaptive cases degraded too much |
| 2026-05-18 | C2 | rejected | `structured_z27_phase2_phi_supervised_b12` | Less aggressive bottleneck replacement using a `12`-D deployable latent | Better than `b8` and strongest nominal low-friction bottleneck branch, but still weaker than `v1` on harder dynamics and switch cases |
| 2026-05-18 | C2 | rejected | `structured_z27_phase2_phi_supervised_resb12` | First residual bottleneck branch keeping the frozen `v1` history adapter and learning only a compact `12`-D correction | Best residual dynamics-leaning alternate; improved blind nominal and OOD dynamics over `v1`, but still lagged on switch robustness and weak-motor edge cases |
| 2026-05-18 | C2 | rejected | `structured_z27_phase2_phi_supervised_resb12_weakmotor_switch` | Final targeted residual follow-up with switch and weak-motor weighting | Best switch-leaning residual alternate; recovered switch average above `v1`, but gave back too much static OOD dynamics capacity to replace the canonical baseline |
| 2026-05-15 | C2 | frozen | `adapt_v3_dyn_only_structured_z27_phase1_stage_a_final.pt` | First structured C2 root rebuild that replaces the old free-form 32-D dyn-only latent with a 27-D dynamics-shaped contract | Strong locomotion plus a genuinely alive structured privileged contract; nominal ablations show fallback behavior, but OOD ablations confirm real latent value; frozen as the new root for structured Phase 2 recovery |
| 2026-05-12 | C1 | frozen | `c1_ethlike_v3_model_400_candidate` | First StageA blind-history Candidate 1 branch trained from `Teacher V4 model_300` and validated end-to-end through export, Isaac deploy rehearsal, and MuJoCo OOD | Current canonical Candidate 1 deployment artifact; history is load-bearing on Isaac switch/push, main remaining weakness is lateral push under continuous corridor geometry |
| 2026-05-07 | C1 | bridge | `adapt_v3_terrain_lite_phase2_stage_a_final.pt` | Closest current blind terrain+dynamics privileged student matching the original C1 definition | Real blind student, but visually posture-fragile and not yet final |
| 2026-05-07 | C1 | active reference | `blind_baseline2_warmstart_final.pt` | Clean blind baseline and deployment-simplicity anchor | Best clean blind baseline reference |
| 2026-05-07 | C1 | active reference | `blind_baseline3_warmstart_imitation_final.pt` | Robustness-leaning blind baseline anchor | Useful comparison for toughness / ETH-style robustness flavor |
| 2026-05-08 | C1 | frozen | `c1_ethlike_v1_model_700_candidate` | First successful robustness-first blind history finalist for the no-adaptation line | Selected from the `400 / 700 / 1200` sweep; won blind + OOD evaluation; canonical MuJoCo limit rerun completed with hidden-env coverage and confirmed lateral-push as the main weakness |
| 2026-05-07 | C2 | active baseline | `adapt_v3_dyn_only_phase2_recovery_low_switch_latent_reg_final.pt` | Best current adaptive MuJoCo challenger | Remains the active adaptive baseline |
| 2026-05-07 | C2 | active deployment winner reference | `adapt_v3_dyn_only_phase2_stage_a_final.pt` | Best practical deployment-side artifact | Still stronger practically than current adaptive branches |
| 2026-05-07 | supporting infrastructure | planned | ETH `terrain-generator` reference | Candidate terrain-generation upgrade path for later adaptation and OOD terrain work | Not needed for current C1 pace, but worth revisiting for future C2 terrain diversity and structured OOD evaluation |

## Current Interpretation

### Candidate 1

Current truth:

- the strict historical bridge artifact is `terrain-lite`
- the earlier frozen C1 finalist is `c1_ethlike_v1_model_700_candidate`
- the current canonical C1 deployment artifact is `c1_ethlike_v3_model_400_candidate`

That means Candidate 1 should be understood as:

- bridge artifact historically
- earlier V1 finalist as an important baseline
- StageA `model_400` as the current deployment-facing blind-history winner

### Candidate 2

Current truth:

- bounded-latent recovery remains the active adaptive baseline
- structured offline Phase 2 now has a canonical working pipeline artifact
- later scalar penalty refinements and residual follow-ups did not replace it
- archived restart attempts did not replace it

That means Candidate 2 should be understood as:

- active adaptive baseline now
- structured offline `v1` pipeline now works end-to-end
- `v1` remains the promoted structured C2 artifact after bottleneck and residual follow-ups

## How To Append

When a meaningful branch is opened, frozen, or rejected, append one row with:

- date
- line (`C1`, `C2`, or supporting infrastructure)
- status
- identity
- why it exists
- outcome / meaning

Keep entries short.
Detailed analysis should live in the dedicated branch notes.

## Related Docs

- `docs/TWO_FINAL_CANDIDATES_ROADMAP.md`
- `docs/CANDIDATE1_BLIND_REACTIVE_PLAN.md`
- `docs/ADAPT_V3_ACTIVE_ROADMAP.md`
- `docs/ADAPTIVE_SIM2SIM_REFINEMENT_PLAN.md`
