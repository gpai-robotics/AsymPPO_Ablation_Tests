# Adapt-V3 Final Candidate Comparison

This note records the first clean head-to-head comparison between the two
finalized blind-student `Adapt-V3` candidates:

1. `adapt_v3_dyn_only_phase2_stage_a_final.pt`
2. `adapt_v3_terrain_lite_phase2_stage_a_final.pt`

It should be read together with:

- `docs/FINAL_CANDIDATE_COMPARISON_RUBRIC.md`
- `docs/DEPLOYMENT_AUDIT_ADAPT_V3.md`

## Comparison Setup

Both candidates were evaluated under the same post-freeze battery:

- blind suite
- geometry OOD
- dynamics OOD
- push OOD
- switch OOD
- gait diagnostics

Artifact roots:

- dyn-only:
  - `artifacts/evaluations/adapt_v3_dyn_only/`
  - `artifacts/ood_evaluations/adapt_v3_dyn_only/`
- terrain-lite:
  - `artifacts/evaluations/adapt_v3_terrain_lite/`
  - `artifacts/ood_evaluations/adapt_v3_terrain_lite/`

## Headline Verdict

The current winner is:

- `adapt_v3_dyn_only_phase2_stage_a_final.pt`

Reason:

- it wins more scenarios overall
- it achieves better suite-level aggregate scores in every mandatory family
- it survives more cleanly with fewer `base_height` and
  `base_orientation` failures
- it shows better forward-motion composure in the gait diagnostics

Terrain-lite remains a legitimate finalist, but not the current deployment-side
leader.

## Suite-Level Results

Mean suite scores:

- blind suite:
  - dyn-only: `8.3622`
  - terrain-lite: `7.6511`
- geometry OOD:
  - dyn-only: `9.6562`
  - terrain-lite: `9.5124`
- dynamics OOD:
  - dyn-only: `7.4559`
  - terrain-lite: `7.3041`
- push OOD:
  - dyn-only: `7.4487`
  - terrain-lite: `7.2104`
- switch OOD:
  - dyn-only: `7.4499`
  - terrain-lite: `7.3478`

Scenario win counts:

- blind:
  - dyn-only wins `7 / 9`
  - terrain-lite wins `2 / 9`
- geometry:
  - dyn-only wins `3 / 4`
  - terrain-lite wins `1 / 4`
- dynamics:
  - dyn-only wins `3 / 4`
  - terrain-lite wins `1 / 4`
- push:
  - dyn-only wins `3 / 4`
  - terrain-lite wins `1 / 4`
- switch:
  - dyn-only wins `2 / 4`
  - terrain-lite wins `2 / 4`

## What Terrain-Lite Actually Improved

Terrain-lite is not a fake branch. It delivered a real benefit:

- lower `vel_err_step_mean` in every suite family
- lower `yaw_err_step_mean` in every suite family

Examples:

- blind:
  - velocity error:
    - dyn-only: `0.1878`
    - terrain-lite: `0.1021`
  - yaw error:
    - dyn-only: `0.0806`
    - terrain-lite: `0.0599`
- geometry:
  - velocity error:
    - dyn-only: `0.1882`
    - terrain-lite: `0.0889`
- push:
  - velocity error:
    - dyn-only: `0.1979`
    - terrain-lite: `0.0962`

Terrain-lite also wins a few niche scenarios:

- blind `nominal_slope_up_l5`
- geometry `ood_stairs_up_l5`
- dynamics `ood_very_weak_motor_random_rough_l5`
- switch `ood_switch_low_friction_heavy_random_rough_l5`
- switch `ood_switch_very_weak_motor_random_rough_l5`

This supports the claim that compact terrain privilege is meaningful and not
just decorative.

## Why Dyn-Only Still Wins

Terrain-lite pays a large price in posture/composure failure modes.

Representative suite-level averages:

- blind:
  - `base_orientation_events_per_env`
    - dyn-only: `0.5295`
    - terrain-lite: `1.4931`
  - `base_height_events_per_env`
    - dyn-only: `1.1007`
    - terrain-lite: `2.1493`
- geometry:
  - `base_orientation_events_per_env`
    - dyn-only: `0.6445`
    - terrain-lite: `1.2930`
- dynamics:
  - `base_height_events_per_env`
    - dyn-only: `2.9922`
    - terrain-lite: `7.2344`
- switch:
  - `base_height_events_per_env`
    - dyn-only: `4.7969`
    - terrain-lite: `9.7930`

Timeout behavior also favors dyn-only:

- blind timeout fraction:
  - dyn-only: `0.3376`
  - terrain-lite: `0.1287`
- geometry timeout fraction:
  - dyn-only: `0.3039`
  - terrain-lite: `0.1769`
- push timeout fraction:
  - dyn-only: `0.1971`
  - terrain-lite: `0.0499`

Interpretation:

- terrain-lite often tracks better before failure
- dyn-only actually survives and composes itself better over the full rollout

## Gait / Composure Comparison

Standstill is effectively a tie.

Forward motion is not.

Dyn-only forward gait:

- `base_height_mean = 0.2978`
- `foot_slip_contact_mean = 0.1441`
- `base_orientation_events_per_env = 0.6875`
- `base_height_events_per_env = 0.0625`
- `diagonal_trot_score = 0.1119`

Terrain-lite forward gait:

- `base_height_mean = 0.2802`
- `foot_slip_contact_mean = 0.1602`
- `base_orientation_events_per_env = 1.6875`
- `base_height_events_per_env = 0.3750`
- `diagonal_trot_score = 0.0705`

Interpretation:

- neither candidate is a beautiful trot specialist
- terrain-lite is clearly more crouched and failure-prone in forward gait
- this matches the training-time observation that terrain-lite settled into a
  more conservative lower-base strategy

## Scientific Interpretation

The progression now looks like:

1. dense terrain latent:
   - too privileged
   - too hard for the student
2. terrain-lite v1:
   - student-trainable
   - real blind terrain-aware branch
   - but not yet the best deployment candidate

So terrain-lite is a success in the research sense:

- it proves compact terrain privilege can survive into a blind student

But it is not yet a success in the final-selection sense:

- it does not clearly beat the simpler dyn-only branch

## Current Decision

Current deployment-side ordering:

1. `adapt_v3_dyn_only_phase2_stage_a_final.pt`
2. `adapt_v3_terrain_lite_phase2_stage_a_final.pt`

Use this ordering for:

- Sim2Sim prioritization
- export/bundle rehearsal
- deployment-path dry runs

Keep terrain-lite as:

- the terrain-aware comparison artifact
- the base for future terrain-summary refinement

## What This Means Next

Dyn-only should move first into deployment-oriented follow-up.

Terrain-lite should not be discarded. It points to a narrower next problem:

- refine the `terrain_lite` summary so it preserves the tracking benefit
  without inducing the conservative low-base posture failure mode

That makes the next terrain-aware iteration much more tractable than the older
dense-terrain line.
