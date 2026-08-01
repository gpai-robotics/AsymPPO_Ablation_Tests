# Adapt-V3 Dynamics-Only Structured Z27 Phase 1 Stage A Freeze

This file freezes the canonical structured dyn-only `Phase 1 Stage A` root for
the current Candidate 2 rebuild.

## Identity

- canonical name:
  - `adapt_v3_dyn_only_structured_z27_phase1_stage_a_final`
- checkpoint:
  - `rma_go2_lab/policies/adapt_v3_dyn_only_structured_z27_phase1_stage_a_final.pt`
- source run:
  - `/home/bhuvan/tools/IsaacLab/logs/rsl_rl/go2_adapt_v3_phase1_stage_a_dyn_only_structured_z27/2026-05-15_15-56-32`
- selected source checkpoint:
  - `model_1999.pt`
- freeze date:
  - `2026-05-15`

## Purpose

This checkpoint freezes the first successful structured C2 root rebuild.

It exists to replace the older free-form dyn-only `32`-D latent contract with a
smaller, dynamics-shaped `27`-D contract:

- static friction
- dynamic friction
- base mass ratio
- joint stiffness scale (`12`)
- joint damping scale (`12`)

Total:

- `27`

## Training Definition

- task:
  - `RMA-Go2-Adapt-V3-DynOnly-StructuredZ27-Phase1-StageA`
- env config:
  - `rma_go2_lab/envs/adaptation/rough_history_stage_a_cfg.py`
- PPO config:
  - `rma_go2_lab/models/adaptation/adapt_v3_ppo_cfg.py`
- actor-critic:
  - `rma_go2_lab/models/adaptation/rma_v3_actor_critic.py`
- algorithm:
  - `rma_go2_lab/models/adaptation/ppo_rma_v3_phase1.py`

Key design choices:

- warm-started actor / critic trunk from:
  - `rma_go2_lab/policies/blind_baseline2_warmstart_final.pt`
- structured `mu` as direct `27 -> 27`
- structured dynamics decoder as direct `27 -> 27`
- identity initialization for the structured privileged path
- temporary blind-policy imitation scaffold during bootstrap
- within-episode adaptation switch disabled for `Stage A`

## Final Behavior Summary

Late-run snapshot near freeze:

- reward:
  - `40.40`
- episode length:
  - `993.12`
- curriculum:
  - `terrain_levels = 5.9116`
- tracking error:
  - `error_vel_xy = 0.1367`
  - `error_vel_yaw = 0.1664`
- terminations:
  - `time_out = 0.9689`
  - `base_contact = 0.0000`
  - `base_orientation = 0.0007`
  - `base_height = 0.0208`
  - `low_progress = 0.0105`

Structured latent-health snapshot near freeze:

- `latent_anchor = 0.0000`
- `dynamics_prediction = 0.0000`
- `latent_batch_std = 0.9995`
- `latent_variation_bonus = 0.0000`
- `latent_pairwise = 0.0000`
- imitation fully off:
  - `flat_imitation_coef = 0.0000`

## Meaning

This is the first C2 Phase 1 root in the current cycle that clearly achieved
both:

- strong locomotion
- and a genuinely alive, load-bearing structured privileged contract

That makes it the correct root to freeze before attempting the corresponding
structured `Phase 2` recovery line.

## Phase 1 Validation Summary

Nominal evaluation:

- gait remains healthy on `random_rough l5 forward`
  - `diagonal_trot_score = 0.5960`
  - `foot_slip_contact_mean = 0.0601`
  - `base_height_mean = 0.3843`
  - `timeout_fraction_of_terminals = 1.0`
- blind-suite average:
  - `12.5966`

Latent ablations:

- nominal `zero` latent is slightly worse than `normal`, but still survivable
- nominal `shuffled` latent degrades gait and blind-suite performance more
  clearly than `zero`
- this means the policy has a strong safe fallback when latent is absent, but
  is harmed by inconsistent latent corrections

Targeted OOD ablations:

- heavy mass shift:
  - `normal` best
  - `zero` slightly worse
  - `shuffled` clearly worse
- weak motors:
  - `normal` clearly best
  - both `zero` and `shuffled` degrade performance
- ultra-low friction:
  - all modes are weak
  - `shuffled` is the least stable of the three

Interpretation:

- the structured latent is not uniformly indispensable in nominal conditions
- but it becomes more clearly load-bearing in harder OOD dynamics regimes
- that is sufficient to promote this Phase 1 artifact into structured
  `Phase 2`

## Freeze Statement

`adapt_v3_dyn_only_structured_z27_phase1_stage_a_final` is now frozen.

Do not change:

- its checkpoint identity
- its source run lineage
- its task/config association
- its role as the structured dyn-only `Phase 1 Stage A` root for the active C2
  rebuild

Any future stronger result should be introduced as a later continuation or an
explicitly versioned successor, not by silently replacing this freeze.
