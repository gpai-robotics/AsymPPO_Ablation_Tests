# Adapt-V3 Phase 1 Stage A Freeze

This file freezes the canonical `Stage A` privileged base for the `Adapt-V3`
branch.

`Stage A` is now frozen. Do not retune or overwrite it in place. Any future
changes should create a new explicitly versioned variant or continue into
`Stage B` / Phase 2 from this frozen base.

## Identity

- canonical name:
  - `adapt_v3_phase1_stage_a_final`
- checkpoint:
  - `rma_go2_lab/policies/adapt_v3_phase1_stage_a_final.pt`
- source run:
  - `/home/bhuvan/tools/IsaacLab/logs/rsl_rl/go2_adapt_v3_phase1_stage_a/2026-04-27_14-27-05`
- selected source checkpoint:
  - `model_1500.pt`
- freeze date:
  - `2026-04-27`

## Purpose

`Stage A` is the locomotion-bootstrap phase of true `Adapt-V3`.

It represents:

- privileged Phase 1 training with `mu(e_t) -> z_t`
- actor-side latent use that remained real under debug checks
- stationary randomized episodes without within-episode switches
- critic-only warm-start from `B2`
- temporary blind-policy imitation scaffold during early training
- the first successful privileged base in the repo that is both:
  - behaviorally strong
  - architecturally faithful to the intended `mu / pi / phi` direction

It is used to answer:

- whether the repo can train a real load-bearing `mu + pi` privileged base
- whether locomotion can be recovered without restoring the old actor-side
  latent bypass
- which frozen base should seed later `Stage B` and Phase 2 work

## Training Definition

- task:
  - `RMA-Go2-Adapt-V3-Phase1-StageA`
- env config:
  - `rma_go2_lab/envs/adaptation/rough_history_stage_a_cfg.py`
- PPO config:
  - `rma_go2_lab/models/adaptation/adapt_v3_ppo_cfg.py`
- actor-critic:
  - `rma_go2_lab/models/adaptation/rma_v3_actor_critic.py`
- algorithm:
  - `rma_go2_lab/models/adaptation/ppo_rma_v3_phase1.py`

Key retained design choices:

- actor fresh
- extrinsics encoder fresh
- adaptation module fresh
- critic-only warm-start from:
  - `rma_go2_lab/policies/blind_baseline2_warmstart_final.pt`
- temporary blind-policy imitation scaffold used only during early bootstrap
- within-episode adaptation switch disabled for `Stage A`

## Selection Rationale

`model_1500.pt` was selected because:

- it sits in the strong late-run regime after imitation had already turned off
- locomotion remained strong and stable
- terrain curriculum remained high rather than collapsing
- debug still showed the actor using `mu(e_t)` meaningfully
- the later checkpoint family was at least as strong as the earlier `1188`
  candidate while providing cleaner evidence that the scaffold had fully faded

## Final Behavior Summary

Late-run training snapshot near freeze:

- reward:
  - `32.87`
- episode length:
  - `874.13`
- curriculum:
  - `terrain_levels = 4.9657`
- tracking error:
  - `error_vel_xy = 0.1427`
  - `error_vel_yaw = 0.1957`
- terminations:
  - `time_out = 0.8275`
  - `base_contact = 0.0000`
  - `base_orientation = 0.0109`
  - `base_height = 0.1569`
  - `low_progress = 0.0055`

Latent-health snapshot near freeze:

- `latent_anchor = 0.0085`
- `latent_batch_std = 0.9786`
- `latent_pairwise = 0.0012`
- imitation fully off:
  - `flat_imitation_coef = 0.0000`

## Debug Validation Summary

Canonical debug artifact:

- `artifacts/debug/adapt_v3_stage_a_model1500_debug.json`

What it confirmed:

- actor-path contract remained correct:
  - `inference_vs_pi_x_mu_e = 0.0`
- latent remained strongly alive:
  - `extrinsics_latent std = 1.3845` pre-switch stress
- zeroing the latent changed action materially:
  - `masked_inference_vs_pi_x_zero_latent = 0.2418`
- shuffling the latent across envs changed action too:
  - `masked_inference_vs_pi_x_shuffled_mu_e = 0.0248`
- history latent remained distinct from privileged latent, as expected in
  Phase 1 / Stage A

Interpretation:

- locomotion did not recover by silently recreating the old latent bypass
- `Stage A` is a real privileged base rather than only a numerically healthy
  PPO run

## Branch Meaning

This freeze is not the final deployable `Adapt-V3` policy.

It is the frozen privileged base that should support:

- later `Stage B` continuation if needed
- Phase 2 adaptation training:
  - `phi(history) -> z_hat`
  - `pi(obs, z_hat) -> action`

So `Stage A` should be read as:

- a foundational `V3` milestone
- not the final adaptive student

## Freeze Statement

`adapt_v3_phase1_stage_a_final` is now frozen.

Do not change:

- its checkpoint identity
- its source run lineage
- its task/config association
- its role as the canonical locomotion-bootstrap base for `Adapt-V3`

Any future stronger `V3` result must be introduced as:

- `Stage B`
- Phase 2
- or an explicitly new `V3` variant

not by silently replacing this frozen base.
