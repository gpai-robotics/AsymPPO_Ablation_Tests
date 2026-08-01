# Adapt-V3 Dynamics-Only Phase 1 Stage A Freeze

This file freezes the canonical dynamics-only `Stage A` privileged base for
the active `Adapt-V3` reboot.

This freeze does not replace the earlier terrain-plus-dynamics `Stage A`
artifact. That older artifact remains as historical lineage. This file defines
the active canonical checkpoint for the dynamics-only line.

## Identity

- canonical name:
  - `adapt_v3_dyn_only_phase1_stage_a_final`
- checkpoint:
  - `rma_go2_lab/policies/adapt_v3_dyn_only_phase1_stage_a_final.pt`
- source run:
  - `/home/bhuvan/tools/IsaacLab/logs/rsl_rl/go2_adapt_v3_phase1_stage_a_dyn_only/2026-04-28_14-18-53`
- selected source checkpoint:
  - `model_1220.pt`
- freeze date:
  - `2026-04-28`

## Purpose

This checkpoint freezes the dynamics-only reboot of `Adapt-V3 Phase 1 Stage A`.

It exists to answer a narrower and more realistic question than the earlier
terrain-plus-dynamics line:

- can a blind history student later recover hidden dynamics factors reliably?

The privileged latent `mu(e_t)` for this line includes only:

- friction
- base mass / payload
- motor strength / joint scaling

It explicitly does **not** include terrain privilege in the active actor path.

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
- temporary blind-policy imitation scaffold during bootstrap
- within-episode adaptation switch disabled for `Stage A`
- active latent target narrowed to `dynamics_privileged` only

## Selection Rationale

`model_1220.pt` was selected because:

- it sits in a strong late-run regime after imitation had fully decayed to zero
- locomotion remained strong and stable
- debug validation confirmed the actor still depended materially on `mu(e_t)`
- latent scale looked cleaner and more controlled than the later `1520`
  candidate

## Final Behavior Summary

Late-run training snapshot near freeze:

- reward:
  - `32.88`
- episode length:
  - `882.41`
- curriculum:
  - `terrain_levels = 4.9684`
- tracking error:
  - `error_vel_xy = 0.1458`
  - `error_vel_yaw = 0.2019`
- terminations:
  - `time_out = 0.8012`
  - `base_contact = 0.0000`
  - `base_orientation = 0.0214`
  - `base_height = 0.1676`
  - `low_progress = 0.0100`

Latent-health snapshot near freeze:

- `latent_anchor = 0.0160`
- `dynamics_prediction = 0.0009`
- `latent_batch_std = 0.9132`
- imitation fully off:
  - `flat_imitation_coef = 0.0000`

## Debug Validation Summary

Canonical debug artifact:

- `artifacts/debug/adapt_v3_dyn_only_stage_a_model1220_debug.json`

What it confirmed:

- actor-path contract remained correct:
  - `inference_vs_pi_x_mu_e = 0.0`
- active actor privilege was dynamics-only:
  - `policy_obs_groups = ["policy", "dynamics_privileged"]`
  - `terrain_privileged_stats = null`
- latent remained strongly alive:
  - `extrinsics_latent std = 1.5591`
- zeroing the latent changed action materially:
  - `masked_inference_vs_pi_x_zero_latent = 0.1025`
- shuffling the latent across envs changed action too:
  - `masked_inference_vs_pi_x_shuffled_mu_e = 0.0297`

Interpretation:

- the dynamics-only reboot preserved a real load-bearing latent
- locomotion success did not come from silently bypassing `mu`
- this is a valid canonical base for the active rebooted `V3` line

## Branch Meaning

This freeze is not the final deployable `Adapt-V3` policy.

It is the frozen privileged base for the active dynamics-only reboot and should
support the next staged question:

- whether later `phi(history) -> z_hat` training becomes more stable once the
  latent target is limited to hidden dynamics rather than terrain geometry plus
  dynamics

## Freeze Statement

`adapt_v3_dyn_only_phase1_stage_a_final` is now frozen.

Do not change:

- its checkpoint identity
- its source run lineage
- its task/config association
- its role as the canonical dynamics-only `Stage A` base for the active
  `Adapt-V3` reboot

Any future stronger result should be introduced as a later continuation or an
explicitly versioned new variant, not by silently replacing this checkpoint.
