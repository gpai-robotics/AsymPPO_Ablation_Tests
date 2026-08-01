# Adapt-V3 Terrain-Lite Phase 1 Freeze

This file freezes the canonical terrain-lite privileged `Phase 1` base for the
active `Adapt-V3` line.

It is the first successful terrain-aware reboot after replacing the earlier
raw-terrain privileged target with the compact `terrain_lite_privileged`
representation.

## Identity

- canonical name:
  - `adapt_v3_terrain_lite_phase1_stage_a_final`
- checkpoint:
  - `rma_go2_lab/policies/adapt_v3_terrain_lite_phase1_stage_a_final.pt`
- source run:
  - `/home/bhuvan/tools/IsaacLab/logs/rsl_rl/go2_adapt_v3_phase1_per_episode_terrain_lite/2026-04-29_14-10-51`
- selected source checkpoint:
  - `model_1999.pt`
- freeze date:
  - `2026-04-30`

## Purpose

This checkpoint freezes the terrain-aware privileged/base policy before
history-only Phase 2 training.

Its job is to answer:

- can a compact terrain-aware privileged target be trained cleanly without
  recreating the older terrain-latent instability?

The privileged latent `mu(e_t)` for this line includes:

- `terrain_lite_privileged`
- `dynamics_privileged`

The terrain input is not the raw `187`-dim height scan. It is the compact
`13`-dim terrain-lite descriptor derived from that scan.

## Training Definition

- task:
  - `RMA-Go2-Adapt-V3-TerrainLite-Phase1-PerEpisode`
- env config:
  - `rma_go2_lab/envs/adaptation/rough_history_per_episode_terrain_lite_cfg.py`
- PPO config:
  - `rma_go2_lab/models/adaptation/adapt_v3_ppo_cfg.py`
- actor-critic:
  - `rma_go2_lab/models/adaptation/rma_v3_actor_critic.py`
- algorithm:
  - `rma_go2_lab/models/adaptation/ppo_rma_v3_phase1.py`

Key design choices:

- actor/critic trunk warm-started from:
  - `rma_go2_lab/policies/adapt_v3_dyn_only_phase1_stage_a_final.pt`
- dynamics portion of `mu` warm-started from the same dynamics-only Phase 1
  checkpoint
- terrain-lite portion of `mu` initialized fresh
- terrain summary decoder active
- temporary flat-policy imitation scaffold during bootstrap
- per-episode domain randomization
- no within-episode adaptation switch during training

Naming note:

- the frozen file keeps the historical `stage_a_final` suffix because the
  terrain-lite Phase 2 config expects this canonical filename
- the successful source run itself used the per-episode terrain-lite task alias

## Selection Rationale

`model_1999.pt` was selected because:

- the run completed the intended full `2000` iteration schedule
- terrain-aware privileged training stayed strong after imitation fully decayed
  to zero
- the terrain-lite privileged target remained stable and easy to fit
- rough-terrain locomotion stayed in the same serious-performance band as the
  successful dynamics-only line

## Final Behavior Summary

Final training snapshot at `Learning iteration 1999/2000`:

- reward:
  - `32.37`
- episode length:
  - `858.19`
- curriculum:
  - `terrain_levels = 4.6535`
- tracking error:
  - `error_vel_xy = 0.1367`
  - `error_vel_yaw = 0.2029`
- terminations:
  - `time_out = 0.8248`
  - `base_contact = 0.0000`
  - `base_orientation = 0.0107`
  - `base_height = 0.1559`
  - `low_progress = 0.0091`

Latent/privileged-health snapshot:

- `latent_anchor = 0.0009`
- `dynamics_prediction = 0.0000`
- `terrain_summary_prediction = 0.0000`
- `latent_batch_std = 0.9868`
- `latent_pairwise = 0.0000`
- imitation fully off:
  - `flat_imitation_coef = 0.0000`
  - `flat_imitation = 0.0000`

## Interpretation

This run provides strong evidence that `terrain-lite` fixed the old
terrain-side Phase 1 failure mode.

What it means:

- compact terrain privilege can be included without destabilizing the
  privileged/base policy
- the terrain-lite target is far more student-compatible than the earlier
  raw-terrain latent line
- the project now has a real terrain-aware Phase 1 base worth carrying into
  history-only Phase 2

What it does not prove yet:

- that `phi(history)` can recover the terrain-lite latent cleanly in Phase 2
- that the terrain-aware student will beat the dyn-only student in final eval

## Branch Meaning

This checkpoint is the canonical terrain-aware privileged/base artifact for the
active `terrain-lite` branch.

It is suitable as:

- the Phase 1 reference for terrain-lite Phase 2
- the teacher/base side of the terrain-aware blind-student comparison
- the canonical answer to whether terrain-lite privileged training itself works

## Freeze Statement

`adapt_v3_terrain_lite_phase1_stage_a_final` is now frozen.

Do not silently replace this checkpoint. Any later stronger terrain-lite
privileged/base result should be added as an explicitly versioned continuation
or documented successor.
