# C2 Terrain-Lite DAgger Failed Branch

Date:

- 2026-05-15

Branch:

- `RMA-Go2-Adapt-V3-TerrainLite-Phase2-Recovery-LowSwitch-DaggerPhi`

Purpose:

- reopen the terrain-aware C2 path directly
- keep the more conservative DAgger-style Phase 2 recipe learned from the
  dyn-only branch
- test whether richer privileged extrinsics were the missing ingredient

Outcome:

- real terrain-lite + dynamics adaptation was active
- the branch still reproduced the old terrain-lite crouch / posture-fragility
  tendency
- locomotion quality degraded while the latent side kept improving
- it did not replace the bounded-latent dyn-only baseline

Meaning:

- the terrain-lite issue is not solved by Phase 2 recipe changes alone
- future terrain-aware C2 work should be treated as a deeper restart from a
  stronger root, not a continuation of the historical terrain-lite line
