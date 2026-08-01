# Combined AsymPPO Staged Baseline

This document is the short operating reference for the combined AsymPPO branch.
For the full pipeline record, use:

```text
docs/COMBINED_ASYMPPO_END_TO_END_PIPELINE.md
```

## Rule For This Branch

When describing a trained policy, use the saved run YAML as truth:

```text
logs/rsl_rl/<experiment>/<run>/params/env.yaml
logs/rsl_rl/<experiment>/<run>/params/agent.yaml
```

Do not infer checkpoint training parameters from the current Python cfg files.
The cfg files can change after a run; the YAML next to the checkpoint records
what actually trained that checkpoint.

## Current Candidate

The current combined stairs candidate under discussion is:

```text
logs/rsl_rl/go2_blind_rough_combined_asymppo_steps_v1/2026-07-16_10-14-29/model_5099.pt
```

YAML lineage:

| Stage | Run |
| --- | --- |
| Stage 1 flat prior | `go2_combined_flat_mjlab_prior_v1/2026-07-02_10-59-02` |
| Stage 2 rough warm-start | `go2_blind_rough_combined_asymppo_rough_v1/2026-07-07_17-11-12` |
| Stage 3 stairs candidate | `go2_blind_rough_combined_asymppo_steps_v1/2026-07-16_10-14-29` |

Status:

| Item | Status |
| --- | --- |
| Flat prior training | Done |
| Rough/slopes warm-start training | Done |
| Stairs-only training | Done for `model_5099.pt` |
| Isaac Sim normal-realistic playback | Passed visual smoke test on 2026-07-20 |
| Exported deployment bundle | Done for `go2_blind_rough_combined_asymppo_steps_v1_candidate` |
| Deployment gate | Passed on 2026-07-20 |
| MuJoCo/FSM sim2sim | Runtime staged and FSM audit passed |
| Hardware deployment | Pending for combined candidate |

The validated Go2 hardware baseline remains separate:

```text
Go2-Blind-Rough-MJLAB-AsymPPO-V1
```

Do not treat the combined candidate as a replacement for the validated hardware
baseline until it also passes controlled hardware bring-up.

## Training Structure

```text
Stage 1: combined flat prior
Stage 2: rough/slopes AsymPPO warm-started from Stage 1
Stage 3: stairs/inverted-stairs AsymPPO warm-started from Stage 2
```

## Stage Summary From Saved YAML

| Parameter | Stage 1 Flat | Stage 2 Rough | Stage 3 Stairs |
| --- | --- | --- | --- |
| Task | `Go2-Combined-Flat-MJLAB-Prior-V1` | `Go2-Blind-Rough-MJLAB-Combined-AsymPPO-Rough-V1` | `Go2-Blind-Rough-MJLAB-Combined-AsymPPO-Steps-V1` |
| Experiment | `go2_combined_flat_mjlab_prior_v1` | `go2_blind_rough_combined_asymppo_rough_v1` | `go2_blind_rough_combined_asymppo_steps_v1` |
| Run used here | `2026-07-02_10-59-02` | `2026-07-07_17-11-12` | `2026-07-16_10-14-29` |
| Warm start | none | Stage 1 `model_1499.pt` | Stage 2 `model_1999.pt` |
| Policy class | `ActorCritic` | `TemporalBlindActorCritic` | `TemporalBlindActorCritic` |
| Actor history | none | `100` steps | `100` steps |
| Env count | `4096` | `4096` | `4096` |
| Physics dt | `0.005s` | `0.005s` | `0.005s` |
| Control dt | `0.020s` | `0.020s` | `0.020s` |
| Episode length | `20s` | `20s` | `20s` |
| Terrain | plane | rough + slopes | stairs + inverted stairs |
| Terrain curriculum | none | enabled | enabled |
| Max iterations | `1500` | `2000` | `3000` |
| Steps per env | `24` | `32` | `32` |
| Learning rate | `1e-3` | `1e-4` | `1e-4` |
| Entropy coef | `0.01` | `0.002` | `0.002` |

## Terrain Distribution From Saved YAML

| Terrain | Stage 1 Flat | Stage 2 Rough | Stage 3 Stairs |
| --- | --- | --- | --- |
| Plane | active | disabled | disabled |
| `random_rough` | n/a | `0.20` | `0.0` |
| `hf_pyramid_slope` | n/a | `0.10` | `0.0` |
| `hf_pyramid_slope_inv` | n/a | `0.10` | `0.0` |
| `pyramid_stairs` | n/a | `0.0` | `0.50` |
| `pyramid_stairs_inv` | n/a | `0.0` | `0.50` |
| `boxes` | n/a | `0.0` | `0.0` |
| Stair height | n/a | disabled | `(0.03, 0.12)` |
| Stair width | n/a | disabled | `0.30m` |
| Platform width | n/a | disabled | `3.0m` |

## Randomization From Saved YAML

| Randomization | Stage 1 Flat | Stage 2 Rough | Stage 3 Stairs |
| --- | --- | --- | --- |
| Static friction | `(0.5, 1.1)` | `(0.1, 2.0)` | `(0.1, 2.0)` |
| Dynamic friction | `(0.4, 1.0)` | `(0.1, 2.0)` | `(0.1, 2.0)` |
| Base mass add | disabled | `(-2.0, 4.0) kg` | `(-2.0, 4.0) kg` |
| Base COM x/y | disabled | `(-0.03, 0.03)m` | `(-0.03, 0.03)m` |
| Base COM z | disabled | `(-0.01, 0.01)m` | `(-0.01, 0.01)m` |
| Pushes | disabled | interval `6-10s` | interval `6-10s` |
| Push x/y velocity | disabled | `(-0.35, 0.35)m/s` | `(-0.35, 0.35)m/s` |
| Push yaw velocity | disabled | `(-0.4, 0.4)rad/s` | `(-0.4, 0.4)rad/s` |
| Motor stiffness scale | disabled | `(0.6, 1.4)` | `(0.6, 1.4)` |
| Motor damping scale | disabled | `(0.6, 1.4)` | `(0.6, 1.4)` |

## Reward Differences From Saved YAML

| Reward | Stage 1 Flat | Stage 2 Rough | Stage 3 Stairs |
| --- | --- | --- | --- |
| `track_lin_vel_xy_exp` | `1.5` | `1.5` | `1.5` |
| `track_ang_vel_z_exp` | `0.5` | `0.75` | `0.75` |
| `lin_vel_z_l2` | `-1.0` | `-0.1` | `-0.5` |
| `ang_vel_xy_l2` | `-0.05` | `-0.075` | `-0.075` |
| `flat_orientation_l2` | `-2.5` | `-1.0` | `-1.0` |
| `feet_air_time` | `0.3` | `0.5` | `0.5` |
| `feet_slide` | `-0.1` | `-0.05` | `-0.05` |
| `stand_still_joint_deviation` | `-0.35` | `-0.2` | `-0.2` |
| `stand_still_foot_motion` | `-0.1` | `-0.05` | `-0.05` |
| `hip_joint_deviation` | `-0.08` | `-0.1` | `-0.1` |
| `joint_deviation` | `-0.02` | not used | not used |
| `air_time_variance` | not used | `-0.05` | `-0.05` |
| `stable_progress` | not used | not used in this YAML run | `0.5` |
| `adaptive_swing_recovery` | not used | not used | `0.25` |
| `feet_height_body` | not used | not used in this YAML run | not used in this YAML run |

## Current Interpretation

The important finding is narrower than the current Python cfg might suggest:

```text
flat prior -> validated-style rough history policy -> stairs-only recovery fine-tune
```

For `model_5099.pt`, the stair improvement came from adding a focused
stairs/inverted-stairs stage with `stable_progress` and
`adaptive_swing_recovery`, not from adding `feet_height_body` to the final saved
run.

Older experimental variants with stronger fixed foot-height shaping produced a
galloping gait and hind-leg-stuck failure mode. Keep those variants as negative
evidence, not as the current baseline.

## Isaac Sim Smoke Test Result

Operator-reported visual smoke tests passed on 2026-07-20 for:

| Test | Result | Notes |
| --- | --- | --- |
| Flat plane, nominal dynamics | Passed visually | 1200-step rollout completed |
| Stairs, nominal dynamics | Passed visually | 1200-step rollout completed |
| Inverted stairs, nominal dynamics | Passed visually | 1200-step rollout completed |

The attached playback logs confirmed the intended nominal evaluation envelope:

| Parameter | Confirmed value |
| --- | --- |
| Checkpoint | `model_5099.pt` |
| Task | `Go2-Blind-Rough-MJLAB-Combined-AsymPPO-Steps-V1` |
| Number of envs | `16` |
| Static friction | `1.0` |
| Dynamic friction | `1.0` |
| Restitution | `0.0` |
| Added mass | `0.0` |
| Base COM offset | `(0.0, 0.0, 0.0)` |
| Pushes | disabled |
| Motor stiffness scale | `1.0` |
| Motor damping scale | `1.0` |

This was the necessary Isaac Sim sanity pass before export and runtime staging.

## Deployment Gate Result

The combined candidate was exported into:

```text
rma_go2_lab/policies/exported/go2_blind_rough_combined_asymppo_steps_v1_candidate/
```

The non-GUI deployment gate passed on 2026-07-20:

| Gate item | Result |
| --- | --- |
| Bundle structural validation | Passed |
| Tensor contract | Passed: `policy_obs_dim=45`, `history_length=100`, `action_dim=12` |
| Actor observation order | Passed: `base_ang_vel`, `projected_gravity`, `velocity_commands`, `joint_pos_rel`, `joint_vel_rel`, `last_action` |
| Deployment command ranges | Passed: `vx=[-0.8, 0.8]`, `vy=[-0.3, 0.3]`, `yaw=[-0.6, 0.6]` |
| TorchScript forward smoke | Passed |
| Checkpoint vs TorchScript parity | Passed, max abs error `0.0` |
| ONNX C++ golden inference parity | Passed, max abs error `1.07288e-06` |
| MuJoCo preflight | Passed |
| Unitree MJLAB FSM runtime audit | Passed |

Report path:

```text
artifacts/deployment_validation/go2_blind_rough_combined_asymppo_steps_v1_candidate/validation_gate_report.json
```

## Immediate Next Direction

1. Run a controlled MuJoCo/FSM visual sim2sim session with the active combined
   runtime.
2. If that is stable, run DDS probe and stance-only hardware bring-up.
3. Only after stance and low-speed policy takeover are stable, attempt stair
   hardware tests.

Validation command source:

```text
docs/RUN_COMMANDS.md -> Experimental Combined AsymPPO Staged Training
```

Until hardware bring-up passes, the validated AsymPPO deployment candidate
remains the production-safe branch.
