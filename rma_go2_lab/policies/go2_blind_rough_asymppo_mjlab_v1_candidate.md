# Go2 Blind Rough AsymPPO MJLAB V1 Freeze

## Identity

- policy: `go2_blind_rough_asymppo_mjlab_v1_candidate`
- task: `Go2-Blind-Rough-MJLAB-AsymPPO-V1`
- source run:
  `/home/bhuvan/tools/IsaacLab/logs/rsl_rl/go2_blind_rough_asymppo_mjlab_v1/2026-06-04_10-31-03`
- source PPO checkpoint: `model_1999.pt`
- source PPO checkpoint status: unavailable in this repo / lost from local logs
- bundle:
  `rma_go2_lab/policies/exported/go2_blind_rough_asymppo_mjlab_v1_candidate`

## Role

This is the canonical deployable rough omnidirectional policy after successful
Isaac Sim, MuJoCo, and initial real-Go2 validation.

It is a blind history actor trained with asymmetric PPO. It is not an RMA
teacher and it does not require a separately distilled student.

Important artifact distinction:

- `go2_blind_rough_asymppo_mjlab_v1_candidate.torchscript.pt` is a deployment
  export, not the raw RSL-RL PPO checkpoint.
- The original raw PPO checkpoint was
  `/home/bhuvan/tools/IsaacLab/logs/rsl_rl/go2_blind_rough_asymppo_mjlab_v1/2026-06-04_10-31-03/model_1999.pt`
  and is currently unavailable.
- Treat the tracked exported bundle as the validated deployment artifact.
- Do not use the TorchScript `.pt` as a training-resume checkpoint.

## Frozen Contract

- actor observation: 45
- history: `100 x 45`
- history layout: `isaaclab_term_major`
- action: 12 joint-position targets
- action scale: `0.25`
- control rate: `50 Hz`
- actor `base_lin_vel`: absent
- gait phase: absent
- deployed terrain privilege: absent

Actor terms:

```text
base_ang_vel
projected_gravity
velocity_commands
joint_pos_rel
joint_vel_rel
last_action
```

## Architecture

- actor MLP: `[512, 256, 128]`, ELU
- critic MLP: `[512, 256, 128]`, ELU
- temporal channels: `[64, 64]`
- temporal kernel: `3`
- history feature: `64`
- history target: `128`
- flat initialization:
  `go2_flat_mjlab_prior_v1/2026-06-02_14-30-48/model_1499.pt`

## Training Envelope

- commands:
  - X: `+/-0.8 m/s`
  - Y: `+/-0.3 m/s`
  - yaw: `+/-0.6 rad/s`
- friction: `0.1-2.0`
- added base mass: `-2 to +4 kg`
- stiffness/damping scale: `0.6-1.4`
- COM X/Y: `+/-0.03 m`
- COM Z: `+/-0.01 m`
- push interval: `6-10 s`
- push X/Y delta velocity: `+/-0.35 m/s`
- push yaw delta: `+/-0.4 rad/s`

## Validation

- structural deployment gate: pass
- source-checkpoint/TorchScript parity: was validated at export time, but the
  raw source checkpoint is no longer available for rerun verification
- C++ ONNX parity: pass
- Isaac/MuJoCo report parity: comparable
- initial real hardware: stable, with no obvious earlier FR asymmetry

Detailed evidence:

- `docs/ASYMPPO_SIM2REAL_SUCCESS_RETROSPECTIVE_20260612.md`
- `docs/DEPLOYMENT_VALIDATION_GATE.md`
- `docs/CROSS_SIMULATOR_VALIDATION_CONTRACT.md`

## Canonical Runtime

Use only:

```bash
bash scripts/deploy/run_unitree_mjlab_sim_deploy.sh activate
bash scripts/deploy/run_unitree_mjlab_sim_deploy.sh validate
bash scripts/deploy/run_unitree_mjlab_sim_deploy.sh hardware <network-interface>
```

For concurrent read-only `LowState` and `LowCmd` capture:

```bash
bash scripts/deploy/run_unitree_mjlab_sim_deploy.sh \
  monitor <network-interface> asymppo_walk
```

The monitor writes schema-v2 policy-ordered telemetry under
`artifacts/go2_realtime_monitor/`.

## Freeze Statement

Do not overwrite this bundle.

Future changes must use a new versioned candidate and compare against this
policy under the same validation profile and hardware protocol.
