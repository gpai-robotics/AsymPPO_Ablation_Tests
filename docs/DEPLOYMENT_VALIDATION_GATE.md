# Deployment Validation Gate

This repo uses a fixed validation ladder for every deployable Go2 policy. The
goal is to stop changing validation criteria per training run.

## Contract First

Every exported bundle must define the runtime contract:

- `policy_obs_dim`
- `policy_order`
- `policy_history_length`
- `policy_history_dim`
- `action_dim`
- `action_scale`
- `default_joint_pos`
- `joint_order`
- `control_dt`
- `joint_stiffness`
- `joint_damping`
- `effort_limit`

For the current AsymPPO candidate:

```text
policy_obs_dim: 45
policy_history_length: 100
policy_history_dim: 4500
action_dim: 12
policy_order:
  base_ang_vel
  projected_gravity
  velocity_commands
  joint_pos_rel
  joint_vel_rel
  last_action
```

This actor intentionally does not consume `base_lin_vel`.

## Gate Ladder

### Gate 0: Bundle contract

Checks that the bundle manifest, metadata, deploy config, TorchScript artifact,
observation dimensions, and history dimensions are internally consistent.

### Gate 1: TorchScript smoke

Runs one zero-input forward pass through the exported TorchScript policy and
checks output shape and finite action values.

### Gate 1b: Golden inference parity

Generates deterministic nonzero policy/history vectors and requires agreement
between the reconstructed training-checkpoint actor, exported TorchScript, and
the same C++ ONNX Runtime `OrtRunner` used by the Unitree controller:

```bash
/home/bhuvan/miniconda3/envs/rma-mujoco/bin/python \
  scripts/deploy/validate_policy_inference_parity.py \
  --bundle-dir rma_go2_lab/policies/exported/go2_blind_rough_asymppo_mjlab_v1_candidate
```

The report and reusable golden vectors are written under:

```text
artifacts/deployment_validation/golden_inference/<policy_name>/
```

### Gate 2: MuJoCo preflight

Checks the MuJoCo runtime bridge, robot model path, exported metadata, and
runtime compatibility. This is a blocker if the `mujoco` Python package is not
available in the active environment.

### Gate 3: Hardware mapping audit

Validates the `unitree_rl_mjlab` Go2 FSM runtime config. This does not touch
DDS or the robot. The C++ FSM is the source of truth for hardware bring-up:

- `FSM.Passive`
- `FSM.FixStand`
- `FSM.Velocity`
- `Velocity.policy_dir`
- staged `params/deploy.yaml`
- staged `exported/policy.onnx`

Run directly with:

```bash
python scripts/deploy/validate_unitree_mjlab_go2_fsm_runtime.py \
  --strict-fixstand-gains \
  --expected-policy-name go2_blind_rough_asymppo_mjlab_v1_candidate
```

For Go2, keep the two directions explicit:

```text
Unitree hardware order:
  FR_hip, FR_thigh, FR_calf,
  FL_hip, FL_thigh, FL_calf,
  RR_hip, RR_thigh, RR_calf,
  RL_hip, RL_thigh, RL_calf

Policy order:
  FL_hip, FR_hip, RL_hip, RR_hip,
  FL_thigh, FR_thigh, RL_thigh, RR_thigh,
  FL_calf, FR_calf, RL_calf, RR_calf

hardware indices gathered into policy order:
  [3, 0, 9, 6, 4, 1, 10, 7, 5, 2, 11, 8]

policy indices gathered into hardware order:
  [1, 5, 9, 0, 4, 8, 3, 7, 11, 2, 6, 10]
```

The stationary observation audit is required to pass both a stable-pose
proximity check and a bounded shadow-action check. This prevents a finite but
wrongly permuted observation from passing.

Real deployment uses the validated Unitree MJLAB FSM controller path:

```text
Passive -> FixStand -> Velocity
```

### Gate 4: MuJoCo scenario suite

Runs the fixed MuJoCo scenario suite with `--actuator-model isaac_dc_motor`.
MuJoCo torque clipping is aligned to the exported bundle `effort_limit`, not
the raw XML actuator range.

### Gate 5: IsaacSim matched traces

Run Isaac traces on fixed commands/scenarios for comparison against MuJoCo
summary metrics. IsaacSim remains the training simulator; MuJoCo is the
independent cross-engine sanity check.

### Gate 6: Hardware bring-up

Only after Gates 0-4 pass:

- dry run
- read-only stationary observation audit
- stance-only
- forward-only low-speed walking
- full omni walking

The stationary audit subscribes only to `rt/lowstate`, reconstructs the
candidate observation/history contract, and runs shadow policy inference
without creating a `LowCmd` publisher:

```bash
python scripts/deploy/run_go2_stationary_observation_audit.py \
  --net-if enp0s31f6
```

## Standard AsymPPO Gate Command

```bash
python scripts/deploy/run_deployment_validation_gate.py \
  --bundle-dir rma_go2_lab/policies/exported/go2_blind_rough_asymppo_mjlab_v1_candidate \
  --expected-policy-obs-dim 45 \
  --expected-history-length 100
```

If the active Python environment has MuJoCo installed, run the rollout suite:

```bash
python scripts/deploy/run_deployment_validation_gate.py \
  --bundle-dir rma_go2_lab/policies/exported/go2_blind_rough_asymppo_mjlab_v1_candidate \
  --expected-policy-obs-dim 45 \
  --expected-history-length 100 \
  --run-mujoco-suite \
  --mujoco-suite mujoco_disturb_v2_moderate \
  --mujoco-rollouts 3 \
  --mujoco-max-steps 900
```

Report path:

```text
artifacts/deployment_validation/<bundle_name>/validation_gate_report.json
```

## Interpretation

Passing IsaacSim alone means the policy works in the simulator family it was
trained in.

Passing MuJoCo means the exported runtime contract and policy are not obviously
fragile to a second physics engine, different contact model, and different
robot XML.

Passing MuJoCo does not prove hardware success. Failing MuJoCo does not
automatically prove the policy is bad; it may reveal an engine/model mismatch.
But no policy should go to hardware before its bundle contract and runtime
bridge pass this gate.
