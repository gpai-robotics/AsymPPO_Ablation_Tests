# Active AsymPPO Run Commands

This document contains only the canonical working pipeline.

```bash
cd /path/to/rma_go2_lab
export REPO=$PWD
export ISAACLAB_ROOT=/opt/IsaacLab
export GO2_NET_IF=<robot-facing-interface>
export GO2_ETH_IF=<ethernet-interface>
export GO2_WIFI_IF=<wifi-interface>
export RMA_MUJOCO_PYTHON=python
export MUJOCO_PYTHON=$RMA_MUJOCO_PYTHON
export GO2_HW_PYTHON=python3
export GO2_USD_PATH=/path/to/go2.usd
export ASYMPPO_CKPT=/path/to/restored/go2_blind_rough_asymppo_mjlab_v1/model_1999.pt
export ASYMPPO_BUNDLE=$REPO/rma_go2_lab/policies/exported/go2_blind_rough_asymppo_mjlab_v1_candidate
export COMBINED_STEPS_CKPT=$REPO/logs/rsl_rl/go2_blind_rough_combined_asymppo_steps_v1/2026-07-16_10-14-29/model_5099.pt
export COMBINED_BUNDLE=$REPO/rma_go2_lab/policies/exported/go2_blind_rough_combined_asymppo_steps_v1_candidate
```

Notes:

- `ASYMPPO_CKPT` is only needed for IsaacLab playback or re-export from a raw
  RSL-RL training checkpoint. The validated source checkpoint was
  `go2_blind_rough_asymppo_mjlab_v1/2026-06-04_10-31-03/model_1999.pt`, but it
  is currently unavailable unless restored from backup.
- Unitree MJLAB sim/hardware deployment uses `ASYMPPO_BUNDLE`, but exported
  bundles are runtime artifacts. Restore/download the bundle explicitly before
  deployment; do not assume a fresh source clone contains trained policy
  exports.
- Do not point `ASYMPPO_CKPT` at
  `go2_blind_rough_asymppo_mjlab_v1_candidate.torchscript.pt`; that file is a
  deployment export, not a PPO checkpoint.
- On the shared workstation, set `RMA_MUJOCO_PYTHON` to the MuJoCo/SDK Python
  environment if `python` is not already that environment.
- Set `GO2_HW_PYTHON` to the hardware SDK Python environment before using
  `dds-probe`, `monitor`, or `hardware`.
- Set `GO2_USD_PATH` to the local Go2 USD used for IsaacLab training/eval on
  machines that do not have the historical workstation asset path.

## Workstation Preflight

Run this before Isaac Sim playback on the shared `/opt` IsaacLab install:

```bash
bash $REPO/scripts/isaaclab_user.sh -p -c \
  "import torch, rma_go2_lab, gymnasium as gym; print(torch.__version__); print(gym.spec('Go2-Blind-Rough-MJLAB-AsymPPO-V1').id)"
```

Expected output includes:

```text
2.7.0+cu128
Go2-Blind-Rough-MJLAB-AsymPPO-V1
```

Do not call `$ISAACLAB_ROOT/isaaclab.sh` directly for active playback on this
workstation. The wrapper sets user-writable Kit cache/log/temp folders and
exposes Isaac Sim's bundled CUDA libraries to PyTorch.

## Isaac Sim

This section requires the raw PPO checkpoint. Skip it if `ASYMPPO_CKPT` has not
been restored. Use the Unitree MuJoCo FSM and hardware deployment sections only
after restoring an exported deployment bundle.

```bash
bash $REPO/scripts/isaaclab_user.sh -p $REPO/scripts/eval/play_policy.py \
  --task Go2-Blind-Rough-MJLAB-AsymPPO-V1 \
  --checkpoint $ASYMPPO_CKPT \
  --num_envs 16 \
  --teleop-keyboard
```

Optional clean nominal playback:

```bash
bash $REPO/scripts/isaaclab_user.sh -p $REPO/scripts/eval/play_policy.py \
  --task Go2-Blind-Rough-MJLAB-AsymPPO-V1 \
  --checkpoint $ASYMPPO_CKPT \
  --num_envs 16 \
  --teleop-keyboard \
  --nominal-env \
  --terrain-type plane
```

## Experimental Combined AsymPPO Staged Training

This is not the frozen validated sim2real candidate. It follows the staged
combined pipeline described in `docs/COMBINED_ASYMPPO_BASELINE.md`:

```text
combined flat prior -> combined rough -> combined stairs
```

Stage 1, flat prior:

```bash
bash $REPO/scripts/isaaclab_user.sh -p "$ISAACLAB_ROOT/scripts/reinforcement_learning/rsl_rl/train.py" \
  --task Go2-Combined-Flat-MJLAB-Prior-V1 \
  --headless
```

Stage 2, rough. Set this to a Stage 1 checkpoint:

```bash
export COMBINED_FLAT_PRIOR_CKPT=/path/to/go2_combined_flat_mjlab_prior_v1/<run>/model_<iter>.pt

bash $REPO/scripts/isaaclab_user.sh -p "$ISAACLAB_ROOT/scripts/reinforcement_learning/rsl_rl/train.py" \
  --task Go2-Blind-Rough-MJLAB-Combined-AsymPPO-Rough-V1 \
  --headless
```

Stage 3, stairs. Set this to a Stage 2 checkpoint:

```bash
export COMBINED_ROUGH_CKPT=/path/to/go2_blind_rough_combined_asymppo_rough_v1/<run>/model_<iter>.pt

bash $REPO/scripts/isaaclab_user.sh -p "$ISAACLAB_ROOT/scripts/reinforcement_learning/rsl_rl/train.py" \
  --task Go2-Blind-Rough-MJLAB-Combined-AsymPPO-Steps-V1 \
  --headless
```

Expected log root:

```text
logs/rsl_rl/go2_combined_flat_mjlab_prior_v1/
logs/rsl_rl/go2_blind_rough_combined_asymppo_rough_v1/
logs/rsl_rl/go2_blind_rough_combined_asymppo_steps_v1/
```

### Combined Candidate Isaac Sim Validation

Use this before exporting or attempting MuJoCo/FSM deployment for the combined
candidate.

Normal realistic Go2 flat playback:

```bash
bash $REPO/scripts/isaaclab_user.sh -p $REPO/scripts/eval/play_policy.py \
  --task Go2-Blind-Rough-MJLAB-Combined-AsymPPO-Steps-V1 \
  --checkpoint $COMBINED_STEPS_CKPT \
  --num_envs 16 \
  --nominal-env \
  --terrain-type plane \
  --fixed-command \
  --cmd-vx 0.3 \
  --cmd-vy 0.0 \
  --cmd-yaw 0.0 \
  --max-steps 1200 \
  --print-env-info
```

Normal realistic stairs playback inside the checkpoint's trained stair-height
range:

```bash
bash $REPO/scripts/isaaclab_user.sh -p $REPO/scripts/eval/play_policy.py \
  --task Go2-Blind-Rough-MJLAB-Combined-AsymPPO-Steps-V1 \
  --checkpoint $COMBINED_STEPS_CKPT \
  --num_envs 16 \
  --nominal-env \
  --terrain-type pyramid_stairs \
  --terrain-level 4 \
  --step-height 0.12 \
  --step-width 0.30 \
  --platform-width 3.0 \
  --fixed-command \
  --cmd-vx 0.3 \
  --cmd-vy 0.0 \
  --cmd-yaw 0.0 \
  --max-steps 1200 \
  --print-env-info
```

Inverted stairs:

```bash
bash $REPO/scripts/isaaclab_user.sh -p $REPO/scripts/eval/play_policy.py \
  --task Go2-Blind-Rough-MJLAB-Combined-AsymPPO-Steps-V1 \
  --checkpoint $COMBINED_STEPS_CKPT \
  --num_envs 16 \
  --nominal-env \
  --terrain-type pyramid_stairs_inv \
  --terrain-level 4 \
  --step-height 0.12 \
  --step-width 0.30 \
  --platform-width 3.0 \
  --fixed-command \
  --cmd-vx 0.3 \
  --cmd-vy 0.0 \
  --cmd-yaw 0.0 \
  --max-steps 1200 \
  --print-env-info
```

Rough/sloped sanity check:

```bash
bash $REPO/scripts/isaaclab_user.sh -p $REPO/scripts/eval/play_policy.py \
  --task Go2-Blind-Rough-MJLAB-Combined-AsymPPO-Steps-V1 \
  --checkpoint $COMBINED_STEPS_CKPT \
  --num_envs 16 \
  --nominal-env \
  --terrain-type random_rough \
  --terrain-level 4 \
  --roughness-amplitude 0.04 \
  --roughness-noise 0.01 \
  --fixed-command \
  --cmd-vx 0.3 \
  --cmd-vy 0.0 \
  --cmd-yaw 0.0 \
  --max-steps 1200 \
  --print-env-info
```

`--nominal-env` means deterministic normal playback: pushes disabled, friction
fixed to `1.0`, restitution fixed to `0.0`, added mass fixed to `0.0`, COM
offset fixed to zero, and motor stiffness/damping scale fixed to `1.0`.

### Combined Candidate Export And Gate

Use this after the Isaac Sim visual smoke tests pass for a combined checkpoint.

Create or refresh the deployment manifest:

```bash
python $REPO/scripts/deploy/package_candidate.py \
  --policy-name go2_blind_rough_combined_asymppo_steps_v1_candidate \
  --source-checkpoint $COMBINED_STEPS_CKPT \
  --task Go2-Blind-Rough-MJLAB-Combined-AsymPPO-Steps-V1 \
  --phase combined-asymppo-stairs-v1 \
  --policy-kind blind_history_policy \
  --observation-groups policy,policy_history \
  --control-rate-hz 50 \
  --bundle-dir $COMBINED_BUNDLE \
  --freeze-note "Combined AsymPPO stairs candidate."
```

Export TorchScript, ONNX, and deployment metadata. This uses Isaac Sim's Python
directly because `export_policy.py` does not launch Isaac Sim and should not
receive `AppLauncher`/Kit arguments:

```bash
ML_ARCHIVE=$ISAACLAB_ROOT/_isaac_sim/exts/omni.isaac.ml_archive/pip_prebundle
ML_LIB_PATHS=$(find "$ML_ARCHIVE" -path '*/lib' -type d 2>/dev/null | paste -sd: -)

env TERM=xterm MPLCONFIGDIR=/tmp/matplotlib LD_LIBRARY_PATH="$ML_LIB_PATHS:${LD_LIBRARY_PATH:-}" \
  $ISAACLAB_ROOT/_isaac_sim/python.sh $REPO/scripts/deploy/export_policy.py \
  --policy-name go2_blind_rough_combined_asymppo_steps_v1_candidate \
  --checkpoint $COMBINED_STEPS_CKPT \
  --task Go2-Blind-Rough-MJLAB-Combined-AsymPPO-Steps-V1 \
  --phase combined-asymppo-stairs-v1 \
  --bundle-dir $COMBINED_BUNDLE \
  --policy-kind blind_history_policy \
  --observation-groups policy,policy_history \
  --policy-history-length 100 \
  --command-lin-vel-x -0.8 0.8 \
  --command-lin-vel-y -0.3 0.3 \
  --command-ang-vel-z -0.6 0.6 \
  --format torchscript \
  --format onnx
```

Run the non-GUI deployment gate:

```bash
$RMA_MUJOCO_PYTHON \
  $REPO/scripts/deploy/run_deployment_validation_gate.py \
  --bundle-dir $COMBINED_BUNDLE \
  --expected-policy-obs-dim 45 \
  --expected-history-length 100 \
  --expected-action-dim 12 \
  --python-exe $RMA_MUJOCO_PYTHON
```

Current `model_5099.pt` status: this gate passed on 2026-07-20 for structural
bundle validation, TorchScript forward smoke, ONNX C++ inference parity, MuJoCo
preflight, and Unitree MJLAB FSM runtime audit. The exported deployment command
ranges are omni-enabled: `vx=[-0.8, 0.8]`, `vy=[-0.3, 0.3]`,
`yaw=[-0.6, 0.6]`.

## Deployment Gate

```bash
$RMA_MUJOCO_PYTHON \
  $REPO/scripts/deploy/run_deployment_validation_gate.py \
  --bundle-dir $ASYMPPO_BUNDLE \
  --expected-policy-obs-dim 45 \
  --expected-history-length 100 \
  --run-mujoco-suite \
  --mujoco-suite mujoco_disturb_v2_moderate \
  --mujoco-rollouts 3 \
  --mujoco-max-steps 900
```

## Unitree MuJoCo FSM

If `reference_repos/unitree_rl_mjlab` was restored from upstream or deleted
during cleanup, reapply the local simulated-joystick/runtime patch first:

```bash
cd $REPO/reference_repos/unitree_rl_mjlab
git apply ../../patches/unitree_rl_mjlab/go2_scripted_controller.patch
cd $REPO
```

Build the C++ controller and simulator if they are missing:

```bash
bash $REPO/scripts/deploy/build_unitree_mjlab_runtime.sh all
```

Recovery details and troubleshooting:
`docs/UNITREE_MJLAB_RUNTIME_BUILD.md`

Activate and validate the frozen AsymPPO runtime:

```bash
bash $REPO/scripts/deploy/run_unitree_mjlab_sim_deploy.sh activate asym
bash $REPO/scripts/deploy/run_unitree_mjlab_sim_deploy.sh validate
```

For the combined stairs candidate, use:

```bash
bash $REPO/scripts/deploy/run_unitree_mjlab_sim_deploy.sh activate combined
bash $REPO/scripts/deploy/run_unitree_mjlab_sim_deploy.sh validate
```

Start these in separate terminals:

```bash
bash $REPO/scripts/deploy/run_unitree_mjlab_sim_deploy.sh controller
```

Flat plane:

```bash
bash $REPO/scripts/deploy/run_unitree_mjlab_sim_deploy.sh sim
```

Available terrain scene with rough boxes, stairs, slope, and heightfields:

```bash
bash $REPO/scripts/deploy/run_unitree_mjlab_sim_deploy.sh sim-unitree-terrain
```

MJLAB procedural terrain path:

```bash
bash $REPO/scripts/deploy/run_unitree_mjlab_sim_deploy.sh sim-terrain random_rough 0.5 42
bash $REPO/scripts/deploy/run_unitree_mjlab_sim_deploy.sh sim-terrain stairs 0.5 42
```

`sim-terrain` requires the `mjlab.terrains` Python package to be available to
`MUJOCO_PYTHON`. If that package is missing, use `sim-unitree-terrain` for the
current local terrain smoke test.

## Real Go2 Network Selection

The controller binds CycloneDDS to one interface at process startup. Transport
switching must be done while the robot is in Passive and the controller is
stopped. There is intentionally no live failover during torque control.

## Combined AsymPPO Model 5099 Validation

Canonical validation plan:

```bash
cat docs/COMBINED_ASYMPPO_MODEL5099_VALIDATION.md
```

Machine-readable manifest:

```bash
cat configs/validation/go2_combined_asymppo_steps_v1_model5099.json
```

Run the non-GUI MuJoCo terrain/robustness suites:

```bash
cd /home/bhuvan/projects/rma/rma_go2_lab
bash scripts/deploy/run_combined_asymppo_model5099_mujoco_validation.sh
```

## Immediate Hardware Deployment Over Ethernet

Use this when the robot is on Ethernet and you want to deploy the active
AsymPPO bundle immediately.

```bash
cd /path/to/rma_go2_lab
export REPO=$PWD
export GO2_ETH_IF=<ethernet-interface>
export GO2_HW_PYTHON=<python-with-unitree-sdk2py-and-cyclonedds>
export MUJOCO_PYTHON=<python-with-mujoco-runtime-deps>
```

Preflight:

```bash
bash $REPO/scripts/deploy/run_unitree_mjlab_sim_deploy.sh activate asym
bash $REPO/scripts/deploy/run_unitree_mjlab_sim_deploy.sh validate
bash $REPO/scripts/deploy/run_unitree_mjlab_sim_deploy.sh network-status ethernet
bash $REPO/scripts/deploy/run_unitree_mjlab_sim_deploy.sh dds-probe ethernet
```

Start the hardware controller:

```bash
bash $REPO/scripts/deploy/run_unitree_mjlab_sim_deploy.sh hardware ethernet
```

Remote sequence:

```text
L2 + up  -> FixStand
R2 + A   -> start AsymPPO Velocity policy
L2 + B   -> return Passive / stop
```

Optional read-only monitor in a second terminal:

```bash
cd /path/to/rma_go2_lab
export GO2_HW_PYTHON=<python-with-unitree-sdk2py-and-cyclonedds>

bash scripts/deploy/run_unitree_mjlab_sim_deploy.sh monitor ethernet asymppo_walk
```

Do not start `hardware` unless `dds-probe ethernet` receives `rt/lowstate`.

Inspect both links:

```bash
bash $REPO/scripts/deploy/run_unitree_mjlab_sim_deploy.sh network-status ethernet
bash $REPO/scripts/deploy/run_unitree_mjlab_sim_deploy.sh network-status wifi
```

### Ethernet

Run the read-only DDS preflight:

```bash
bash $REPO/scripts/deploy/run_unitree_mjlab_sim_deploy.sh dds-probe ethernet
```

Then start the canonical C++ FSM:

```bash
bash $REPO/scripts/deploy/run_unitree_mjlab_sim_deploy.sh hardware ethernet
```

### Wi-Fi

The USB dongle on the Go2 and the laptop adapter must both be configured.
Simply inserting the dongle does not connect the robot to a WLAN.

Keep Ethernet attached and configure the Go2-side dongle over SSH:

```bash
bash $REPO/scripts/deploy/run_unitree_mjlab_sim_deploy.sh \
  robot-wifi-connect <GO2_ETHERNET_IP> <SSID>
```

The command prints the robot's new Wi-Fi IP. Then connect the laptop adapter
to the same WLAN. Credentials are interactive and are not stored in this repo:

```bash
bash $REPO/scripts/deploy/run_unitree_mjlab_sim_deploy.sh \
  wifi-scan $GO2_WIFI_IF

bash $REPO/scripts/deploy/run_unitree_mjlab_sim_deploy.sh \
  wifi-connect <GO2_DONGLE_SSID> $GO2_WIFI_IF
```

Verify direct peer reachability before testing DDS:

```bash
bash $REPO/scripts/deploy/run_unitree_mjlab_sim_deploy.sh network-status wifi
bash $REPO/scripts/deploy/run_unitree_mjlab_sim_deploy.sh \
  wifi-peer <GO2_WIFI_IP> $GO2_WIFI_IF
bash $REPO/scripts/deploy/run_unitree_mjlab_sim_deploy.sh dds-probe wifi
```

Only after the read-only probe receives LowState packets:

```bash
bash $REPO/scripts/deploy/run_unitree_mjlab_sim_deploy.sh hardware wifi
```

The Wi-Fi access point must allow multicast and must not enable client/AP
isolation. A valid IPv4 connection without LowState packets is not sufficient.
Campus and public guest WLANs usually violate these requirements. Prefer a
dedicated router/hotspot, or a lab IoT WLAN explicitly configured for
client-to-client multicast. The laptop adapter and the dongle installed on the
Go2 must join that same WLAN.

Interpretation:

- `wifi-peer` fails: robot is not on the WLAN, has the wrong IP, or the WLAN
  isolates clients.
- `wifi-peer` passes but `dds-probe` fails: the WLAN filters DDS multicast, or
  the Go2 DDS services are not exposing the Wi-Fi interface.
- both pass: Wi-Fi is ready for the hardware controller.

### Explicit Interface Or Auto

The old direct-interface form remains valid:

```bash
bash $REPO/scripts/deploy/run_unitree_mjlab_sim_deploy.sh hardware $GO2_NET_IF
```

`auto` prefers a ready Ethernet link, then Wi-Fi:

```bash
bash $REPO/scripts/deploy/run_unitree_mjlab_sim_deploy.sh hardware auto
```

Remote transitions:

```text
L2 + up  -> FixStand
R2 + A   -> Velocity
L2 + B   -> Passive
```

## Live Low-Level Telemetry

Start this read-only monitor in a separate terminal before starting the
hardware controller:

```bash
bash $REPO/scripts/deploy/run_unitree_mjlab_sim_deploy.sh \
  monitor ethernet asymppo_walk
```

For Wi-Fi:

```bash
bash $REPO/scripts/deploy/run_unitree_mjlab_sim_deploy.sh \
  monitor wifi asymppo_wifi_walk
```

It subscribes to `rt/lowstate`, `rt/lowcmd`, and `rt/sportmodestate`. It does
not publish commands or switch robot modes. Stop it with `Ctrl-C` after the
robot has returned to Passive.

Captured JSONL files are written to:

```text
artifacts/go2_realtime_monitor/
```

Analyze one capture:

```bash
$RMA_MUJOCO_PYTHON \
  scripts/deploy/analyze_go2_realtime_monitor.py \
  --jsonl artifacts/go2_realtime_monitor/<capture>.jsonl
```

Run the mirrored-leg analysis:

```bash
$RMA_MUJOCO_PYTHON \
  scripts/deploy/analyze_go2_leg_mirror_pairs.py \
  --jsonl artifacts/go2_realtime_monitor/<capture>.jsonl
```

Telemetry schema version 2 stores all motor vectors in deployed policy order:

```text
FL_hip FR_hip RL_hip RR_hip
FL_thigh FR_thigh RL_thigh RR_thigh
FL_calf FR_calf RL_calf RR_calf
```

Older monitor captures are automatically remapped from raw SDK order by the
analysis scripts.

## Two-Robot Hardware Differential

Use this when the same policy bundle works on one Go2 but destabilizes on a
second Go2 after Velocity/policy takeover. The workflow captures standing
blueprints and read-only LowState/LowCmd streams, then compares target commands,
gains, tracking error, estimated torque, joint velocity, foot-force balance, and
IMU response.

Full protocol:

```text
docs/GO2_TWO_ROBOT_DEPLOYMENT_DIFFERENTIAL.md
```

Quick command shape:

```bash
scripts/deploy/run_go2_readonly_signature_check.sh capture-blueprint go2_a_stand enp0s31f6 8
scripts/deploy/run_go2_readonly_signature_check.sh capture-blueprint go2_b_stand enp0s31f6 8

scripts/deploy/run_go2_readonly_signature_check.sh capture-dynamic-lowcmd go2_a_policy_takeover enp0s31f6 30 0.05
scripts/deploy/run_go2_readonly_signature_check.sh capture-dynamic-lowcmd go2_b_policy_takeover enp0s31f6 30 0.05

scripts/deploy/run_go2_readonly_signature_check.sh compare-dynamic \
  artifacts/go2_readonly_signatures/<a>_go2_a_policy_takeover_series.jsonl \
  artifacts/go2_readonly_signatures/<a>_go2_a_policy_takeover_lowcmd_stream.jsonl \
  artifacts/go2_readonly_signatures/<b>_go2_b_policy_takeover_series.jsonl \
  artifacts/go2_readonly_signatures/<b>_go2_b_policy_takeover_lowcmd_stream.jsonl
```




<!-- bash $REPO/scripts/isaaclab_user.sh -p $REPO/scripts/eval/play_policy.py \
  --task Go2-Blind-Rough-MJLAB-Combined-AsymPPO-Steps-V1 \
  --checkpoint $COMBINED_STEPS_CKPT \
  --num_envs 16 \
  --terrain-type pyramid_stairs_inv \
  --terrain-level 0 \
  --step-height 0.17 \
  --step-width 0.30 \
  --platform-width 3.0 \
  --disable-pushes \
  --static-friction 1.0 \
  --dynamic-friction 1.0 \
  --restitution 0.0 \
  --added-mass 0.0 \
  --com-x 0.0 \
  --com-y 0.0 \
  --com-z 0.0 \
  --motor-stiffness-scale 1.0 \
  --motor-damping-scale 1.0 \
  --teleop-keyboard -->



<!-- bhuvan@deepan-Precision-7960-Tower:~$ cd /home/bhuvan/projects/rma/rma_go2_lab
python scripts/eval/print_combined_asymppo_model5099_isaac_commands.py
cd /home/bhuvan/projects/rma/rma_go2_lab -->

<!-- # normal_flat_forward
bash \
  scripts/isaaclab_user.sh \
  -p \
  scripts/eval/play_policy.py \
  --task Go2-Blind-Rough-MJLAB-Combined-AsymPPO-Steps-V1 \
  --checkpoint /home/bhuvan/projects/rma/rma_go2_lab/logs/rsl_rl/go2_blind_rough_combined_asymppo_steps_v1/2026-07-16_10-14-29/model_5099.pt \
  --num_envs 16 \
  --nominal-env \
  --terrain-type plane \
  --fixed-command \
  --cmd-vx 0.35 \
  --cmd-vy 0.0 \
  --cmd-yaw 0.0 \
  --max-steps 800 \
  --print-env-info \
  --eval-json-out /home/bhuvan/projects/rma/rma_go2_lab/artifacts/isaac_eval/go2_blind_rough_combined_asymppo_steps_v1_candidate_5099/normal_flat_forward.json -->

<!-- # normal_random_rough_forward
bash \
  scripts/isaaclab_user.sh \
  -p \
  scripts/eval/play_policy.py \
  --task Go2-Blind-Rough-MJLAB-Combined-AsymPPO-Steps-V1 \
  --checkpoint /home/bhuvan/projects/rma/rma_go2_lab/logs/rsl_rl/go2_blind_rough_combined_asymppo_steps_v1/2026-07-16_10-14-29/model_5099.pt \
  --num_envs 16 \
  --nominal-env \
  --terrain-type random_rough \
  --fixed-command \
  --cmd-vx 0.35 \
  --cmd-vy 0.0 \
  --cmd-yaw 0.0 \
  --max-steps 800 \
  --print-env-info \
  --eval-json-out /home/bhuvan/projects/rma/rma_go2_lab/artifacts/isaac_eval/go2_blind_rough_combined_asymppo_steps_v1_candidate_5099/normal_random_rough_forward.json \
  --terrain-level 4 -->

<!-- # normal_stairs_up_forward
bash \
  scripts/isaaclab_user.sh \
  -p \
  scripts/eval/play_policy.py \
  --task Go2-Blind-Rough-MJLAB-Combined-AsymPPO-Steps-V1 \
  --checkpoint /home/bhuvan/projects/rma/rma_go2_lab/logs/rsl_rl/go2_blind_rough_combined_asymppo_steps_v1/2026-07-16_10-14-29/model_5099.pt \
  --num_envs 16 \
  --nominal-env \
  --terrain-type pyramid_stairs \
  --fixed-command \
  --cmd-vx 0.35 \
  --cmd-vy 0.0 \
  --cmd-yaw 0.0 \
  --max-steps 800 \
  --print-env-info \
  --eval-json-out /home/bhuvan/projects/rma/rma_go2_lab/artifacts/isaac_eval/go2_blind_rough_combined_asymppo_steps_v1_candidate_5099/normal_stairs_up_forward.json \
  --terrain-level 4 -->

<!-- # normal_stairs_down_forward
bash \
  scripts/isaaclab_user.sh \
  -p \
  scripts/eval/play_policy.py \
  --task Go2-Blind-Rough-MJLAB-Combined-AsymPPO-Steps-V1 \
  --checkpoint /home/bhuvan/projects/rma/rma_go2_lab/logs/rsl_rl/go2_blind_rough_combined_asymppo_steps_v1/2026-07-16_10-14-29/model_5099.pt \
  --num_envs 16 \
  --nominal-env \
  --terrain-type pyramid_stairs_inv \
  --fixed-command \
  --cmd-vx 0.3 \
  --cmd-vy 0.0 \
  --cmd-yaw 0.0 \
  --max-steps 800 \
  --print-env-info \
  --eval-json-out /home/bhuvan/projects/rma/rma_go2_lab/artifacts/isaac_eval/go2_blind_rough_combined_asymppo_steps_v1_candidate_5099/normal_stairs_down_forward.json \
  --terrain-level 4 -->

<!-- # normal_slope_forward
bash \
  scripts/isaaclab_user.sh \
  -p \
  scripts/eval/play_policy.py \
  --task Go2-Blind-Rough-MJLAB-Combined-AsymPPO-Steps-V1 \
  --checkpoint /home/bhuvan/projects/rma/rma_go2_lab/logs/rsl_rl/go2_blind_rough_combined_asymppo_steps_v1/2026-07-16_10-14-29/model_5099.pt \
  --num_envs 16 \
  --nominal-env \
  --terrain-type hf_pyramid_slope \
  --fixed-command \
  --cmd-vx 0.35 \
  --cmd-vy 0.0 \
  --cmd-yaw 0.0 \
  --max-steps 800 \
  --print-env-info \
  --eval-json-out /home/bhuvan/projects/rma/rma_go2_lab/artifacts/isaac_eval/go2_blind_rough_combined_asymppo_steps_v1_candidate_5099/normal_slope_forward.json \
  --terrain-level 4 -->

<!-- # normal_omni_diagonal
bash \
  scripts/isaaclab_user.sh \
  -p \
  scripts/eval/play_policy.py \
  --task Go2-Blind-Rough-MJLAB-Combined-AsymPPO-Steps-V1 \
  --checkpoint /home/bhuvan/projects/rma/rma_go2_lab/logs/rsl_rl/go2_blind_rough_combined_asymppo_steps_v1/2026-07-16_10-14-29/model_5099.pt \
  --num_envs 16 \
  --nominal-env \
  --terrain-type plane \
  --fixed-command \
  --cmd-vx 0.3 \
  --cmd-vy 0.15 \
  --cmd-yaw 0.2 \
  --max-steps 800 \
  --print-env-info \
  --eval-json-out /home/bhuvan/projects/rma/rma_go2_lab/artifacts/isaac_eval/go2_blind_rough_combined_asymppo_steps_v1_candidate_5099/normal_omni_diagonal.json -->

<!-- # normal_flat_lateral_left
bash \
  scripts/isaaclab_user.sh \
  -p \
  scripts/eval/play_policy.py \
  --task Go2-Blind-Rough-MJLAB-Combined-AsymPPO-Steps-V1 \
  --checkpoint /home/bhuvan/projects/rma/rma_go2_lab/logs/rsl_rl/go2_blind_rough_combined_asymppo_steps_v1/2026-07-16_10-14-29/model_5099.pt \
  --num_envs 16 \
  --nominal-env \
  --terrain-type plane \
  --fixed-command \
  --cmd-vx 0.0 \
  --cmd-vy 0.2 \
  --cmd-yaw 0.0 \
  --max-steps 800 \
  --print-env-info \
  --eval-json-out /home/bhuvan/projects/rma/rma_go2_lab/artifacts/isaac_eval/go2_blind_rough_combined_asymppo_steps_v1_candidate_5099/normal_flat_lateral_left.json -->

<!-- # normal_flat_lateral_right
bash \
  scripts/isaaclab_user.sh \
  -p \
  scripts/eval/play_policy.py \
  --task Go2-Blind-Rough-MJLAB-Combined-AsymPPO-Steps-V1 \
  --checkpoint /home/bhuvan/projects/rma/rma_go2_lab/logs/rsl_rl/go2_blind_rough_combined_asymppo_steps_v1/2026-07-16_10-14-29/model_5099.pt \
  --num_envs 16 \
  --nominal-env \
  --terrain-type plane \
  --fixed-command \
  --cmd-vx 0.0 \
  --cmd-vy -0.2 \
  --cmd-yaw 0.0 \
  --max-steps 800 \
  --print-env-info \
  --eval-json-out /home/bhuvan/projects/rma/rma_go2_lab/artifacts/isaac_eval/go2_blind_rough_combined_asymppo_steps_v1_candidate_5099/normal_flat_lateral_right.json -->

<!-- # normal_flat_yaw_left
bash \
  scripts/isaaclab_user.sh \
  -p \
  scripts/eval/play_policy.py \
  --task Go2-Blind-Rough-MJLAB-Combined-AsymPPO-Steps-V1 \
  --checkpoint /home/bhuvan/projects/rma/rma_go2_lab/logs/rsl_rl/go2_blind_rough_combined_asymppo_steps_v1/2026-07-16_10-14-29/model_5099.pt \
  --num_envs 16 \
  --nominal-env \
  --terrain-type plane \
  --fixed-command \
  --cmd-vx 0.0 \
  --cmd-vy 0.0 \
  --cmd-yaw 0.35 \
  --max-steps 800 \
  --print-env-info \
  --eval-json-out /home/bhuvan/projects/rma/rma_go2_lab/artifacts/isaac_eval/go2_blind_rough_combined_asymppo_steps_v1_candidate_5099/normal_flat_yaw_left.json -->

<!-- # normal_flat_yaw_right
bash \
  scripts/isaaclab_user.sh \
  -p \
  scripts/eval/play_policy.py \
  --task Go2-Blind-Rough-MJLAB-Combined-AsymPPO-Steps-V1 \
  --checkpoint /home/bhuvan/projects/rma/rma_go2_lab/logs/rsl_rl/go2_blind_rough_combined_asymppo_steps_v1/2026-07-16_10-14-29/model_5099.pt \
  --num_envs 16 \
  --nominal-env \
  --terrain-type plane \
  --fixed-command \
  --cmd-vx 0.0 \
  --cmd-vy 0.0 \
  --cmd-yaw -0.35 \
  --max-steps 800 \
  --print-env-info \
  --eval-json-out /home/bhuvan/projects/rma/rma_go2_lab/artifacts/isaac_eval/go2_blind_rough_combined_asymppo_steps_v1_candidate_5099/normal_flat_yaw_right.json -->

<!-- # normal_random_rough_diagonal
bash \
  scripts/isaaclab_user.sh \
  -p \
  scripts/eval/play_policy.py \
  --task Go2-Blind-Rough-MJLAB-Combined-AsymPPO-Steps-V1 \
  --checkpoint /home/bhuvan/projects/rma/rma_go2_lab/logs/rsl_rl/go2_blind_rough_combined_asymppo_steps_v1/2026-07-16_10-14-29/model_5099.pt \
  --num_envs 16 \
  --nominal-env \
  --terrain-type random_rough \
  --fixed-command \
  --cmd-vx 0.3 \
  --cmd-vy 0.12 \
  --cmd-yaw 0.15 \
  --max-steps 800 \
  --print-env-info \
  --eval-json-out /home/bhuvan/projects/rma/rma_go2_lab/artifacts/isaac_eval/go2_blind_rough_combined_asymppo_steps_v1_candidate_5099/normal_random_rough_diagonal.json \
  --terrain-level 4 -->

<!-- # normal_slope_yaw
bash \
  scripts/isaaclab_user.sh \
  -p \
  scripts/eval/play_policy.py \
  --task Go2-Blind-Rough-MJLAB-Combined-AsymPPO-Steps-V1 \
  --checkpoint /home/bhuvan/projects/rma/rma_go2_lab/logs/rsl_rl/go2_blind_rough_combined_asymppo_steps_v1/2026-07-16_10-14-29/model_5099.pt \
  --num_envs 16 \
  --nominal-env \
  --terrain-type hf_pyramid_slope \
  --fixed-command \
  --cmd-vx 0.25 \
  --cmd-vy 0.0 \
  --cmd-yaw 0.2 \
  --max-steps 800 \
  --print-env-info \
  --eval-json-out /home/bhuvan/projects/rma/rma_go2_lab/artifacts/isaac_eval/go2_blind_rough_combined_asymppo_steps_v1_candidate_5099/normal_slope_yaw.json \
  --terrain-level 4 -->
