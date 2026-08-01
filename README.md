# RMA-Go2 Lab

Main RL research and deployment codebase for Unitree Go2 rough-terrain
locomotion.

This repository is not only the AsymPPO deployment repo. It contains the
project history from flat-prior training, blind rough baselines, teacher lines,
adaptation/RMA-style attempts, cross-simulator validation, and the currently
successful AsymPPO sim2real path.

Use this README as the orientation map. Use the linked docs for exact command
sheets and deeper branch-specific details.

## Current Mental Model

```text
flat prior
  -> blind rough baselines
  -> privileged teachers
  -> adaptation / RMA-style branches
  -> blind AsymPPO MJLAB branch
  -> Isaac Sim validation
  -> MuJoCo / Unitree MJLAB FSM validation
  -> Go2 hardware deployment
```

The current deployable winner is:

```text
go2_blind_rough_asymppo_mjlab_v1_candidate
```

That does not erase the older baselines. Those baselines explain how we got
here, what failed, and what should not be mixed back into the active deployment
path without a deliberate experiment.

## Baseline And Branch Status

| Line | Status | What It Proved | Current Role |
| --- | --- | --- | --- |
| Flat prior | Complete | Stable nominal locomotion backbone | Root warm-start reference |
| Blind rough B1 scratch | Complete / frozen in project history | Rough blind locomotion from scratch is possible | Baseline lower bound |
| Blind rough B2 warm-start | Complete / frozen in project history | Flat warm-start improves blind rough training | Canonical warm-start comparison |
| Blind rough B3 warm-start + imitation | Complete / frozen in project history | Imitation helps some robustness axes, not all | Comparison artifact |
| Privileged teacher lineage | Complete / auditable, not final deployment path | Privileged training can provide upper-bound/reference behavior | Teacher/reference history |
| Early adaptation V0/V1/V2 | Complete / archived | History-to-latent adaptation can be trained | Historical adaptation baseline |
| Adapt-V3 / C2 | Partially complete, not sim2real winner | Structured `mu / pi / phi` path exists but remains deployment-risky | Research branch, not active hardware path |
| Blind MJLAB AsymPPO | Complete / active | Blind history AsymPPO survived Isaac Sim, MuJoCo FSM, and real Go2 bring-up | Current deployment mainline |
| Unitree MJLAB runtime bridge | Recovered / active | Two-terminal sim and hardware FSM deployment works again | Required for active deployment |

Important artifact distinction:

- Freeze notes and branch documentation are retained in this repo.
- Some older raw `.pt` checkpoints may live only in local logs, archives, or
  external storage after cleanup.
- Exported deployment bundles are runtime artifacts, not normal source files.
  They should be shared through an explicit release/package when someone needs
  the exact trained policy.
- The original AsymPPO raw PPO checkpoint
  `go2_blind_rough_asymppo_mjlab_v1/2026-06-04_10-31-03/model_1999.pt` is
  currently unavailable. Any TorchScript `.pt` or ONNX file in
  `rma_go2_lab/policies/exported/` is a runtime export, not a PPO training
  checkpoint.

## Active Deployable Candidate

Historical validated runtime bundle, if restored locally or downloaded from a
release package:

```text
rma_go2_lab/policies/exported/go2_blind_rough_asymppo_mjlab_v1_candidate/
```

Required files:

```text
bundle_manifest.json
export_request.json
go2_blind_rough_asymppo_mjlab_v1_candidate.torchscript.pt
go2_blind_rough_asymppo_mjlab_v1_candidate.onnx
go2_blind_rough_asymppo_mjlab_v1_candidate.export_metadata.json
go2_blind_rough_asymppo_mjlab_v1_candidate.deploy_config.json
go2_blind_rough_asymppo_mjlab_v1_candidate.deploy.yaml
```

Deployment contract:

| Field | Value |
| --- | --- |
| Task | `Go2-Blind-Rough-MJLAB-AsymPPO-V1` |
| Policy kind | blind history policy |
| Policy obs dim | `45` |
| History length | `100` |
| History obs dim | `4500` |
| Action dim | `12` |
| Control dt | `0.020 s` |
| Control rate | `50 Hz` |
| Action type | joint position targets |
| Action scale | `0.25` |
| Deployment gains | `kp=25`, `kd=0.5` |
| Actor base linear velocity | not used |

The raw PPO checkpoint would be useful for retraining or re-exporting, but the
validated source checkpoint is currently unavailable. The exported bundle is
what the Unitree MJLAB runtime uses, but it is treated as an external artifact
rather than source-controlled code.

## Code Layout

Core environment families:

```text
rma_go2_lab/envs/priors/
  Flat prior environments.

rma_go2_lab/envs/blind/
  Earlier blind rough and C1-style baselines.

rma_go2_lab/envs/teacher/
  Privileged teacher and active AsymPPO rough environment.

rma_go2_lab/envs/adaptation/
  Adaptation / RMA-style student environments.

rma_go2_lab/envs/asymppo/
  Validated MJLAB AsymPPO V1 rough/history building blocks.

rma_go2_lab/envs/combined_asymppo/
  Experimental combined AsymPPO building blocks for Go2 + Trakr-inspired
  rough/stair baselines.
```

Core model families:

```text
rma_go2_lab/models/priors/
  Flat prior PPO configs.

rma_go2_lab/models/blind/
  Blind baseline actor-critic and PPO variants.

rma_go2_lab/models/teacher/
  Privileged teacher and active AsymPPO PPO config.

rma_go2_lab/models/adaptation/
  RMA-style adaptation modules and PPO variants.

rma_go2_lab/models/asymppo/
  Validated MJLAB AsymPPO V1 blind-history actor-critic.

rma_go2_lab/models/combined_asymppo/
  Experimental combined AsymPPO temporal actor-critic and policy config.
```

Deployment and validation:

```text
scripts/eval/
  IsaacLab playback, gait checks, clips, and evaluation helpers.

scripts/deploy/
  Policy export, bundle validation, MuJoCo sim2sim, Unitree MJLAB runtime,
  hardware monitor, and real Go2 deployment utilities.

patches/unitree_rl_mjlab/
  Repo-owned patches required to recover the local Unitree MJLAB runtime.
```

Historical material:

```text
docs/archive/
rma_go2_lab/archive/
scripts/archive/
rma_go2_lab/policies/archive/
```

Treat archived material as reference knowledge. Do not silently mix it into the
active launch path.

## First-Party Code Vs Dependencies

This repo owns the IsaacLab task code, model configs, deployment scripts,
validation scripts, docs, and patches.

This repo does not normally own generated training/deployment artifacts:
checkpoints, TensorBoard logs, MuJoCo result dumps, ONNX exports, TorchScript
exports, and zipped deployment bundles should stay local or be distributed as
explicit release/package artifacts.

It does not own the large simulator/runtime mirrors under `reference_repos/`.
That directory is ignored by git and should be treated as a local dependency
workspace. The active AsymPPO deployment path currently depends on:

```text
reference_repos/unitree_rl_mjlab/
reference_repos/unitree_sdk2/
```

MuJoCo validation and terrain tooling may also require:

```text
reference_repos/mujoco_menagerie/
reference_repos/unitree_mujoco/
reference_repos/mjlab/
```

Python hardware diagnostics require `unitree_sdk2py` to be importable either
from the selected hardware Python environment or from a local
`reference_repos/sim2real_unitree_sdk2py/` mirror.

The detailed dependency boundary is documented in:

```text
docs/DEPENDENCIES.md
```

## First-Time Setup

This repo assumes Isaac Sim / IsaacLab is installed separately.

Recommended workstation convention:

```bash
export REPO=$(pwd)
export ISAACLAB_ROOT=/opt/IsaacLab
export GO2_USD_PATH=/path/to/go2.usd
export MUJOCO_PYTHON=python
export GO2_HW_PYTHON=python3
```

Use the repo wrapper instead of calling IsaacLab directly on shared
workstations:

```bash
bash scripts/isaaclab_user.sh -p scripts/check_tasks.py
```

The wrapper isolates Kit/IsaacLab logs and cache paths per user. This avoids
the shared `/opt` permission failures we saw when multiple users used the same
IsaacLab installation.

If `torch` fails inside IsaacLab, fix the IsaacLab installation. Do not install
an arbitrary PyPI `torch` into Isaac Sim Python; it can break CUDA/NCCL library
compatibility.

`GO2_USD_PATH` is required when the local Go2 USD is not available at the
historical workstation path baked into older configs. New machines should set
it explicitly rather than editing environment source files.

## Training Pipeline

Use IsaacLab's RSL-RL train script for this private codebase:

```bash
bash scripts/isaaclab_user.sh -p "$ISAACLAB_ROOT/scripts/reinforcement_learning/rsl_rl/train.py" \
  --task <TASK_NAME> \
  --headless
```

Important registered tasks include:

| Task | Purpose | Status |
| --- | --- | --- |
| `RMA-Go2-Flat` | Flat locomotion prior | Complete historical root |
| `RMA-Go2-Blind-Baseline-Rough` | B1 scratch rough baseline | Complete historical baseline |
| `RMA-Go2-Blind-Baseline-Rough-WarmStart` | B2 warm-start baseline | Complete historical baseline |
| `RMA-Go2-Blind-Baseline-Rough-WarmStart-Imitation` | B3 warm-start + imitation baseline | Complete historical baseline |
| `RMA-Go2-Privileged-Teacher-Rough-V3` | Privileged teacher reference | Historical teacher path |
| `RMA-Go2-Privileged-Teacher-Rough-V4` | Later privileged teacher reference | Historical teacher path |
| `Go2-Blind-Rough-MJLAB-AsymPPO-V1` | Active blind AsymPPO rough policy | Current sim2real winner |
| `Go2-Combined-Flat-MJLAB-Prior-V1` | Combined branch flat deployable prior | Active experiment stage 1 |
| `Go2-Blind-Rough-MJLAB-Combined-AsymPPO-Rough-V1` | Combined branch rough/slopes stage | Active experiment stage 2 |
| `Go2-Blind-Rough-MJLAB-Combined-AsymPPO-Steps-V1` | Combined branch stair fine-tune | Active experiment stage 3 |

Before retraining `Go2-Blind-Rough-MJLAB-AsymPPO-V1`, check:

```text
rma_go2_lab/models/teacher/ppo_mjlab_asymppo_cfg.py
```

The warm-start source is currently controlled by `MJLAB_FLAT_PRIOR_CKPT`. Update
that constant if the historical flat-prior checkpoint is not available on the
machine.

## Evaluation Pipeline

IsaacLab visual playback:

```bash
bash scripts/isaaclab_user.sh -p scripts/eval/play_policy.py \
  --task Go2-Blind-Rough-MJLAB-AsymPPO-V1 \
  --checkpoint /path/to/model_<iter>.pt \
  --num_envs 16 \
  --teleop-keyboard
```

This playback command requires a raw RSL-RL PPO checkpoint. The tracked
TorchScript export inside the bundle is for deployment inference and cannot be
used with this IsaacLab checkpoint playback command.

For the full command sheet, use:

```text
docs/RUN_COMMANDS.md
```

For cross-simulator validation and parity rules, use:

```text
docs/CROSS_SIMULATOR_VALIDATION_CONTRACT.md
docs/MUJOCO_SIM2SIM_VALIDATION.md
docs/DEPLOYMENT_VALIDATION_GATE.md
```

## Exporting A Candidate

Do not deploy directly from an untracked training log path. Export a candidate
bundle first:

```bash
bash scripts/isaaclab_user.sh -p scripts/deploy/export_policy.py \
  --policy-name go2_blind_rough_asymppo_mjlab_v1_candidate \
  --checkpoint /path/to/model_<iter>.pt \
  --task Go2-Blind-Rough-MJLAB-AsymPPO-V1 \
  --phase asymppo \
  --bundle-dir rma_go2_lab/policies/exported/go2_blind_rough_asymppo_mjlab_v1_candidate \
  --policy-kind blind_history_policy \
  --observation-groups policy,policy_history \
  --format torchscript \
  --format onnx \
  --policy-history-length 100
```

Then validate the bundle before deployment:

```bash
python scripts/deploy/run_deployment_validation_gate.py \
  --bundle-dir rma_go2_lab/policies/exported/go2_blind_rough_asymppo_mjlab_v1_candidate \
  --expected-policy-obs-dim 45 \
  --expected-history-length 100
```

## Unitree MJLAB Deployment Runtime

The active sim2real path depends on a local `unitree_rl_mjlab` mirror:

```text
reference_repos/unitree_rl_mjlab/
```

That mirror is not the main source tree. Our required runtime changes are
tracked as:

```text
patches/unitree_rl_mjlab/go2_scripted_controller.patch
```

If `reference_repos/unitree_rl_mjlab` is restored from upstream or deleted
during cleanup, reapply the patch:

```bash
cd reference_repos/unitree_rl_mjlab
git apply ../../patches/unitree_rl_mjlab/go2_scripted_controller.patch
cd ../..
```

The patch restores:

- scripted two-terminal sim/controller startup
- keyboard teleop in the controller terminal
- `unitree_mujoco --use_joystick=0/1`
- ONNX Runtime dynamic batch shape handling

Build, activate, and validate:

```bash
bash scripts/deploy/build_unitree_mjlab_runtime.sh all
bash scripts/deploy/run_unitree_mjlab_sim_deploy.sh activate
bash scripts/deploy/run_unitree_mjlab_sim_deploy.sh validate
```

Strict runtime gate:

```bash
python scripts/deploy/validate_unitree_mjlab_go2_fsm_runtime.py \
  --strict-fixstand-gains \
  --json-out artifacts/deployment_validation/active_unitree_mjlab_fsm_runtime_audit.json
```

## Unitree MJLAB Sim2Sim

Run in two terminals.

Terminal 1:

```bash
bash scripts/deploy/run_unitree_mjlab_sim_deploy.sh controller
```

Terminal 2:

```bash
bash scripts/deploy/run_unitree_mjlab_sim_deploy.sh sim
```

Expected controller signal:

```text
Using scripted controller: FixStand@0.1s Velocity@4.0s repeat=12.0s command=[0.3, 0.0, 0.0]
FSM: Change state from Passive to FixStand
FSM: Change state from FixStand to Velocity
```

Keyboard teleop is available in the controller terminal:

```text
W/S or arrow up/down: vx
A/D or arrow left/right: vy
Q/E: yaw
```

## Real Go2 Deployment

Real hardware deployment sends low-level commands. Use it only with the robot in
a safe area, remote in hand, and an operator ready to stop.

Ethernet preflight:

```bash
bash scripts/deploy/run_unitree_mjlab_sim_deploy.sh network-status ethernet
bash scripts/deploy/run_unitree_mjlab_sim_deploy.sh dds-probe ethernet
```

Start hardware deployment:

```bash
bash scripts/deploy/run_unitree_mjlab_sim_deploy.sh hardware ethernet
```

Remote FSM controls:

```text
L2 + up: enter FixStand
R2 + A : enter Velocity policy control
L2 + B : return to Passive / stop
```

Wi-Fi deployment is supported only when the laptop and Go2 dongle are on the
same WLAN and DDS multicast/client-to-client traffic is allowed:

```bash
bash scripts/deploy/run_unitree_mjlab_sim_deploy.sh network-status wifi
bash scripts/deploy/run_unitree_mjlab_sim_deploy.sh dds-probe wifi
bash scripts/deploy/run_unitree_mjlab_sim_deploy.sh hardware wifi
```

Guest networks and many phone hotspots commonly block DDS multicast.

## Hardware Monitoring

Read-only realtime telemetry:

```bash
bash scripts/deploy/run_unitree_mjlab_sim_deploy.sh monitor ethernet asymppo_walk
```

Stationary observation audit:

```bash
python scripts/deploy/run_go2_stationary_observation_audit.py \
  --net-if <robot-facing-interface> \
  --duration 15
```

These tools are intended for diagnosis and should remain separate from policy
actuation unless a script explicitly states otherwise.

## Reading Order For Interns

Use this order to understand the repo without getting lost in old branches:

1. `README.md`
2. `docs/PROJECT_PROGRESS_TIMELINE.md`
3. `docs/DEPENDENCIES.md`
4. `TRAINING_SOP.md`
5. `rma_go2_lab/policies/README.md`
6. `docs/ACTIVE_PATHS.md`
7. `docs/RUN_COMMANDS.md`
8. `docs/ACTIVE_ASYMPPO_DEPLOYMENT_ARTIFACTS.md`
9. `docs/UNITREE_MJLAB_RUNTIME_BUILD.md`
10. `docs/ASYMPPO_SIM2REAL_SUCCESS_RETROSPECTIVE_20260612.md`

For adaptation/RMA-style work, then read:

```text
docs/ADAPT_V3_ACTIVE_ROADMAP.md
docs/C2_ADAPT_V3_BASELINE_CANDIDATE_CARD.md
docs/C2_STRUCTURED_OFFLINE_FINAL_BASELINE_CARD.md
```

## Recovery Checklist

If the active deployment pipeline breaks:

1. Confirm the AsymPPO exported bundle exists.
2. Confirm `reference_repos/unitree_rl_mjlab/` exists.
3. Reapply `patches/unitree_rl_mjlab/go2_scripted_controller.patch` if the mirror was restored.
4. Run `bash scripts/deploy/build_unitree_mjlab_runtime.sh all`.
5. Run `bash scripts/deploy/run_unitree_mjlab_sim_deploy.sh activate`.
6. Run `bash scripts/deploy/run_unitree_mjlab_sim_deploy.sh validate`.
7. Run `validate_unitree_mjlab_go2_fsm_runtime.py --strict-fixstand-gains`.
8. Verify two-terminal `controller` + `sim` reaches `FixStand -> Velocity`.
9. Verify hardware `dds-probe` receives `rt/lowstate` before running `hardware`.

## Policy Rule

This repo is allowed to preserve many research branches. It should not expose
all of them as active deployment paths.

When adding a new branch, document:

- what changed
- what checkpoint or bundle is the source of truth
- whether it is complete, incomplete, archived, or active
- which previous branch it replaces, if any

If a command is not represented in `docs/RUN_COMMANDS.md`, `TRAINING_SOP.md`,
or a candidate-specific card, treat it as legacy until proven otherwise.
