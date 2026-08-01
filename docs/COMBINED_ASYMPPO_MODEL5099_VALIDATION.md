# Combined AsymPPO Model 5099 Validation Plan

This document freezes the validation plan for:

```text
logs/rsl_rl/go2_blind_rough_combined_asymppo_steps_v1/2026-07-16_10-14-29/model_5099.pt
```

The goal is to convert the current qualitative observation, "it looks robust in
MuJoCo on rough/random/stairs terrain", into a repeatable validation record.

## Candidate Identity

| Item | Value |
| --- | --- |
| Candidate label | `go2_blind_rough_combined_asymppo_steps_v1_candidate_5099` |
| IsaacLab task | `Go2-Blind-Rough-MJLAB-Combined-AsymPPO-Steps-V1` |
| Checkpoint | `logs/rsl_rl/go2_blind_rough_combined_asymppo_steps_v1/2026-07-16_10-14-29/model_5099.pt` |
| Training env truth | `logs/rsl_rl/go2_blind_rough_combined_asymppo_steps_v1/2026-07-16_10-14-29/params/env.yaml` |
| Training agent truth | `logs/rsl_rl/go2_blind_rough_combined_asymppo_steps_v1/2026-07-16_10-14-29/params/agent.yaml` |
| Bundle | `rma_go2_lab/policies/exported/go2_blind_rough_combined_asymppo_steps_v1_candidate/` |
| Validation manifest | `configs/validation/go2_combined_asymppo_steps_v1_model5099.json` |

## Deployment Command Contract

The exported deployment bundle must allow the command range learned during the
combined AsymPPO pipeline:

| Command | Range |
| --- | --- |
| `lin_vel_x` | `[-0.8, 0.8]` |
| `lin_vel_y` | `[-0.3, 0.3]` |
| `ang_vel_z` | `[-0.6, 0.6]` |

If MuJoCo or hardware only responds to forward motion, first inspect the
candidate deploy YAML/JSON command ranges. If those are correct, the remaining
limitation is likely the terminal keyboard teleop path, not the policy.

## Stage 1: Bundle And Runtime Gate

Run this before quantitative terrain testing:

```bash
cd /home/bhuvan/projects/rma/rma_go2_lab

/home/bhuvan/miniconda3/envs/rma-mujoco/bin/python \
  scripts/deploy/run_deployment_validation_gate.py \
  --bundle-dir rma_go2_lab/policies/exported/go2_blind_rough_combined_asymppo_steps_v1_candidate \
  --expected-policy-obs-dim 45 \
  --expected-history-length 100 \
  --expected-action-dim 12 \
  --python-exe /home/bhuvan/miniconda3/envs/rma-mujoco/bin/python
```

Expected report:

```text
artifacts/deployment_validation/go2_blind_rough_combined_asymppo_steps_v1_candidate/validation_gate_report.json
```

Current status:

```text
pass
```

This gate checks bundle structure, TorchScript forward execution, checkpoint vs
TorchScript vs ONNX golden-inference parity, MuJoCo runtime preflight, and the
active Unitree MJLAB FSM runtime wiring.

## Stage 2: Quantitative MuJoCo Terrain Suites

Run the non-GUI MuJoCo validation batch:

```bash
cd /home/bhuvan/projects/rma/rma_go2_lab

bash scripts/deploy/run_combined_asymppo_model5099_mujoco_validation.sh
```

This runs:

| Suite | Purpose |
| --- | --- |
| `mujoco_nominal_v1` | Flat nominal and basic dynamics perturbation sanity check |
| `mujoco_disturb_v2_moderate` | Command transitions and moderate push disturbances |
| `mujoco_rough_v1` | Rough blocks, hfields, stairs, and moderate terrain perturbations |
| `mujoco_rough_v2_hard` | Harder technical terrain recipes |

Summary outputs:

```text
artifacts/mujoco_eval/go2_blind_rough_combined_asymppo_steps_v1_candidate/<suite>/suite_summary.csv
artifacts/mujoco_eval/go2_blind_rough_combined_asymppo_steps_v1_candidate/<suite>/suite_summary.json
artifacts/mujoco_eval/go2_blind_rough_combined_asymppo_steps_v1_candidate/model5099_validation_report.md
```

Current metric contract:

```text
mujoco_runtime_named_obs_v2
```

Older MuJoCo summaries without this metric contract are stale. They were
generated before the runtime metric extraction used named observation slices, so
tracking and projected-gravity tilt metrics should not be interpreted. Rerun the
suite before using the numeric pass/fail result.

Primary metrics to inspect:

| Metric | Why it matters |
| --- | --- |
| `successful_rollouts / rollout_count` | Survival and runtime stability |
| `base_height_mean` | Collapse/crouch detection |
| `base_tilt_projected_gravity_xy_mean` | Rollover/tilt stability |
| `vel_err_step_mean` | Linear command tracking |
| `yaw_err_step_mean` | Yaw command tracking |
| `joint_vel_abs_mean` | Violent joint motion indicator |
| `ctrl_abs_mean` | Control effort / saturation proxy |
| `first_event_step_mean` | Disturbance recovery timing |

Acceptance thresholds are recorded in:

```text
configs/validation/go2_combined_asymppo_steps_v1_model5099.json
```

## Stage 3: Visual MuJoCo Smoke Tests

Use these only for visual satisfaction and qualitative debugging. They do not
replace the quantitative suites above.

Terminal 1:

```bash
cd /home/bhuvan/projects/rma/rma_go2_lab
bash scripts/deploy/run_unitree_mjlab_sim_deploy.sh activate combined
bash scripts/deploy/run_unitree_mjlab_sim_deploy.sh controller
```

Terminal 2, flat:

```bash
cd /home/bhuvan/projects/rma/rma_go2_lab
bash scripts/deploy/run_unitree_mjlab_sim_deploy.sh sim
```

Terminal 2, stock Unitree terrain:

```bash
cd /home/bhuvan/projects/rma/rma_go2_lab
bash scripts/deploy/run_unitree_mjlab_sim_deploy.sh sim-unitree-terrain
```

Terminal 2, generated recipe:

```bash
cd /home/bhuvan/projects/rma/rma_go2_lab
/home/bhuvan/miniconda3/envs/rma-mujoco/bin/python \
  scripts/deploy/materialize_unitree_mujoco_terrain_recipes.py

bash scripts/deploy/run_unitree_mjlab_sim_deploy.sh sim \
  reference_repos/unitree_mujoco/unitree_robots/go2/scene_eval_forward_technical_e.xml
```

## Stage 4: IsaacSim Parity Commands

Do not launch these from Codex. Run them manually when a GUI/IsaacSim session is
intended.

Generate the current command list from the manifest:

```bash
cd /home/bhuvan/projects/rma/rma_go2_lab
python scripts/eval/print_combined_asymppo_model5099_isaac_commands.py
```

The generated commands write JSON metrics under:

```text
artifacts/isaac_eval/go2_blind_rough_combined_asymppo_steps_v1_candidate_5099/
```

After all generated IsaacSim runs are complete, reduce the JSONs into one
pass/fail report:

```bash
cd /home/bhuvan/projects/rma/rma_go2_lab
python scripts/eval/check_combined_asymppo_model5099_isaac_eval.py
```

Expected report:

```text
artifacts/isaac_eval/go2_blind_rough_combined_asymppo_steps_v1_candidate_5099/model5099_isaac_parity_report.md
```

The IsaacSim matrix has two separate purposes:

- forward-aligned terrain traversal on random rough, slope, stairs, and
  inverted stairs,
- command-authority checks for lateral, yaw, and diagonal commands.

Do not interpret the stair cases as full omni stair validation. The generated
stair terrains are aligned to forward travel, so stair cases intentionally use
forward commands. Omni authority is checked separately on flat and selected
rough/slope cases.

## Stage 5: Hardware Bring-Up Gate

Hardware should only start after MuJoCo and IsaacSim parity are acceptable.

Read-only DDS probe:

```bash
cd /home/bhuvan/projects/rma/rma_go2_lab
conda activate go2-hw

bash scripts/deploy/run_unitree_mjlab_sim_deploy.sh dds-probe ethernet enp0s31f6
```

Activate combined runtime:

```bash
cd /home/bhuvan/projects/rma/rma_go2_lab
bash scripts/deploy/run_unitree_mjlab_sim_deploy.sh activate combined
```

Hardware controller:

```bash
cd /home/bhuvan/projects/rma/rma_go2_lab
conda activate go2-hw

bash scripts/deploy/run_unitree_mjlab_sim_deploy.sh hardware ethernet enp0s31f6
```

Recommended logged hardware sequence:

Terminal 1, read-only LowState/LowCmd monitor:

```bash
cd /home/bhuvan/projects/rma/rma_go2_lab
conda activate go2-hw

bash scripts/deploy/run_unitree_mjlab_sim_deploy.sh \
  monitor ethernet combined_model5099_flat_bringup enp0s31f6
```

Terminal 2, controller:

```bash
cd /home/bhuvan/projects/rma/rma_go2_lab
conda activate go2-hw

bash scripts/deploy/run_unitree_mjlab_sim_deploy.sh hardware ethernet enp0s31f6
```

After the robot returns to Passive, analyze the monitor capture:

```bash
cd /home/bhuvan/projects/rma/rma_go2_lab
/home/bhuvan/miniconda3/envs/rma-mujoco/bin/python \
  scripts/deploy/analyze_go2_realtime_monitor.py \
  --jsonl artifacts/go2_realtime_monitor/<capture>.jsonl

/home/bhuvan/miniconda3/envs/rma-mujoco/bin/python \
  scripts/deploy/analyze_go2_leg_mirror_pairs.py \
  --jsonl artifacts/go2_realtime_monitor/<capture>.jsonl
```

Bring-up order:

1. Passive / lowstate connection only.
2. FixStand only.
3. Low-speed forward command on flat ground.
4. Low-speed yaw and lateral commands on flat ground.
5. Small obstacle/step only after flat/omni passes.

Do not treat violent rollover recovery in sim as acceptable hardware behavior.
Hardware should transition to damping/passive if it leaves the safe locomotion
envelope.

## Success Record

After running the matrix, update this section with:

| Backend | Evidence | Status |
| --- | --- | --- |
| Deployment gate | `artifacts/deployment_validation/.../validation_gate_report.json` | Passed |
| MuJoCo nominal | `artifacts/mujoco_eval/.../mujoco_nominal_v1/suite_summary.csv` and `model5099_validation_report.md` | Passed |
| MuJoCo disturbance | `artifacts/mujoco_eval/.../mujoco_disturb_v2_moderate/suite_summary.csv` and `model5099_validation_report.md` | Passed |
| MuJoCo rough | `artifacts/mujoco_eval/.../mujoco_rough_v1/suite_summary.csv` and `model5099_validation_report.md` | Passed |
| MuJoCo hard rough/stairs | `artifacts/mujoco_eval/.../mujoco_rough_v2_hard/suite_summary.csv` and `model5099_validation_report.md` | Passed |
| Isaac flat command authority | `artifacts/isaac_eval/.../model5099_isaac_parity_report.md` | Passed |
| Isaac random rough/slope | `artifacts/isaac_eval/.../model5099_isaac_parity_report.md` | Passed |
| Isaac stairs up/down | `artifacts/isaac_eval/.../model5099_isaac_parity_report.md` | Passed |
| Hardware bring-up | read-only -> stance -> low speed -> terrain | Pending |
