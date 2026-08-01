# Active AsymPPO Deployment Artifacts

This document is the recovery checklist for the current working AsymPPO
sim2real path.

## Active Policy

- policy name: `go2_blind_rough_asymppo_mjlab_v1_candidate`
- task: `Go2-Blind-Rough-MJLAB-AsymPPO-V1`
- policy kind: `blind_history_policy`
- actor inputs:
  - `policy_obs`: 45
  - `policy_history`: 4500
- action dim: 12
- control rate: 50 Hz
- deployment gains: `kp=25`, `kd=0.5`
- action scale: `0.25`
- source PPO checkpoint:
  `/home/bhuvan/tools/IsaacLab/logs/rsl_rl/go2_blind_rough_asymppo_mjlab_v1/2026-06-04_10-31-03/model_1999.pt`
- source PPO checkpoint status: unavailable / not tracked

Tracked bundle:

```text
rma_go2_lab/policies/exported/go2_blind_rough_asymppo_mjlab_v1_candidate/
```

Required files:

```text
bundle_manifest.json
go2_blind_rough_asymppo_mjlab_v1_candidate.torchscript.pt
go2_blind_rough_asymppo_mjlab_v1_candidate.onnx
go2_blind_rough_asymppo_mjlab_v1_candidate.export_metadata.json
go2_blind_rough_asymppo_mjlab_v1_candidate.deploy_config.json
go2_blind_rough_asymppo_mjlab_v1_candidate.deploy.yaml
```

Known-good hashes:

```text
454afb874ef196cf7187775729165a30224daf0e99dc89ffc5ce709e02ec7f55  torchscript.pt
64839040e43fc19b2f158a953a124d852e8e2f98a93e11641628837f213f0ee7  onnx
1d164e393091b752c7c7157d86e58f2ee24b2faeb64825039af64827954f5015  export_metadata.json
ef9017b32b4fc46031bdc515d0adbe2b50fa8ee287447c65dd7529acd254bab4  deploy_config.json
aa64f2b7ec2cd6a92fb140da61bc5c3e10c5c61851bedf9420cdde99f0f02c41  deploy.yaml
de517ff7b2dc0b90edc5f4a95c2fade3b119946200dd594135afaf74db4b2fd1  bundle_manifest.json
570f2445b70c1f170227d17036323c9a3c28c29d3265deaab5bc768f7ac3c932  export_request.json
```

The original PPO `.pt` checkpoint would be useful for further training or
re-exporting, but it is currently unavailable and is not tracked here. The
deployable source of truth is the tracked bundle.

Do not confuse these two artifacts:

- `go2_blind_rough_asymppo_mjlab_v1_candidate.torchscript.pt`
  - deployment TorchScript export
  - valid for runtime inference
  - not valid for resuming PPO training
- `model_1999.pt`
  - original RSL-RL PPO checkpoint
  - validated source checkpoint at export time
  - currently lost/unavailable

## Runtime Patch

`reference_repos/unitree_rl_mjlab` is a local dependency mirror and is not
tracked as source. The local runtime delta is tracked here:

```text
patches/unitree_rl_mjlab/go2_scripted_controller.patch
```

This patch restores:

- two-terminal scripted sim/controller startup
- keyboard teleop in the controller terminal
- `unitree_mujoco --use_joystick=0/1`
- ONNX Runtime dynamic batch dimension handling

Reapply after restoring a fresh `unitree_rl_mjlab` mirror:

```bash
cd reference_repos/unitree_rl_mjlab
git apply ../../patches/unitree_rl_mjlab/go2_scripted_controller.patch
cd ../..
```

## Recovery Commands

```bash
bash scripts/deploy/build_unitree_mjlab_runtime.sh all
bash scripts/deploy/run_unitree_mjlab_sim_deploy.sh activate
bash scripts/deploy/run_unitree_mjlab_sim_deploy.sh validate
```

Then run in two terminals:

```bash
bash scripts/deploy/run_unitree_mjlab_sim_deploy.sh controller
```

```bash
bash scripts/deploy/run_unitree_mjlab_sim_deploy.sh sim
```

Expected controller signal:

```text
Using scripted controller: FixStand@0.1s Velocity@4.0s repeat=12.0s command=[0.3, 0.0, 0.0]
FSM: Change state from Passive to FixStand
FSM: Change state from FixStand to Velocity
```
