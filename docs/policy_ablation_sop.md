# ABLATION TEST SOP

## Pipeline Pending for Ablation Tests
- `/home/bhuvan/projects/rma/rma_go2_lab/docs/COMBINED_ASYMPPO_END_TO_END_PIPELINE.md`

## Evaluation Harness

Use `scripts/eval/play_policy.py` as the primary Isaac Sim robustness harness. Environment overrides are centralized in `_apply_cli_environment_overrides(env_cfg, args_cli)`, so new ablation knobs should be added there instead of patching task config files for one-off tests.

Example:
```bash
env TERM=xterm /opt/IsaacLab/isaaclab.sh -p scripts/eval/play_policy.py \
  --task Go2-Blind-Rough-MJLAB-Combined-AsymPPO-Rough-V1 \
  --checkpoint /path/to/model.pt \
  --num_envs 16 \
  --terrain-type pyramid_stairs \
  --terrain-level 4 \
  --step-height 0.17 \1
  --step-width 0.30 \
  --static-friction 0.6 \
  --dynamic-friction 0.5 \
  --added-mass 2.0 \
  --motor-stiffness-scale 0.8 \
  --motor-damping-scale 0.8 \
  --cmd-vx 0.3 \
  --cmd-vy 0.0 \
  --cmd-yaw 0.0 \
  --print-env-info
```

Supported override categories:
- Terrain family and terrain level.
- Stair geometry: fixed/ranged step height, fixed step width, fixed platform width.
- Random rough height amplitude and noise step.
- Boxes/random-grid height and grid width.
- Slope coefficient for slope terrains.
- Static/dynamic friction and restitution ranges.
- Added base mass, base COM offset, global motor stiffness/damping scales.
- Push disabling, fixed push interval, and velocity-push magnitudes.
- Observation noise scaling where the task exposes additive noise terms.
- Fixed velocity commands during rollout.
- Reset spawn yaw/roll/pitch/height where `reset_base.pose_range` supports those keys.

Known unsupported CLI requests are intentionally reported at startup:
- `--num-steps`: IsaacLab pyramid stairs derive step count from terrain size, step width, and platform width.
- `--roughness-frequency`, `--roughness-scale`, `--box-spacing`, `--obstacle-density`: not exposed by the active IsaacLab terrain configs.
- `--push-force`, `--push-torque`: current Go2 tasks use velocity pushes, not force/torque impulse pushes.
- `--obs-delay`, `--action-delay`: current env stack has no generic delay-buffer config.
- Per-joint-group gain overrides warn unless a future task exposes matching actuator randomization events.

## LEVEL 1: PIPELINE DESIGN VALIDATION

### PURPOSE
- Is the staging of pipeline even necessary?
- What if same performance can be achieved via alternate minimal design?

### Current Active Design
```Flat Training -> Rough Training -> Stairs Training```

### Tests (Re-training Required)
- Flat -> Stairs
- Flat -> Rough + Stairs
- All terrains combined in a single training environment.



## LEVEL 2: ENVIRONMENT VALIDATION

### PURPOSE
- Which environment components are actually required?

### Current Components
- friction DR *
- COM DR *
- motor DR *
- pushes
- terrain curriculum
- command curriculum
- privileged critic *
- temporal history *

### Ablation Tests
- Test by removing only one component at a time.
- Start be testing ablations of important components first (marked with *), then move to standard ones.


## LEVEL 3: REWARD VALIDATION

### PURPOSE
- Which rewards drive the behaviour and are major contributors in locomotion success?
- Which rewards are absolutely necessary and cannot be compromised upon?

### Current Components
- track_lin_vel_xy_exp
- track_ang_vel_z_exp
- action_rate_l2
- dof_torques_l2
- dof_acc_l2
- dof_pos_limits
- feet_slide
- air_time_variance *
- stand-still joint and foot-motion penalties *
- stable_progress *
- adaptive_swing_recovery *

### Ablation Tests
- Test by removing only one component at a time. 
- Start be testing ablations of novel rewards first (marked with *), then move to standard rewards.
