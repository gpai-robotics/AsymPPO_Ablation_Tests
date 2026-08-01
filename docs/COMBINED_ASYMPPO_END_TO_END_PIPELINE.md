# Combined AsymPPO End-to-End Pipeline

This document records the final combined AsymPPO locomotion pipeline as an
engineering reference. It is separate from the older outline in
`docs/Final_asymppo.md` and should be treated as the concrete implementation
record for the staged blind rough/stair locomotion branch.

The core result of this branch is a deployable blind locomotion pipeline that:

- keeps the actor deployment contract compatible with the validated Go2 MJLAB
  AsymPPO path,
- trains from flat locomotion to rough terrain and then stairs,
- uses privileged information only for the critic during training,
- learns robust rough-terrain locomotion under broad dynamics randomization,
- learns stair climbing from proprioception and history without actor-side
  terrain sensing,
- remains compatible with the Isaac Sim -> MuJoCo/FSM -> hardware deployment
  workflow used by the validated AsymPPO candidate.

## Source Of Truth For Training Configs

For any trained policy, the saved run YAML is the source of truth:

```text
logs/rsl_rl/<experiment>/<run>/params/env.yaml
logs/rsl_rl/<experiment>/<run>/params/agent.yaml
```

The Python config files describe the current code path and may evolve after a
run. When documenting a specific checkpoint, use the YAML saved next to that
checkpoint, not the current Python config.

The concrete combined checkpoint discussed here is:

```text
logs/rsl_rl/go2_blind_rough_combined_asymppo_steps_v1/2026-07-16_10-14-29/model_5099.pt
```

Its saved YAML lineage is:

| Stage | Run YAML |
| --- | --- |
| Stage 1 flat prior | `logs/rsl_rl/go2_combined_flat_mjlab_prior_v1/2026-07-02_10-59-02/params/{env,agent}.yaml` |
| Stage 2 rough warm-start | `logs/rsl_rl/go2_blind_rough_combined_asymppo_rough_v1/2026-07-07_17-11-12/params/{env,agent}.yaml` |
| Stage 3 stairs checkpoint | `logs/rsl_rl/go2_blind_rough_combined_asymppo_steps_v1/2026-07-16_10-14-29/params/{env,agent}.yaml` |

The original validated AsymPPO hardware checkpoint YAML is not currently present
under the old `go2_blind_rough_asymppo_mjlab_v1` log path in this checkout. Do
not reconstruct its exact training numbers from memory or current cfg scripts.
Treat that path as validated through its exported bundle and deployment record
until its original YAML is restored.

## Executive Summary

The combined AsymPPO branch is not an RMA teacher-student pipeline. It is a
blind asymmetric PPO pipeline:

```text
actor: deployable proprioception + command + last action + history
critic: actor inputs + privileged terrain/dynamics/state
```

Training is staged:

```text
Stage 1: flat MJLAB prior
Stage 2: rough/slopes AsymPPO
Stage 3: stairs-only fine-tune
```

The design preserves the parts that made the validated Go2 AsymPPO branch work
in sim2real:

- no actor-side `base_lin_vel`,
- no actor-side terrain height scan,
- 100-step proprioceptive history,
- MJLAB-compatible action and observation contract,
- moderate actuator gain randomization,
- push and COM disturbances,
- MuJoCo/FSM deployment validation before hardware.

The branch adds the parts that helped solve blind stair behavior in the Stage 3
YAML-backed run:

- staged rough-to-stair transfer,
- `stable_progress` reward,
- `adaptive_swing_recovery` reward,
- stairs-only Stage 3 terrain distribution.

## Source Files

| Component | Path |
| --- | --- |
| Flat Stage 1 env | `rma_go2_lab/envs/priors/combined_flat_mjlab_prior_cfg.py` |
| Flat Stage 1 runner | `rma_go2_lab/models/priors/combined_flat_mjlab_prior_runner_cfg.py` |
| Rough/stair base rewards and randomization | `rma_go2_lab/envs/combined_asymppo/rough_base_cfg.py` |
| Rough/stair history config | `rma_go2_lab/envs/combined_asymppo/rough_history_base_cfg.py` |
| Rough/stair privileged critic config | `rma_go2_lab/envs/combined_asymppo/rough_privileged_history_cfg.py` |
| Rough/stair omni command config | `rma_go2_lab/envs/combined_asymppo/rough_omni_cfg.py` |
| Rough Stage 2 final env | `rma_go2_lab/envs/teacher/combined_rough_blind_mjlab_asymppo_cfg.py` |
| Stair Stage 3 final env | `rma_go2_lab/envs/teacher/combined_steps_blind_rough_mjlab_asymppo_cfg.py` |
| Rough Stage 2 runner | `rma_go2_lab/models/teacher/combined_rough_ppo_mjlab_asymppo_cfg.py` |
| Stair Stage 3 runner | `rma_go2_lab/models/teacher/combined_steps_ppo_mjlab_asymppo_cfg.py` |
| Stage checkpoint resolver | `rma_go2_lab/models/teacher/combined_stage_checkpoints.py` |
| Temporal actor-critic | `rma_go2_lab/models/combined_asymppo/history_actor_critic.py` |
| Policy config | `rma_go2_lab/models/combined_asymppo/policy_cfg.py` |
| MJLAB observation contract | `rma_go2_lab/envs/mjlab_contract.py` |

## Task Names

| Stage | Task | Experiment |
| --- | --- | --- |
| Stage 1 | `Go2-Combined-Flat-MJLAB-Prior-V1` | `go2_combined_flat_mjlab_prior_v1` |
| Stage 2 | `Go2-Blind-Rough-MJLAB-Combined-AsymPPO-Rough-V1` | `go2_blind_rough_combined_asymppo_rough_v1` |
| Stage 3 | `Go2-Blind-Rough-MJLAB-Combined-AsymPPO-Steps-V1` | `go2_blind_rough_combined_asymppo_steps_v1` |

The validated hardware baseline remains separate:

```text
Go2-Blind-Rough-MJLAB-AsymPPO-V1
```

Do not mix checkpoints, tasks, or deployment claims between the validated branch
and the combined branch without explicitly documenting the handoff.

## Actor Observation Contract

The deployable actor uses the MJLAB-style blind observation contract:

| Term | Source | Notes |
| --- | --- | --- |
| `base_ang_vel` | IMU angular velocity | Scaled in IsaacLab parent task |
| `projected_gravity` | IMU orientation / projected gravity | Gives body attitude without global pose |
| `velocity_commands` | commanded `vx`, `vy`, `yaw_rate` | Runtime command input |
| `joint_pos_rel` | joint position relative to default pose | Deployment-safe |
| `joint_vel_rel` | joint velocity | Deployment-safe |
| `last_action` | previous policy action | Deployment-safe and important for temporal consistency |

The actor explicitly does **not** use:

- base linear velocity,
- terrain height scan,
- depth/camera/lidar,
- privileged terrain parameters,
- randomized dynamics parameters.

The actor also receives a `policy_history` group. In rough/stair stages the
history length is `100`, flattened and consumed by a temporal encoder.

## Critic Privilege

The critic is allowed to see training-only information:

| Group | Purpose |
| --- | --- |
| `critic_privileged` | Adds critic-only base linear velocity through the MJLAB critic contract |
| `dynamics_privileged` | Tracks randomized dynamics/material/mass information |
| `terrain_privileged` | Terrain height scan through critic-side raycast grid |

This is asymmetric PPO. The critic learns with privileged state, but the actor
that is exported and deployed remains blind.

## Model Architecture

The rough/stair stages use `TemporalBlindActorCritic`.

| Parameter | Value |
| --- | --- |
| Actor MLP | `[512, 256, 128]` |
| Critic MLP | `[512, 256, 128]` |
| Activation | `elu` |
| Initial action noise | `0.35` |
| Actor obs normalization | `False` |
| Critic obs normalization | `False` |
| History group | `policy_history` |
| Temporal channels | `[64, 64]` |
| Temporal kernel size | `3` |
| History feature dim | `64` |
| History target dim | `128` |
| History target hidden dims | `[128]` |

Stage 1 flat prior uses the regular RSL-RL actor-critic with the MJLAB flat
prior contract:

| Parameter | Value |
| --- | --- |
| Actor MLP | `[512, 256, 128]` |
| Critic MLP | `[512, 256, 128]` |
| Activation | `elu` |
| Initial action noise | `1.0` |

## PPO Parameters

The values below are from the saved YAML runs listed in the source-of-truth
section.

### Stage 1 Flat Prior

| Parameter | Value |
| --- | --- |
| `num_steps_per_env` | `24` |
| `max_iterations` | `1500` |
| `save_interval` | `50` |
| `value_loss_coef` | `1.0` |
| `clip_param` | `0.2` |
| `entropy_coef` | `0.01` |
| `num_learning_epochs` | `5` |
| `num_mini_batches` | `4` |
| `learning_rate` | `1e-3` |
| `schedule` | `adaptive` |
| `gamma` | `0.99` |
| `lam` | `0.95` |
| `desired_kl` | `0.01` |
| `max_grad_norm` | `1.0` |

### Stage 2 Rough and Stage 3 Stairs

| Parameter | Stage 2 Rough | Stage 3 Stairs |
| --- | --- | --- |
| `num_steps_per_env` | `32` | `32` |
| `max_iterations` | `2000` | `3000` |
| `save_interval` | `50` | `50` |
| `value_loss_coef` | `1.0` | `1.0` |
| `clip_param` | `0.2` | `0.2` |
| `entropy_coef` | `0.002` | `0.002` |
| `num_learning_epochs` | `5` | `5` |
| `num_mini_batches` | `4` | `4` |
| `learning_rate` | `1e-4` | `1e-4` |
| `schedule` | `adaptive` | `adaptive` |
| `gamma` | `0.99` | `0.99` |
| `lam` | `0.95` | `0.95` |
| `desired_kl` | `0.01` | `0.01` |
| `max_grad_norm` | `1.0` | `1.0` |

## Stage 1: Flat MJLAB Prior

Stage 1 trains a clean deployable flat prior under the same MJLAB actor contract
used by the rough and stair stages.

Purpose:

- establish stable deployable flat locomotion,
- avoid using old flat checkpoints with a mismatched actor input contract,
- provide a clean actor warm-start for Stage 2.

Terrain:

```text
plane only
```

Saved YAML run:

```text
logs/rsl_rl/go2_combined_flat_mjlab_prior_v1/2026-07-02_10-59-02
```

Core simulator settings from YAML:

| Parameter | Value |
| --- | --- |
| Number of envs | `4096` |
| Physics dt | `0.005s` |
| Control decimation | `4` |
| Control dt | `0.020s` |
| Episode length | `20s` |
| Terrain type | `plane` |

Command curriculum:

```text
initial: narrow flat omni command range
limit:   vx ±1.0, vy ±0.4, yaw ±1.0
```

Flat prior randomization:

| Parameter | Value |
| --- | --- |
| Static friction | `(0.5, 1.1)` |
| Dynamic friction | `(0.4, 1.0)` |
| Base mass randomization | disabled |
| Base COM randomization | disabled |
| Pushes | disabled |
| Motor strength/gain randomization | disabled |

Important Stage 1 reward weights:

| Reward | Weight |
| --- | --- |
| `track_lin_vel_xy_exp` | `1.5` |
| `track_ang_vel_z_exp` | `0.5` |
| `flat_orientation_l2` | `-2.5` |
| `lin_vel_z_l2` | `-1.0` |
| `ang_vel_xy_l2` | `-0.05` |
| `action_rate_l2` | `-0.003` |
| `dof_torques_l2` | `-2e-4` |
| `dof_acc_l2` | `-5e-7` |
| `feet_air_time` | `0.3` |
| `feet_slide` | `-0.1` |
| `stand_still_joint_deviation` | `-0.35` |
| `stand_still_foot_motion` | `-0.1` |
| `hip_joint_deviation` | `-0.08` |
| `joint_deviation` | `-0.02` |

Command:

```bash
export ISAACLAB_ROOT=/opt/IsaacLab

bash scripts/isaaclab_user.sh -p "$ISAACLAB_ROOT/scripts/reinforcement_learning/rsl_rl/train.py" \
  --task Go2-Combined-Flat-MJLAB-Prior-V1 \
  --headless
```

The Stage 1 checkpoint is passed explicitly into Stage 2:

```bash
export COMBINED_FLAT_PRIOR_CKPT=/path/to/go2_combined_flat_mjlab_prior_v1/<run>/model_<iter>.pt
```

Observed finding:

- The flat policy may look mediocre mid-run after command curriculum expands,
  especially in yaw tracking.
- By the end of training it recovered: full episode length, low base contact,
  reduced feet slide, and usable tracking.
- Do not judge Stage 1 too early if timeout rate is high and base-contact
  termination remains low.

## Stage 2: Rough and Slopes

Stage 2 warm-starts from Stage 1 and trains rough/sloped blind locomotion.

Purpose:

- learn robust proprioceptive rough-terrain locomotion,
- train under wide friction/mass/gain/COM/push randomization,
- preserve the deployment contract before stair specialization.

Saved YAML run used by `model_5099.pt` as actor warm-start:

```text
logs/rsl_rl/go2_blind_rough_combined_asymppo_rough_v1/2026-07-07_17-11-12
```

This run is closer to the validated rough AsymPPO setup than the later
experimental combined-rough cfg. It does **not** include `stable_progress`,
`feet_height_body`, or `adaptive_swing_recovery` in the saved reward YAML.

Terrain distribution:

| Terrain | Proportion |
| --- | --- |
| `random_rough` | `0.20` |
| `hf_pyramid_slope` | `0.10` |
| `hf_pyramid_slope_inv` | `0.10` |
| `pyramid_stairs` | `0.0` |
| `pyramid_stairs_inv` | `0.0` |
| `boxes` | `0.0` |

Core simulator settings from YAML:

| Parameter | Value |
| --- | --- |
| Number of envs | `4096` |
| Physics dt | `0.005s` |
| Control decimation | `4` |
| Control dt | `0.020s` |
| Episode length | `20s` |
| Terrain curriculum | enabled |
| Max init terrain level | `2` |

Command range:

```text
initial: vx ±0.1, vy ±0.1, yaw ±0.1
limit:   vx ±0.8, vy ±0.3, yaw ±0.6
```

Domain randomization:

| Parameter | Value |
| --- | --- |
| Static friction | `(0.1, 2.0)` |
| Dynamic friction | `(0.1, 2.0)` |
| Base mass | `(-2.0, 4.0)` added mass when event exists |
| Motor stiffness scale | `(0.6, 1.4)` |
| Motor damping scale | `(0.6, 1.4)` |
| Push interval | every `6-10s` |
| Push velocity | `x/y ±0.35`, `yaw ±0.4` |
| Base COM x/y | `±0.03m` |
| Base COM z | `±0.01m` |

Important Stage 2 reward weights:

| Reward | Weight |
| --- | --- |
| `track_lin_vel_xy_exp` | `1.5` |
| `track_ang_vel_z_exp` | `0.75` |
| `flat_orientation_l2` | `-1.0` |
| `lin_vel_z_l2` | `-0.1` |
| `ang_vel_xy_l2` | `-0.075` |
| `action_rate_l2` | `-0.001` |
| `dof_torques_l2` | `-5e-5` |
| `dof_acc_l2` | `-1e-7` |
| `dof_pos_limits` | `-0.05` |
| `feet_air_time` | `0.5` |
| `feet_slide` | `-0.05` |
| `stand_still_joint_deviation` | `-0.2` |
| `stand_still_foot_motion` | `-0.05` |
| `hip_joint_deviation` | `-0.1` |
| `air_time_variance` | `-0.05` |

Command:

```bash
export COMBINED_FLAT_PRIOR_CKPT=/path/to/stage1/model_<iter>.pt

bash scripts/isaaclab_user.sh -p "$ISAACLAB_ROOT/scripts/reinforcement_learning/rsl_rl/train.py" \
  --task Go2-Blind-Rough-MJLAB-Combined-AsymPPO-Rough-V1 \
  --headless
```

Observed finding:

- The Stage 2 rough run used 4096 envs and the same broad randomization envelope
  as the later stair stage.
- It deliberately kept stair-specific shaping out of the rough warm-start.
- This matters because the successful `model_5099.pt` lineage learned rough
  robustness first, then added stair-specific recovery in Stage 3.

## Stage 3: Stairs

Stage 3 warm-starts from Stage 2 and specializes the policy on stairs.

Purpose:

- focus the already-rough policy on discontinuous terrain,
- train stair recovery without adding actor-side terrain sensing,
- avoid training from scratch on stairs,
- force the policy to learn proprioceptive recovery rather than merely surviving
  rough terrain.

Saved YAML run:

```text
logs/rsl_rl/go2_blind_rough_combined_asymppo_steps_v1/2026-07-16_10-14-29
```

Terrain distribution:

| Terrain | Proportion |
| --- | --- |
| `random_rough` | `0.0` |
| `hf_pyramid_slope` | `0.0` |
| `hf_pyramid_slope_inv` | `0.0` |
| `pyramid_stairs` | `0.5` |
| `pyramid_stairs_inv` | `0.5` |
| `boxes` | `0.0` |

Stair geometry:

| Parameter | Value |
| --- | --- |
| Step height range | `(0.03, 0.12)` |
| Step width | `0.30m` |
| Platform width | `3.0m` |
| Initial terrain level | `1` |

Core simulator settings from YAML:

| Parameter | Value |
| --- | --- |
| Number of envs | `4096` |
| Physics dt | `0.005s` |
| Control decimation | `4` |
| Control dt | `0.020s` |
| Episode length | `20s` |
| Terrain curriculum | enabled |

Stage 3 modifies the Stage 2 reward setup:

| Reward | Stage 3 value |
| --- | --- |
| `feet_air_time` | `0.5` |
| `lin_vel_z_l2` | `-0.5` |
| `stable_progress` | `0.5` |
| `adaptive_swing_recovery` | `0.25` |
| `air_time_variance` | `-0.05` |

Command:

```bash
export COMBINED_ROUGH_CKPT=/path/to/stage2/model_<iter>.pt

bash scripts/isaaclab_user.sh -p "$ISAACLAB_ROOT/scripts/reinforcement_learning/rsl_rl/train.py" \
  --task Go2-Blind-Rough-MJLAB-Combined-AsymPPO-Steps-V1 \
  --headless
```

Observed finding:

- Early versions with strong `feet_height_body` and `stable_progress` produced
  a galloping behavior that could fail when the hind legs became stuck.
- The `model_5099.pt` YAML-backed stair formulation moved away from fixed
  `feet_height_body` shaping and used `adaptive_swing_recovery` instead.
- The successful behavior is closer to failed-progress recovery: if commanded
  motion is high but actual progress is poor, distal forward motion is only
  useful when paired with upward lift.

## Reward Design

### Tracking Rewards

Tracking rewards keep the policy commandable:

- `track_lin_vel_xy_exp`,
- `track_ang_vel_z_exp`.

For the `model_5099.pt` lineage, Stage 2 keeps linear tracking at `1.5`. The
stair improvement should therefore not be attributed to a stronger rough-stage
linear tracking weight.

### Regularization Rewards

Regularization keeps the policy smooth and deployable:

- `action_rate_l2`,
- `dof_torques_l2`,
- `dof_acc_l2`,
- `dof_pos_limits`,
- `feet_slide`,
- `air_time_variance`,
- stand-still joint and foot-motion penalties.

The rough/stair branch keeps these weaker than the flat prior because excessive
regularization can suppress recovery behavior on difficult terrain.

### `stable_progress`

`stable_progress` rewards forward progress in the commanded direction while
penalizing unstable roll/pitch-rate behavior through an exponential stability
gate.

Conceptually:

```text
progress = projected robot velocity along command direction
stability = exp(-2 * (roll_rate^2 + pitch_rate^2))
reward = max(progress, 0) * stability
```

This reward is morphology-agnostic. It does not depend on a specific leg name,
gait phase, or stair geometry.

### `adaptive_swing_recovery`

`adaptive_swing_recovery` is the stair-stage recovery term.

It is not a stair detector. It is a failed-progress detector:

1. Check whether there is meaningful commanded motion.
2. Compare achieved progress against commanded speed.
3. When progress is poor, inspect distal-body motion relative to the root.
4. Penalize forward distal motion that stays low.
5. Reward forward distal motion paired with upward lift.

This targets the observed failure mode where the robot bulldozes into the stair
or traps a leg instead of lifting and recovering.

The reward is intentionally based on:

- command direction,
- root velocity,
- distal body relative velocity,
- upward component of distal motion.

It does not use:

- stair height,
- terrain class,
- height scan in actor,
- hand-coded leg identity,
- scripted stepping logic.

## Terminations

The combined branch uses stricter rough/stair failure detection than vanilla
flat locomotion:

| Termination | Purpose |
| --- | --- |
| `time_out` | Normal successful episode end |
| `base_contact` | Terminate when base contacts terrain |
| `base_orientation` | Terminate when body orientation exceeds limit |
| `base_height` | Terminate when root clearance above local foot support plane is too low |
| `low_progress` | Terminate when nontrivial command produces insufficient displacement/speed |

The `base_height` term uses the local foot support plane rather than just env
origin height. This is important for steps and slopes because absolute terrain
height changes during valid traversal.

## Why The Pipeline Is Staged

The staged design is not cosmetic. Each stage solves a narrower problem:

| Stage | What it solves | Why it matters |
| --- | --- | --- |
| Flat | Stable deployable action/obs contract | Avoids actor-contract mismatch and unstable rough warm-starts |
| Rough | General rough/slope robustness under randomization | Builds robust body control and recovery before stairs |
| Stairs | Discontinuous terrain specialization | Forces lift/recovery behavior without terrain sensing |

Directly training the combined rough/stair task from scratch was unstable. The
flat -> rough -> stairs sequence proved more reliable and easier to diagnose.

## Important Discoveries

### 1. Do Not Put `base_lin_vel` In The Actor

The deployable actor should not consume base linear velocity. It is not a
reliable hardware-side signal in this deployment stack. Removing it from the
actor while keeping it in the critic was a major part of the successful AsymPPO
deployment path.

### 2. The Critic Can Be Privileged Without Making The Actor Undeployable

The critic can use terrain scans, dynamics randomization state, and base linear
velocity during training. Since the critic is discarded at deployment, these
signals improve training without polluting the hardware actor contract.

### 3. Moderate Gain Randomization Worked Better Than Wide Gain Randomization

The successful range stayed at:

```text
stiffness scale: (0.6, 1.4)
damping scale:   (0.6, 1.4)
```

Wider gain randomization previously hurt terrain progression and produced worse
policies.

### 4. Pushes Are Useful But Should Not Be Overdone

The working rough setup uses interval pushes every `6-10s` with moderate
velocity perturbations. Removing pushes did not improve the failed run. Keeping
pushes preserved recovery pressure.

### 5. Fixed Foot-Height Shaping Can Create Bad Gaits

Strong `feet_height_body` shaping can push the robot toward a galloping style.
That may look dynamic, but it can fail when hind legs get trapped. Stair
behavior improved when recovery was framed around failed progress and adaptive
upward swing instead.

### 6. Stairs Need Their Own Stage

Rough terrain training alone produced robust locomotion, but stair climbing
required a focused stair-only stage. The policy needed direct exposure to
stair/inverted-stair discontinuities after learning rough locomotion.

### 7. MuJoCo/FSM Validation Is Still Required

Isaac Sim success is not enough. The validated deployment workflow requires a
second check through the Unitree MJLAB/FSM runtime before real hardware.

## Evaluation And Validation Pipeline

The intended validation path is:

```text
Isaac Sim visual evaluation
-> Isaac Sim metric evaluation
-> export policy bundle
-> deployment bundle validation
-> MuJoCo/FSM sim2sim
-> low-speed hardware stance/policy takeover
-> controlled hardware locomotion
```

### Isaac Sim Evaluation

Use Isaac Sim to inspect:

- standing posture,
- flat command following,
- rough/sloped terrain survival,
- stair ascent/descent,
- stuck-leg recovery,
- base contact/orientation/height terminations,
- velocity tracking errors,
- terrain-level curriculum progression.

### Export And Bundle Validation

The validated deployment workflow uses an exported policy bundle rather than
directly deploying from a training log directory.

For the validated AsymPPO branch, the tracked source-of-truth bundle is:

```text
rma_go2_lab/policies/exported/go2_blind_rough_asymppo_mjlab_v1_candidate/
```

For the combined branch, the current `model_5099.pt` candidate has been
exported into:

```text
rma_go2_lab/policies/exported/go2_blind_rough_combined_asymppo_steps_v1_candidate/
```

The non-GUI deployment gate passed on 2026-07-20: structural bundle validation,
TorchScript smoke, ONNX C++ inference parity, MuJoCo preflight, and Unitree
MJLAB FSM runtime audit.

### MuJoCo / Unitree MJLAB FSM Validation

The deployment bridge uses the recovered Unitree MJLAB-style two-terminal FSM:

```text
Passive -> FixStand -> Velocity
```

Typical setup:

```bash
bash scripts/deploy/build_unitree_mjlab_runtime.sh all
bash scripts/deploy/run_unitree_mjlab_sim_deploy.sh activate combined
bash scripts/deploy/run_unitree_mjlab_sim_deploy.sh validate
```

Sim/controller flow:

```bash
# terminal 1
bash scripts/deploy/run_unitree_mjlab_sim_deploy.sh sim

# terminal 2
bash scripts/deploy/run_unitree_mjlab_sim_deploy.sh controller
```

The FSM validation is important because it checks the runtime contract that is
closer to hardware:

- observation ordering,
- action scaling,
- joint ordering,
- policy update rate,
- default pose,
- PD command semantics,
- command interface,
- startup state.

### Hardware Deployment

Hardware deployment should only happen after:

- bundle validation passes,
- DDS probe receives `rt/lowstate`,
- MuJoCo/FSM sim works,
- stance-only bring-up is stable,
- operator and physical test area are ready.

Network/DDS checks:

```bash
bash scripts/deploy/run_unitree_mjlab_sim_deploy.sh network-status ethernet
bash scripts/deploy/run_unitree_mjlab_sim_deploy.sh dds-probe ethernet
```

Hardware run:

```bash
bash scripts/deploy/run_unitree_mjlab_sim_deploy.sh hardware ethernet
```

Remote/FSM sequence:

```text
L2 + up -> FixStand
R2 + A  -> Velocity policy control
```

For debugging hardware differences between Go2 units, use:

```bash
scripts/deploy/run_go2_readonly_signature_check.sh capture-blueprint <label> <net_if> 8
scripts/deploy/run_go2_readonly_signature_check.sh capture-dynamic-lowcmd <label> <net_if> 20 0.25
```

## Deployment Contract

Deployment depends on the actor-only contract:

```text
target_joint_position = default_pose + action_scale * action
```

The active AsymPPO deployment bundle records:

- observation order,
- history layout,
- action scale,
- default pose,
- PD gains,
- control timestep,
- joint remapping,
- command mapping.

The combined branch must export the same kind of bundle before it can be called
deployment-ready.

## Current State Of The Work

Completed:

- validated rough Go2 AsymPPO deployment pipeline,
- combined flat prior stage,
- combined rough/slopes stage,
- combined stairs-only stage design,
- combined `model_5099.pt` Isaac Sim nominal visual smoke test on flat, stairs,
  and inverted stairs,
- combined `model_5099.pt` deployment export,
- combined `model_5099.pt` non-GUI deployment gate through MuJoCo preflight and
  Unitree MJLAB FSM runtime audit,
- stair recovery reward design,
- documentation of staged branch contract,
- MuJoCo/FSM runtime recovery for deployment validation.

Demonstrated findings:

- robust rough terrain locomotion is achievable with blind proprioceptive
  history and asymmetric critic privilege,
- wide domain randomization can be tolerated when staged correctly,
- stairs require a focused stage after rough training,
- adaptive recovery shaping is preferable to overly strong fixed foot-height
  shaping,
- the deployment contract must remain clean and blind.

Still required before claiming final hardware deployment for the combined stair
policy:

- run Isaac Sim metric validation beyond the initial visual smoke test,
- run MuJoCo/FSM sim2sim validation,
- run stance-only and low-speed hardware bring-up,
- collect repeated hardware logs if comparing against the validated AsymPPO
  candidate.

## Commands Summary

Stage 1:

```bash
bash scripts/isaaclab_user.sh -p "$ISAACLAB_ROOT/scripts/reinforcement_learning/rsl_rl/train.py" \
  --task Go2-Combined-Flat-MJLAB-Prior-V1 \
  --headless
```

Stage 2:

```bash
export COMBINED_FLAT_PRIOR_CKPT=/path/to/stage1/model_<iter>.pt

bash scripts/isaaclab_user.sh -p "$ISAACLAB_ROOT/scripts/reinforcement_learning/rsl_rl/train.py" \
  --task Go2-Blind-Rough-MJLAB-Combined-AsymPPO-Rough-V1 \
  --headless
```

Stage 3:

```bash
export COMBINED_ROUGH_CKPT=/path/to/stage2/model_<iter>.pt

bash scripts/isaaclab_user.sh -p "$ISAACLAB_ROOT/scripts/reinforcement_learning/rsl_rl/train.py" \
  --task Go2-Blind-Rough-MJLAB-Combined-AsymPPO-Steps-V1 \
  --headless
```

MuJoCo/FSM validation:

```bash
bash scripts/deploy/build_unitree_mjlab_runtime.sh all
bash scripts/deploy/run_unitree_mjlab_sim_deploy.sh activate combined
bash scripts/deploy/run_unitree_mjlab_sim_deploy.sh validate
bash scripts/deploy/run_unitree_mjlab_sim_deploy.sh sim
bash scripts/deploy/run_unitree_mjlab_sim_deploy.sh controller
```

Hardware:

```bash
bash scripts/deploy/run_unitree_mjlab_sim_deploy.sh dds-probe ethernet
bash scripts/deploy/run_unitree_mjlab_sim_deploy.sh hardware ethernet
```

## Paper-Oriented Claim

The strongest defensible claim from this branch is:

> A blind Go2 locomotion policy can acquire robust rough-terrain locomotion and
> stair-climbing behavior through staged asymmetric PPO training, using a
> deployable proprioceptive actor with temporal history and a privileged critic
> during training, without actor-side terrain sensing or runtime stair-specific
> heuristics.

A stronger claim about hardware stair deployment should only be made after the
combined Stage 3 policy passes controlled real Go2 deployment validation.
