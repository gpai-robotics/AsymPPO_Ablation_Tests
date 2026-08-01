# GO2 Blind Student Actuation Diagnosis 2026-05-29

This note records the current diagnosis for the old-robot frozen blind student
deployment, with emphasis on the chain:

`command -> policy action -> joint position targets -> PD tracking -> body motion`

It is intended to answer two questions:

1. Is the deploy/runtime stack obviously misconfigured?
2. Which deployment patches are likely covering for training-side gaps?

## Scope

Runtime examined:

- frozen old-robot deploy config:
  - `reference_repos/unitree_rl_lab/deploy/robots/go2/config/policy/velocity/c1_blind_rough_omni_usable_v1_final/params/deploy_frozen_old_robot_20260528.yaml`
- representative frozen-runtime log:
  - `logs/go2_ctrl/go2_ctrl_20260529_114655.log`

Training artifact examined:

- student policy freeze note:
  - `rma_go2_lab/policies/c1_blind_rough_omni_usable_v1_final.md`
- student env config:
  - `rma_go2_lab/envs/blind/c1_blind_rough_omni_usable_cfg.py`
- inherited blind rough base config:
  - `rma_go2_lab/envs/blind/blind_rough_forward_cfg.py`
- runner config:
  - `rma_go2_lab/models/blind/blind_rough_runner_cfg.py`

## Frozen Deploy Chain

### Commands

From the frozen deploy YAML:

- command deadband: `0.03`
- command slew:
  - `lin_vel_x: 0.4`
  - `lin_vel_y: 0.3`
  - `ang_vel_z: 0.45`
- command ranges:
  - `lin_vel_x: [-0.15, 0.2]`
  - `lin_vel_y: [-0.2, 0.2]`
  - `ang_vel_z: [-1.0, 1.0]`

### Actions

Policy output is turned into joint targets in
`reference_repos/unitree_rl_lab/deploy/include/isaaclab/envs/mdp/actions/joint_actions.h`
as:

- `processed_action = raw_action * scale + offset`
- scale: `0.25` for all 12 joints
- offsets:
  - hips: `+/-0.1`
  - thighs: `0.8` front, `1.0` rear
  - calves: `-1.5`
- clip is effectively non-binding: `[-100, 100]`

In the frozen runtime there is:

- no joint target slew limiting
- no neutral action compensation
- no pose compatibility offset blending

### PD Handoff

From
`reference_repos/unitree_rl_lab/deploy/include/FSM/State_RLBase.h`,
runtime sends:

- `motor_cmd.q = commanded joint position`
- `motor_cmd.kp = joint_stiffness`
- `motor_cmd.kd = joint_damping`

The frozen deploy config uses:

- `Kp = 25.0` for all 12 joints
- `Kd = 0.5` for all 12 joints

Those values are loaded into the env in
`reference_repos/unitree_rl_lab/deploy/include/isaaclab/envs/manager_based_rl_env.h`.

## On-Paper Sim/Deploy Agreement

The obvious actuator settings do match across training/export/deploy:

- deploy YAML:
  - `Kp 25.0`, `Kd 0.5`, action scale `0.25`
- exported deploy metadata:
  - `reference_repos/unitree_rl_lab/deploy/robots/go2/config/policy/velocity/c1_blind_rough_omni_usable_v1_final/params/deploy_config.json`
  - same `joint_stiffness`, `joint_damping`, `default_joint_pos`
- Unitree asset config:
  - `reference_repos/unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py`
  - `UNITREE_GO2_CFG` actuator also uses `stiffness=25.0`, `damping=0.5`
- exported policy YAML:
  - `rma_go2_lab/policies/exported/c1_blind_rough_omni_usable_v1_final/c1_blind_rough_omni_usable_v1_final.deploy.yaml`
  - same action scale `0.25`, same offsets, same nominal gains

Conclusion:

- this does not look like a simple deploy typo such as wrong PD gains or wrong
  action scale
- the mismatch is more likely in how the real robot responds to the target
  stream than in the nominal config values alone

## What the Raw Log Shows

Representative log:

- `logs/go2_ctrl/go2_ctrl_20260529_114655.log`

### Key observation

Small filtered command can coexist with large joint targets and large body
motion.

Examples:

- `11:47:09.043`
  - filtered command:
    - `cmd=[+0.188, +0.000, +0.000]`
  - body state already active:
    - `lin_vel=[+0.030, +0.000, +0.000]`
  - relative joint targets already sizable:
    - FL `[-0.054, +0.003, +0.284]`
    - FR `[+0.027, +0.089, +0.307]`
    - RL `[-0.162, +0.086, +0.234]`
    - RR `[+0.195, +0.026, +0.336]`

- `11:47:10.223`
  - filtered command has decayed back to:
    - `cmd=[+0.024, +0.000, +0.000]`
  - but measured motion is large:
    - `lin_vel=[-0.372, +0.351, +0.000]`
    - `base_ang=[+2.388, +0.466, +0.017]`
  - RL calf relative target is extremely large:
    - RL `[-0.100, -0.118, +0.807]`
  - torque estimate spikes:
    - FL calf `+14.983`
    - RL calf `+8.061`

- `11:47:10.603`
  - filtered command still only:
    - `cmd=[-0.096, +0.000, +0.000]`
  - measured motion:
    - `lin_vel=[-0.461, +0.371, +0.000]`
    - `imu_wz=+0.369`
  - torque estimates are again high:
    - FL calf `+22.617`
    - RL calf `+11.095`

### Interpretation

This pattern strongly suggests:

- command clamping is working
- the policy is still producing assertive joint targets under small commands
- the PD loop is tracking those targets strongly enough to create large body
  motion
- the real robot then couples axes and carries motion beyond the intended
  command

This is consistent with:

- “small command, big movement”
- “release command, robot keeps going”
- “lateral or yaw induces forward/backward leakage”

## Training-Side Assumptions

### Command curriculum

The frozen blind student is not a pure low-speed policy.

From `rma_go2_lab/envs/blind/c1_blind_rough_omni_usable_cfg.py`:

- initial ranges:
  - `lin_vel_x = (-0.1, 0.1)`
  - `lin_vel_y = (-0.1, 0.1)`
  - `ang_vel_z = (-0.1, 0.1)`
- curriculum expands toward limit ranges:
  - `lin_vel_x = (-0.8, 0.8)`
  - `lin_vel_y = (-0.3, 0.3)`
  - `ang_vel_z = (-0.6, 0.6)`

The policy card in
`rma_go2_lab/policies/c1_blind_rough_omni_usable_v1_final.md`
describes that same “usable command envelope”.

So although training begins narrow, the final student is explicitly shaped to
handle much stronger commands than the small frozen deploy test values.

### Action smoothness regularization

From `rma_go2_lab/envs/blind/blind_rough_forward_cfg.py`:

- `track_lin_vel_xy_exp.weight = 1.5`
- `track_ang_vel_z_exp.weight = 0.75`
- `action_rate_l2.weight = -0.001`
- `dof_torques_l2.weight = -5e-5`
- `dof_acc_l2.weight = -1e-7`

Interpretation:

- tracking is strongly rewarded
- action-rate and torque regularization are present, but weak

This makes it plausible that the student learned an aggressive target stream
that is acceptable in sim but too lively on hardware.

### Standstill structure

The student does have some standstill shaping:

- `stand_still_joint_deviation`
- `stand_still_foot_motion`

But both are thresholded around command magnitude `0.15` to `0.2`.

That means:

- very small commands are not aggressively trained as a “hard stop” regime
- release behavior may remain soft if the policy prefers to keep locomotion
  dynamics alive

### Omni command shaping

The student is curriculum-expanded using
`rma_go2_lab/envs/blind/blind_omni_command_curriculums.py`.

This increases command range when tracking reward is good enough. That is a
reasonable design, but it encourages success under a broad command family, not
necessarily crisp stop-and-decouple teleoperation behavior.

## Likely Root Cause

The best current explanation is:

- not a simple command-range bug
- not a simple PD gain typo
- not obvious joystick leakage

Instead:

- the frozen blind student generates joint targets that are too aggressive or
  too coupled for this real robot under the current actuator reality
- the PD controller then honestly enforces those targets
- the real robot responds with overshoot, carryover, and axis coupling

In short:

- the issue likely lives in the `policy target stream x real actuator response`
  mismatch

## What Other Deployments Suggest

The most useful local comparison is:

- `reference_repos/go2-sim2real-deploy/example/go2/low_level/final/go2_policy_walk.py`

That deployment adds runtime protection we do not have in the frozen stack:

- explicit per-step joint target slew limiting:
  - `MAX_STEP_RAD = 0.1`
- optional runtime stiffness shaping / per-leg stiffness logic

That supports the idea that post-policy target shaping is a practical and common
hardware safeguard, even when nominal PD gains look reasonable.

## Training-Side Candidates

Deployment patches we tested earlier map naturally back to training questions:

1. Stop on release
   - stronger zero-command reward structure
   - more command pulse / release curriculum
   - stronger penalty on residual motion when command is near zero

2. Cross-axis leakage
   - explicit penalties on non-commanded planar velocity
   - explicit penalties on translation during pure yaw
   - explicit penalties on yaw during pure translation

3. Action aggressiveness
   - stronger `action_rate_l2`
   - possibly stronger torque or acceleration regularization
   - actuator-aware or target-slew-aware training

4. Real actuator mismatch
   - richer actuator/domain randomization
   - training-time target slew constraint if we decide it is a permanent deploy
     mechanism

## Current Bottom Line

The current evidence supports the following working diagnosis:

- frozen deploy command filtering is not the main problem
- frozen PD gains are not obviously wrong on paper
- the blind student likely produces a target stream that is too aggressive for
  the real old robot, causing the PD loop to drive large coupled motion from
  small commands

This is why:

- command caps alone do not reliably slow the robot
- stop behavior remains poor
- lateral/yaw commands leak into forward/backward motion

That makes this a good candidate for both:

- deployment-side post-policy safeguards
- training-side sim2real cleanup focused on stop behavior, axis decoupling, and
  action smoothness
