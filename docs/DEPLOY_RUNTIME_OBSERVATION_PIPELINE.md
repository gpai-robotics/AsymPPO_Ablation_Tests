# Go2 Deploy Runtime Observation Pipeline

This document explains the real deployment-time data path for the current Go2
low-level RL controller in this repo.

It is intentionally specific to the active `unitree_rl_lab` deploy stack and the
current Go2 bundle format used by `go2_ctrl`.

## Why This Matters

The deployed policy does **not** read raw Unitree DDS packets directly.

Instead, the runtime:

1. reads a subset of robot state
2. reconstructs the IsaacLab deploy observation contract
3. filters a few important channels
4. stacks history
5. runs the ONNX policy
6. converts the policy action into joint position targets
7. republishes those targets at a higher low-level rate

Understanding this exact path is important because many deployment issues come
from:

- observation mismatch vs training
- delayed or filtered commands
- imperfect base linear velocity estimation
- hidden memory from history and `last_action`
- confusion about what runs at 50 Hz vs 1 kHz

## Runtime Loops

There are two important loops in the current hardware controller.

### 1. FSM / low-level publish loop

This loop runs at **1 kHz**.

Source:
- `reference_repos/unitree_rl_lab/deploy/include/FSM/CtrlFSM.h`

Key fact:
- `dt = 0.001`

Each tick does:

1. `lowstate->update()`
2. current FSM state's `run()`
3. `lowcmd->unlockAndPublish()`

So low-level motor commands are published every **1 ms**.

### 2. Policy loop

This loop runs at **50 Hz**.

Sources:
- `reference_repos/unitree_rl_lab/deploy/include/FSM/State_RLBase.h`
- `reference_repos/unitree_rl_lab/deploy/robots/go2/config/policy/velocity/c1_blind_rough_omni_usable_v1_final/params/deploy.yaml`

Key facts:
- `step_dt: 0.02`
- the policy thread sleeps for `env->step_dt`

Each policy step does:

1. `robot->update()`
2. `observation_manager->compute()`
3. `alg->act(obs)` on the ONNX policy
4. `action_manager->process_action(action)`

So the policy updates every **20 ms**, while low-level commands are published
every **1 ms**.

## High-Level Data Flow

Per policy step, the data path is:

1. Unitree lowstate is refreshed by the 1 kHz FSM loop
2. `robot->update()` copies selected state into the deploy articulation
3. observation terms are built from that articulation state
4. history buffers are updated
5. ONNX inference runs
6. action post-processing converts network output to joint targets
7. those joint targets are held and republished by the 1 kHz loop

## What Comes From Unitree Lowstate

Source:
- `reference_repos/unitree_rl_lab/deploy/include/unitree_articulation.h`

The runtime reads only a subset of the robot state directly from Unitree
lowstate:

- IMU gyroscope -> `root_ang_vel_b`
- IMU quaternion -> `root_quat_w`
- gravity projected into the body frame -> `projected_gravity_b`
- motor positions `q` -> `joint_pos`
- motor velocities `dq` -> `joint_vel`

This means the policy is not consuming raw DDS messages directly. It consumes
the reconstructed deploy articulation state.

## Base Linear Velocity

Base linear velocity is special because it is **not** present in `rt/lowstate`
in the way the policy wants it.

The deploy stack supports two sources:

- `zero`
- `odometry`

Source:
- `reference_repos/unitree_rl_lab/deploy/include/unitree_articulation.h`

### Zero mode

If the source is `zero`, then:

- `root_lin_vel_b = [0, 0, 0]`

### Odometry mode

If the source is `odometry`, then the deploy runtime reads body-frame linear
velocity from the UDP odometry bridge.

Current odometry endpoint:

- `udp://127.0.0.1:5560`

Source:
- `reference_repos/unitree_rl_lab/deploy/robots/go2/main.cpp`

### Odometry filtering and clamping

Before feeding odometry to the policy, the runtime applies:

1. sanity clamp
2. low-pass filtering

Current clamps:

- `x` clamped to `[-0.8, 0.8]`
- `y` clamped to `[-0.6, 0.6]`
- `z` clamped to `[-0.08, 0.08]`

Current low-pass filter:

- `alpha_xy = 0.2`
- `alpha_z = 0.05`

Update equation:

```text
filtered = filtered + alpha * (raw - filtered)
```

Important:

- this filter is applied when `robot->update()` runs
- in the current stack that means it is effectively updated at the **50 Hz
  policy rate**

## Observation Contract

The policy consumes the deploy observation contract assembled in:

- `reference_repos/unitree_rl_lab/deploy/include/isaaclab/envs/mdp/observations/observations.h`

For the current Go2 history policy bundle, `policy_obs` is **48D**:

- `base_lin_vel`: 3
- `base_ang_vel`: 3
- `projected_gravity`: 3
- `velocity_commands`: 3
- `joint_pos_rel`: 12
- `joint_vel_rel`: 12
- `last_action`: 12

This ordering is mirrored in the bundle export metadata and deploy config.

## Observation Terms

### `base_ang_vel`

Source:
- IMU gyroscope

Transform:
- copied directly from lowstate into `root_ang_vel_b`

Filtering:
- none in this deploy path

### `projected_gravity`

Source:
- IMU quaternion

Transform:
- rotate world gravity into the body frame using the conjugate quaternion

Filtering:
- none in this deploy path

### `joint_pos_rel`

Source:
- motor joint positions

Transform:

```text
joint_pos_rel = joint_pos - default_joint_pos
```

Filtering:
- none

### `joint_vel_rel`

Source:
- motor joint velocities

Transform:
- passed through directly

Filtering:
- none

### `last_action`

Source:
- previous raw action stored in the action manager

Transform:
- direct feedback of the last network action

Filtering:
- none

### `velocity_commands`

This is one of the most important transformed observations.

The policy does **not** see raw joystick axes directly.

Source:
- wireless remote joystick axes

Pipeline:

1. read raw stick values
2. apply deadband
3. apply optional directional scaling
4. clamp to configured command ranges
5. apply extra deploy-time shaping
6. slew-limit to produce `filtered_velocity_command`
7. feed the filtered command to the policy

Source:
- `reference_repos/unitree_rl_lab/deploy/include/isaaclab/envs/mdp/observations/observations.h`

## Current Command Configuration

From the current deploy bundle:

- deadband: `0.05`
- configured ranges:
  - `lin_vel_x`: `[-0.3, 0.3]`
  - `lin_vel_y`: `[-0.18, 0.18]`
  - `ang_vel_z`: `[-0.4, 0.4]`
- configured slew rates:
  - `lin_vel_x`: `0.4`
  - `lin_vel_y`: `0.3`
  - `ang_vel_z`: `0.45`

## Extra Deploy-Time Command Shaping

The current deploy code adds extra shaping for translational commands:

- forward command scale: `0.5`
- hard cap on `vx`: `0.15`
- `vx` ramp-up cap: `0.15`
- `vx` ramp-down cap: `0.45`
- `vy` ramp-down cap: `0.45`
- release snap threshold:
  - `vx`: `0.03`
  - `vy`: `0.03`

This logic exists specifically in deploy code and is not just read from the
bundle.

## Slew-Rate Math

Per policy step:

```text
delta_limit = slew_limit * step_dt
```

With `step_dt = 0.02`, the current effective command rate limits are:

- `vx` ramp-up:
  - `0.15 * 0.02 = 0.003` per step
- `vx` ramp-down:
  - `0.45 * 0.02 = 0.009` per step
- `vy` ramp-down:
  - `0.45 * 0.02 = 0.009` per step
- `wz`:
  - `0.45 * 0.02 = 0.009` per step

So the policy sees a **stateful filtered command**, not an instantaneous target.

## History Contract

The current exported bundle is a **blind history policy**.

Sources:
- `reference_repos/unitree_rl_lab/deploy/robots/go2/config/policy/velocity/c1_blind_rough_omni_usable_v1_final/params/deploy.yaml`
- `reference_repos/unitree_rl_lab/deploy/include/isaaclab/manager/observation_manager.h`
- `reference_repos/unitree_rl_lab/deploy/include/isaaclab/manager/manager_term_cfg.h`

### History length

Each `policy_history` term uses:

- `history_length: 100`

At 50 Hz:

- `100 * 0.02 = 2.0 s`

So the policy can see **2 seconds of history** for each term.

### What is stored in history

The same 48D observation channels are stored in history:

- `base_lin_vel`
- `base_ang_vel`
- `projected_gravity`
- `velocity_commands`
- `joint_pos_rel`
- `joint_vel_rel`
- `last_action`

### History layout

The runtime uses `use_gym_history: true`.

That means flattening is done **term by term**, not timestep by timestep.

So the final history vector is arranged like:

1. all `base_lin_vel` history, oldest -> newest
2. all `base_ang_vel` history, oldest -> newest
3. all `projected_gravity` history, oldest -> newest
4. all `velocity_commands` history, oldest -> newest
5. all `joint_pos_rel` history, oldest -> newest
6. all `joint_vel_rel` history, oldest -> newest
7. all `last_action` history, oldest -> newest

### Current deploy-side memory flush patch

The current runtime includes a deploy-side patch that clears some translational
memory when commands are near zero.

Current rule:

- if `policy_history.velocity_commands` is near zero in all 3 axes
- then history for:
  - `velocity_commands`
  - `last_action`
  is reset to zeros

Current threshold:

- `abs(cmd_x) < 0.03`
- `abs(cmd_y) < 0.03`
- `abs(cmd_z) < 0.03`

This patch was added to reduce long command tails from the 2-second history
window.

## Action Path

After ONNX inference, the runtime does not send raw policy output directly to
the robot.

Source:
- `reference_repos/unitree_rl_lab/deploy/include/isaaclab/envs/mdp/actions/joint_actions.h`

Per action dimension:

1. optional raw bias subtraction
2. multiply by scale
3. add offset
4. clip

For the current Go2 bundle:

- action scale: `0.25` for all 12 joints
- action offset: default standing joint pose
- clip: effectively inactive at `[-100, 100]`

So, approximately:

```text
processed_joint_target = 0.25 * raw_action + default_pose
```

## Startup Blending

There is one important interpolation stage at startup.

Source:
- `reference_repos/unitree_rl_lab/deploy/robots/go2/src/State_RLBase.cpp`

When the RL state is entered:

- the current measured joint positions are captured
- for the first `1.5 s`, commanded joint positions are blended from the current
  robot pose to the policy target

Blend equation:

```text
alpha = clamp((now - blend_start) / 1.5, 0, 1)
q_cmd = q_initial + alpha * (q_policy - q_initial)
```

This is **startup interpolation only**.

## Is There Interpolation Between Policy Steps?

This is an important point.

### Yes

There is interpolation during the **1.5 s startup blend** described above.

### No

There is **no steady-state action interpolation** between consecutive 50 Hz
policy outputs.

Instead:

- the policy outputs a new joint target every `20 ms`
- that target is then **held**
- the 1 kHz FSM loop republishes that same target every `1 ms` until the next
  policy step updates it

So the deploy stack is a **sample-and-hold controller at 50 Hz**, republished at
1 kHz.

## Publish Behavior

Low-level publishing happens in:

- `reference_repos/unitree_rl_lab/deploy/include/FSM/FSMState.h`
- `reference_repos/unitree_rl_lab/deploy/robots/go2/include/Types.h`

Every 1 kHz FSM tick:

- current motor command message is CRC-updated
- it is published on `rt/lowcmd`

There is no additional motor-target interpolation layer in `LowCmd_t`.

## Signal Summary

| Signal | Source | Runtime Rate | Transform | Filtered? | Used by Policy |
| --- | --- | --- | --- | --- | --- |
| `base_ang_vel` | IMU gyro | 50 Hz snapshot of latest lowstate | copy | no | yes |
| `projected_gravity` | IMU quat | 50 Hz snapshot | quaternion rotate gravity | no | yes |
| `joint_pos_rel` | motor `q` | 50 Hz snapshot | subtract default pose | no | yes |
| `joint_vel_rel` | motor `dq` | 50 Hz snapshot | copy | no | yes |
| `base_lin_vel` | odometry UDP or zero | 50 Hz snapshot | clamp + low-pass or zero-fill | yes | yes |
| `velocity_commands` | joystick axes | 50 Hz | deadband + clamp + slew + release snap | yes | yes |
| `last_action` | previous raw action | 50 Hz | copy | no | yes |
| `policy_history` | observation manager buffers | 50 Hz | oldest->newest stacking | implicit memory | yes |
| joint command publish | processed action | 1 kHz | hold latest target | no interpolation | sent to robot |

## Practical Interpretation

The deploy controller has four main memory / smoothing sources:

1. filtered velocity command
2. filtered odometry-based base linear velocity
3. `last_action`
4. 100-step policy history

The deploy controller does **not** smooth motion primarily by interpolating
joint targets between policy steps.

Instead, smoothness comes from:

- command filtering
- history and temporal dependence inside the policy
- action scaling
- holding each 50 Hz target while republishing it at 1 kHz

## Why Deployment Bugs Can Feel Strange

Because the policy sees a transformed observation contract rather than raw robot
packets, behavior can go wrong even when DDS communication itself is fine.

Typical failure modes include:

- sticky stop behavior from filtered commands and history
- poor braking from missing or noisy `base_lin_vel`
- mismatch between hardware command semantics and training command semantics
- action bias from pose or neutral-action mismatch

In other words:

- the policy is not simply "reading the robot and acting"
- it is acting on a reconstructed deploy-time interface

That interface must be correct in:

- timing
- filtering
- ordering
- units
- offsets
- history semantics

for the deployed behavior to match training expectations.

