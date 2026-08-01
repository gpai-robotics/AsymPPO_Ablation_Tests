# Unitree RL MJLab Deployment Audit

This audit covers the checked-in Go2 path under
`reference_repos/unitree_rl_mjlab`. It distinguishes mechanisms present in
the repository from assumptions about policies deployed by other users.

## Bottom Line

The repository does not hide a command filter, command slew limiter, action
smoother, latency estimator, or online adaptation module. Its low-effort
deployment path comes mainly from using one consistent MuJoCo model and one
generated policy contract from training through simulation deployment and
real deployment.

The supplied Go2 deployment contract most closely matches the flat velocity
policy. The rough task gives the actor a terrain height scan, while the
checked-in hardware `deploy.yaml` does not provide a height scan. A claim that
a policy was deployed directly from this repository therefore needs the exact
task, checkpoint, ONNX input names, and deployment YAML before it can be used
as evidence for blind rough-terrain transfer.

## Checked-In Go2 Runtime

| Property | Repository behavior |
|---|---|
| Policy format | ONNX, exported whenever a checkpoint is saved |
| Policy rate | 50 Hz (`step_dt: 0.02`) |
| Physics rate in training | 200 Hz (`timestep: 0.005`, decimation 4) |
| Action | Joint position target |
| Action transform | `q_target = default_joint_pos + 0.25 * action` |
| Policy gains | Hip/thigh `Kp=20, Kd=1`; calf `Kp=40, Kd=2` |
| Command input | Raw gamepad axes, range-clamped |
| Command deadband | None in the deployment runtime |
| Command slew/filter | None |
| Action smoothing | None |
| Action clipping | Not enabled in the supplied Go2 YAML |
| Observation history | One frame in the supplied deployment YAML |
| Safety | Passive damping state, interpolated fixed-stand state, orientation fallback, LowState timeout fallback |

The joint order is explicitly remapped from policy order to Unitree SDK order:

```text
policy: FL, FR, RL, RR
SDK:    FR, FL, RR, RL
map:    [3,4,5, 0,1,2, 9,10,11, 6,7,8]
```

## What Actually Reduces Deployment Work

1. **Training and deployment share actuator semantics.** The Go2 model uses
   MuJoCo built-in position actuators with the same nominal gains, effort
   limits, default pose, action scale, and 50 Hz policy interval written into
   the hardware deployment contract.
2. **The runtime reconstructs observations directly from the same contract.**
   Joint ordering, offsets, scales, history length, and ONNX input names are
   explicit instead of being manually reimplemented for each policy.
3. **The bring-up state machine is conservative.** It enters passive damping,
   interpolates from the current pose through a crouch into the policy stance,
   and only then enables the policy.
4. **The deployable policy is simple.** The supplied contract uses current
   angular velocity, projected gravity, command, gait phase, joint position,
   joint velocity, and previous action. There is no estimator or long history
   buffer whose layout can diverge across runtimes.
5. **Simulation deployment uses the same C++ controller.** The hardware binary
   is first connected to `unitree_mujoco`, so the DDS, FSM, ONNX, observation,
   action, gain, and joystick code is exercised before connecting to hardware.

## Training Robustness Present in MJLab

The base velocity task trains with:

- IMU angular velocity noise: `[-0.2, 0.2]`
- Projected-gravity noise: `[-0.05, 0.05]`
- Joint-position noise: `[-0.01, 0.01]` rad
- Joint-velocity noise: `[-1.5, 1.5]` rad/s
- Foot friction: `[0.3, 1.6]`
- Encoder bias: `[-0.015, 0.015]` rad
- Base COM offsets: `[-0.05, 0.05]` m on each axis
- Pushes every `5-6` seconds, implemented as randomized base velocity changes
- Rough-terrain curriculum and command curriculum

It does not configure actuator command delay or observation delay in this Go2
task, even though the underlying MJLab framework supports actuator lag.

## Important Rough/Flat Distinction

The rough Go2 actor includes a `1.6 m x 1.0 m` terrain height scan at `0.1 m`
resolution. That observation is not available in the checked-in C++ hardware
runtime. The flat task removes this scan, and its remaining actor observations
match the supplied Go2 deployment YAML.

Before comparing our blind rough policy with an MJLab deployment report, obtain:

- the exact task ID (`Unitree-Go2-Flat` or `Unitree-Go2-Rough`);
- the ONNX model input names and shapes;
- the deployment YAML used on hardware;
- the robot XML and actuator configuration revision;
- whether the deployed actor had terrain scan input removed or distilled.

## Comparison With Our Current Asymmetric-PPO Candidate

Our candidate already has several equivalent safeguards:

- 50 Hz policy execution;
- explicit hardware/policy joint remapping;
- position-target action reconstruction;
- startup pose interpolation;
- passive/damping exit behavior;
- observation and action contract auditing.

The major differences are:

- our actor consumes 100 frames of 45-dimensional history, while their supplied
  deployment contract consumes a single frame plus a two-dimensional gait phase;
- our exported gains are uniform `Kp=25, Kd=0.5`, while their Go2 gains are
  joint-class-specific `20/1`, `20/1`, and `40/2`;
- our policy was trained in Isaac Sim and is validated in MuJoCo before hardware,
  while their policy is trained and simulation-deployed in MuJoCo;
- our runtime includes a `0.05` joystick deadband, which is a benign input
  cleanup rather than the source of the earlier heavy command smoothing.

## Recommended Next Steps

1. Verify TorchScript-versus-ONNX action parity on recorded observations.
2. Run the staged exact hardware controller against `unitree_mujoco`:

   ```bash
   # Terminal 1
   bash scripts/deploy/run_unitree_mjlab_sim_deploy.sh sim

   # Terminal 2
   bash scripts/deploy/run_unitree_mjlab_sim_deploy.sh controller
   ```

   Use the simulated gamepad/FSM sequence: `LT + Up` for FixStand, then
   `RT + A` for Velocity.
3. Add a deployment-contract test that checks joint map, default pose, gains,
   action scale, observation order, history layout, and policy rate before any
   motor command is published.
4. Evaluate the current policy with both its trained gains and the MJLab
   joint-class gains in simulation. Do not change real-hardware gains until the
   cross-simulator result is understood.
5. Keep command slew disabled for policy evaluation. Retain only command range
   clipping and a small joystick deadband so policy recovery ability is measured
   rather than hidden.

## Source Files

- `reference_repos/unitree_rl_mjlab/deploy/robots/go2/src/State_RLBase.cpp`
- `reference_repos/unitree_rl_mjlab/deploy/robots/go2/config/config.yaml`
- `reference_repos/unitree_rl_mjlab/deploy/robots/go2/config/policy/velocity/v0/params/deploy.yaml`
- `reference_repos/unitree_rl_mjlab/deploy/include/FSM/State_RLBase.h`
- `reference_repos/unitree_rl_mjlab/deploy/include/FSM/State_FixStand.h`
- `reference_repos/unitree_rl_mjlab/deploy/include/isaaclab/envs/mdp/observations/observations.h`
- `reference_repos/unitree_rl_mjlab/deploy/include/isaaclab/envs/mdp/actions/joint_actions.h`
- `reference_repos/unitree_rl_mjlab/src/tasks/velocity/velocity_env_cfg.py`
- `reference_repos/unitree_rl_mjlab/src/tasks/velocity/config/go2/env_cfgs.py`
- `reference_repos/unitree_rl_mjlab/src/assets/robots/unitree_go2/go2_constants.py`
