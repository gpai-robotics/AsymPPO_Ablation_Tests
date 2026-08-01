# AsymPPO Sim2Real Success Retrospective

Date: 2026-06-12

## Result

`go2_blind_rough_asymppo_mjlab_v1_candidate` is the first policy in this
repository that has simultaneously shown:

- strong rough-terrain behavior in Isaac Sim
- stable independent MuJoCo behavior
- matched Isaac/MuJoCo contract validation
- clean real Go2 locomotion
- no obvious recurrence of the earlier FR-side gait asymmetry during the
  operator's initial hardware evaluation
- no need for deployment-side command slew, directional scaling, neutral-action
  compensation, or actor-side body-velocity estimation

This is a successful baseline, not proof that every rough-terrain or hardware
failure mode is solved.

## Frozen Identity

- task: `Go2-Blind-Rough-MJLAB-AsymPPO-V1`
- source run:
  `/home/bhuvan/tools/IsaacLab/logs/rsl_rl/go2_blind_rough_asymppo_mjlab_v1/2026-06-04_10-31-03`
- checkpoint: `model_1999.pt`
- checkpoint SHA-256:
  `1f826597f29f94d25dd27dc6c48ea0079af8c5ae808edaa85a7af3eb493ca0a8`
- exported bundle:
  `rma_go2_lab/policies/exported/go2_blind_rough_asymppo_mjlab_v1_candidate`
- ONNX SHA-256:
  `64839040e43fc19b2f158a953a124d852e8e2f98a93e11641628837f213f0ee7`
- TorchScript SHA-256:
  `454afb874ef196cf7187775729165a30224daf0e99dc89ffc5ce709e02ec7f55`

The captured training parameters under the source run's `params/` directory
are the historical source of truth. Mutable Python configs may change later.

## What The C1 A/B Test Proved

The frozen C1 policy was staged into the current `unitree_rl_mjlab` FSM with:

- the same MuJoCo simulator
- the same Go2 model
- the same `Passive -> FixStand -> Velocity` FSM
- the same policy-to-SDK joint map
- the same 50 Hz policy loop
- the same nominal policy gains, `kp=25`, `kd=0.5`
- the C1 checkpoint's required 100-frame history layout

C1 reproduced its earlier failure behavior in simulation. AsymPPO remained
stable in that same stack.

Therefore the following are not sufficient explanations for the improvement:

- switching to the new FSM alone
- fixing the joint map alone
- using nominal `25/0.5` gains alone
- removing old command slew alone
- changing only the startup posture

The decisive improvement is inside the policy/training contract.

The A/B test does not isolate one individual training change. C1 still required
actor-side `base_lin_vel`; the no-odometry MJLAB runtime supplied explicit
zeros for that legacy term. That is itself evidence that the C1 observation
contract was unsuitable for this deployment path.

## Highest-Confidence Success Factors

### 1. Deploy-Honest Actor Observations

The deployed actor consumes exactly:

```text
base_ang_vel          3
projected_gravity     3
velocity_commands     3
joint_pos_rel        12
joint_vel_rel        12
last_action          12
total                45
```

It does not consume:

- `base_lin_vel`
- terrain heights
- friction
- mass
- COM
- actuator scales
- external forces

All six deployed observation terms come directly from LowState, the remote
command, or previous policy state. No external odometry estimator is needed.

This removes the old C1 dependency on simulator-perfect body velocity and the
hardware-side odometry filtering, clamping, deadband, dropout, and latency
path.

### 2. Direct Asymmetric PPO Instead Of Teacher-Student Imitation

The actor was optimized directly for the rough-terrain task using PPO.

The critic additionally received:

- true base linear velocity
- terrain height scan
- static and dynamic friction
- base mass ratio
- stiffness scale
- damping scale

The privileged information improves value estimation during training but is
not part of deployed inference.

Unlike C1, this run did not combine PPO with teacher-action imitation and
latent-regression schedules. The deployed actor is the actor that solved the
training task, not a downstream approximation of another policy.

This is a strong likely contributor, but it has not yet been isolated from the
observation-contract and randomization changes.

### 3. Symmetric Recovery Pressure During Training

The successful run used:

- push interval: `6-10 s`
- push delta velocity X/Y: `[-0.35, 0.35] m/s`
- push delta yaw: `[-0.4, 0.4] rad/s`
- base COM X/Y: `[-0.03, 0.03] m`
- base COM Z: `[-0.01, 0.01] m`
- added base mass: `[-2, 4] kg`
- stiffness scale: `[0.6, 1.4]`
- damping scale: `[0.6, 1.4]`
- static and dynamic friction: `[0.1, 2.0]`

These perturbations force the actor to repeatedly redistribute support and
recover from errors instead of relying on one exact symmetric simulated plant.
This is a plausible reason the real FR-thigh weakness no longer dominates the
visible gait.

No joint-specific FR impairment was added. The result came from broad,
symmetrically sampled robustness pressure.

### 4. Long Proprioceptive History Was Retained

The successful actor uses:

- current observation: `45`
- history length: `100`
- history input: `4500`
- history duration: `2.0 s` at 50 Hz
- history layout: `isaaclab_term_major`
- temporal channels: `[64, 64]`
- temporal kernel: `3`
- history feature: `64`
- history target: `128`

History was not removed. It gives the actor enough temporal evidence to infer
motion, loading, contact transitions, actuator response, and recovery state
without explicit actor-side base velocity.

The history vector is clean during training: corruption is enabled for the
current actor observation, but disabled for the stored history group.

### 5. A Proven Flat Locomotion Initialization

The rough actor was initialized from:

`go2_flat_mjlab_prior_v1/2026-06-02_14-30-48/model_1499.pt`

This supplied a working omnidirectional locomotion backbone before rough
terrain, privileged critic learning, pushes, COM variation, and wider dynamics
were introduced.

Warm starting reduced the amount of exploration needed to rediscover basic
standing and gait behavior. Earlier failed rough runs demonstrated that
initialization and staging could not be treated casually.

## Supporting Design Choices

### Tracking-Dominant Reward

The reward does not prescribe one rigid gait:

- linear tracking: `1.5`
- yaw tracking: `0.75`
- feet air time: `0.5`
- flat orientation: `-1.0`
- torque: `-5e-5`
- acceleration: `-1e-7`
- action rate: `-0.001`
- feet slide: `-0.05`
- air-time variance: `-0.05`

Regularization is present but weak relative to task tracking. This leaves the
actor freedom to change foot timing and body strategy under terrain and
disturbances.

### Practical Omnidirectional Curriculum

Commands begin at `+/-0.1` and expand to:

- X velocity: `[-0.8, 0.8] m/s`
- Y velocity: `[-0.3, 0.3] m/s`
- yaw rate: `[-0.6, 0.6] rad/s`

The target is a robust usable envelope rather than the maximum possible Go2
speed.

### Terrain Scope Matched Blind Locomotion

The successful run used:

- random rough height: `0.01-0.06 m`
- forward and inverted slopes: `0-0.4`
- terrain difficulty: `0-1`
- initial terrain level cap: `2`

It did not train on:

- stairs
- inverted stairs
- boxes

This avoided asking a blind reactive policy to anticipate abrupt obstacles
that fundamentally benefit from exteroception.

## Deployment Contract That Preserved The Result

- physics training step: `0.005 s`
- policy step: `0.020 s`
- control rate: `50 Hz`
- action: joint position target
- action scale: `0.25`
- nominal policy gains: `kp=25`, `kd=0.5`
- effort limit: `23.5 Nm`
- default joint pose:
  `[0.1, -0.1, 0.1, -0.1, 0.8, 0.8, 1.0, 1.0, -1.5, -1.5, -1.5, -1.5]`
- hardware map:
  `[3, 0, 9, 6, 4, 1, 10, 7, 5, 2, 11, 8]`
- FSM: `Passive -> FixStand -> Velocity`

The runtime does not apply policy command slew, directional compensation, or
neutral-action compensation.

## Validation Evidence

### Export parity

- checkpoint vs TorchScript max error: `0`
- C++ ONNX vs expected max error: `1.07288e-06`
- tolerance: `1e-05`

### Cross-simulator contract

The strict parity report is `comparable` with zero failures for:

- forward
- lateral left/right
- yaw left/right
- 50 N push left/right

The matched suite used:

- 50 Hz policy control
- 5 ms physics step
- one-step observation, action, and command delays
- identical reset distribution
- identical named scenarios and seeds

### Hardware

Initial operator evaluation on 2026-06-12 reported:

- stable standing and walking
- large improvement over C1
- no obvious FR-side asymmetry
- no need for the previous deployment smoothing and compensation stack

This is qualitative evidence. A logged repeated-run hardware comparison is
still required before claiming the asymmetry is statistically eliminated.

## What Was Not Required

The successful training run did not use:

- actor-side base linear velocity
- gait phase
- encoder bias
- observation delay
- action delay
- command delay
- stairs
- boxes
- joint-specific impairment randomization
- MJLAB split `20/1`, `40/2` actuator priors
- runtime command slew
- teacher-action imitation
- a separately deployed student

Do not retroactively attribute success to these features.

## Confidence Ranking

High confidence:

1. The improvement is policy/training-side, not primarily the FSM.
2. Removing the hardware-unreliable actor `base_lin_vel` contract was valuable.
3. Exact history ordering, observation ordering, and joint mapping are required.
4. The exported ONNX is numerically faithful to the trained checkpoint.

Medium confidence:

1. Direct asymmetric PPO was better suited than the old imitation ladder.
2. pushes, COM, mass, friction, and gain variation taught the actor to absorb
   the real robot's localized weakness.
3. the two-second history substitutes effectively for explicit body velocity.
4. flat-prior initialization prevented the collapse seen in earlier rough runs.

Not yet isolated:

1. the individual contribution of pushes versus COM randomization
2. whether history can be shortened without losing hardware robustness
3. whether privileged terrain or privileged dynamics contributed more
4. whether the real FR anomaly is hidden, compensated, or absent in this run

## Freeze Rules

Do not change these together:

- observation contract
- history length or layout
- training randomization
- critic privilege
- actuator model
- nominal deployment gains
- command envelope

Any next experiment must change one named axis and preserve this candidate as
the control.

Recommended first ablations:

1. repeated logged hardware runs to quantify mirror-pair tracking
2. history length `100` versus a shorter history, with all else frozen
3. push randomization on/off
4. COM randomization on/off
5. encoder bias and sensor delay only after the baseline is fully measured

## Canonical Commands

Activate and validate:

```bash
bash scripts/deploy/run_unitree_mjlab_sim_deploy.sh activate
bash scripts/deploy/run_unitree_mjlab_sim_deploy.sh validate
```

Hardware:

```bash
bash scripts/deploy/run_unitree_mjlab_sim_deploy.sh hardware enp0s31f6
```

The Go2-facing NIC must have an address on `192.168.123.0/24`.
