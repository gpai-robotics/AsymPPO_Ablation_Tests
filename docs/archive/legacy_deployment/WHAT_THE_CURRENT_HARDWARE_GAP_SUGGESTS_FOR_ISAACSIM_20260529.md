# What The Current Hardware Gap Suggests For IsaacSim

This note turns the current old-robot deployment findings into a concrete
IsaacSim-side improvement agenda.

It is intentionally downstream-facing:

- start from what the real robot is doing
- compare that against what IsaacSim and MuJoCo are **not** reproducing strongly
- infer which categories of simulator realism are most likely missing

This is not yet a reward-edit plan. It is the step before that.

## Current Observed Hardware Gap

On the real old robot, the frozen blind student shows:

1. small commanded velocity can produce disproportionately large body motion
2. after joystick release, the robot can continue locomoting for longer than
   expected
3. longer held commands can lead to longer continuation after release
4. lateral and yaw commands can leak into forward/backward motion
5. deployment-side joint target slew limiting improves responsiveness quality,
   but does not fully solve persistence

At the same time:

- IsaacSim teleop/playback does not show the same severity
- MuJoCo pulse tests do not show the same severity
- MuJoCo history ablations suggest policy history contributes somewhat, but not
  enough to explain the hardware gap by itself

That means the missing realism is likely in a category shared by:

- neither IsaacSim enough
- nor MuJoCo enough

or at least not represented strongly enough in either.

## Most Likely Missing IsaacSim-Side Categories

### 1. Actuator/target-tracking realism

This is currently the strongest suspect.

Why:

- the policy outputs joint position targets
- those targets are tracked by a PD-like actuator model in simulation
- on real hardware, the same target stream appears to create larger or more
  persistent body motion than either simulator predicts

That suggests the real actuator-plus-mechanics chain is not fully captured by
the training environment.

Potential IsaacSim-side improvements:

- broaden actuator gain randomization beyond simple gain scaling
- test whether the simulated actuator should include target lag, rate limits, or
  more explicit actuation dynamics
- evaluate whether action-to-target mapping should include a training-time target
  slew model if deployment consistently benefits from it
- compare simulated joint tracking error envelopes against real `q_cmd - q_meas`
  traces

Concrete project tie-in:

- deployment-side `joint_actions.h` experiments were useful because they operate
  directly on the policy-target stream
- that strongly suggests the missing realism may live near the action/target
  interface rather than only in high-level command shaping

### 2. Low-speed stop / settle behavior

This is the second strongest suspect.

Why:

- on hardware, release-to-zero does not reliably produce a crisp stop
- in simulation, release looks cleaner
- that means the simulator may under-represent the continuation dynamics of the
  locomotion state after command removal

Potential IsaacSim-side improvements:

- explicitly test short pulse and long pulse command scenarios in IsaacSim
- measure decay time of planar speed after zero command
- compare that decay curve against hardware logs
- add a training evaluation suite for:
  - command pulse on
  - command pulse off
  - residual motion duration
- if the sim settles too quickly, consider whether actuator/contact damping or
  stop-related reward structure is too forgiving

### 3. Contact and friction dissipation realism

Why:

- both stop persistence and off-axis leakage can be strongly affected by how
  contact impulses and ground interaction dissipate motion
- if real contact behavior is less clean than simulation, the robot may keep
  moving or cross-couple axes more aggressively

Potential IsaacSim-side improvements:

- broaden evaluation specifically for low-speed command-release cases on
  different friction settings
- compare flat-surface friction and damping assumptions against hardware
  conditions
- inspect whether the training terrain/contact family is too optimistic for
  “settle back to standstill” behavior
- study whether body slip and foot slip during release are lower in sim than in
  hardware

Important nuance:

- this does not mean “just randomize more friction”
- it means we should specifically ask whether release-phase contact dissipation
  is unrealistically clean in simulation

### 4. State estimation realism

This is easy to underestimate.

In deployment, the policy can consume:

- IMU angular velocity
- filtered odometry linear velocity
- joint states from hardware

In simulation, those values are usually cleaner and more immediate.

Why it matters:

- a history-bearing policy is sensitive not just to instantaneous state, but to
  how state evolves over time
- if real odometry drifts, lags, or smooths differently, the policy may stay in
  a locomotion regime longer than it does in simulation

Potential IsaacSim-side improvements:

- introduce deploy-like observation filtering for evaluation mode
- compare raw sim-state policy playback against filtered/noisy playback
- create an Isaac evaluation mode that mimics:
  - base velocity filtering
  - low-speed deadbands
  - realistic angular velocity noise

This is especially relevant because our real deploy path uses filtered odometry
in:

- `reference_repos/unitree_rl_lab/deploy/include/unitree_articulation.h`

If the simulator is feeding near-perfect base velocity while hardware feeds a
filtered/noisy estimate, the history trajectory seen by the policy is not the
same.

### 5. Cross-axis decoupling realism

Observed on hardware:

- lateral and yaw can induce forward/backward motion

Possible interpretation:

- the policy has some coupling
- but the real actuator/contact loop amplifies it more than simulation

Potential IsaacSim-side improvements:

- create structured isolated-axis evaluation suites:
  - pure `vx`
  - pure `vy`
  - pure `wz`
  - pulse and release for each
- track non-commanded motion explicitly:
  - `vx` during pure `vy`
  - planar translation during pure yaw
- compare those metrics in sim vs hardware

This should become a first-class sim2real evaluation lens rather than a casual
teleop observation.

## What This Does Not Automatically Mean

These findings do **not** automatically imply:

- the policy is fundamentally bad
- MuJoCo is unnecessary
- IsaacSim is wrong
- retraining should start immediately

Instead, they imply:

- the remaining error budget is likely in realism around actuator/contact/state
  evolution
- that realism gap should be characterized more explicitly in IsaacSim
- only then should reward/curriculum changes be chosen, unless a purely
  deployment-side safeguard already solves most of the problem

## Recommended IsaacSim Evaluation Additions

Before retraining, the most useful additions are evaluation additions, not
immediate task rewrites.

### Add deterministic pulse tests

Add IsaacSim-side tests for:

- short forward pulse then zero
- long forward pulse then zero
- short lateral pulse then zero
- short yaw pulse then zero

Measure:

- speed decay after command-off
- residual yaw after command-off
- cross-axis leakage during and after pulse

### Add deploy-like observation realism mode

Create an evaluation mode that mimics deploy observations more closely:

- filtered base linear velocity
- low-speed deadband
- realistic gyro noise
- optional observation lag

This should not necessarily become the default training mode immediately, but it
should exist for sim2real diagnosis.

### Add actuator-stress evaluation mode

Evaluate the frozen policy under:

- reduced motor strength
- increased damping mismatch
- target lag / target slew constraints
- altered contact dissipation

The goal is not to “patch the simulator until hardware matches by eye,” but to
find which realism category reproduces the hardware-style persistence.

### Add cross-axis coupling metrics

For each isolated-axis command family, record:

- commanded axis tracking
- non-commanded axis motion
- body tilt
- joint target error
- action magnitude and action delta

This will make “why does yaw cause vx on hardware?” a measurable question in
sim, not just a qualitative one.

## When To Move From Evaluation To Training Changes

Only after the above, or after a deploy-side actuator-proximal safeguard proves
necessary, should we lock in training modifications.

Likely triggers for training changes:

1. A deploy-side joint target slew guard is the only thing that makes hardware
   usable.
   - then training should probably internalize that smoothness or target-rate
     realism

2. IsaacSim deploy-like observation mode reproduces persistence.
   - then training should probably strengthen zero-command stabilization and
     stop behavior under that observation realism

3. Actuator-stress evaluation reproduces hardware leakage.
   - then training should probably use richer actuator/domain randomization

4. Cross-axis metrics reveal weak decoupling even in stressed sim.
   - then training should explicitly penalize non-commanded motion

## Most Likely First IsaacSim Improvements

If we prioritize by current evidence, the most promising IsaacSim-side
improvements are:

1. deploy-like observation realism evaluation
2. actuator/target-tracking realism evaluation
3. deterministic pulse-and-release evaluation suite
4. isolated-axis coupling evaluation metrics

Only after those:

5. reward/curriculum changes for stop behavior and decoupling

## Bottom Line

The current hardware gap suggests that IsaacSim is probably missing realism most
strongly in:

- actuator/target-tracking behavior
- low-speed stop/settle dynamics
- deploy-like state-estimation behavior

That means the best next simulator-side work is not “switch simulators” or
“immediately change rewards.”

It is:

- make IsaacSim better at reproducing the deploy-time bridge and the release
  behavior that hardware is exposing

Once that is done, retraining will be far more likely to fix the real problem
instead of just improving a simulator-specific metric.
