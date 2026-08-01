# Why MuJoCo If We Already Have IsaacSim?

This note answers a practical project question, not a philosophical one:

> If our training stack already lives in IsaacSim / IsaacLab, and our target is
> real deployment, why do a MuJoCo Sim2Sim step at all? Why not just improve
> IsaacSim and go straight to hardware?

The short answer is:

- we **can** go straight from IsaacSim to real
- MuJoCo is **not** a mandatory step in the sim2real pipeline
- MuJoCo is valuable mainly as an **independent validation and diagnosis
  environment**
- IsaacSim should still remain the **primary environment where we close the
  actual sim2real gap**

This note explains that in technical detail and ties it to the current Go2
deployment findings.

## Executive Summary

For this repo, the roles are best understood as:

- **IsaacSim / IsaacLab**
  - primary simulator
  - primary training environment
  - primary place to improve robot realism for sim2real
  - source of truth for curriculum, rewards, observations, actuation model, and
    export contract

- **MuJoCo**
  - secondary simulator
  - secondary runtime bridge
  - independent deploy-contract check
  - cross-simulator robustness and falsification tool
  - useful for testing whether a policy’s good behavior depends too heavily on
    Isaac-specific dynamics or implementation details

So the right mental model is:

- IsaacSim is where we should try to **fix** sim2real
- MuJoCo is where we can try to **disprove false confidence**

## The Core Distinction: Solution Simulator vs Validation Simulator

There are two different questions in locomotion deployment work.

### 1. Where do we train and model the problem?

This is the **solution** question.

For us, that is clearly IsaacSim / IsaacLab:

- our environment configs live there
- our observation contracts are defined there
- our command curriculum is defined there
- our actuation assumptions are defined there
- our policy export path originates there

Examples from this repo:

- training env configs:
  - `rma_go2_lab/envs/blind/c1_blind_rough_omni_usable_cfg.py`
  - `rma_go2_lab/envs/blind/blind_rough_forward_cfg.py`
- IsaacLab-side robot/task configs:
  - `reference_repos/unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/go2/velocity_env_cfg.py`
  - `reference_repos/unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py`

If the real robot is showing behavior that simulation misses, the long-term fix
should usually be to make IsaacSim’s modeled problem closer to reality.

### 2. How do we test whether our policy is overfit to that simulator?

This is the **validation** question.

That is where MuJoCo becomes useful.

If the policy:

- behaves well in IsaacSim
- exports successfully
- behaves reasonably in MuJoCo
- but still fails on hardware

then we learn something very useful:

- the remaining gap is probably not just “the policy only works in Isaac”
- instead, it is more likely a shared sim-to-real miss:
  - actuator response realism
  - contact/friction realism
  - state estimation / odometry mismatch
  - low-speed stop / settle behavior not captured by either simulator

That is exactly the kind of narrowing information a second simulator gives us.

## What MuJoCo Is Actually Buying Us In This Repo

MuJoCo is not acting here as a replacement training simulator. It is acting as:

1. an exported-policy runtime bridge
2. an independent physics engine
3. an independent actuator/runtime implementation
4. a cross-check on the deploy contract

Each of those matters.

### 1. Exported-policy runtime bridge check

Our MuJoCo runtime:

- `scripts/deploy/mujoco_runtime.py`

does not use the Isaac training env directly. It reconstructs a deployable
runtime bridge around the exported policy artifact.

It loads:

- exported TorchScript policy
- exported deploy config / metadata
- MuJoCo robot model

and then reconstructs:

- policy observations
- policy history
- joint target semantics
- PD-like actuation behavior

That means MuJoCo can tell us:

- does the exported artifact still behave sensibly outside the training loop?
- is the observation contract actually self-consistent?
- is the policy only “working” because the Isaac training runtime gives it
  hidden conveniences?

This is especially important for history-bearing policies like ours:

- `policy_kind = blind_history_policy`
- `policy_history_dim = 4800`
- `policy_obs_dim = 48`

Those contracts can quietly break in deployment even when the model file itself
loads fine.

### 2. Independent physics engine

IsaacSim and MuJoCo do not share the same contact solver, integration details,
collision handling, or rigid-body dynamics implementation.

So if a policy behaves well in both:

- IsaacSim
- MuJoCo

that gives more confidence that the behavior is not purely an artifact of one
engine’s contact or solver peculiarities.

This does **not** prove hardware success.

But it helps distinguish between:

- simulator-specific success
- cross-simulator success

That distinction is valuable when deciding whether to retrain or to look harder
at deploy/runtime mismatch.

### 3. Independent actuator/runtime implementation

The MuJoCo runtime is not identical to the real deploy runtime.

That is not a flaw. It is part of its value.

It has its own:

- command handling
- history handling
- target generation
- actuator application path

Examples from `scripts/deploy/mujoco_runtime.py`:

- command comes from `self.command`
- history can be ablated:
  - `normal`
  - `zero`
  - `frozen`
- action target is formed as:
  - `q_target = action_offset + action_scale * action`
- actuator model can be:
  - `simple_pd`
  - `isaac_dc_motor`

So MuJoCo lets us ask:

- if we rebuild a clean runtime bridge outside the real robot SDK path, does the
  policy still show the same pathology?

That is exactly why the recent history-ablation experiment was informative.

### 4. A controlled place for structured ablations

MuJoCo is extremely convenient for experiments that are hard or risky on
hardware, such as:

- zeroing policy history
- freezing policy history
- injecting exact command pulses
- changing ground friction
- changing passive damping / motor strength
- changing actuator model assumptions

In hardware, these tests are:

- slower
- noisier
- riskier
- harder to isolate

In MuJoCo, they can be done quickly and deterministically.

That means MuJoCo is a very good **hypothesis testing environment**, even if it
is not the place where the final sim2real fix should live.

## Why MuJoCo Is Not The Main Place To Solve Sim2Real

This is equally important.

If our final goal is:

- train in IsaacLab
- export from IsaacLab
- deploy to real robot

then MuJoCo should not become the main place where we invent compensations.

Why not?

### 1. It is not the training simulator

The policy was trained in IsaacSim / IsaacLab.

That means:

- the command curriculum was defined there
- the actuation assumptions were defined there
- the observation scaling and noise were defined there
- the randomization family was defined there

If hardware mismatch persists, the most principled long-term fix is to improve
that original modeling environment, not to bolt on a third parallel truth.

### 2. MuJoCo has its own bridge assumptions

Our MuJoCo runtime is not a perfect mirror of real deploy.

Important examples:

- `mujoco_runtime.py` uses local body velocity directly from the simulator
- the real runtime can use filtered odometry:
  - `reference_repos/unitree_rl_lab/deploy/include/unitree_articulation.h`
- MuJoCo teleop and scheduled command injection are runtime conveniences, not
  literal replicas of the hardware joystick bridge
- MuJoCo actuator behavior is still a model, not the physical SDK motor loop

So if we “fix” everything in MuJoCo but do not reflect it back into IsaacSim or
real deploy, we risk solving the wrong problem.

### 3. The final deployment contract still lives elsewhere

The actual production chain is closer to:

- IsaacLab training/export
- Unitree RL Lab deploy runtime
- real robot SDK / motor interface

MuJoCo sits beside that chain, not inside it.

So it is better viewed as:

- a test bench

than as:

- the authoritative source of deployment behavior

## What The Current Project Evidence Says

This project already gives a concrete answer to the “why MuJoCo?” question.

### Observation 1: Hardware shows strong post-release persistence

On the old robot, we observed:

- after releasing the joystick, the robot keeps walking
- longer held commands lead to longer continuation
- lateral and yaw can leak into forward/backward motion

### Observation 2: MuJoCo does not reproduce the severity

Using deterministic pulse scenarios:

- `scripts/deploy/scenarios/go2_forward_pulse_short.json`
- `scripts/deploy/scenarios/go2_forward_pulse_long.json`

and history ablations:

- `normal`
- `zero`

we found:

- history contributes somewhat
- but MuJoCo still settles relatively quickly
- zeroing history helps a bit, but does not produce a dramatic qualitative
  shift

Interpretation:

- policy history is not the whole story
- the real hardware issue is more severe than what MuJoCo predicts

This is exactly the kind of insight a second simulator is good for.

It tells us:

- the remaining issue is probably not just “the policy is fundamentally broken”
- and probably not just “history policy persistence alone”
- instead, the real gap likely lives in:
  - actuator response
  - contact/friction dissipation
  - state estimation / odometry
  - real motor/PD tracking dynamics

That is very useful guidance for what to do next in IsaacSim and in deployment.

### Observation 3: Both simulators looking good does not guarantee hardware

This is the most important current lesson.

If:

- IsaacSim looks good
- MuJoCo looks good
- hardware still looks bad

then the missing realism is likely in a category that both simulators are still
under-representing.

Typical suspects:

- real joint target tracking is harsher than modeled
- real actuator/transmission compliance is different
- real friction/collision dissipation differs from both simulators
- deploy-time state estimation differs from simulator ground truth

So MuJoCo did not solve the issue, but it narrowed it in a helpful way.

## Command -> Policy -> PD Bridge: Isaac vs MuJoCo vs Real

To understand why behaviors can differ, it helps to compare the full bridge.

### IsaacSim / IsaacLab

Approximate path:

1. command manager generates command
2. simulator provides state:
   - base linear velocity
   - base angular velocity
   - projected gravity
   - joint positions/velocities
3. policy observes:
   - `policy`
   - `policy_history`
   - `last_action`
4. `JointPositionActionCfg(scale=0.25, use_default_offset=True)` forms targets
5. simulator actuator model applies those targets
6. next simulator state is produced

Important properties:

- tight integration
- simulator-native state
- simulator-native actuator semantics
- no real sensor latency / noise path unless explicitly modeled

### MuJoCo runtime bridge

Approximate path:

1. command comes from runtime:
   - direct command vector
   - teleop keyboard or scheduled pulses
2. MuJoCo provides simulator state
3. runtime reconstructs observations:
   - local velocity
   - projected gravity
   - joint states
   - `last_action`
   - history buffer
4. policy outputs action
5. runtime forms:
   - `q_target = offset + scale * action`
6. runtime applies actuator model:
   - `simple_pd` or `isaac_dc_motor`
7. MuJoCo physics advances

Important properties:

- still simulated
- but not the same bridge as IsaacSim training/runtime
- useful for contract checking and ablations

### Real deployment runtime

Approximate path:

1. real joystick input
2. deploy observation bridge:
   - deadband
   - directional scale
   - command clamp
   - optional command slew
3. policy ONNX output
4. deploy action bridge:
   - scale
   - offset
   - optional joint slew
5. low-level PD commands:
   - `q`
   - `kp`
   - `kd`
6. real motors, real contacts, real drivetrain, real body
7. real state estimate:
   - IMU
   - motor states
   - optional filtered odometry

Important properties:

- real estimation path
- real actuator response
- real contacts
- real mechanical dissipation and coupling
- real timing and transport effects

This is why matching only the policy network is not enough.

The bridge matters.

## Why IsaacSim Should Stay The Primary Place To Improve Sim2Real

Even after all of the above, IsaacSim should remain the main simulator where we
improve fidelity for training and deployment.

Reasons:

1. It is where the policy is trained.
2. It is where reward/curriculum changes will be made.
3. It is where actuator randomization and dynamics modeling should be improved.
4. It is where export semantics originate.
5. It is the cleanest place to make the policy learn the behaviors we want.

So if the real robot is showing:

- poor stop behavior
- cross-axis leakage
- persistence after release

the long-term action is still:

- improve IsaacSim-side realism and training structure

not:

- move the core training story to MuJoCo

## So When Is MuJoCo Worth It?

MuJoCo is worth it when we want to answer questions like:

- does the exported policy still behave sensibly outside Isaac?
- is a failure Isaac-specific or cross-simulator?
- does history ablation change behavior?
- does a hidden-dynamics probe break the policy similarly in a second engine?
- are we seeing a deployment-contract bug or a real control issue?

MuJoCo is less useful when the question is:

- what should the training task definition be?
- what actuator randomization should we add to the primary simulator?
- what reward terms should shape stop-on-release?

Those remain IsaacSim / IsaacLab questions.

## Current Project Conclusion

For this Go2 project, MuJoCo has already justified itself as:

- a deploy-contract verification tool
- a history-ablation testbed
- a second-opinion dynamics/runtime environment

But the current evidence also says:

- MuJoCo is **not** where the final sim2real fix should primarily be developed
- the remaining gap is likely in the real actuator/state/contact path
- therefore the next true long-term fix belongs back in IsaacSim-side realism
  and training design

In short:

- **Use IsaacSim to solve the problem**
- **Use MuJoCo to check whether the solution is simulator-specific**

That is the cleanest technical justification for keeping both, without confusing
their roles.

## Practical Guidance Going Forward

Given the current evidence, the most sensible workflow is:

1. Keep IsaacSim / IsaacLab as the main training and modeling environment.
2. Use MuJoCo only for:
   - export/runtime checks
   - structured ablations
   - cross-simulator confidence checks
3. Use hardware findings to identify the missing realism category.
4. Push the long-term fixes into:
   - Isaac actuator modeling
   - reward shaping
   - command transition curriculum
   - stop/decoupling behavior
5. Keep deployment-side patches minimal and well documented so we know what
   training should later absorb.

That way, MuJoCo remains a sharp tool rather than becoming a second competing
source of truth.
