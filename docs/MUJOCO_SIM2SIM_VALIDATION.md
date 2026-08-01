# MuJoCo Sim2Sim Validation Contract

This note documents the current MuJoCo Sim2Sim validation setup for the Go2 deployment bundle, with an emphasis on:

- what is inherited directly from the IsaacLab export contract,
- what parameters are explicitly tunable in the MuJoCo bridge,
- what is changed by default versus only changed when an override is passed,
- and what simulator-level differences remain even when the interface is matched exactly.

The goal is to make it easy to answer:

- "Did we actually scale anything?"
- "Did we only tune the PD gains?"
- "Which knobs were identity by default?"
- "Why do we treat MuJoCo as a stronger pre-deployment check, but not as hardware truth?"

## Scope

This document describes the repo-owned MuJoCo validation path:

- [scripts/deploy/run_sim2sim.py](/home/bhuvan/projects/rma/rma_go2_lab/scripts/deploy/run_sim2sim.py)
- [scripts/deploy/mujoco_runtime.py](/home/bhuvan/projects/rma/rma_go2_lab/scripts/deploy/mujoco_runtime.py)

and the frozen exported deployment contract:

- [deploy_config.json](/home/bhuvan/projects/rma/rma_go2_lab/reference_repos/unitree_rl_lab/deploy/robots/go2/config/policy/velocity/c1_blind_rough_omni_usable_v1_final/params/deploy_config.json)
- [deploy.yaml](/home/bhuvan/projects/rma/rma_go2_lab/reference_repos/unitree_rl_lab/deploy/robots/go2/config/policy/velocity/c1_blind_rough_omni_usable_v1_final/params/deploy.yaml)

It does **not** describe the real robot runtime in full, and it does **not** claim MuJoCo is the final source of truth over hardware.

## Executive Summary

For the current Go2 Sim2Sim validation setup:

- The **policy/update timing** is matched exactly to the IsaacLab export contract:
  - physics dt = `0.005 s`
  - decimation = `4`
  - control/policy dt = `0.02 s`
- The **observation layout** is reconstructed to match the deploy contract:
  - `base_lin_vel`
  - `base_ang_vel`
  - `projected_gravity`
  - `velocity_commands`
  - `joint_pos_rel`
  - `joint_vel_rel`
  - `last_action`
  - plus `policy_history`
- The **action post-processing** is matched to the export contract:
  - `q_target = action_offset + action_scale * action`
- The main **MuJoCo-only tuning knobs** are:
  - `ground_friction`
  - `foot_friction`
  - `base_mass_scale`
  - `motor_strength_scale`
  - `joint_damping_scale`
  - `passive_joint_damping_scale`
  - `passive_joint_frictionloss_scale`
  - `actuator_model`
  - `dc_motor_velocity_limit`

Most importantly:

- If you run the bridge **without passing any of those overrides**, then those physical scaling knobs remain at identity and you are **not** silently changing friction, mass, or motor strength.
- The bridge is still **not Isaac Sim**. Even with identity overrides, MuJoCo differs in contact handling, passive dissipation, actuator realization, XML model details, and runtime implementation choices.

## 1. IsaacLab Timing Baseline

The Go2 rough locomotion family uses:

- `decimation = 4`
- `sim.dt = 0.005`

in:

- [velocity_env_cfg.py](/home/bhuvan/projects/rma/rma_go2_lab/reference_repos/unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/go2/velocity_env_cfg.py)

That gives:

- physics rate = `1 / 0.005 = 200 Hz`
- policy/env step rate = `1 / (0.005 * 4) = 50 Hz`

IsaacLab computes rewards once per environment step, after the decimation loop, not every physics substep:

- [manager_based_rl_env.py](/home/bhuvan/tools/IsaacLab/source/isaaclab/isaaclab/envs/manager_based_rl_env.py)
- [reward_manager.py](/home/bhuvan/tools/IsaacLab/source/isaaclab/isaaclab/managers/reward_manager.py)

So during training:

- simulation advances at `200 Hz`
- policy updates at `50 Hz`
- rewards are computed at `50 Hz`

## 2. Exported Deploy Contract

The exported bundle currently records:

- `physics_dt = 0.005`
- `step_dt = 0.02`
- `decimation = 4`

in:

- [deploy_config.json](/home/bhuvan/projects/rma/rma_go2_lab/reference_repos/unitree_rl_lab/deploy/robots/go2/config/policy/velocity/c1_blind_rough_omni_usable_v1_final/params/deploy_config.json)

The same timing also appears in:

- [deploy.yaml](/home/bhuvan/projects/rma/rma_go2_lab/reference_repos/unitree_rl_lab/deploy/robots/go2/config/policy/velocity/c1_blind_rough_omni_usable_v1_final/params/deploy.yaml)

### Exported action contract

From the current frozen `deploy_config.json`:

- `action scale = 0.25` for all 12 joints
- `action offset = [0.1, -0.1, 0.1, -0.1, 0.8, 0.8, 1.0, 1.0, -1.5, -1.5, -1.5, -1.5]`

The bridge applies:

- `q_target = action_offset + action_scale * action`

in:

- [mujoco_runtime.py](/home/bhuvan/projects/rma/rma_go2_lab/scripts/deploy/mujoco_runtime.py)

### Exported joint-space control metadata

The export path records:

- joint stiffness = `25.0`
- joint damping = `0.5`

These values are loaded from the deploy config by the MuJoCo bridge when the bundle includes a deploy config, which it does in the current setup.

Important nuance:

- [BridgeConfig](/home/bhuvan/projects/rma/rma_go2_lab/scripts/deploy/mujoco_runtime.py) also defines fallback `kp=50.0` and `kd=3.5`
- those are only fallback values used if a deploy config is absent
- for the current exported Go2 bundle, the bridge loads the deploy config, so the effective control metadata comes from the bundle, not the fallback dataclass defaults

### Exported observation contract

The frozen deploy contract for the policy path is currently:

- `policy_dim = 48`
- `policy_history_length = 100`

Observation order in the export metadata is:

1. `base_lin_vel`
2. `base_ang_vel`
3. `projected_gravity`
4. `velocity_commands`
5. `joint_pos_rel`
6. `joint_vel_rel`
7. `last_action`

This is read from:

- [deploy_config.json](/home/bhuvan/projects/rma/rma_go2_lab/reference_repos/unitree_rl_lab/deploy/robots/go2/config/policy/velocity/c1_blind_rough_omni_usable_v1_final/params/deploy_config.json)

## 3. What the MuJoCo Bridge Changes By Default

The most important answer:

### By default, the bridge does **not** apply physical scaling overrides

The MuJoCo bridge defaults are:

- `ground_friction = 0.0`
- `foot_friction = 0.0`
- `base_mass_scale = 1.0`
- `motor_strength_scale = 1.0`
- `joint_damping_scale = 1.0`
- `passive_joint_damping_scale = 1.0`
- `passive_joint_frictionloss_scale = 1.0`
- `latent_clamp_max_abs = 0.0`

from:

- [mujoco_runtime.py](/home/bhuvan/projects/rma/rma_go2_lab/scripts/deploy/mujoco_runtime.py)
- [run_sim2sim.py](/home/bhuvan/projects/rma/rma_go2_lab/scripts/deploy/run_sim2sim.py)

That means:

- friction is unchanged unless you explicitly pass `--ground-friction` or `--foot-friction`
- base mass/inertia are unchanged unless you pass `--base-mass-scale`
- action scale is unchanged unless you pass `--motor-strength-scale`
- deploy damping is unchanged unless you pass `--joint-damping-scale`
- MuJoCo passive joint damping/frictionloss are unchanged unless you pass the passive scaling flags

### What *is* changed by default

Even with all scaling knobs at identity, the bridge still changes the execution environment in these ways:

1. The simulator backend is MuJoCo, not Isaac Sim / PhysX.
2. The model comes from the MuJoCo scene XML:
   - [scene.xml](/home/bhuvan/projects/rma/rma_go2_lab/reference_repos/mujoco_menagerie/unitree_go2/scene.xml)
3. The robot is stepped by the repo-owned bridge implementation:
   - [mujoco_runtime.py](/home/bhuvan/projects/rma/rma_go2_lab/scripts/deploy/mujoco_runtime.py)
4. The bridge uses MuJoCo body/joint state to reconstruct the deploy observation.
5. The bridge does **not** recompute the original Isaac training rewards; it only logs a simple velocity proxy.

So the default MuJoCo run is still a changed system, just not changed by the explicit scaling knobs above.

## 4. Exact MuJoCo Runtime Tuning Knobs

These are the knobs you can truthfully say were tuned if you passed them in the CLI or scenario JSON.

### Contact and friction

- `--ground-friction`
  - overrides tangential friction on ground geoms
- `--foot-friction`
  - overrides tangential friction on foot geoms

Implemented in:

- [mujoco_runtime.py](/home/bhuvan/projects/rma/rma_go2_lab/scripts/deploy/mujoco_runtime.py)

### Mass / inertia

- `--base-mass-scale`
  - multiplies base body mass and inertia

### Motor / action authority

- `--motor-strength-scale`
  - multiplies `action_scale`
  - effectively changes the mapping from policy action to target joint position amplitude

This is important:

- this is **not** "just tuning KP/KD"
- this changes how large the commanded joint target excursion is for the same policy action

### Deploy damping

- `--joint-damping-scale`
  - scales the deploy damping loaded from the bundle config

### Passive MuJoCo dissipation

- `--passive-joint-damping-scale`
  - scales MuJoCo DOF damping on robot joints
- `--passive-joint-frictionloss-scale`
  - scales MuJoCo DOF friction loss on robot joints

These are simulator-side passive effects, separate from deploy PD damping.

### Actuator realization

- `--actuator-model {simple_pd, isaac_dc_motor}`
- `--dc-motor-velocity-limit`

The bridge supports two actuator realizations:

- `simple_pd`
  - torque = `Kp * (q_target - q) + Kd * (0 - dq)`
  - then clipped to actuator control range
- `isaac_dc_motor`
  - same nominal torque law
  - then clipped with an Isaac-like velocity-dependent DC motor saturation model

So if you change `--actuator-model`, that is a substantial sim2sim change and should be documented explicitly.

### Reset stochasticity

- `--reset-pos-xy-jitter`
- `--reset-yaw-jitter-deg`
- `--reset-joint-pos-jitter`
- `--reset-joint-vel-jitter`

These do not change the nominal dynamics, but they do change how hard the validation setup is.

### History and latent debugging

- `--history-ablation {normal, zero, frozen}`
- `--latent-clamp-max-abs`

These are debug knobs, not physical tuning knobs, but they still change runtime behavior and should be logged if used.

## 5. What Counts as "No Tuning"

You can honestly say the MuJoCo validation was run without explicit physical retuning if all of these stayed at default:

- `--ground-friction 0.0`
- `--foot-friction 0.0`
- `--base-mass-scale 1.0`
- `--motor-strength-scale 1.0`
- `--joint-damping-scale 1.0`
- `--passive-joint-damping-scale 1.0`
- `--passive-joint-frictionloss-scale 1.0`
- `--actuator-model simple_pd`
- `--dc-motor-velocity-limit 30.0`

and if you did not use a scenario JSON that changed them.

That means:

- no friction scaling
- no mass scaling
- no action amplitude scaling
- no extra passive damping scaling
- no actuator saturation-model swap beyond the bridge default

## 6. Important Non-Obvious Difference: Bundle Default Command

The current frozen `deploy_config.json` still contains:

- `commands.base_velocity.default = [0.5, 0.0, 0.0]`

This matters because the MuJoCo bridge initializes the command from the bundle default unless you explicitly pass:

- `--command-x`
- `--command-y`
- `--command-yaw`

So for clean teleop validation, the safer launch is:

```bash
python scripts/deploy/run_sim2sim.py \
  --bundle-dir /home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/policies/exported/c1_blind_rough_omni_usable_v1_final \
  --execute-runtime \
  --viewer \
  --teleop-keyboard \
  --command-x 0.0 \
  --command-y 0.0 \
  --command-yaw 0.0
```

This avoids confusing "the simulator keeps walking forward" with an actual policy bug.

## 7. What the MuJoCo Bridge Matches Exactly

The bridge intentionally matches these parts of the deploy contract:

- control rate:
  - `physics_dt = 0.005`
  - `control_dt = 0.02`
  - `substeps_per_control = 4`
- joint ordering
- action scale and offset
- default joint pose
- policy observation dimension and ordering
- policy history dimension and length
- last-action feedback in the observation

This is why MuJoCo is useful:

- it lets us isolate "contract bugs" from "simulator differences"

If behavior diverges after these are aligned, the remaining gap is more likely to come from dynamics, contact, or actuator differences rather than tensor ordering mistakes.

## 8. State Model Differences: Isaac Training vs MuJoCo Runtime vs Hardware Deploy

This section is specifically about the **state model**, not just the simulator backend.

There are three distinct layers to keep separate:

1. the **full simulator state**
2. the **policy-visible observation state**
3. the **command state** that gets injected into the observation

These are easy to conflate, but they differ in important ways.

### 8.1 Full simulator state

At the full-state level, Isaac Sim and MuJoCo are substantially different.

#### IsaacLab / Isaac Sim full state

IsaacLab training uses asset and sensor state from the Isaac/PhysX stack, such as:

- root pose
- root linear velocity in body frame
- root angular velocity in body frame
- projected gravity in body frame
- joint positions
- joint velocities
- contact sensor state
- terrain scanner state
- command manager state
- action manager state

Much of this is surfaced through:

- [observations.py](/home/bhuvan/tools/IsaacLab/source/isaaclab/isaaclab/envs/mdp/observations.py)
- [velocity_env_cfg.py](/home/bhuvan/tools/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py)

#### MuJoCo full state

The MuJoCo bridge instead derives state from:

- `self.data.qpos`
- `self.data.qvel`
- `self.data.xmat`
- `self.data.xpos`
- `self.data.xquat`
- `self.data.cvel`
- `self.data.contact`
- `self.model.*` actuator and DOF parameters

in:

- [mujoco_runtime.py](/home/bhuvan/projects/rma/rma_go2_lab/scripts/deploy/mujoco_runtime.py)

The important point is:

- the policy may see the same **shape** of state,
- but that state is being computed from a different physics engine, different solver, and different robot model.

So at the full-state level:

- **IsaacLab state model and MuJoCo state model are not equivalent**

### 8.2 Policy-visible observation state

At the deployable student-policy level, the two runtimes are much closer.

#### IsaacLab training observation group

For the locomotion policy group, the default IsaacLab velocity environment uses:

1. `base_lin_vel`
2. `base_ang_vel`
3. `projected_gravity`
4. `velocity_commands`
5. `joint_pos_rel`
6. `joint_vel_rel`
7. `last_action`

configured in:

- [velocity_env_cfg.py](/home/bhuvan/tools/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py#L126)

and implemented via:

- [base_lin_vel](/home/bhuvan/tools/IsaacLab/source/isaaclab/isaaclab/envs/mdp/observations.py#L54)
- [base_ang_vel](/home/bhuvan/tools/IsaacLab/source/isaaclab/isaaclab/envs/mdp/observations.py#L64)
- [projected_gravity](/home/bhuvan/tools/IsaacLab/source/isaaclab/isaaclab/envs/mdp/observations.py#L74)
- [joint_pos_rel](/home/bhuvan/tools/IsaacLab/source/isaaclab/isaaclab/envs/mdp/observations.py#L212)
- [joint_vel_rel](/home/bhuvan/tools/IsaacLab/source/isaaclab/isaaclab/envs/mdp/observations.py#L257)
- [last_action](/home/bhuvan/tools/IsaacLab/source/isaaclab/isaaclab/envs/mdp/observations.py#L657)

#### MuJoCo runtime observation group

The repo-owned MuJoCo runtime reconstructs the policy observation as:

1. body-frame linear velocity
2. body-frame angular velocity
3. projected gravity
4. command
5. `joint_pos - default_joint_pos`
6. joint velocity
7. last action

in:

- [mujoco_runtime.py](/home/bhuvan/projects/rma/rma_go2_lab/scripts/deploy/mujoco_runtime.py#L537)

So at the **observation layout** level, MuJoCo is intentionally trying to match the deploy contract.

The current deploy observation contract is frozen in:

- [deploy_config.json](/home/bhuvan/projects/rma/rma_go2_lab/reference_repos/unitree_rl_lab/deploy/robots/go2/config/policy/velocity/c1_blind_rough_omni_usable_v1_final/params/deploy_config.json)

with:

- `policy_dim = 48`
- `policy_history_length = 100`

### 8.3 Side-by-side comparison table

| Aspect | IsaacLab training | MuJoCo runtime | Hardware deploy |
|---|---|---|---|
| Root linear velocity | `asset.data.root_lin_vel_b` | derived from `data.cvel` and base rotation | from `robot->data.root_lin_vel_b` in deploy runtime |
| Root angular velocity | `asset.data.root_ang_vel_b` | derived from `data.cvel` and base rotation | from `robot->data.root_ang_vel_b` |
| Gravity projection | `asset.data.projected_gravity_b` | computed from MuJoCo world gravity and base rotation | from `robot->data.projected_gravity_b` |
| Joint position relative | `joint_pos - default_joint_pos` | same semantic reconstruction | same semantic reconstruction |
| Joint velocity relative | `joint_vel - default_joint_vel` | currently just joint velocity, since default joint vel is zero in practice | current deploy path uses joint velocity directly |
| Last action | Isaac action manager state | local cached `last_action` | deploy action manager state |
| Command fed to obs | command manager generated command | bridge `self.command` | filtered joystick command in `observations.h` |
| Observation noise | yes by default in training config | no | no explicit training-noise injection |
| History | Isaac observation manager history | bridge-owned history buffer | deploy observation manager history |

### 8.4 Exact observation-term semantic differences

#### `base_lin_vel`

IsaacLab:

- directly uses `asset.data.root_lin_vel_b`

MuJoCo:

- computes base spatial velocity from `data.cvel`
- takes the linear part
- rotates from world frame into base frame using `xmat.T`

This is semantically aligned, but numerically it may differ because:

- MuJoCo and Isaac do not integrate the base the same way
- contact resolution changes the underlying body motion

#### `base_ang_vel`

IsaacLab:

- directly uses `asset.data.root_ang_vel_b`

MuJoCo:

- uses `data.cvel`
- extracts angular velocity
- rotates into the local base frame

Again, semantically aligned, but not guaranteed numerically identical.

#### `projected_gravity`

IsaacLab:

- directly uses `asset.data.projected_gravity_b`

MuJoCo:

- normalizes model gravity
- rotates it into the base frame using the base rotation matrix

This is very close semantically and is one of the cleaner matched channels.

#### `joint_pos_rel`

IsaacLab:

- `joint_pos - default_joint_pos`

MuJoCo:

- `self.data.qpos[joint_qpos_indices] - self.default_joint_pos`

This is very closely matched as long as:

- joint order is correct
- default joint pose is the same

#### `joint_vel_rel`

IsaacLab:

- `joint_vel - default_joint_vel`

MuJoCo:

- uses raw joint velocity directly in the observation construction

In practice this is usually close because default joint velocity is typically zero, but it is still worth naming precisely:

- this is a **semantic near-match**, not a strict symbolic copy of the Isaac term

#### `last_action`

IsaacLab:

- comes from the action manager

MuJoCo:

- comes from the locally cached previously applied policy action

This is conceptually aligned.

### 8.5 Command-state differences

This is one of the most important current mismatches.

#### IsaacLab training command state

Training uses:

- `generated_commands(...)`

from the Isaac command manager:

- [generated_commands](/home/bhuvan/tools/IsaacLab/source/isaaclab/isaaclab/envs/mdp/observations.py#L665)

This is not a joystick. It is the training-time command process generated by the environment command manager.

#### MuJoCo runtime command state

MuJoCo feeds:

- `self.command`

which comes from:

- CLI defaults
- bundle default command
- teleop keyboard updates
- optional command schedule

This means MuJoCo command injection is much more direct and simpler than the training command manager.

#### Hardware deploy command state

Hardware deploy is different again. The deploy runtime builds the command observation in:

- [observations.h](/home/bhuvan/projects/rma/rma_go2_lab/reference_repos/unitree_rl_lab/deploy/include/isaaclab/envs/mdp/observations/observations.h)

The current hardware command path includes:

- joystick sampling
- deadband
- directional scaling
- explicit clamp to configured ranges
- extra forward scaling
- hard forward cap
- asymmetric slew limiting for `vx`
- generic slew limiting for `vy` and `yaw`

So command state currently differs across the three environments:

- training command manager
- MuJoCo keyboard/CLI command
- hardware filtered joystick command

This is not a minor detail. It can materially affect:

- acceleration feel
- stopping behavior
- whether small commands look aggressive
- whether mirrored commands produce mirrored trajectories

### 8.6 Observation noise differences

Another important difference is observation corruption.

IsaacLab training injects observation noise in the default velocity env config:

- `base_lin_vel` noise
- `base_ang_vel` noise
- `projected_gravity` noise
- `joint_pos_rel` noise
- `joint_vel_rel` noise

visible in:

- [velocity_env_cfg.py](/home/bhuvan/tools/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py#L126)

MuJoCo runtime currently does **not** inject that same observation noise.

Hardware deploy also does **not** explicitly replay the IsaacLab observation-noise process.

So for observation noise:

- training: noisy
- MuJoCo: clean
- hardware deploy: clean but physically noisy through sensors/estimation

This is one reason a policy can look sharper or more brittle in validation than in training.

### 8.7 History-state differences

The policy-history channel also needs to be separated from the instantaneous observation.

#### IsaacLab training history

History is managed by the Isaac observation manager and updated at env-step rate.

#### MuJoCo runtime history

History is a bridge-owned NumPy buffer:

- initialized from the current observation
- shifted each control step
- flattened before policy evaluation

in:

- [mujoco_runtime.py](/home/bhuvan/projects/rma/rma_go2_lab/scripts/deploy/mujoco_runtime.py)

#### Hardware deploy history

History is managed by the deploy observation manager.

We also fixed one runtime history-layout bug earlier:

- `policy_history` flattening needed to be term-major rather than step-major

So the current state is:

- history semantics are intended to match,
- but they are implemented by different codepaths in all three runtimes.

### 8.8 Privileged-state differences

For teacher and adaptation variants, the difference is much larger than for the blind deployable student.

IsaacLab training may include:

- terrain height scan
- privileged terrain observations
- privileged dynamics observations

MuJoCo deploy validation does **not** reconstruct privileged teacher state for the deployable student runtime.

Hardware deploy also does not have those privileged channels.

So:

- for the blind deployable student, the observation-model match is relatively strong
- for privileged teacher/adaptation training state, the match is intentionally incomplete

### 8.9 Practical interpretation

If someone asks "how different are the IsaacLab Go2 state and MuJoCo state models?", the precise answer is:

- **full simulator state:** very different
- **student policy observation layout:** deliberately similar
- **student policy observation numerics:** only approximately matched
- **command-state semantics:** noticeably different
- **observation noise model:** different
- **history implementation:** different codepath, same intended contract

So MuJoCo is best thought of as:

- a good validator for deploy-contract correctness,
- a decent validator for qualitative student behavior,
- but not a numerically identical replacement for IsaacLab state evolution.

## 9. What the MuJoCo Bridge Does **Not** Match Exactly

Even with no explicit tuning overrides, the bridge is not Isaac Sim.

### Simulator-level differences

#### Contact model and solver

Isaac Sim / PhysX and MuJoCo do not resolve contact in the same way.

This affects:

- foot touchdown timing
- slip onset
- lateral impulse production
- base bounce/compliance feel
- how marginal contacts stabilize or destabilize the gait

#### Passive dynamics

MuJoCo joints and actuators carry XML-defined:

- damping
- frictionloss
- ctrl ranges

These are not the same implementation as Isaac's actuator stack and PhysX integration.

#### Actuator realization

The bridge reconstructs deploy execution using:

- joint targets
- PD logic
- optional DC motor saturation approximation

That is a validation bridge, not the original training simulator actuator pipeline.

#### Robot model asset

Training and deployment export originate from Isaac/Usd-based robot assets, while MuJoCo validation uses:

- MuJoCo menagerie Go2 XML scene

Even with matching names and similar nominal geometry, these are not guaranteed to have identical inertias, contacts, or collision details.

#### Reward implementation

The MuJoCo bridge does **not** compute the original Isaac training reward terms.

Instead, it logs:

- velocity error
- yaw error
- base tilt
- target tracking error
- a simple `reward_proxy`

So MuJoCo validation is for:

- rollout sanity
- stability
- contract correctness
- relative robustness

not for claiming exact reward parity with Isaac training.

## 10. Why MuJoCo Is Useful Before Sim2Real

MuJoCo is valuable because it is a stronger external check than "it works in the same Isaac stack it was trained in."

### Why it helps

- It is an independent dynamics engine.
- It forces us to reconstruct the deploy contract explicitly.
- It exposes contact, damping, friction, and actuator assumptions more clearly.
- It is easy to perturb with structured OOD sweeps.
- It is easier to introspect in a single-scene, single-robot setting than GPU-batched Isaac training.

### Why it is not hardware truth

MuJoCo is still a simulator.

It does not automatically know:

- real foot rubber behavior
- real floor surface
- real motor temperature/current limits
- real cable drag
- real gearbox friction/stiction
- real Unitree onboard state-estimation quirks
- real communication latency and packet jitter

So the right phrasing is:

- MuJoCo is a **stronger pre-deployment validation proxy**
- hardware remains the final truth

## 11. Recommended Reporting Template

When documenting a MuJoCo Sim2Sim run, record:

### A. Frozen contract

- bundle name
- checkpoint
- action scale/offset
- joint stiffness/damping from deploy config
- physics dt
- control dt
- decimation
- policy history length

### B. Runtime overrides

- ground friction
- foot friction
- base mass scale
- motor strength scale
- joint damping scale
- passive joint damping scale
- passive joint frictionloss scale
- actuator model
- DC motor velocity limit
- reset jitters
- history ablation
- latent clamp

### C. Scenario setup

- scene XML
- teleop or scripted commands
- command schedule
- wrench schedule
- real-time factor

### D. Outcome metrics

- velocity tracking
- yaw tracking
- base tilt
- joint target tracking
- control saturation fraction
- qualitative gait notes

## 12. Current Bottom Line

For the current repo-owned MuJoCo Sim2Sim path:

- the **control/observation/action contract** is intentionally aligned with the exported IsaacLab deploy contract
- the **physical scaling knobs default to identity**
- any explicit tuning should be attributable to the CLI/runtime override flags listed above
- MuJoCo should be treated as a **second-opinion validator**, not the final source of truth over hardware

If you need a one-line summary:

- "By default we did not silently rescale friction, mass, or motor strength; the main changes are the simulator backend and the repo-owned MuJoCo execution bridge, with optional runtime overrides available and individually attributable."
