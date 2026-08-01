# Sim2Real Reference Map

This note explains which local reference repos are actually useful for our
sim2real path, and which problems they do or do not solve.

The goal is simple:

- reuse what is already good
- avoid rebuilding solved pieces
- avoid importing the wrong architecture just because it exists nearby

## Current Repo Role

The active repo is now the **source of truth** for:

- candidate selection
- frozen bundle export
- deployable observation contract
- Isaac runtime parity
- MuJoCo runtime/OOD validation

That means external repos should support the path, not define it.

## Reference Roles

## 1. `sim2real_unitree_sdk2py`

Path:

- `reference_repos/sim2real_unitree_sdk2py`

Best use:

- real Unitree SDK2 low-level DDS shell
- mode switch flow
- wireless remote start/stop semantics
- stance gate and operator safety flow

Most useful files:

- `example/go2/low_level/deploy/v1/dep_walk_v1.1.py`
- `example/go2/low_level/debug/mode_switch.py`
- `example/go2/low_level/common/remote_controller.py`
- `example/go2/low_level/common/command_helper.py`

What it solved well:

- getting a policy onto the real Go2 at all
- practical low-level bring-up sequence
- operator-controlled start and stop

What it does **not** solve for us anymore:

- bundle-driven deployment
- multi-policy contract handling
- history-bearing deployment contracts
- canonical export / metadata / validation flow

Takeaway:

- keep the **operational shell**
- drop the hard-coded policy and standalone YAML assumptions

## 2. `mujoco_playground`

Path:

- `reference_repos/mujoco_playground`

Best use:

- clean MuJoCo-side policy replay ideas
- ONNX inference path examples
- joystick/gamepad input patterns
- structured MuJoCo environment randomization examples

Most useful files:

- `mujoco_playground/experimental/sim2sim/README.md`
- `mujoco_playground/experimental/sim2sim/play_go1_joystick.py`
- `mujoco_playground/experimental/sim2sim/gamepad_reader.py`
- `mujoco_playground/_src/locomotion/go1/randomize.py`
- `mujoco_playground/_src/wrapper_torch.py`

What it solved well:

- simple policy-in-MuJoCo replay loop
- ONNX runtime inference instead of a heavy training stack
- gamepad-driven command control
- disciplined randomization examples inside MuJoCo-side locomotion tasks

What it does **not** solve for us:

- real Go2 hardware deployment
- Unitree SDK2 low-level DDS shell
- our exported `blind_history_policy` contract directly
- our IsaacLab-based candidate freeze pipeline

Most valuable lesson for us:

- if we want a lighter deployment runtime later, an **ONNX export path** is a
  reasonable future improvement
- but `mujoco_playground` is a **runtime/replay reference**, not our hardware
  deploy framework

## 3. `unitree_mujoco`

Path:

- `reference_repos/unitree_mujoco`

Best use:

- Unitree-flavored MuJoCo robot scenes
- simulator-side SDK2 bridge ideas
- terrain generation tooling

Most useful files:

- `simulate_python/unitree_mujoco.py`
- `simulate_python/unitree_sdk2py_bridge.py`
- `terrain_tool/terrain_generator.py`
- `terrain_tool/readme.md`

What it solved well:

- publishing/subscribing Unitree SDK2 topics inside MuJoCo
- scene-side terrain generation
- a familiar Unitree robotics simulator structure

What it does **not** solve for us cleanly:

- our export bundle contract
- our history-policy deployment semantics
- our canonical MuJoCo evaluation structure

Takeaway:

- use it as a **scene/bridge/terrain reference**
- do not let it replace the repo-owned deployment surface

## 4. `mujoco_menagerie`

Path:

- `reference_repos/mujoco_menagerie`

Best use:

- canonical MuJoCo robot assets
- baseline scene fidelity reference

What it solved well:

- reliable robot XML assets
- repeatable MuJoCo loading path

What it does **not** solve:

- deployment logic
- joystick/runtime shell
- evaluation taxonomy

Important repo lesson:

- menagerie was the best runtime baseline for our C1 MuJoCo bridge
- but we still had to tune passive damping to reduce the overly stiff feel

## 5. `unitree_rl_lab` deploy folder

Path:

- `reference_repos/unitree_rl_lab/deploy`

Best use:

- understanding how an IsaacLab-style policy is expected to land on the robot
- operator-facing FSM structure for hardware deployment
- ONNXRuntime-based deploy backend example

Most useful files:

- `deploy/robots/go2/config/config.yaml`
- `deploy/robots/go2/src/State_RLBase.cpp`
- `deploy/include/FSM/State_RLBase.h`
- `deploy/include/FSM/CtrlFSM.h`
- `deploy/include/param.h`
- `source/unitree_rl_lab/unitree_rl_lab/utils/export_deploy_cfg.py`

What it clarifies for us:

- deployment is treated as an explicit FSM:
  - Passive
  - FixStand
  - RL velocity state
- the RL state runs in its own fixed-rate thread
- policy inference uses ONNXRuntime
- the exported policy directory is expected to contain:
  - `exported/policy.onnx`
  - `params/deploy.yaml`

Why this matters for our repo:

- it validates the idea that our bundle should be the source of truth for deploy
  metadata
- it makes ONNX a much more credible future backend for us
- it confirms that a stance gate plus explicit FSM transitions is normal, not
  over-cautious

What it does **not** solve directly:

- our `blind_history_policy` contract out of the box
- our exact exported bundle schema
- our MuJoCo evaluation path

Most important repo lesson:

- our current `run_go2_hardware.py` is directionally right on operator flow
- but `unitree_rl_lab` shows a stronger long-term target:
  - FSM-managed hardware states
  - ONNX deploy backend
  - deploy config generated alongside the policy
- because of that, the repo should preserve a compatibility path toward:
  - `policy.onnx`
  - `deploy.yaml`
  - robot-facing FSM deployment

## 6. `IsaacLab`

Path:

- `/home/bhuvan/tools/IsaacLab`

Best use:

- training truth
- export metadata truth
- source runtime trace truth

Takeaway:

- IsaacLab remains the primary truth for:
  - policy definition
  - observation contract
  - source-side behavior
- MuJoCo and sim2real should adapt around that truth, not fork it casually

## 7. ETH rough-terrain supplementary repo

Path:

- `reference_repos/learning_quadrupedal_locomotion_over_challenging_terrain_supplementary`

Best use:

- blind-history locomotion design reference
- terrain/task diversity ideas

What it helps with:

- C1 identity and training design

What it does not directly provide:

- modern export/deployment bundle practices
- Unitree hardware shell

## What We Should Reuse Immediately

### Keep from `sim2real_unitree_sdk2py`

- mode-switch flow
- DDS topic shell
- operator start/stop semantics
- safe stance gate

### Keep from `mujoco_playground`

- ONNX deployment idea for lighter runtime
- gamepad input design patterns
- simple replay-loop structure
- randomization examples for MuJoCo-side env stress

### Keep from `unitree_mujoco`

- terrain generation tooling
- optional simulator-side SDK2 bridge ideas
- additional traversable scenes

### Keep from `unitree_rl_lab`

- explicit hardware FSM structure
- FixStand -> RL state operational flow
- ONNXRuntime robot-side inference idea
- generated deploy-config pattern

### Keep from `mujoco_menagerie`

- canonical robot XML baseline

## What We Should Not Rebuild Again

- ad hoc policy contract files outside the bundle
- one-off MuJoCo viewers with no scenario identity
- hard-coded policy filenames in hardware deploy scripts
- separate old-vs-new deployment logic paths that disagree on observation order

## Recommended Near-Term Sim2Real Roadmap

1. keep current repo as the deployment contract truth
2. use the new `run_go2_hardware.py` shell as the repo-native hardware entrypoint
3. port only the necessary Unitree SDK2 operational pieces forward
4. later, add optional ONNX export/runtime support if we want a lighter deploy path
5. keep MuJoCo evaluation canonical and separate from real-hardware bring-up

## Practical Conclusion

The best blend is now:

- **this repo**
  - candidate selection
  - bundle export
  - deployment contract
- **`sim2real_unitree_sdk2py`**
  - operator-safe Go2 DDS shell
- **`mujoco_playground`**
  - ideas for lightweight runtime replay and future ONNX deployment
- **`unitree_mujoco`**
  - terrain/scene and SDK2-bridge references
- **`unitree_rl_lab` deploy**
  - the cleanest example of IsaacLab-style policy packaging landing in an
    operator-facing robot FSM with ONNXRuntime

That is the shortest path to better sim2real work without wasting effort.
