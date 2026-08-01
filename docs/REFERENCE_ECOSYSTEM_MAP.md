# Reference Ecosystem Map

This note consolidates the reference repos currently living under
`reference_repos/`.

Its purpose is not to duplicate those repos. Its purpose is to keep our own
project coherent while we borrow ideas from multiple lineages:

- ETH blind locomotion
- original RMA / CMS-style adaptation
- LeggedGym / Unitree RL Gym
- IsaacLab / Unitree RL Lab
- MuJoCo model and runtime stacks

Use this file when the question is:

- which repo should influence which branch of our project?
- what is conceptually similar versus structurally different?
- where should we copy ideas from, and where should we avoid mixing contracts?

## Short Answer

The reference repos fall into six different roles:

1. ETH blind locomotion reference
2. RMA / privileged-to-blind adaptation reference
3. LeggedGym fixed-policy locomotion reference
4. IsaacLab locomotion and deployment-surface reference
5. MuJoCo model/runtime reference
6. terrain-generation / future OOD reference

The biggest coherence rule for our repo is:

- `C1` should stay closest to the ETH blind-reactive line
- `C2` should stay closest to the RMA line
- deployment tooling should stay closest to the IsaacLab + MuJoCo contract
- robot-model selection should stay closest to Menagerie unless runtime
  integration requires Unitree-specific bridges

## Repo Roles

### 1. ETH blind locomotion

Primary repo:

- `reference_repos/learning_quadrupedal_locomotion_over_challenging_terrain_supplementary`

What it contributes:

- blind locomotion over challenging terrain
- explicit `state + history -> action` student
- strong temporal-history emphasis
- teacher/student distinction without forcing an explicit runtime latent

Most important files we checked:

- `README.md`
- `include/graph/Policy.hpp`
- `include/environment/environment_c100.hpp`
- `applications/test_c100.cpp`
- `applications/test_c010.cpp`

Key structural lessons:

- the blind student is history-conditioned, not history-free
- deployment uses direct history-reactive control
- their history length is long (`100` in the test apps)
- the student observation payload is richer than our current blind policy

How it should affect us:

- this is the strongest conceptual reference for `C1`
- this is the reason `C1-ETHLike-V1` is a direct blind history policy rather
  than another explicit-latent branch

What not to borrow blindly:

- exact Raisim environment details
- exact ANYmal observation dimensionality
- exact actuator stack

### 2. RMA / privileged-to-blind adaptation

Primary repo:

- `reference_repos/rl_locomotion`

What it contributes:

- a practical RMA-derived training line
- privileged policy training first
- blind distillation/adaptation second
- explicit use of history-conditioned student inference

Most important files we checked:

- `README.md`
- `raisimGymTorch/env/envs/rsg_a1_task/runner.py`
- `raisimGymTorch/env/envs/rsg_a1_task/Environment.hpp`
- `raisimGymTorch/env/envs/dagger_a1/dagger.py`
- `raisimGymTorch/env/envs/dagger_a1/Environment.hpp`
- `raisimGymTorch/algo/ppo/module.py`

Key structural lessons:

- privileged training and blind deployment are separated deliberately
- the teacher side uses encoded privileged information
- the blind side is trained through DAgger-style supervision and history
- history is not just an evaluation add-on; it is part of the student design

Important nuance:

- this repo is not a pure original-RMA reproduction
- it is already an application-specific derivative
- that makes it useful as a practical adaptation reference, but not the only
  source of truth for what "RMA" means

How it should affect us:

- strongest conceptual reference for `C2`
- useful reference for teacher/student mechanics in `C1`
- good reminder that privileged training and blind deployment can stay cleanly
  separated

What not to copy blindly:

- exact A1/Raisim environment contracts
- exact latent dimensionalities
- exact reward and curriculum choices

### 3. LeggedGym fixed-policy locomotion

Primary repos:

- `reference_repos/unitree_rl_gym`
- `reference_repos/walk-these-ways-go2`

What they contribute:

- the mainstream Go2 fixed-policy locomotion recipe
- observation/action conventions that are close to the broader ecosystem
- deployment-side habits used by many practical Go2 projects

Important files we checked:

Unitree RL Gym:

- `legged_gym/envs/go2/go2_config.py`
- `legged_gym/envs/base/legged_robot.py`
- `legged_gym/scripts/train.py`
- `legged_gym/scripts/play.py`

Walk These Ways Go2:

- `go2_gym/envs/wrappers/history_wrapper.py`
- `go2_gym_learn/ppo_cse/actor_critic.py`
- `go2_gym_deploy/scripts/deploy_policy.py`
- `go2_gym_deploy/utils/deployment_runner.py`
- `go2_gym_deploy/envs/history_wrapper.py`
- `go2_gym/envs/go2/go2_config.py`

Key structural lessons:

- the fixed-policy observation contract is still the ecosystem default:
  proprio + command + previous action
- history wrappers are a common deployment-side pattern
- deployment often physically separates:
  - body network
  - adaptation/history module
  - runtime runner
- Go2 deployments care heavily about:
  - calibration
  - action scaling
  - state-estimator coupling
  - emergency-stop behavior

How it should affect us:

- use these repos as reference for fixed-policy locomotion hygiene
- use them to sanity-check our deployable observation/action conventions
- use the history-wrapper / deployment-runner pattern as a practical runtime
  design reference

What not to mix carelessly:

- `walk-these-ways` uses an explicit adaptation module over observation history
- that makes it closer to a practical `C2` runtime shape than to a pure `C1`
  blind-reactive controller
- so we should borrow deployment mechanics from it more than scientific
  interpretation

### 4. IsaacLab locomotion and deployment surface

Primary repo:

- `reference_repos/unitree_rl_lab`

What it contributes:

- a clean IsaacLab-native Unitree stack
- manager-based observation/reward/config style
- deployment-config export discipline

Important files we checked:

- `README.md`
- `source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/observations.py`
- `source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/rewards.py`
- `source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/agents/rsl_rl_ppo_cfg.py`
- `source/unitree_rl_lab/unitree_rl_lab/utils/export_deploy_cfg.py`

Key structural lessons:

- the IsaacLab stack is much more explicit about environment contracts
- deployment export should include:
  - joint map
  - control step
  - action scaling
  - observation terms
- reward/observation definitions are modular and named
- gait-phase observations are ecosystem-normal in some lines, not exotic

How it should affect us:

- this is the strongest style reference for our repo infrastructure
- use it to keep our env/config/deploy code legible and exportable
- use it as the main reference for how deploy config should be serialized

What not to over-assume:

- Unitree RL Lab is not our scientific north star for `C1` or `C2`
- it is mainly our engineering/style reference for IsaacLab integration

### 5. MuJoCo model and runtime

Primary repos:

- `reference_repos/mujoco`
- `reference_repos/mujoco_menagerie`
- `reference_repos/unitree_mujoco`

What they contribute:

- MuJoCo engine/runtime
- curated neutral robot models
- Unitree-oriented runtime bridge examples

Important files we checked:

- `mujoco/README.md`
- `mujoco_menagerie/README.md`
- `mujoco_menagerie/unitree_go2/README.md`
- `mujoco_menagerie/unitree_go2/scene.xml`
- `unitree_mujoco/readme.md`

Key structural lessons:

- Menagerie is the cleanest neutral model source
- Unitree MuJoCo is the more practical Unitree-specific runtime bridge
- model choice and runtime choice should not be conflated

How it should affect us:

- use Menagerie `unitree_go2/scene.xml` as the canonical clean model reference
- use `unitree_mujoco` as an integration/runtime reference when hardware-side
  semantics matter
- keep our Sim2Sim logic policy-centric, not tied too tightly to one runtime
  stack

### 6. Terrain generation and future OOD work

Primary repo:

- `reference_repos/terrain-generator`

What it contributes:

- structured terrain generation
- stronger terrain-family control
- a likely future upgrade path for adaptation/OOD work

How it should affect us:

- not a blocker for current `C1`
- promising future input for `C2`, OOD suites, and adaptation stress testing

## How The Ecosystem Maps To Our Branches

### Candidate 1

`C1` should primarily align with:

1. ETH blind locomotion repo
2. LeggedGym / Go2 fixed-policy conventions
3. IsaacLab deployment/export discipline

That means `C1` should be:

- blind at deployment
- history-conditioned if useful
- robust-training-first
- not forced into an explicit latent adaptation story

### Candidate 2

`C2` should primarily align with:

1. original RMA idea
2. `rl_locomotion` as a practical derivative reference
3. our own `Adapt-V3` architecture docs

That means `C2` should preserve:

- privileged encoder `mu`
- base policy `pi`
- history adaptation `phi`
- explicit deployment-time online adaptation story

### Deployment / Sim2Sim

Our deployment stack should primarily align with:

1. IsaacLab / Unitree RL Lab export discipline
2. Menagerie model hygiene
3. Unitree MuJoCo runtime semantics where needed
4. practical deploy runners from `walk-these-ways-go2`

That means our deployment pipeline should stay explicit about:

- deployable observation groups
- action scaling
- joint ordering
- history update semantics
- runtime parity between source and export

## Main Coherence Rules For Our Repo

1. Do not use ETH blind-locomotion references to justify explicit-latent `C2`
   decisions.
2. Do not use RMA references to justify turning `C1` into a half-adaptive
   controller.
3. Do not use deployment/runtime repos as the main scientific reference for
   branch identity.
4. Do use IsaacLab/Unitree RL Lab style as an engineering reference for
   environment, config, and export hygiene.
5. Do treat Menagerie as the cleaner neutral MuJoCo model source unless a
   Unitree-specific runtime bridge is explicitly needed.

## Practical Interpretation For Current Work

Current branch meanings should be:

- `C1-ETHLike-V1`
  - ETH blind-reactive direction
  - teacher-supported during training
  - blind history student at deployment

- `Adapt-V3`
  - RMA-style candidate line
  - explicit online adaptation line
  - should remain separate from the `C1` story

- `scripts/deploy/`
  - should stay aligned with IsaacLab-style explicit contracts
  - should not silently depend on training-only assumptions

## Bottom Line

The ecosystem is coherent if we divide responsibilities clearly:

- ETH blind locomotion tells us what `C1` should feel like
- RMA / `rl_locomotion` tells us what `C2` should preserve
- LeggedGym tells us what the broader quadruped locomotion baseline contract
  looks like
- IsaacLab / Unitree RL Lab tells us how to keep training and deployment code
  structured cleanly
- Menagerie + Unitree MuJoCo tell us how to think about Sim2Sim and runtime
  bridging

The repo becomes incoherent only when we mix those roles without naming which
one we are borrowing from.
