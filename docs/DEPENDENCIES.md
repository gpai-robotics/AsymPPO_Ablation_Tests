# Dependency Boundary

This repo is the first-party RL codebase. Large third-party repositories,
simulators, SDKs, generated logs, and raw training checkpoints are intentionally
not treated as normal source files.

## First-Party Code In This Repo

These paths are owned by this project and should be reviewed/committed here:

```text
rma_go2_lab/
  IsaacLab task configs, model configs, policy code, exported active bundles.

scripts/eval/
  IsaacLab playback and evaluation utilities.

scripts/deploy/
  Export, bundle validation, MuJoCo validation, Unitree runtime preparation,
  network helpers, hardware probes, and deployment wrappers.

configs/
  Cross-simulator validation profiles and scenario contracts.

patches/
  Local patches required to make external reference repos match this project.

docs/
  Project decisions, runbooks, branch cards, retrospectives, and onboarding.
```

## External Runtime Dependencies

These must be installed separately on the machine:

```text
Isaac Sim / IsaacLab
  Required for training, IsaacLab playback, and Isaac-side validation.
  This repo assumes IsaacLab is available through scripts/isaaclab_user.sh.

Go2 IsaacLab USD asset
  Required for IsaacLab tasks. Set GO2_USD_PATH when the asset is not at the
  historical workstation path.

MuJoCo Python environment
  Required for scripts/deploy/run_sim2sim.py, MuJoCo OOD suites, terrain
  materialization, and cross-simulator audits.

Unitree C++ SDK / DDS stack
  Required for Unitree MJLAB C++ sim and hardware controller builds.

Unitree Python SDK2 bindings
  Required only for Python read-only hardware probes/monitors and the older
  Python hardware runner.
```

Do not add `torch` to this repo's `pyproject.toml`. Isaac Sim Python owns the
PyTorch/CUDA/NCCL stack. Installing arbitrary PyPI `torch` into Isaac Sim Python
has already caused CUDA library mismatch failures.

## `reference_repos/`

`reference_repos/` is ignored by git and is a local dependency workspace. It is
not part of this repo's committed source tree.

Required for the active AsymPPO deployment runtime:

```text
reference_repos/unitree_rl_mjlab/
  Unitree MJLAB C++ FSM controller and simulator.
  Required by scripts/deploy/run_unitree_mjlab_sim_deploy.sh.
  Must have patches/unitree_rl_mjlab/go2_scripted_controller.patch applied.

reference_repos/unitree_sdk2/
  C++ Unitree SDK source/install prefix used by build_unitree_mjlab_runtime.sh.
  The build script can clone/build this if it is missing and network access is
  available.
```

Required for MuJoCo validation and terrain tooling:

```text
reference_repos/mujoco_menagerie/
  Canonical MuJoCo Go2 scene used by several sim2sim/audit scripts.

reference_repos/unitree_mujoco/
  Unitree rough terrain assets and terrain recipe generation helpers.

reference_repos/mjlab/
  MJLAB terrain generator package used by materialize_mjlab_mujoco_terrain.py.
```

Required for Python hardware diagnostics if not installed into the selected
hardware Python environment:

```text
reference_repos/sim2real_unitree_sdk2py/
  Provides unitree_sdk2py imports for read-only probes and Python monitor tools.
```

Historical/reference-only mirrors may exist locally, but they are not required
for the current active deployment path unless a branch-specific document says
so.

## Reference Repo Resolution

Some Python deployment tools resolve reference repos in this order:

```text
1. Explicit environment variable
2. <repo>/reference_repos/<name>
3. shared workspace RefRepo/<name>
```

Supported override variables:

```bash
export RMA_UNITREE_RL_MJLAB_ROOT=/path/to/unitree_rl_mjlab
export RMA_MUJOCO_MENAGERIE_ROOT=/path/to/mujoco_menagerie
export RMA_UNITREE_MUJOCO_ROOT=/path/to/unitree_mujoco
export RMA_MJLAB_ROOT=/path/to/mjlab
export RMA_UNITREE_SDK2PY_ROOT=/path/to/sim2real_unitree_sdk2py
```

The active shell launcher currently expects the Unitree MJLAB runtime at:

```text
reference_repos/unitree_rl_mjlab/
```

So for the active deployment path, keep that local mirror in place unless the
launcher is updated to use the resolver.

## Active AsymPPO Deployment Minimum

For the recovered active AsymPPO path, the minimum practical dependency set is:

```text
Committed in this repo:
  scripts/deploy/run_unitree_mjlab_sim_deploy.sh
  scripts/deploy/build_unitree_mjlab_runtime.sh
  scripts/deploy/prepare_unitree_rl_mjlab_go2_runtime.py
  scripts/deploy/validate_unitree_mjlab_go2_fsm_runtime.py
  patches/unitree_rl_mjlab/go2_scripted_controller.patch

Local external dependencies:
  restored/downloaded exported policy bundle under rma_go2_lab/policies/exported/
  reference_repos/unitree_rl_mjlab/
  reference_repos/unitree_sdk2/ or /opt/unitree_robotics
  C++ build deps: cmake, g++, build-essential, yaml-cpp, Boost, Eigen3, fmt
  ONNX Runtime files bundled under the Unitree MJLAB deploy thirdparty folder
```

For training or IsaacLab playback, also require:

```text
Isaac Sim / IsaacLab
Go2 USD asset via GO2_USD_PATH
raw RSL-RL checkpoint for playback/export, if not using a restored exported bundle
```

For MuJoCo validation, also require:

```text
MuJoCo Python environment
reference_repos/mujoco_menagerie/
reference_repos/unitree_mujoco/
reference_repos/mjlab/ when using MJLAB terrain materialization
```
