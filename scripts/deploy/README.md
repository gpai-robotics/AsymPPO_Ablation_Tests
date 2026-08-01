# Deployment Scripts

> Active-path notice: the canonical deployment surface is the AsymPPO
> candidate through `run_unitree_mjlab_sim_deploy.sh`. Build the C++ runtime
> with `build_unitree_mjlab_runtime.sh`. Recovery guide:
> `docs/UNITREE_MJLAB_RUNTIME_BUILD.md`. Use `docs/RUN_COMMANDS.md` for
> copy-paste commands.

This folder is the isolated deployment surface for the project.

Its purpose is to keep hardware-facing work separate from:

- training scripts in `IsaacLab`
- simulation evaluation scripts in `scripts/eval/`
- exploratory utilities in `rma_go2_lab/tools/`

## Intended Scope

Use `scripts/deploy/` for deployment-specific tasks only:

1. exporting frozen policy checkpoints into deployment-ready formats
2. validating deployable observation contracts
3. running simulation-side deployment rehearsals with the exact deployed I/O
4. running MuJoCo Sim2Sim transfer rehearsal
5. building policy bundles for robot-side runtime use
6. hardware logging / replay / inspection helpers

Do not put general evaluation or training code here.

## Planned Script Roles

These filenames are the intended stable roles for this folder:

- `export_policy.py`
  - export a frozen checkpoint to deployment-ready artifacts
  - expected outputs:
    - TorchScript
    - ONNX if needed
    - sidecar metadata
    - unitree_rl_lab-compatible `deploy.yaml` compatibility artifact

- `validate_bundle.py`
  - verify that a deployment bundle is self-consistent
  - check:
    - checkpoint identity
    - task identity
    - observation contract
    - latent mode
    - policy type

- `play_deploy_policy.py`
  - simulation-side rehearsal using the exact deployable I/O contract
  - meant to replace one-off ad hoc “play” scripts for final candidates

- `run_sim2sim.py`
  - MuJoCo transfer rehearsal entrypoint
  - currently validates:
    - deployable observation contract
    - action scaling
    - control/update timing
    - history update semantics for adaptive students
    - backend/model readiness for the chosen MuJoCo path
  - when `mujoco` is available, it can also call the repo-owned runtime bridge
    for a real rehearsal attempt

- `run_mujoco_ood_suite.py`
  - structured MuJoCo-side OOD / harsh evaluation runner
  - uses named scenario manifests instead of ad hoc viewer testing
  - intended to mirror the discipline of Isaac-side checkpoint and OOD sweeps

- `mujoco_ood_scenarios.py`
  - canonical MuJoCo scenario registry
  - defines nominal, disturbance, continuous-terrain, hidden-env, and limit suites

- `run_harsh_sim2sim_suite.py`
  - legacy compatibility wrapper only
  - forwards to the canonical MuJoCo suite runner
  - do not use for new evaluation work

- `package_candidate.py`
  - collect the frozen deployment candidate into one versioned directory
  - include:
    - exported policy
    - metadata
    - freeze note reference
    - expected control-rate info

- `materialize_unitree_rl_lab_layout.py`
  - convert a frozen bundle into a `unitree_rl_lab`-style layout
  - emits:
    - `exported/policy.onnx`
    - `params/deploy.yaml`
    - copied bundle metadata sidecars

- `inspect_log.py`
  - offline helper for deployment logs
  - summarize:
    - command tracking
    - posture drift
    - abort causes

- `run_go2_hardware.py`
  - first repo-native Go2 hardware bring-up runner
  - reads the exported bundle contract directly
  - mirrors the old SDK2 start/stop safety flow while using the new runtime surface

## Current State

Right now the repo already has a few deployment-adjacent pieces outside this
folder:

- `docs/DEPLOYMENT_PLAN.md`
- `rma_go2_lab/tools/keyboard_play_go2.py`
- `rma_go2_lab/policies/exported/`

Those are legacy starting points.

The goal from here is:

- keep deployment planning in `docs/`
- keep deployment scripts in `scripts/deploy/`
- keep frozen exported artifacts in `rma_go2_lab/policies/exported/`

## Current Scaffold

The initial stable deployment entrypoints now exist:

- `export_policy.py`
- `package_candidate.py`
- `validate_bundle.py`
- `play_deploy_policy.py`
- `run_sim2sim.py`
- `run_mujoco_ood_suite.py`
- `inspect_log.py`

These started lightweight, but the active dyn-only path now has:

- real export
- deploy-side rehearsal
- source-vs-export parity smoke
- Sim2Sim preflight gating

The remaining missing layer is the actual MuJoCo runtime bridge.

That bridge now exists in first form as:

- `mujoco_runtime.py`

and now supports:

- runtime parity traces
- keyboard teleop for viewer-side command testing
- scheduled command switches
- scheduled external wrench disturbances
- scenario-driven MuJoCo OOD suites

It still depends on having the MuJoCo Python backend available in the
active environment.

## Active Candidate Order

The current deployment candidate order is defined by:

- `README.md`
- `docs/ACTIVE_PATHS.md`
- `docs/RUN_COMMANDS.md`
- `docs/ACTIVE_ASYMPPO_DEPLOYMENT_ARTIFACTS.md`

As of now, that means:

1. Unitree default controller as the hardware sanity reference
2. `go2_blind_rough_asymppo_mjlab_v1_candidate` as the active learned policy
3. archived C1, blind-baseline, teacher, and Adapt-V3 paths only when a
   deliberate comparison or recovery task requires them

Do not treat older `B2`, `studentNA`, or Adapt-V3 bundles as the default
hardware deployment candidates. They are historical references unless a
candidate card explicitly promotes them back into the active path.

## Canonical MuJoCo Eval Surface

Use only these for MuJoCo evaluation work:

1. `run_sim2sim.py`
   - single scenario / viewer / trace rehearsal
2. `mujoco_ood_scenarios.py`
   - named scenario definitions
3. `run_mujoco_ood_suite.py`
   - batch suite execution and compact summaries
   - supports repeated seeded rollouts per scenario
   - supports controlled reset variation for less trajectory-fragile MuJoCo evaluation
4. `compare_runtime_traces.py`
   - Isaac vs MuJoCo parity comparison

Treat the following as legacy/debug-only:

- `run_harsh_sim2sim_suite.py`
- ad hoc files under `artifacts/debug/`
- old MuJoCo outputs under `artifacts/ood_evaluations/c1_mujoco_*`

## Canonical MuJoCo Eval Outputs

New MuJoCo eval outputs should live under:

`artifacts/mujoco_eval/<bundle_name>/<suite_name>/`

Expected structure:

- `suite_summary.json`
- `suite_summary.csv`
- `scenario_defs/`
- `scenario_runs/`

This keeps:

- scenario definitions
- scenario results
- suite-level ranking

separate and readable.

Current canonical suite families:

- `mujoco_nominal_v1`
- `mujoco_disturb_v1`
- `mujoco_continuous_v1`
- `mujoco_hidden_env_v1`
- `mujoco_limit_v1`
