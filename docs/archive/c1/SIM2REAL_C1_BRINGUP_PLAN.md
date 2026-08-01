# C1 Sim2Real Bring-Up Plan

This is the first repo-native bring-up note for taking the frozen C1 finalist
from export bundle to real Unitree Go2 hardware.

The target policy is:

- `rma_go2_lab/policies/exported/c1_ethlike_v1_model_700_candidate/`

This note exists so the hardware path does not remain trapped in the old
`sim2real_unitree_sdk2py` repo assumptions.

Cross-reference:

- `docs/SIM2REAL_REFERENCE_MAP.md`

## Purpose

The current repo is already stronger than the old deploy repo at:

- candidate selection
- frozen contract export
- Isaac runtime parity
- MuJoCo runtime and OOD evaluation

The remaining job is to combine that with the practical Go2 hardware shell from
the old SDK2 bring-up path:

- low-level DDS mode switch
- wireless remote start/stop
- stance gate before policy start
- safe operator-controlled abort

Related deployment reference:

- `reference_repos/unitree_rl_lab/deploy`
  - useful because it shows the stronger long-term target:
    - FSM-managed hardware states
    - generated deploy config
    - ONNXRuntime robot-side inference

## What Changes Relative To The Old Repo

The old repo deployed a hard-coded policy file and an external YAML contract.

The new repo should deploy from the exported bundle directly:

- `bundle_manifest.json`
- `*.deploy_config.json`
- `*.export_metadata.json`
- `*.torchscript.pt`

That means the hardware runner should not assume:

- hard-coded `48D` policy forever
- hard-coded policy filename
- separate handwritten policy contract

Instead, it should read the bundle and enforce:

- policy kind
- observation contract
- history length
- action scale / offset
- default joint pose
- control timestep

## First Hardware Contract

For frozen C1 the expected hardware contract is:

```text
inputs:
  policy_obs       [48]
  policy_history   [4800]

output:
  action           [12]
```

Policy kind:

- `blind_history_policy`

Meaning:

- no privileged inputs
- no latent adaptation state at deploy time
- per-step history update must match the exported runtime contract

## Repo-Native Hardware Runner

The first scaffolded runner now lives at:

- `scripts/deploy/run_go2_hardware.py`

Current responsibilities:

- read the frozen bundle
- validate `blind_history_policy`
- resolve TorchScript + deploy config
- switch robot into low-level DDS mode
- wait for state stream
- zero-torque gate
- move to default stance
- hold default until operator presses `A`
- run the policy at exported `50 Hz`
- stop when operator presses `SELECT`

## Important Current Limitation

The first runner intentionally uses:

- IMU gyro
- projected gravity
- joint position / velocity
- last action

But it currently fills:

- base linear velocity

with zeros on hardware.

That is acceptable for a first scaffold because it makes the missing part
explicit instead of silently pretending we already have a clean state estimator.

Before real floor walking, we should decide the first hardware velocity source:

1. safe temporary approximation for indoor bring-up only
2. Unitree state estimate if trustworthy
3. repo-owned estimator later if needed

## Recommended Bring-Up Ladder

### Stage 0: Dry contract bring-up

Run:

```bash
python scripts/deploy/run_go2_hardware.py \
  --bundle-dir rma_go2_lab/policies/exported/c1_ethlike_v1_model_700_candidate \
  --dry-run
```

Success means:

- bundle resolves
- policy loads
- dimensions match
- no DDS dependency needed yet

### Stage 1: DDS shell only

Goal:

- verify low-level mode switch and state stream
- do not start floor walking yet

Use:

- zero torque
- move to default
- hold default

### Stage 2: Bench / hanger sanity

Goal:

- verify observation ordering
- verify joint ordering
- verify action scale and stance behavior

Do not evaluate locomotion quality here.

### Stage 3: Flat indoor first locomotion

Goal:

- forward walk only
- short duration
- operator on immediate abort

### Stage 4: Controlled flat stress

Goal:

- stop / restart
- mild command changes
- confirm the policy is controllable enough for real floor use

## What To Port From The Old Repo

Keep:

- mode switch flow
- DDS topic setup
- operator start / stop semantics
- stance gate before run

Drop:

- hard-coded policy filename
- handwritten standalone YAML policy contract
- old repo-local assumption that deployment equals one flat 48D policy forever

## What Still Needs To Be Added

Before first serious C1 floor trials, we still want:

- explicit hardware logging in the new runner
- command / observation / action CSV traces
- better hardware-side velocity handling
- optional limited-step safety timeout
- optional torque / joint-limit monitoring

## Decision Rule

If C1 clears the first flat indoor hardware stages safely, it becomes the fixed
no-adaptation real-robot reference while we prepare the adaptive C2 line.
