# Unitree RL Lab Go2 Path

This is the safer long-term deployment path for the frozen Go2 baseline.

Why we are taking this path:

- `unitree_rl_lab` already gives us a real FSM runtime:
  - `Passive`
  - `FixStand`
  - `Velocity`
- the policy is executed through ONNX in a dedicated control process
- bad-orientation checks already exist in the runtime
- this is a better final deployment surface than continuing to stretch the
  repo-native Python shell into a full robot controller

## Current Status

The frozen bundle already contains the two key artifacts needed by
`unitree_rl_lab`:

- `policy.onnx`
- bundle-side deploy metadata

We now also have a repo bridge that converts the frozen bundle into the actual
`unitree_rl_lab` runtime layout and observation-group format:

- [materialize_unitree_rl_lab_layout.py](/home/bhuvan/projects/rma/rma_go2_lab/scripts/deploy/materialize_unitree_rl_lab_layout.py)

Important detail:

- our old bundle `.deploy.yaml` was only a compatibility artifact
- it was **not** a drop-in `unitree_rl_lab` runtime config
- the bridge now fixes:
  - Go2 `joint_ids_map` into Unitree SDK motor order
  - multi-input observation groups for `policy_obs` and `policy_history`
  - timestep-major history layout for the history input

## Known Conservative Limitation

For this blind-history Go2 bundle, `base_lin_vel` is still conservative:

- the materialized runtime config includes `base_lin_vel`
- the local `unitree_rl_lab` runtime patch accepts it
- but it is still zero-filled today unless we later wire in a trusted
  estimator

So this path improves runtime structure and safety architecture, but it does
not magically solve the missing hardware linear-velocity source by itself.

## Stage The Frozen Bundle

Use the repo helper:

```bash
cd /home/bhuvan/projects/rma/rma_go2_lab

python scripts/deploy/prepare_unitree_rl_lab_go2_runtime.py \
  --bundle-dir /home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/policies/exported/c1_blind_rough_omni_usable_v1_final \
  --force
```

This stages the runtime under:

- `reference_repos/unitree_rl_lab/deploy/robots/go2/config/policy/velocity/<policy_name>`

The Go2 config now defaults `Velocity.policy_dir` to:

- `config/policy/velocity`

which matches how `unitree_rl_lab` already discovers the latest staged runtime.

## Raw Materialization Primitive

If you want the lower-level primitive directly:

```bash
cd /home/bhuvan/projects/rma/rma_go2_lab

python scripts/deploy/materialize_unitree_rl_lab_layout.py \
  --bundle-dir /home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/policies/exported/c1_blind_rough_omni_usable_v1_final \
  --output-dir /tmp/c1_unitree_rl_lab_go2 \
  --force
```

This emits:

- `/tmp/c1_unitree_rl_lab_go2/exported/policy.onnx`
- `/tmp/c1_unitree_rl_lab_go2/params/deploy.yaml`
- copied metadata sidecars
- `/tmp/c1_unitree_rl_lab_go2/params/bundle_compat.deploy.yaml`

The copied `bundle_compat.deploy.yaml` is just for traceability. The runtime
should use `params/deploy.yaml`.

## Go2 Runtime Policy Directory

Go2 runtime config:

- [config.yaml](/home/bhuvan/projects/rma/rma_go2_lab/reference_repos/unitree_rl_lab/deploy/robots/go2/config/config.yaml)

Recommended `Velocity.policy_dir`:

- `config/policy/velocity`

## Build The Runtime

From Unitree’s own README, the expected shape is:

1. install `unitree_sdk2`
2. compile `go2_ctrl`

Main files:

- [main.cpp](/home/bhuvan/projects/rma/rma_go2_lab/reference_repos/unitree_rl_lab/deploy/robots/go2/main.cpp)
- [State_RLBase.cpp](/home/bhuvan/projects/rma/rma_go2_lab/reference_repos/unitree_rl_lab/deploy/robots/go2/src/State_RLBase.cpp)
- [CtrlFSM.h](/home/bhuvan/projects/rma/rma_go2_lab/reference_repos/unitree_rl_lab/deploy/include/FSM/CtrlFSM.h)

## Recommended Rollout Order

1. Keep using the read-only DDS probe first.
2. Use the repo-native Python shell only for:
   - mapping inspection
   - DDS reconnaissance
   - limited debugging
3. Stage the frozen bundle into `unitree_rl_lab`.
4. Validate `FixStand` first.
5. Only then try `Velocity`.

## Why This Is Better

This path gives us:

- a real state machine
- clearer state ownership
- cleaner transition boundaries
- ONNX runtime integration that already matches how Unitree ships the deploy
  architecture
- an easier place to add hard safety checks without burying everything inside a
  one-off Python script
