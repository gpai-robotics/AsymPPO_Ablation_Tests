# Active Path

The default repository surface exposes one locomotion lineage:

## Blind Rough MJLAB Asymmetric PPO

This is the canonical successful Isaac Sim -> MuJoCo -> Unitree C++ FSM ->
real-Go2 path.

Active Gym tasks:

- `RMA-Go2-Flat`
- `Go2-Blind-Rough-MJLAB-AsymPPO-V1`
- `Go2-Combined-Flat-MJLAB-Prior-V1` experimental stage 1, not frozen
- `Go2-Blind-Rough-MJLAB-Combined-AsymPPO-Rough-V1` experimental stage 2, not frozen
- `Go2-Blind-Rough-MJLAB-Combined-AsymPPO-Steps-V1` experimental stage 3, not frozen

Primary files:

- `rma_go2_lab/envs/priors/flat_mjlab_prior_cfg.py`
- `rma_go2_lab/models/priors/flat_mjlab_prior_runner_cfg.py`
- `rma_go2_lab/envs/priors/combined_flat_mjlab_prior_cfg.py`
- `rma_go2_lab/models/priors/combined_flat_mjlab_prior_runner_cfg.py`
- `rma_go2_lab/envs/teacher/blind_rough_mjlab_asymppo_cfg.py`
- `rma_go2_lab/models/teacher/ppo_mjlab_asymppo_cfg.py`
- `rma_go2_lab/envs/teacher/combined_rough_blind_mjlab_asymppo_cfg.py`
- `rma_go2_lab/models/teacher/combined_rough_ppo_mjlab_asymppo_cfg.py`
- `rma_go2_lab/envs/teacher/combined_steps_blind_rough_mjlab_asymppo_cfg.py`
- `rma_go2_lab/models/teacher/combined_steps_ppo_mjlab_asymppo_cfg.py`
- `rma_go2_lab/models/asymppo/history_actor_critic.py`
- `rma_go2_lab/envs/mjlab_contract.py`
- `rma_go2_lab/policies/go2_blind_rough_asymppo_mjlab_v1_candidate.md`
- `scripts/deploy/run_unitree_mjlab_sim_deploy.sh`
- `scripts/deploy/build_unitree_mjlab_runtime.sh`

C++ runtime build and recovery guide:
`docs/UNITREE_MJLAB_RUNTIME_BUILD.md`

`rma_go2_lab/models/rma_actor_critic.py` is a two-symbol compatibility shim
for an eager import in the locally patched RSL-RL installation. The active
task does not instantiate either RMA class.

Current frozen candidate artifact, if restored locally or downloaded from a
release/package:

```text
rma_go2_lab/policies/exported/go2_blind_rough_asymppo_mjlab_v1_candidate
```

Use `docs/RUN_COMMANDS.md` for canonical copy-paste commands.

Combined baseline design note:
`docs/COMBINED_ASYMPPO_BASELINE.md`

## Archived Paths

C1, teacher/student, adaptation, RMA, and failed experimental branches remain
available as historical knowledge under:

- `rma_go2_lab/archive/attempts`
- `rma_go2_lab/policies/archive`
- `scripts/archive`
- `docs/archive`
- `artifacts/archive/legacy_workspace`

They are not imported, registered, or exposed by active launchers. Restoring
one is an explicit development task: move its complete dependency set back,
repair imports, and register a distinct task without modifying the active
AsymPPO contract.
