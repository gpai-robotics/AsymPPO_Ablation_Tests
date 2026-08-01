# Artifact Archive Status

This note records the cleanup pass performed on `2026-05-09` to keep the
active project surface small and reduce confusion from superseded branch
attempts.

## Active Artifacts Kept In Place

These remain in active folders because they still represent current or
canonical working surfaces:

- frozen baseline / teacher / adaptation / `Adapt-V3` evaluation folders
- current C1 V2 history-ablation run:
  - `artifacts/evaluations/checkpoint_history_ablation/2026-05-08_19-49-41`
- current teacher root-audit run:
  - `artifacts/evaluations/teacher_dependency_watch/2026-05-09_10-34-56`
- canonical MuJoCo C1 surface:
  - `artifacts/mujoco_eval/c1_ethlike_v1_model_700_candidate`
- deployment compatibility surface:
  - `artifacts/deploy_compat/c1_ethlike_v1_model_700_candidate_unitree_rl_lab`

## Archived On 2026-05-09

The following folders were archived because they were superseded, exploratory,
or failed-attempt surfaces that were no longer meant to be treated as active:

### `artifacts/evaluations/archive/2026-05-09_failed_attempts`

- `history_ablation`
- `history_ablation_v2`
- `history_ablation_v3`
- `checkpoint_history_ablation_manual`
- `checkpoint_history_ablation_2026-05-08_12-42-05`
- `checkpoint_history_ablation_tmp`
- `c1_ethlike_v1_ckpt_sweep`
- `evaluations_tmp`

### `artifacts/debug/archive/2026-05-09_failed_attempts`

- `c1_fixcheck_model_500`
- `adapt_v3_dyn_only_phase2_recovery_low_switch_latent_reg_model220_runtime_compare_plots`

### `artifacts/ood_evaluations/archive/2026-05-09_failed_attempts`

- `c1_ethlike_v1_ckpt_sweep`

## Interpretation Rule

Going forward:

- active folders should contain only current working or canonical-final
  surfaces
- exploratory, superseded, or failed-attempt branches should be moved to an
  archive folder once their conclusions are documented
- if a question depends on archived material, the archive path should be cited
  explicitly instead of leaving that material mixed into active surfaces
