# C2 Structured Offline Pipeline Runbook

This is the shortest practical runbook for the working structured Candidate 2
pipeline.

Use this when the goal is:

- reproduce the working structured C2 path
- continue from the canonical offline baseline
- avoid reopening the old online PPO Phase 2 collapse story by accident

Helper script:

- `scripts/adaptation/prepare_structured_phase2_offline_cycle.py`
- `scripts/adaptation/prepare_structured_phase2_final_baseline_validation.py`

If you want one generated shell script instead of manually copying the command
blocks below, use:

```bash
python /home/bhuvan/projects/rma/rma_go2_lab/scripts/adaptation/prepare_structured_phase2_offline_cycle.py \
  --cycle-name structured_z27_phase2_phi_supervised_next
```

That writes:

- `artifacts/pipeline_runs/structured_z27_phase2_phi_supervised_next.sh`

## Canonical Artifacts

Structured Phase 1 root:

- `rma_go2_lab/policies/adapt_v3_dyn_only_structured_z27_phase1_stage_a_final.pt`

Working structured Phase 2 baseline:

- `rma_go2_lab/policies/adapt_v3_dyn_only_structured_z27_phase2_phi_supervised_v1_candidate.pt`

Operational baseline card:

- [C2_STRUCTURED_OFFLINE_FINAL_BASELINE_CARD.md](/home/bhuvan/projects/rma/rma_go2_lab/docs/C2_STRUCTURED_OFFLINE_FINAL_BASELINE_CARD.md)
- [C2_RMA_RESTART_BLUEPRINT.md](/home/bhuvan/projects/rma/rma_go2_lab/docs/C2_RMA_RESTART_BLUEPRINT.md)

## Core Idea

The working pipeline is:

1. keep the structured Phase 1 root frozen
2. collect on-policy history rollouts
3. train only `phi(history)` offline
4. evaluate the resulting student runtime

That means:

- no online PPO actor updates in Phase 2
- no joint policy drift while `phi` is learning

The main architecture explorations beyond `v1` have now already been tested:

- pure bottleneck replacement
- residual bottleneck correction

Those branches were useful diagnostics, but none replaced canonical `v1`
overall. So this runbook should now be read primarily as:

- how to reproduce the working final structured offline pipeline
- how to reopen targeted optimization later if the project explicitly needs it

## Collector

Script:

- `scripts/adaptation/collect_structured_phase2_onpolicy_dataset.py`

Canonical collection command pattern:

```bash
CUDA_VISIBLE_DEVICES=1 env TERM=xterm /home/bhuvan/tools/IsaacLab/isaaclab.sh -p \
  /home/bhuvan/projects/rma/rma_go2_lab/scripts/adaptation/collect_structured_phase2_onpolicy_dataset.py \
  --task RMA-Go2-Adapt-V3-DynOnly-StructuredZ27-Phase2-Recovery-LowSwitch \
  --checkpoint /home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/policies/adapt_v3_dyn_only_structured_z27_phase2_phi_supervised_v1_candidate.pt \
  --output-dir /home/bhuvan/projects/rma/rma_go2_lab/artifacts/datasets/<new_dataset_name> \
  --num-envs 128 \
  --steps 2000 \
  --chunk-steps 250 \
  --headless
```

Optional research branch knobs:

- `--adaptation-bottleneck-dim 8` or `12`
- `--adaptation-residual`

Expected dataset keys:

- `policy`
- `policy_history`
- `dynamics_privileged`
- `teacher_latent`
- `teacher_action`
- `rollout_action`
- `command_active`
- `switch_applied`
- `step_index`

## Offline Phi Training

Script:

- `scripts/adaptation/train_structured_phase2_phi_supervised.py`

Canonical training command pattern:

```bash
CUDA_VISIBLE_DEVICES=1 env TERM=xterm /home/bhuvan/tools/IsaacLab/isaaclab.sh -p \
  /home/bhuvan/projects/rma/rma_go2_lab/scripts/adaptation/train_structured_phase2_phi_supervised.py \
  --task RMA-Go2-Adapt-V3-DynOnly-StructuredZ27-Phase2-Recovery-LowSwitch \
  --dataset-dir /home/bhuvan/projects/rma/rma_go2_lab/artifacts/datasets/<new_dataset_name> \
  --output-dir /home/bhuvan/projects/rma/rma_go2_lab/artifacts/models/<new_model_name> \
  --checkpoint /home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/policies/adapt_v3_dyn_only_structured_z27_phase2_phi_supervised_v1_candidate.pt \
  --epochs 12 \
  --batch-size 2048 \
  --lr 3e-4 \
  --latent-coef 1.0 \
  --action-coef 0.10 \
  --latent-l2-coef 1e-3 \
  --active-only
```

Optional research branch knobs:

- `--adaptation-bottleneck-dim 8` or `12`
- `--adaptation-residual`

Optional weighting knobs now supported:

- `--low-friction-threshold`
- `--low-friction-upweight`
- `--switch-upweight`
- `--very-heavy-threshold`
- `--very-heavy-upweight`
- `--weak-motor-threshold`
- `--weak-motor-upweight`

Use these only for targeted tradeoff exploration, not as the default path.

## Evaluation Gate

Run all four:

1. gait
2. blind nominal suite
3. `ood_dynamics_v1`
4. `ood_switch_v1`

Canonical command pattern:

```bash
CUDA_VISIBLE_DEVICES=1 env TERM=xterm /home/bhuvan/tools/IsaacLab/isaaclab.sh -p \
  /home/bhuvan/projects/rma/rma_go2_lab/scripts/eval/gait.py \
  --task RMA-Go2-Adapt-V3-DynOnly-StructuredZ27-Phase2-Recovery-LowSwitch \
  --checkpoint /home/bhuvan/projects/rma/rma_go2_lab/artifacts/models/<new_model_name>/best.pt \
  --terrain-type random_rough \
  --terrain-level 5 \
  --command-profile forward \
  --forced-lin-x 0.55 \
  --forced-lin-y 0.0 \
  --forced-ang-z 0.0 \
  --steps 1000 \
  --num_envs 64 \
  --seed 999 \
  --json-out /home/bhuvan/projects/rma/rma_go2_lab/artifacts/evaluations/<new_model_name>/gait_best_random_rough_l5_forward.json
```

For bottleneck or residual branches, pass the same shape flags to all four eval
commands so the policy skeleton matches the checkpoint shape:

- `--adaptation-bottleneck-dim`
- `--adaptation-residual`

```bash
CUDA_VISIBLE_DEVICES=1 env TERM=xterm /home/bhuvan/tools/IsaacLab/isaaclab.sh -p \
  /home/bhuvan/projects/rma/rma_go2_lab/scripts/eval/run_isolated_suite.py \
  --task RMA-Go2-Adapt-V3-DynOnly-StructuredZ27-Phase2-Recovery-LowSwitch \
  --checkpoint /home/bhuvan/projects/rma/rma_go2_lab/artifacts/models/<new_model_name>/best.pt \
  --suite blind_baseline_v1 \
  --output-dir /home/bhuvan/projects/rma/rma_go2_lab/artifacts/evaluations/<new_model_name>
```

```bash
CUDA_VISIBLE_DEVICES=1 env TERM=xterm /home/bhuvan/tools/IsaacLab/isaaclab.sh -p \
  /home/bhuvan/projects/rma/rma_go2_lab/scripts/eval_ood/run_ood_suite.py \
  --task RMA-Go2-Adapt-V3-DynOnly-StructuredZ27-Phase2-Recovery-LowSwitch \
  --checkpoint /home/bhuvan/projects/rma/rma_go2_lab/artifacts/models/<new_model_name>/best.pt \
  --suite ood_dynamics_v1 \
  --output-dir /home/bhuvan/projects/rma/rma_go2_lab/artifacts/ood_evaluations/<new_model_name>
```

```bash
CUDA_VISIBLE_DEVICES=1 env TERM=xterm /home/bhuvan/tools/IsaacLab/isaaclab.sh -p \
  /home/bhuvan/projects/rma/rma_go2_lab/scripts/eval_ood/run_ood_suite.py \
  --task RMA-Go2-Adapt-V3-DynOnly-StructuredZ27-Phase2-Recovery-LowSwitch \
  --checkpoint /home/bhuvan/projects/rma/rma_go2_lab/artifacts/models/<new_model_name>/best.pt \
  --suite ood_switch_v1 \
  --output-dir /home/bhuvan/projects/rma/rma_go2_lab/artifacts/ood_evaluations/<new_model_name>
```

## Current Selection Rule

Right now:

- `v1` is the working structured offline winner-for-now
- `v2`, `v3`, `resb12`, and `resb12_weakmotor_switch` are archived alternates

So any future branch should beat `v1`, not merely beat `v2` or `v3`.

## Final Baseline Hardening

If the goal is not another research branch but to revalidate the frozen
structured C2 baseline as an operational artifact, use:

```bash
python /home/bhuvan/projects/rma/rma_go2_lab/scripts/adaptation/prepare_structured_phase2_final_baseline_validation.py
```

That writes:

- `artifacts/pipeline_runs/structured_z27_phase2_phi_supervised_v1_final_baseline_validation.sh`

The generated script performs:

1. bundle manifest packaging
2. deployable export generation
3. bundle validation
4. gait, blind nominal, OOD dynamics, and OOD switch revalidation

## When To Reopen Optimization

Only reopen structured C2 optimization if one of these becomes important:

- weak-motor OOD robustness
- switch robustness under heavy or low-friction cases
- low-friction nominal behavior
- stronger deployment-side performance than the older bounded-latent adaptive baseline
