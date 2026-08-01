# Structured Phase 2 Offline Phi Candidate

## Identity

- candidate name:
  `adapt_v3_dyn_only_structured_z27_phase2_phi_supervised_v1_candidate`
- checkpoint:
  `rma_go2_lab/policies/adapt_v3_dyn_only_structured_z27_phase2_phi_supervised_v1_candidate.pt`
- source model dir:
  `artifacts/models/structured_z27_phase2_phi_supervised_v1`
- selected source checkpoint:
  `best.pt`
- date frozen:
  `2026-05-18`

## Purpose

This is the first successful structured Phase 2 candidate produced with a
true RMA-style offline adaptation path:

- frozen structured Phase 1 teacher/root
- on-policy history collection
- supervised `phi(history) -> z_teacher` training
- no online PPO actor drift during Phase 2

## Upstream Root

This candidate is built on the frozen structured Phase 1 root:

- `rma_go2_lab/policies/adapt_v3_dyn_only_structured_z27_phase1_stage_a_final.pt`

## Training Path

Dataset:

- `artifacts/datasets/structured_z27_phase2_onpolicy_v1`

Collector:

- `scripts/adaptation/collect_structured_phase2_onpolicy_dataset.py`

Offline trainer:

- `scripts/adaptation/train_structured_phase2_phi_supervised.py`

## Offline Training Summary

Across 12 epochs, the offline student improved steadily:

- val total loss:
  `1.0832 -> 0.5256`
- val latent loss:
  `1.0810 -> 0.5244`
- val action loss:
  `0.0214 -> 0.0064`
- val latent cosine:
  `0.2873 -> 0.7185`

This was the first structured Phase 2 path that improved latent alignment
without simultaneously drifting the actor online.

## Runtime Evaluation Summary

### Gait

File:

- `artifacts/evaluations/structured_z27_phase2_phi_supervised_v1/gait_best_random_rough_l5_forward.json`

Key signals:

- gait family:
  `high_duty_diagonal_gait_staggered_touchdown`
- diagonal trot score:
  `0.5866`
- foot slip contact mean:
  `0.0657`
- base height mean:
  `0.3841`
- base tilt projected gravity xy mean:
  `0.0577`

### Blind Nominal Suite

File:

- `artifacts/evaluations/structured_z27_phase2_phi_supervised_v1/isolated_suite_best_blind_baseline_v1_random_rough_levelspread_normal_normal_seed999.json`

Key signals:

- average score:
  `12.5054`
- weakest nominal case:
  `fric_min_random_rough_l5 = 7.6510`

### OOD Dynamics

File:

- `artifacts/ood_evaluations/structured_z27_phase2_phi_supervised_v1/ood_suite_best_ood_dynamics_v1_normal_seed999.json`

Key signals:

- average score:
  `10.3283`
- weakest dynamics case:
  `ood_ultra_low_friction_random_rough_l5 = 7.3297`

### OOD Switch

File:

- `artifacts/ood_evaluations/structured_z27_phase2_phi_supervised_v1/ood_suite_best_ood_switch_v1_normal_seed999.json`

Key signals:

- average score:
  `9.1835`
- weakest switch case:
  `ood_switch_low_friction_heavy_random_rough_l5 = 7.3387`

## Verdict

This candidate is the first structured Phase 2 line in the repo that:

- preserves a healthy locomotion family
- survives Phase 2 without the old online PPO collapse
- shows respectable OOD dynamics and switch behavior

It should now be read as:

- the canonical working structured C2 pipeline artifact
- the winner-for-now for the structured offline `phi` path

Two follow-up refresh rounds were tried after this:

- `v2`
  - improved nominal low-friction behavior
  - regressed some dynamics and switch robustness
- `v3`
  - improved gait quality and some OOD dynamics cases
  - still did not beat this `v1` candidate overall

## Remaining Weakness

Low friction remains the main weakness in both nominal and OOD settings.

That means the next refresh round should prioritize:

- preserving the same healthy gait family
- improving low-friction robustness without destabilizing the offline path
