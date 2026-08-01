# C2 Supervised Phi Null Runtime Result

Date:

- `2026-05-15`

## Branch Intent

This experiment tested a cleaner C2 decomposition inspired by the reference
repos:

- collect rollout data from the current dyn-only bounded-latent baseline
- log deployable observations plus teacher latent / teacher action targets
- train only `phi(history)` offline with frozen `pi`

The goal was to answer:

**if we optimize `phi` cleanly outside the PPO loop, do we get a better runtime
adaptive policy?**

## What Was Built

- dataset collector:
  [collect_adapt_v3_teacher_dataset.py](/home/bhuvan/projects/rma/rma_go2_lab/scripts/adaptation/collect_adapt_v3_teacher_dataset.py)
- offline trainer:
  [train_adapt_v3_phi_supervised.py](/home/bhuvan/projects/rma/rma_go2_lab/scripts/adaptation/train_adapt_v3_phi_supervised.py)

Dataset:

- `artifacts/datasets/c2_dyn_only_teacher_targets_v1`

Model outputs:

- `artifacts/models/c2_dyn_only_phi_supervised_v1/best.pt`
- `artifacts/models/c2_dyn_only_phi_supervised_v1/last.pt`

## Offline Training Result

The offline optimization worked cleanly:

- train loss decreased steadily
- validation loss decreased steadily
- action imitation loss stayed very small

This confirmed:

- the dataset was valid
- the offline `phi(history) -> z_t` fit is learnable

## Runtime Result

Runtime evaluation of `best.pt` and `last.pt` showed:

- identical learned model weights at the end
- no new gait family
- no clear behavioral improvement over the prior runtime baseline
- same broad weak spot:
  - low-friction and low-friction switch cases

Gait remained:

- `high_duty_diagonal_gait_staggered_touchdown`

So the experiment produced:

- a technical offline training success
- but a **null runtime result**

## Meaning

This is important evidence.

It suggests that:

- better offline latent fit alone is not enough
- the remaining bottleneck is probably deeper than the `phi` optimizer

The most likely remaining issues are:

- the root controller / latent contract itself
- the information content of the deployable observation history
- or both

## Conclusion

This experiment should not be promoted as a new C2 winner.

Its value is diagnostic:

- offline supervision is feasible
- but it did not change runtime behavior enough to replace the current
  dyn-only bounded-latent baseline
