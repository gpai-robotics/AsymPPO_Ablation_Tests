# C2 TCN Anchored Failed Branch

This note archives the anchored TCN continuation that followed the first
unanchored TCN failure.

## Identity

- task:
  `RMA-Go2-Adapt-V3-DynOnly-Phase2-Recovery-LowSwitch-LatentReg-TCN-Anchored`
- run root:
  `/home/bhuvan/tools/IsaacLab/logs/rsl_rl/go2_adapt_v3_phase2_recovery_low_switch_dyn_only_latent_reg_tcn_anchored/2026-05-14_13-55-10`
- status:
  retired

## Why It Existed

This branch kept the TCN `phi(history)` idea but tried to stop the previous
late collapse by:

- warm-starting from the stronger recovery latent-reg artifact
- keeping teacher-action anchoring alive much longer
- reducing pure latent-matching pressure

## What It Improved

Early and mid training looked healthier than the first unanchored TCN branch:

- locomotion held together longer
- latent cosine became strong early
- switch exposure was initially live

## Why It Was Retired

The same fundamental late-training failure pattern returned:

- latent alignment kept improving
- locomotion quality decayed badly
- episodes became short and fall-dominated
- switch exposure collapsed because episodes no longer lived long enough

Representative late-stage pattern:

- high `latent_cosine`
- low `latent_regression`
- very low reward
- short episode length
- large `base_height` termination fraction
- very low `adaptation_switch_applied_frac`

## Conclusion

This branch does not replace the current bounded-latent C2 baseline.

It is useful as a negative result because it shows:

- stronger sequence modeling alone is not enough
- longer teacher anchoring alone is not enough
- C2 still needs a better late-stage locomotion-preservation mechanism

## Successor Truth

After this retirement:

- the current canonical Candidate 2 reference returns to:
  `rma_go2_lab/policies/adapt_v3_dyn_only_phase2_recovery_low_switch_latent_reg_final.pt`
- this anchored branch is archived as a failed follow-up, not a live candidate
