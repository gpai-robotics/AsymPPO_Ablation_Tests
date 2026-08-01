# Sim2Sim Stage A vs Adaptive Comparison

This note records the first clean Isaac-vs-MuJoCo runtime-trace comparison
between:

- the stationary dyn-only Stage A winner, and
- the bounded-latent adaptive recovery challenger.

Its purpose is to answer a very specific question:

- is the remaining MuJoCo problem a general bridge failure, or is it primarily
  specific to the adaptive branch?

## Compared Artifacts

### Stationary deployment-side winner

- policy:
  - `rma_go2_lab/policies/adapt_v3_dyn_only_phase2_stage_a_final.pt`
- Isaac trace:
  - `artifacts/debug/adapt_v3_dyn_only_phase2_stage_a_final_isaac_trace.json`
- MuJoCo trace:
  - `artifacts/debug/adapt_v3_dyn_only_phase2_stage_a_final_fulltrace.json`
- comparison:
  - `artifacts/debug/adapt_v3_dyn_only_phase2_stage_a_final_runtime_compare.json`

### Bounded-latent adaptive challenger

- policy:
  - `rma_go2_lab/policies/adapt_v3_dyn_only_phase2_recovery_low_switch_latent_reg_final.pt`
- Isaac trace:
  - `artifacts/debug/adapt_v3_dyn_only_phase2_recovery_low_switch_latent_reg_model220_isaac_trace.json`
- MuJoCo trace:
  - `artifacts/debug/adapt_v3_dyn_only_phase2_recovery_low_switch_latent_reg_model220_fulltrace.json`
- comparison:
  - `artifacts/debug/adapt_v3_dyn_only_phase2_recovery_low_switch_latent_reg_model220_runtime_compare.json`

## Core Finding

The MuJoCo bridge is **not** fundamentally broken.

Why we can say that:

- the stationary Stage A winner keeps a clean alternating gait in MuJoCo
- the Stage A winner retains good tracking and posture in MuJoCo
- the adaptive bounded-latent branch does not collapse catastrophically anymore,
  but still degrades into a sticky over-supported gait

So the current bottleneck is much more specific than “MuJoCo mismatch exists.”

The current bottleneck is:

- the adaptive branch remains more fragile under cross-engine history mismatch
  than the stationary Stage A winner

## Isaac Comparison

In Isaac, the two policies are close.

### Stage A

- `reward_proxy_mean = 0.519`
- `vel_err_step_mean = 0.058`
- `yaw_err_step_mean = 0.085`
- `base_height_mean = 0.358`

### Bounded-latent adaptive

- `reward_proxy_mean = 0.503`
- `vel_err_step_mean = 0.052`
- `yaw_err_step_mean = 0.080`
- `base_height_mean = 0.365`

Interpretation:

- the adaptive branch does not look fundamentally worse in Isaac
- Isaac alone would not explain the large MuJoCo gap

## MuJoCo Comparison

In MuJoCo, the stationary winner is clearly stronger.

### Stage A

- `reward_proxy_mean = 0.411`
- `vel_err_step_mean = 0.155`
- `yaw_err_step_mean = 0.082`
- `base_height_mean = 0.332`
- `base_tilt_projected_gravity_xy_mean = 0.065`

### Bounded-latent adaptive

- `reward_proxy_mean = 0.302`
- `vel_err_step_mean = 0.267`
- `yaw_err_step_mean = 0.163`
- `base_height_mean = 0.267`
- `base_tilt_projected_gravity_xy_mean = 0.103`

Interpretation:

- the bounded-latent fix improved the earlier recovery branch substantially
- but it still does not match the Stage A winner’s MuJoCo robustness

## Contact Pattern Comparison

The contact trace is where the difference becomes most obvious.

### Stage A MuJoCo

- diagonal support fractions:
  - `FL+RR = 0.267`
  - `FR+RL = 0.262`
- all-4 contact fraction:
  - `0.079`
- none-contact fraction:
  - `0.004`

Interpretation:

- this is a reasonably clean alternating gait
- the bridge is capable of supporting graceful locomotion for at least one
  frozen policy family

### Bounded-latent adaptive MuJoCo

- diagonal support fractions:
  - `FL+RR = 0.116`
  - `FR+RL = 0.049`
- all-4 contact fraction:
  - `0.558`
- none-contact fraction:
  - `0.065`

Interpretation:

- the adaptive branch still falls into a sticky, over-supported, asymmetric
  stepping pattern
- this is not a simple “MuJoCo always looks awkward” effect

## Latent Comparison

The latent signal strongly supports the same conclusion.

### Stage A MuJoCo

- `latent_norm_mean = 7.007`
- `latent_norm_max = 7.007`
- `latent_max_abs_mean = 4.840`
- `latent_max_abs_max = 4.840`

### Bounded-latent adaptive MuJoCo

- `latent_norm_mean = 25.767`
- `latent_norm_max = 121.183`
- `latent_max_abs_mean = 12.496`
- `latent_max_abs_max = 63.505`

Interpretation:

- the adaptive branch is still the one carrying the remaining cross-engine
  latent instability
- the bridge itself is not forcing large latent drift on the stationary winner

## Project Meaning

This comparison narrows the current frontier further:

- the next fix should target adaptive-branch Sim2Sim robustness specifically
- it should not be framed as a generic bridge rescue task

The clean repo-level interpretation is now:

- Stage A dyn-only remains the strongest deployment-side candidate
- bounded-latent recovery remains the strongest adaptive refinement base
- the next adaptive work should be judged by how much of the Stage A MuJoCo gap
  it closes without giving up the recovered online-adaptation story
