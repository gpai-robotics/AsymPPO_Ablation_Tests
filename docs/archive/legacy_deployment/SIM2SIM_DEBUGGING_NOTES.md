# Sim2Sim Debugging Notes

This note captures the non-obvious bridge issues discovered while bringing
`adapt_v3_dyn_only_phase2_stage_a_final` from IsaacLab export into the repo-owned
MuJoCo runtime.

These issues were important because the policy could appear "alive" while still
being semantically wrong in ways that would be easy to miss in a casual demo.

## Candidate

- Policy:
  [adapt_v3_dyn_only_phase2_stage_a_final.pt](/home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/policies/adapt_v3_dyn_only_phase2_stage_a_final.pt)
- Bundle:
  [adapt_v3_dyn_only_phase2_stage_a_final](/home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/policies/exported/adapt_v3_dyn_only_phase2_stage_a_final)

## Hidden Issues Found

1. MuJoCo local base velocity was misleading.
- Using `mj_objectVelocity(..., local=1)` produced a body-frame velocity that was
  axis-mismatched relative to the policy's expected `base_lin_vel`.
- Symptom:
  the robot appeared to be moving forward in world coordinates while the policy
  was told it was moving backward in local `x`.
- Fix:
  compute world-frame spatial velocity from `data.cvel` and rotate it into the
  base frame explicitly using `xmat.T`.

2. Reset posture matched joints but not the floating base.
- The MuJoCo bridge originally reset joints to the exported default pose but
  inherited the Menagerie root pose.
- Symptom:
  a quiet but meaningful mismatch at rollout start.
- Fix:
  export and apply the IsaacLab base init pose explicitly:
  `pos=(0, 0, 0.4)`, `quat=(1, 0, 0, 0)`.

3. Training-side joint tensor order did not match the naive deploy guess.
- The blind student did not use leg-major ordering.
- The training order was grouped by joint type:
  hips `FL, FR, RL, RR`, then thighs `FL, FR, RL, RR`, then calves `FL, FR, RL, RR`.
- Symptom:
  the bridge produced plausible-looking but systematically wrong locomotion,
  especially sideways drift and poor posture retention.
- Fix:
  align `joint_names`, `actuator_names`, and `default_joint_pos` in the export
  bundle and MuJoCo runtime to the true training-side order.

4. Rich deploy metadata mattered.
- A minimal export was not enough to debug the bridge reliably.
- Fix:
  emit a deploy config sidecar containing:
  joint names, actuator names, default joint pose, base init pose, gains, action
  semantics, and observation layout.

## Why These Issues Matter

Each issue above is the kind of bug that can survive basic smoke tests:

- export loads
- runtime executes
- robot moves

But the policy can still be semantically wrong.

For adaptive locomotion, those mismatches are especially costly because they
poison both the current observation and the history stream used by `phi(history)`.

## Before/After Signal

Fixed-command comparison under `vx=0.5, vy=0.0, yaw=0.0`.

### MuJoCo Before Key Fixes

- `vel_err_step_mean` around `0.625`
- `yaw_err_step_mean` around `0.614`
- `base_height_mean` around `0.250`
- `base_tilt_projected_gravity_xy_mean` around `0.241`

### MuJoCo After Velocity/Root/Order Fixes

- 40-step run:
  - `vel_err_step_mean = 0.218`
  - `yaw_err_step_mean = 0.083`
  - `base_height_mean = 0.340`
  - `base_tilt_projected_gravity_xy_mean = 0.052`

- 200-step run:
  - `vel_err_step_mean = 0.163`
  - `yaw_err_step_mean = 0.080`
  - `base_height_mean = 0.333`
  - `base_tilt_projected_gravity_xy_mean = 0.063`

### IsaacLab Fixed-Command Reference

- 40-step run:
  - `vel_err_step_mean = 0.143`
  - `yaw_err_step_mean = 0.103`
  - `base_height_mean = 0.389`

- 200-step run:
  - `vel_err_step_mean = 0.071`
  - `yaw_err_step_mean = 0.096`
  - `base_height_mean = 0.383`

## Current Interpretation

The bridge is no longer fundamentally broken.

The remaining gap now looks like ordinary simulator/model mismatch rather than a
core deployment-contract wiring failure.

That is a much healthier place to be for future Sim2Sim and Sim2Real work.

Follow-up comparison note:

- `docs/SIM2SIM_STAGEA_VS_ADAPTIVE_COMPARISON.md`
  - records the later direct runtime-trace comparison showing that the
    stationary Stage A winner keeps a clean MuJoCo gait while the adaptive
    bounded-latent branch remains the more fragile policy family
