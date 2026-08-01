# C1 StageA Model 400 Deploy Card

This note is the canonical deployment card for the current Candidate 1 blind-history policy.

Cross-reference for the next shared transfer step:

- [C1_C2_TRANSFER_EXECUTION_PLAN.md](/home/bhuvan/projects/rma/rma_go2_lab/docs/C1_C2_TRANSFER_EXECUTION_PLAN.md)

Current plain-language C1 vs C2 standing:

- [C1_C2_STATUS_COMPARISON.md](/home/bhuvan/projects/rma/rma_go2_lab/docs/C1_C2_STATUS_COMPARISON.md)

## Identity

- task:
  `RMA-Go2-C1-ETHLike-V3-StageA`
- source checkpoint:
  `/home/bhuvan/tools/IsaacLab/logs/rsl_rl/go2_c1_ethlike_v3_v4teacher300/2026-05-11_13-10-12/model_400.pt`
- exported bundle:
  `rma_go2_lab/policies/exported/c1_ethlike_v3_model_400_candidate/`
- policy kind:
  `blind_history_policy`

## Why This Candidate Matters

This is the first Candidate 1 branch in the repo that simultaneously cleared:

- repaired upstream teacher truth through `Teacher V4 model_300`
- live student history-path training
- student-side history ablations showing `normal > zero/frozen`
- export/source parity validation
- Isaac deployment rehearsal
- MuJoCo nominal and OOD deployment rehearsal

It is the current best deployment-facing blind-history artifact in the repo.

## Upstream Training Story

Teacher of record:

- `Teacher V4 model_300`
- `/home/bhuvan/tools/IsaacLab/logs/rsl_rl/go2_privileged_teacher_rough_v4_terrain_aux/2026-05-09_10-34-56/model_300.pt`

Student task:

- `RMA-Go2-C1-ETHLike-V3-StageA`

Training intent:

- blind deployable policy at inference
- teacher-supported student during training
- explicit history-target supervision
- explicit teacher imitation pressure

What changed relative to earlier C1 work:

- the student now trains from a teacher that is actually validated as using both
  terrain and dynamics on trusted rough-terrain probes
- the history path is not only present in the architecture, but explicitly
  supervised and later behaviorally validated

## History-Use Evidence

Canonical checkpoint:

- `model_400.pt`

Isaac-side history ablation artifacts:

- `artifacts/evaluations/checkpoint_history_ablation/2026-05-11_13-10-12/c1_history_switch_v1_model_400.json`
- `artifacts/evaluations/checkpoint_history_ablation/2026-05-11_13-10-12/c1_history_push_v1_model_400.json`

What those show:

- `normal` beats `frozen`
- `normal` beats `zero`
- strongest wins appear on:
  - friction switch
  - mass switch
  - push recovery

This is the first clear proof that the deployed history pathway is behaviorally load-bearing on the intended temporal-robustness suites.

## Deploy Contract

Frozen bundle contract:

```text
inputs:
  policy_obs       [48]
  policy_history   [4800]

output:
  action           [12]
```

Key deploy config truth:

- history length:
  `100`
- control rate:
  `50 Hz`
- deployable observation groups:
  `policy`, `policy_history`

## Export and Parity Validation

Canonical deploy artifact:

- `rma_go2_lab/policies/exported/c1_ethlike_v3_model_400_candidate/c1_ethlike_v3_model_400_candidate.torchscript.pt`

Important export fixes that were required:

- policy-history length had to be inferred correctly as `100`
- the blind-history export wrapper had to rebuild `history_projection` as
  `Linear -> ELU` to match the trained source policy

Canonical Isaac deploy rehearsal artifact:

- `artifacts/deploy_eval/c1_ethlike_v3_model_400_candidate/isaac_deploy_rehearsal.json`

Source/export parity after the fix:

- `action_abs_diff_mean = 6.0e-09`
- `action_mse_mean = 7.2e-16`
- `action_max_abs_diff = 4.77e-07`

Interpretation:

- source checkpoint and exported TorchScript now match essentially perfectly
- the deployment bundle is trustworthy as a frozen export surface

## Isaac Deploy Rehearsal

Artifact:

- `artifacts/deploy_eval/c1_ethlike_v3_model_400_candidate/isaac_deploy_rehearsal.json`

Key metrics:

- `vel_err_step_mean = 0.0679`
- `base_height_mean = 0.3496`
- `base_tilt_projected_gravity_xy_mean = 0.0667`
- no terminations across the rehearsal

Interpretation:

- exported deployment contract works
- Isaac-side deploy rehearsal is stable
- no immediate deployment-surface crash or parity drift remains

## MuJoCo Runtime Validation

Canonical artifacts:

- nominal runtime:
  `artifacts/deploy_eval/c1_ethlike_v3_model_400_candidate/mujoco_runtime.json`
- hidden-env suite:
  `artifacts/mujoco_eval/c1_ethlike_v3_model_400_candidate/mujoco_hidden_env_v1/suite_summary.json`
- moderate disturbance suite:
  `artifacts/mujoco_eval/c1_ethlike_v3_model_400_candidate/mujoco_disturb_v2_moderate/suite_summary.json`
- continuous corridor suite:
  `artifacts/mujoco_eval/c1_ethlike_v3_model_400_candidate/mujoco_continuous_v1/suite_summary.json`
- MuJoCo history-ablation suites:
  `artifacts/mujoco_eval_history_ablation/c1_ethlike_v3_model_400_candidate/`

### Nominal MuJoCo runtime

- `reward_proxy_mean = 0.4650`
- `vel_err_step_mean = 0.1363`
- `yaw_err_step_mean = 0.0716`
- `base_height_mean = 0.3222`
- `base_tilt_projected_gravity_xy_mean = 0.0594`
- `ctrl_saturation_frac_mean = 0.0`

Interpretation:

- nominal runtime transfer is viable
- no immediate MuJoCo transfer disaster

### Hidden environment mismatch

Flat-ground hidden-parameter cases are broadly healthy.

Examples:

- `hidden_ultra_high_friction_flat` score `8.33`
- `hidden_very_heavy_payload_flat` score `7.74`
- `hidden_ultra_low_friction_flat` score `6.46`
- `hidden_very_weak_motor_flat` score `6.43`

Interpretation:

- hidden friction / mass / motor mismatch is not the primary deployment weakness on flat ground

### Moderate disturbances

The moderate disturbance suite is the canonical deployment-facing disturbance benchmark.

Results:

- `flat_command_step_up_moderate` score `10.21`
- `flat_yaw_pulse_moderate` score `7.99`
- `flat_command_step_down_moderate` score `6.95`
- `flat_lateral_push_moderate` score `6.66`
- `flat_yaw_torque_pulse_moderate` score `6.44`

Interpretation:

- the candidate handles realistic command changes well
- moderate pushes and yaw torque pulses are survivable and useful as deployment-side tests

### Continuous corridor

The continuous-corridor suite is now genuinely validated and no longer blocked by tooling.

Results:

- `continuous_corridor_nominal` score `5.84`
- `continuous_corridor_weak_motor` score `4.27`
- `continuous_corridor_low_friction` score `3.68`
- `continuous_corridor_yaw_pulse` score `3.26`
- `continuous_corridor_lateral_push` score `-0.64`

Interpretation:

- corridor nominal transfer is viable
- corridor low friction and weak motor are degraded but functional
- the main remaining deployment weakness is:
  `continuous_corridor_lateral_push`

### MuJoCo history ablations

Canonical ablation artifacts:

- moderate disturbance:
  - `artifacts/mujoco_eval_history_ablation/c1_ethlike_v3_model_400_candidate/mujoco_disturb_v2_moderate/suite_summary.json`
  - `artifacts/mujoco_eval_history_ablation/c1_ethlike_v3_model_400_candidate/mujoco_disturb_v2_moderate__history_frozen/suite_summary.json`
  - `artifacts/mujoco_eval_history_ablation/c1_ethlike_v3_model_400_candidate/mujoco_disturb_v2_moderate__history_zero/suite_summary.json`
- continuous corridor:
  - `artifacts/mujoco_eval_history_ablation/c1_ethlike_v3_model_400_candidate/mujoco_continuous_v1/suite_summary.json`
  - `artifacts/mujoco_eval_history_ablation/c1_ethlike_v3_model_400_candidate/mujoco_continuous_v1__history_frozen/suite_summary.json`
  - `artifacts/mujoco_eval_history_ablation/c1_ethlike_v3_model_400_candidate/mujoco_continuous_v1__history_zero/suite_summary.json`

Moderate disturbance results are clean:

- `normal > frozen > zero` on:
  - `flat_command_step_up_moderate`
  - `flat_yaw_pulse_moderate`
  - `flat_command_step_down_moderate`
  - `flat_lateral_push_moderate`
  - `flat_yaw_torque_pulse_moderate`

This is the strongest deploy-side evidence that the history pathway remains behaviorally useful after export into MuJoCo.

Continuous corridor results are mixed:

- `normal` clearly wins on:
  - `continuous_corridor_nominal`
  - `continuous_corridor_yaw_pulse`
  - slightly on `continuous_corridor_weak_motor`
- `frozen` wins on:
  - `continuous_corridor_low_friction`
  - `continuous_corridor_lateral_push`
- `zero` is generally worst or near-worst, but not always dramatically so

Interpretation:

- MuJoCo does preserve a real history-use signal
- but the harder corridor contact / disturbance regime is not a uniformly
  monotonic `normal > frozen > zero` story

## Current Honest Conclusion

`C1 StageA model_400` is now the current Candidate 1 deployable blind-history policy.

What is validated:

- history is behaviorally useful on the intended Isaac switch/push suites
- MuJoCo moderate-disturbance ablations show a clean `normal > frozen > zero`
  history-use result
- export parity is fixed and trusted
- Isaac deployment rehearsal is healthy
- MuJoCo nominal runtime is healthy
- MuJoCo hidden-env flat mismatch is reasonably strong
- MuJoCo moderate disturbances are reasonably strong
- MuJoCo continuous corridor transfer is viable in nominal and several OOD variants

Main remaining weakness:

- lateral disturbance recovery under harder corridor geometry
- mixed history-ablation behavior in the hardest corridor contact regimes

This is a real limitation, but it is narrow enough that the policy still counts as a credible deployment candidate rather than an unresolved experiment.

## Freeze Rule

The current evaluation record is now treated as sufficient for freezing
`C1 StageA model_400` as the canonical Candidate 1 artifact.

Do not rerun already completed batteries by default:

- Isaac switch ablations
- Isaac push ablations
- Isaac deploy rehearsal
- MuJoCo nominal runtime
- MuJoCo hidden-env suite
- MuJoCo moderate-disturbance suite
- MuJoCo continuous-corridor suite
- MuJoCo history ablations on those same suites

Only reopen evaluation if something materially changes, such as:

- a new checkpoint intended to replace `model_400`
- a new scene or scenario family
- a new export bundle or runtime-bridge change
- a new corridor design replacing the previous continuous-corridor scene

Otherwise, this card should be treated as the frozen final evidence base for
the current Candidate 1 line.
