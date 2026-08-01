# Go2 Robot Internal State Comparison

Date: 2026-07-01  
Purpose: preserve the working-vs-failing Go2 internal-state comparison used to
debug asymmetric PPO hardware deployment instability.

## Context

The same asymPPO policy bundle and deployment runtime worked on one Go2 but
destabilized on a second Go2 immediately after FSM transition into Velocity
policy control. Joint ordering had already been checked manually. We therefore
captured read-only DDS signatures for both robots and compared:

- static FixStand/default-stand state,
- policy-takeover dynamics,
- LowCmd targets/gains,
- LowState measured joints, IMU, estimated torque, foot force, and temperature.

All captures were read-only DDS subscriptions. No diagnostic capture published
`LowCmd` or switched robot modes.

## Capture Files

### Static Stand Repeats

| Robot | Capture Files |
| --- | --- |
| Working Go2 | `artifacts/go2_readonly_signatures/20260630_154657_working_go2_stand_repeat_1.json` |
| Working Go2 | `artifacts/go2_readonly_signatures/20260630_154706_working_go2_stand_repeat_2.json` |
| Working Go2 | `artifacts/go2_readonly_signatures/20260630_154714_working_go2_stand_repeat_3.json` |
| Failing Go2 | `artifacts/go2_readonly_signatures/20260630_154247_failing_go2_stand_repeat_1.json` |
| Failing Go2 | `artifacts/go2_readonly_signatures/20260630_154255_failing_go2_stand_repeat_2.json` |
| Failing Go2 | `artifacts/go2_readonly_signatures/20260630_154303_failing_go2_stand_repeat_3.json` |

### Policy Takeover

| Robot | Capture File | Type |
| --- | --- | --- |
| Working Go2 | `artifacts/go2_readonly_signatures/20260630_151753_working_go2_policy_takeover_summary.json` | Summary |
| Working Go2 | `artifacts/go2_readonly_signatures/20260630_151753_working_go2_policy_takeover_series.jsonl` | LowState/Sport/LowCmd sampled series |
| Working Go2 | `artifacts/go2_readonly_signatures/20260630_151753_working_go2_policy_takeover_lowcmd_stream.jsonl` | Full-rate LowCmd stream |
| Failing Go2 | `artifacts/go2_readonly_signatures/20260630_152319_failing_go2_policy_takeover_summary.json` | Summary |
| Failing Go2 | `artifacts/go2_readonly_signatures/20260630_152319_failing_go2_policy_takeover_series.jsonl` | LowState/Sport/LowCmd sampled series |
| Failing Go2 | `artifacts/go2_readonly_signatures/20260630_152319_failing_go2_policy_takeover_lowcmd_stream.jsonl` | Full-rate LowCmd stream |

## Joint Order

Static snapshots are stored in Unitree SDK motor order.

Policy-order analysis uses:

```text
FL_hip FR_hip RL_hip RR_hip
FL_thigh FR_thigh RL_thigh RR_thigh
FL_calf FR_calf RL_calf RR_calf
```

## Static Stand Summary

Three repeated static captures were taken for each robot in the same default
standing state.

| Metric | Working Go2 Mean | Failing Go2 Mean | Difference |
| --- | ---: | ---: | ---: |
| LowState rate | 500.04 Hz | 481.41 Hz | -18.63 Hz |
| Body height | 0.3244 m | 0.3229 m | -0.0015 m |
| Foot-force total | 140.0 | 468.0 | +328.0 |
| Foot-force L2 delta | n/a | n/a | 183.4 |
| IMU accel L2 delta | n/a | n/a | 0.715 |
| Joint `q` L2 delta | n/a | n/a | 0.101 rad |
| Joint `dq` L2 delta | n/a | n/a | 0.077 rad/s |

## Static Stand: Foot Force

| Foot | Working Go2 Mean | Failing Go2 Mean | Failing - Working |
| --- | ---: | ---: | ---: |
| FL | 38.0 | 69.3 | +31.3 |
| FR | 41.7 | 94.0 | +52.3 |
| RL | 44.7 | 161.0 | +116.3 |
| RR | 15.7 | 143.7 | +128.0 |
| Total | 140.0 | 468.0 | +328.0 |

The failing robot consistently reports much higher rear-foot loading before
policy takeover. This is the strongest static mismatch.

## Static Stand: IMU

| Signal | Working Go2 Mean | Failing Go2 Mean | Working - Failing |
| --- | --- | --- | --- |
| IMU accel xyz | `[0.110, 0.190, 9.618]` | `[-0.055, -0.502, 9.686]` | `[0.165, 0.692, -0.069]` |
| IMU gyro xyz | `[0.000, 0.003, -0.025]` | `[0.000, -0.005, -0.020]` | approximately small |

The failing robot has a repeatable posture/gravity-projection difference even
in static stance.

## Static Stand: Joint Position

Values are Unitree SDK order.

| Joint Index | Working `q` Mean | Failing `q` Mean | Working - Failing |
| ---: | ---: | ---: | ---: |
| 0 | -0.057 | -0.042 | -0.015 |
| 1 | 0.696 | 0.708 | -0.012 |
| 2 | -1.453 | -1.392 | -0.061 |
| 3 | -0.019 | -0.002 | -0.018 |
| 4 | 0.705 | 0.718 | -0.013 |
| 5 | -1.488 | -1.528 | +0.039 |
| 6 | -0.005 | -0.019 | +0.014 |
| 7 | 0.709 | 0.740 | -0.031 |
| 8 | -1.407 | -1.427 | +0.020 |
| 9 | 0.131 | 0.131 | -0.001 |
| 10 | 0.667 | 0.652 | +0.015 |
| 11 | -1.462 | -1.413 | -0.048 |

Joint-position differences exist but are moderate. The much larger mismatch is
load distribution and IMU/posture.

## Static Stand: Estimated Torque

Values are Unitree SDK order.

| Joint Index | Working `tau_est` Mean | Failing `tau_est` Mean | Working - Failing |
| ---: | ---: | ---: | ---: |
| 0 | 0.256 | 0.503 | -0.247 |
| 1 | 0.676 | -0.198 | +0.874 |
| 2 | 4.014 | 3.683 | +0.331 |
| 3 | -0.140 | 0.363 | -0.503 |
| 4 | 0.272 | 0.107 | +0.165 |
| 5 | 3.983 | 4.315 | -0.332 |
| 6 | 4.189 | 4.181 | +0.008 |
| 7 | 2.655 | 1.806 | +0.849 |
| 8 | 6.622 | 4.346 | +2.276 |
| 9 | -4.238 | -4.230 | -0.008 |
| 10 | 1.039 | 1.839 | -0.800 |
| 11 | 4.394 | 5.706 | -1.312 |

## Policy Takeover Summary

| Metric | Working Go2 | Failing Go2 |
| --- | ---: | ---: |
| Aligned samples | 518 | 494 |
| Engaged samples | 413 | 411 |
| Duration | 29.95 s | 28.57 s |
| LowState sampled rate in series | 19.5 Hz | 19.6 Hz |
| LowCmd stream rate | 658.6 Hz | 684.6 Hz |
| LowCmd pair dt mean | 0.6 ms | 0.5 ms |
| `kp` range | 0.0 to 80.0 | 0.0 to 80.0 |
| `kd` range | 0.5 to 5.0 | 0.5 to 5.0 |
| Foot-force mean | `[35.6, 39.9, 36.0, 16.1]` | `[78.9, 105.1, 165.4, 148.8]` |
| Gyro abs mean | `[0.014, 0.016, 0.017]` | `[0.050, 0.055, 0.097]` |

The two robots receive comparable control gains and valid LowCmd streams. The
failing robot shows larger body motion and much larger load on rear feet during
policy execution.

## Policy Takeover: Top Tracking Errors

### Mean Absolute `q_des - q`

| Rank | Working Go2 | Failing Go2 |
| ---: | --- | --- |
| 1 | RR_calf 0.2032 | RR_calf 0.3145 |
| 2 | FL_calf 0.1671 | FL_calf 0.2780 |
| 3 | RL_calf 0.1398 | RL_calf 0.2573 |
| 4 | FR_calf 0.1009 | FR_calf 0.2320 |
| 5 | FR_thigh 0.0825 | RR_hip 0.1252 |
| 6 | RR_hip 0.0797 | RL_thigh 0.1130 |

### Peak Absolute `q_des - q`

| Rank | Working Go2 | Failing Go2 |
| ---: | --- | --- |
| 1 | RR_calf 0.5057 | RR_calf 1.2807 |
| 2 | FL_calf 0.4586 | RL_calf 1.2712 |
| 3 | RL_calf 0.4083 | FR_calf 1.2557 |
| 4 | FR_calf 0.2725 | FL_calf 1.2512 |
| 5 | RL_thigh 0.2063 | FL_thigh 0.8487 |
| 6 | RR_hip 0.1753 | RL_thigh 0.8129 |

The failing robot’s calf tracking peaks are roughly 2.5x to 4.6x larger than
the working robot during policy takeover.

## Policy Takeover: Largest Dynamic Deltas

| Delta Metric | Largest Joints |
| --- | --- |
| `q_err` mean_abs delta | FR_calf 0.1311, RL_calf 0.1175, RR_calf 0.1114, FL_calf 0.1109 |
| `q_err` peak_abs delta | FR_calf 0.9833, RL_calf 0.8628, FL_calf 0.7926, RR_calf 0.7750 |
| `tau_est` mean_abs delta | FR_calf 1.5508, FR_thigh 1.4542, FR_hip 1.2016, RR_thigh 1.1252 |
| `dq` mean_abs delta | FR_calf 0.1282, FR_thigh 0.1172, RL_hip 0.0891, RL_calf 0.0797 |
| `q_des` mean_abs delta | RR_calf 0.1005, FR_hip 0.0771, FL_calf 0.0725, FR_calf 0.0471 |

## Interpretation

The failing robot is not equivalent to the working robot before policy takeover.
The strongest evidence is static-stance rear-foot loading:

```text
Working rear feet: RL=44.7, RR=15.7
Failing rear feet: RL=161.0, RR=143.7
```

This mismatch is repeatable across three static captures. The policy therefore
starts from a different physical condition on the failing robot.

Current conclusion:

- Not primarily a joint-order issue.
- Not primarily a missing LowCmd stream.
- Not primarily a policy-bundle loading issue.
- More likely a robot-specific hardware/state-entry/calibration issue.

Most likely investigation targets:

1. Foot-force sensor calibration or scaling on the failing robot.
2. IMU/posture calibration difference causing different FixStand load balance.
3. Rear-leg mechanical stiffness, friction, assembly, or motor response issue.
4. Rear motor zero/calibration offset.
5. Battery/load placement or hardware variant difference.

## Next Diagnostic

Lift each robot so all feet are unloaded, then run the same static
`capture-blueprint` command.

Expected outcomes:

- If failing robot still reports high rear foot force while lifted, suspect
  foot-force sensor calibration/offset.
- If lifted foot forces drop near zero, the rear loading is real and likely
  caused by posture, mechanical, motor, or IMU/calibration differences.

Example:

```bash
scripts/deploy/run_go2_readonly_signature_check.sh capture-blueprint working_go2_lifted enp0s31f6 8
scripts/deploy/run_go2_readonly_signature_check.sh capture-blueprint failing_go2_lifted enp0s31f6 8
scripts/deploy/run_go2_readonly_signature_check.sh compare \
  artifacts/go2_readonly_signatures/<working_lifted>.json \
  artifacts/go2_readonly_signatures/<failing_lifted>.json
```
