# Go2 Cross-Simulator Validation Contract

## Scope

Isaac Sim and MuJoCo are not physically identical. They use different solvers,
contact models, and robot assets. A fair sim2sim gate therefore means:

1. Match every controllable policy, timing, actuator, reset, command, latency,
   friction, and disturbance parameter.
2. Record the resolved runtime model parameters in every report.
3. Preserve engine-specific properties instead of pretending they are
   numerically interchangeable.
4. Compare behavior under the same named scenario manifest.

The canonical machine-readable profile is:

`configs/validation/go2_crosssim_validation_v1.json`

Both matched runners now load this profile directly. Commands, push timing,
delays, reset envelopes, seeds, rollout counts, control timing, warmup window,
and MuJoCo equivalence overrides must not be duplicated in runner code.

Every matched report contains:

- profile SHA-256
- canonical scenario SHA-256
- resolved runtime parameters
- explicit terrain comparison mode

Reports are not valid for direct comparison until the strict report-parity
gate returns `comparable`.

## Matched Contract

| Property | Value |
|---|---:|
| Physics step | 0.005 s |
| Control step | 0.020 s |
| Physics steps per action | 4 |
| Control rate | 50 Hz |
| Actor observation | 45 values |
| History | 100 x 45 = 4500 values |
| Action scale | 0.25 |
| Joint stiffness | 25 Nm/rad |
| Joint damping | 0.5 Nm s/rad |
| Effort limit | 23.5 Nm |
| DCMotor torque-speed velocity limit | 30 rad/s |
| Ground and foot friction | 1.0 |
| Restitution | 0.0 |
| Observation delay | 1 control step |
| Action delay | 1 control step |
| Command delay | 1 control step |
| Reset preset | `light` |
| Push test | +/-50 N lateral, steps 300-309 |
| MuJoCo actuator emulation | `isaac_dc_motor` |
| Metric warmup | first 100 control steps excluded |

The joint order, observation order, action offsets, reset envelopes, seven
commands, and push definitions are stored in the profile rather than repeated
here.

Isaac enables the permanent wrench only around each pushed control step and
clears it immediately afterward. This applies the force over all four physics
substeps without keeping an external-wrench buffer active during nominal
walking.

## Deliberate MuJoCo Overrides

The Menagerie model is not nominally equivalent to the Isaac articulation.
The matched suite applies:

- foot tangential friction: `0.8 -> 1.0`
- model timestep: `0.002 s -> 0.005 s`
- actuator torque clipping and velocity weakening from the exported Isaac
  deployment contract

MuJoCo's Menagerie passive joint damping, friction loss, and armature remain at
their native values. A controlled ablation showed that zeroing these terms
causes immediate plant instability while retaining them is stable with both
`simple_pd` and `isaac_dc_motor`. They are therefore recorded as
backend-specific plant properties, not treated as duplicate policy PD gains or
silently removed as an equivalence correction.

The Isaac USD also carries raw simulator-side velocity limits of `30.1 rad/s`
for hip/thigh and `15.7 rad/s` for calf. Those are distinct from the explicit
DCMotor torque-speed limit of `30 rad/s`. Reports preserve both layers rather
than describing them as one limit.

## Non-Equatable Properties

- Isaac uses PhysX TGS; MuJoCo uses Newton with Euler integration.
- Isaac uses the USD asset; MuJoCo uses the Menagerie MJCF asset.
- MuJoCo retains the Menagerie model's passive joint damping, friction loss,
  and armature because these terms are part of that backend's stable plant.
- PhysX material combination does not map exactly to MuJoCo `solmix`.
- PhysX contact offsets do not map exactly to MuJoCo
  `margin`/`gap`/`solref`/`solimp`.
- Equal seeds do not imply equal trajectories across engines.

These values must be recorded and reviewed, but they cannot pass an
"identical numeric value" test.

## Audit

Run the static and compiled-model audit:

```bash
/home/bhuvan/miniconda3/envs/rma-mujoco/bin/python \
  scripts/deploy/audit_crosssim_contract.py \
  --strict \
  --json-out artifacts/diagnostics/go2_crosssim_contract_audit_v1.json
```

The audit checks the exported deploy contract, canonical profile, compiled
Menagerie model, and the actual `mujoco_fr_asymmetry_matched_v2` manifest. It
also hashes both simulator assets and the deploy config.

## Matched Isaac Run

```bash
bash scripts/isaaclab_user.sh -p \
  scripts/eval/eval_isaac_leg_asymmetry.py \
  --profile configs/validation/go2_crosssim_validation_v1.json \
  --task Go2-Blind-Rough-MJLAB-AsymPPO-V1 \
  --checkpoint "$ASYMPPO_CKPT" \
  --json-out artifacts/diagnostics/go2_asymppo_isaac_fr_asymmetry_matched_v2.json
```

Do not repeat profile-owned values on the CLI for a canonical run. Overrides
are allowed for debugging, but the parity gate will reject mismatched reports.

The repository launcher redirects Kit's mutable cache, data, log, global
cache, and global data tokens to `${HOME}/.cache/isaacsim`. This is required
when the shared Isaac Sim binary installation is read-only under `/opt`.
Override `ISAACLAB_ROOT` or `ISAACSIM_USER_ROOT` when those locations differ.

Use one rollout per scenario for the visual seven-robot check. Repeat with
five rollouts per scenario and `--headless` for the final statistical report.
Do not run this concurrently with another Isaac Sim or Kit process because
Kit's cache/datastore lock can leave a window visible while initialization
cannot complete.

## Matched MuJoCo Run

```bash
/home/bhuvan/miniconda3/envs/rma-mujoco/bin/python \
  scripts/deploy/run_mujoco_ood_suite.py \
  --bundle-dir "$ASYMPPO_BUNDLE" \
  --profile configs/validation/go2_crosssim_validation_v1.json \
  --suite mujoco_fr_asymmetry_matched_v2 \
  --output-dir artifacts/mujoco_eval
```

Each rollout report includes the resolved timestep, solver, contact
parameters, body masses and inertias, joint limits, passive terms, gains,
action mapping, actuator limits, delays, noise, reset, and runtime overrides.
MuJoCo excludes the same first 100 control steps from aggregate metrics.

## Mandatory Report-Parity Gate

```bash
python scripts/deploy/compare_crosssim_contracts.py \
  --isaac-report artifacts/diagnostics/go2_asymppo_isaac_fr_asymmetry_matched_v2.json \
  --mujoco-report artifacts/mujoco_eval/go2_blind_rough_asymppo_mjlab_v1_candidate/mujoco_fr_asymmetry_matched_v2/suite_summary.json \
  --strict \
  --json-out artifacts/diagnostics/go2_crosssim_report_parity.json
```

Only reports with `status: comparable` may be used for cross-simulator result
tables.

## Terrain Comparison Rule

The canonical direct comparison uses backend-native flat surfaces. This is an
accepted equivalence class, not identical geometry.

MJLab-generated MuJoCo rough terrain is valid for independent robustness and
OOD testing. It is not a direct Isaac-vs-MuJoCo terrain comparison unless the
same frozen geometry artifact is imported into both engines. Reports from
different terrain generators must be labeled `backend_specific`.

## Training Versus Validation

Training is performed in Isaac Sim with the distribution recorded under
`training_distribution_isaac` in the profile. MuJoCo is an independent
validation backend, not a second training backend. A nominal matched test
should not contain training randomization unless a separately named ablation
explicitly enables it.

Any future change to timing, observations, assets, gains, limits, friction,
delays, reset, commands, disturbances, or solver settings requires a new
profile version. Do not silently edit `v1`.
# History Layout

The deployable history vector must use `isaaclab_term_major` ordering. IsaacLab
applies the 100-step group history independently to every observation term,
flattens each term's history, and then concatenates the terms:

```text
[base_ang_vel history, projected_gravity history, command history,
 joint_pos history, joint_vel history, last_action history]
```

Directly flattening a `[100, 45]` frame buffer is time-major and is not
compatible with checkpoints trained by the current IsaacLab observation
manager. MuJoCo, Python hardware deployment, and generated Unitree runtime
configuration must preserve the term-major contract.
