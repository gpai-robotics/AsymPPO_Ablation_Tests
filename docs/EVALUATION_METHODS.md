# Evaluation Methods

This document explains how the repo measures and ranks trained policies.

It is written from the actual implementation in:

- `scripts/eval/isolated.py`
- `scripts/eval/run_isolated_suite.py`
- `scripts/eval_ood/run_ood_suite.py`
- `scripts/eval/gait.py`
- `scripts/eval_ood/ood_scenarios.py`

The goal is to make the evaluation protocol legible to scientific readers and
to make clear which numbers are:

- directly used for ranking
- diagnostic only
- verification checks that make sure requested overrides were actually applied

## Evaluation Layers

The repo currently has three evaluation layers.

### 1. Isolated scenario evaluation

Script:

- `scripts/eval/isolated.py`

Purpose:

- run a single checkpoint in one controlled scenario
- optionally override friction, mass, motor strength, pushes, or one-shot
  mid-episode switches
- compute a compact metrics dictionary
- compute a single scalar `score`

This is the core ranking primitive. The suite runners call this script under
the hood in separate IsaacLab processes.

### 2. Process-level suite orchestration

Scripts:

- `scripts/eval/run_isolated_suite.py`
- `scripts/eval_ood/run_ood_suite.py`

Purpose:

- define a list of scenarios
- run `isolated.py` once per scenario in a fresh process
- collect per-scenario JSON
- rank scenarios by the isolated `score`
- write suite-level JSON and CSV

Important:

- these scripts do **not** invent a second scoring rule
- they reuse the per-scenario `score` from `isolated.py`

### 3. Gait and controller diagnostics

Script:

- `scripts/eval/gait.py`

Purpose:

- quantify contact pattern, body motion, slip, duty factors, and simple command
  response behavior
- produce interpretable gait/controller diagnostics

Important:

- `gait.py` does **not** currently compute the scalar ranking score used in the
  isolated/OOD suites
- it is a diagnostic script, not the ranking primitive

## Single-Scenario Evaluation: `isolated.py`

## Inputs

`isolated.py` accepts:

- checkpoint
- task
- terrain type
- terrain level
- seed
- number of envs
- rollout length
- optional deterministic overrides:
  - `static_friction`
  - `dynamic_friction`
  - `mass_offset`
  - `motor_stiffness_scale`
  - `motor_damping_scale`
- optional push overrides
- optional one-shot switch overrides:
  - `switch_step`
  - post-switch friction / mass / motor values

## Environment setup

The script does the following before rollout:

1. load env config from the registered task
2. force isolated terrain composition if `--terrain-type` is given
3. disable terrain curriculum if a fixed `--terrain-level` is requested
4. apply deterministic startup overrides to the event terms
5. apply push overrides if requested
6. create the env
7. force terrain level directly in the terrain importer if requested
8. load the trained policy checkpoint

During rollout:

- the script can optionally apply a one-shot mid-episode switch exactly at
  `switch_step`

## What gets measured every step

For each step, the script accumulates:

- `reward`
- planar velocity tracking error
- yaw-rate tracking error
- terrain level
- per-termination-term event counts

The data structure is split into:

- `overall_stats`
- `pre_switch_stats`
- `post_switch_stats`

If no switch is configured, only the overall/pre-switch region is populated.

## How tracking errors are computed

Preferred source:

- env-provided logged metrics, if available

Fallback source:

- direct computation from env state

The direct formulas are:

Planar velocity error:

```text
vel_err_t = mean_env || v_xy(t) - c_xy(t) ||_2
```

where:

- `v_xy(t)` is the robot base linear velocity in body frame, XY only
- `c_xy(t)` is the commanded XY velocity

Yaw-rate error:

```text
yaw_err_t = mean_env | ω_z(t) - c_yaw(t) |
```

where:

- `ω_z(t)` is the base angular velocity about yaw
- `c_yaw(t)` is the commanded yaw rate

These are then averaged across rollout steps:

```text
vel_err_step_mean = mean_t vel_err_t
yaw_err_step_mean = mean_t yaw_err_t
```

## How terrain level is measured

Preferred source:

- logged `Curriculum/terrain_levels`

Fallback source:

- direct readback from `env.unwrapped.scene.terrain.terrain_levels`

If neither is available and a fixed terrain level was explicitly requested, the
script uses the forced value and labels the source as `forced`.

## How termination counts are measured

This part was explicitly validated and corrected.

The script now reads the real per-term fire masks from:

```text
env.unwrapped.termination_manager.get_term(name)
```

for each active termination term:

- `time_out`
- `base_contact`
- `base_orientation`
- `base_height`
- `low_progress`

This means non-timeout terminations are no longer incorrectly lumped into
`base_contact`.

The script reports both:

- aggregate legacy fields:
  - `terminal_dones`
  - `terminal_timeouts`
  - `terminal_base_contacts`
- per-term fields:
  - `terminal_time_out`
  - `terminal_base_contact`
  - `terminal_base_orientation`
  - `terminal_base_height`
  - `terminal_low_progress`

plus per-env normalized versions such as:

- `base_height_events_per_env`
- `time_out_events_per_env`

## Constraint verification: making sure overrides really happened

`isolated.py` includes `constraint_checks` specifically to catch no-op override
bugs.

These are **not** part of the ranking score. They are validation outputs.

The script reads back:

- material properties from the PhysX view:
  - `observed_static_friction_{mean,min,max}`
  - `observed_dynamic_friction_{mean,min,max}`
- body masses:
  - `observed_mass_offset_{mean,min,max}`
- actuator stiffness:
  - `observed_motor_stiffness_scale_{mean,min,max}`
- push interval when applicable

For switch scenarios, the script also records:

- `post_switch_constraint_checks`

This is important because it proves the post-switch env state actually changed
to the requested values.

## Phase split for switch scenarios

If `switch_step` is set and at least one switch override is provided:

- startup metrics are accumulated into `pre_switch_metrics`
- after the one-shot switch is applied, metrics are accumulated into
  `post_switch_metrics`

The script records:

- `switch_applied`
- `switch_applied_step`

If a switch is requested but never applied, the ranking should not be trusted.

## Exact ranking score

The isolated evaluator computes one scalar score.

If a switch was applied and there are valid post-switch samples, scoring uses
the **post-switch** metrics.

Otherwise, scoring uses the full-rollout metrics.

Let the chosen metric bundle be `m`.

The score is:

```text
score =
    3.0  * m.reward_step_mean
  + 1.5  * m.terrain_level_step_mean
  + 6.0  * timeout_fraction(m)
  - 10.0 * m.base_contact_events_per_env
  - 6.0  * m.vel_err_step_mean
  - 1.5  * m.yaw_err_step_mean
```

where:

```text
timeout_fraction(m) =
    m.timeout_fraction_of_terminals    if not None
    0.0                                otherwise
```

### Interpretation of the score terms

Positive terms:

- higher reward is better
- higher terrain level is better
- higher timeout fraction is better
  - because it means a larger share of episode endings were benign timeouts
    instead of failure terminations

Negative terms:

- more base-contact failures per env are worse
- larger planar tracking error is worse
- larger yaw-rate tracking error is worse

### Important nuance

The score currently penalizes `base_contact` explicitly, but does **not**
separately penalize `base_height`, `base_orientation`, or `low_progress` in the
scalar formula.

Those are still logged and should absolutely be inspected when comparing
policies. So:

- the scalar score is a useful ranking summary
- it is **not** the whole scientific conclusion

## Suite evaluation: `run_isolated_suite.py`

This script defines named scenario sets such as:

- `blind_baseline_v1`
- `friction_only`
- `mass_only`
- `motor_only`
- `terrain_only`

Each scenario becomes one separate `isolated.py` run.

The suite runner then:

1. loads each scenario result JSON
2. sorts rows by `score` descending
3. writes:
   - suite JSON
   - suite CSV

So suite ranking is simply:

```text
rank = sort_by_descending(score)
```

The suite CSV is a convenient flattened report. It is not a new evaluator.

## OOD evaluation: `run_ood_suite.py`

This works the same way as `run_isolated_suite.py`, but uses scenario sets from
`scripts/eval_ood/ood_scenarios.py`.

Current OOD families include:

- `ood_geometry_v1`
- `ood_dynamics_v1`
- `ood_combo_v1`
- `ood_push_v1`
- `ood_switch_v1`
- `ood_limit_v1`

Again, each scenario is evaluated by `isolated.py`, so the same score formula
and verification logic apply.

## Gait diagnostics: `gait.py`

`gait.py` is not the ranking script. It is a detailed behavior diagnostic.

It supports command profiles:

- `task`
- `standstill`
- `step`
- `forward`

## Contact-based gait metrics

The script derives foot contact states from contact-force history:

```text
contact_foot = max_history || force || > 1.0
```

From this it computes quantities such as:

- `diagonal_trot_score`
- `diagonal_pair_sync_score`
- `diagonal_antiphase_score`
- `lateral_pair_sync_score`
- `fore_hind_pair_sync_score`
- swing-pair fractions
- all-feet-airborne fraction
- all-feet-stance fraction
- touchdown/liftoff event structure

The diagonal trot score is:

```text
diag_sync = 0.5 * agree(FL, RR) + 0.5 * agree(FR, RL)
diag_antiphase = disagree(FL, FR)
diagonal_trot_score = diag_sync * diag_antiphase
```

where `agree(a,b)` is `1` if the two binary contacts match and `0` otherwise.

This score is a diagnostic indicator of trot-likeness, not the suite ranking
score.

## Body and controller metrics

`gait.py` also reports:

- foot slip during contact
- foot clearance
- base height
- base vertical velocity
- roll/pitch angular velocity
- yaw speed
- projected-gravity tilt
- joint position errors
- duty factor per foot
- forward drift / heading behavior
- step response latency / overshoot for step-command profiles

These metrics are meant to answer:

- is the robot moving cleanly?
- is the gait trot-like or serial?
- is there slip, bounce, or overcorrection?

## Terminations in `gait.py`

Like `isolated.py`, `gait.py` now reads true termination causes from the
termination manager and reports:

- `terminal_time_out`
- `terminal_base_contact`
- `terminal_base_orientation`
- `terminal_base_height`
- `terminal_low_progress`

These are diagnostics. `gait.py` does not compute the suite ranking score.

## What is used for policy ranking today

For ranking policy performance across isolated and OOD cases, the current repo
uses:

- `isolated.py` score
- suite-level sorting of that score

For qualitative and controller-style interpretation, the repo uses:

- `gait.py` metrics

So the current protocol is:

1. use isolated/OOD scores to rank scenarios and compare policies at scale
2. use gait diagnostics to understand *how* the policy is moving
3. inspect constraint checks to ensure the evaluated scenario actually matches
   the intended perturbation
4. inspect raw per-term termination counts before making strong claims

## What was validated explicitly

The following checks were run during evaluation-audit work:

- fixed startup overrides were read back correctly
  - friction
  - mass offset
  - motor stiffness scale
- one-shot switch overrides were read back correctly in
  `post_switch_constraint_checks`
- per-term termination reporting was corrected to avoid collapsing all failures
  into `base_contact`
- gait diagnostics produced consistent JSON and real per-term termination counts

Validation artifacts:

- `artifacts/debug/eval_validation_fixed.json`
- `artifacts/debug/eval_validation_switch.json`
- `artifacts/debug/gait_validation_forward.json`

## Current limitations

The evaluation stack is much more trustworthy after the audit, but a few caveats
remain:

- the scalar isolated score is still a hand-designed ranking summary
  - it is useful
  - but it is not a substitute for looking at the component metrics
- the score penalizes `base_contact` explicitly but not all other failure modes
  explicitly
- short rollouts can produce zero termination counts and therefore hide failure
  structure
- the suite runners are process orchestrators; they assume `isolated.py`
  remains the canonical source of truth

## Recommended scientific reporting practice

When comparing trained policies, report at least:

- scalar suite score
- `reward_step_mean`
- `vel_err_step_mean`
- `yaw_err_step_mean`
- `terrain_level_step_mean`
- per-term termination counts
- relevant constraint checks
- representative gait diagnostics for nominal and switched cases

That gives both:

- a compact ranking signal
- enough methodological transparency for scientific readers
