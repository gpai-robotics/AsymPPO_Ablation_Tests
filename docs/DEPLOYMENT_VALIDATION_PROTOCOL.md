# Deployment Validation Protocol

This document turns the current Go2 hardware phase into a repeatable
deployment-qualification routine.

The goal is not just to answer "did the robot move?" The goal is to make
deployment runs comparable across:

- bundles
- firmware / runtime changes
- surfaces and terrain features
- battery state
- operator sessions

This protocol assumes the current `go2_ctrl` runtime is the canonical
deployment surface and that the startup / handoff path is already frozen unless
there is a separate safety reason to revisit it.

## Scope

This protocol is for:

- deployment-phase hardware validation
- post-run quality analysis
- bundle-to-bundle comparison
- regression detection after runtime changes

This protocol is not for:

- training-time evaluation
- MuJoCo OOD sweep ranking
- informal teleop exploration without logs

## Qualification Philosophy

A deployment run should answer four different questions:

1. Did the runtime launch cleanly and stay in the intended control mode?
2. Did the robot obey commands with acceptable composure and stop behavior?
3. Did the robot remain robust on the intended terrain family?
4. Was the result repeatable enough to trust as a deployment-quality signal?

Those questions map to four stages:

1. pre-deploy gate
2. flat-ground acceptance
3. robustness qualification
4. endurance and consistency

## Required Artifacts Per Run

Every qualified hardware run should capture:

- bundle identifier
- git commit or dirty-tree note
- active `deploy.yaml` path
- robot-facing NIC
- surface / terrain description
- battery state if available
- operator notes
- raw `go2_ctrl` terminal log
- video filename if recorded

Recommended run naming:

- `YYYYMMDD_bundle_surface_trialNN`

Example:

- `20260525_c1_blind_rough_omni_usable_v1_final_flat_concrete_trial03`

## Pre-Deploy Gate

Run this before the robot is asked to locomote.

### Contract checks

Verify:

- exact bundle path
- exact TorchScript / ONNX artifact path
- `step_dt = 0.02`
- policy rate `50 Hz`
- observation contract matches the export
- command caps match the intended trial
- no accidental runtime-only command hacks are enabled unless documented

### Runtime checks

Verify:

- `go2_ctrl` receives `rt/lowstate`
- the controller reaches the intended FSM state
- no sport-mode ownership conflict is visible
- zero-command stance is stable
- no unexplained drift appears before command input

### Operator checks

Verify:

- spotter is present
- clear stop zone is available
- emergency stop path is understood
- intended command sequence is agreed before the run

If the run fails this gate, it does not count as an acceptance run.

## Flat-Ground Acceptance

This stage is the first trusted behavioral gate.

Use a high-traction flat surface first.

### Command suite

Run the following sequence with pauses between segments:

1. idle hold
2. forward hold
3. release to idle
4. backward hold
5. release to idle
6. left strafe hold
7. release to idle
8. right strafe hold
9. release to idle
10. left yaw hold
11. release to idle
12. right yaw hold
13. release to idle
14. gentle diagonal forward-left
15. gentle diagonal forward-right

Recommended hold durations:

- idle baseline: `3 s`
- command hold: `2-3 s`
- release observation: `2-3 s`

### Primary questions

This stage should answer:

- does the robot track the commanded direction?
- does it stop in a reasonable time after release?
- is there obvious mirrored-command asymmetry?
- do the joints track commands without a runaway leg or side?
- does the robot look composed enough to proceed to harder terrain?

### Minimum pass criteria

At a minimum, the run should show:

- clean entry into `Velocity`
- no fall
- no forced operator rescue
- no persistent drift after final release
- no unexplained runaway escalation in `JointDiag err`
- no obvious command inversion

## Robustness Qualification

Only start this after flat-ground acceptance passes.

### Terrain suite

Use a fixed progression such as:

1. wood ramp up
2. wood ramp down
3. uneven board / plank transition
4. rough but traction-friendly ground
5. obstacle edge or shallow threshold

If you want more aggressive trials later, add them as a separate stress tier
rather than mixing them into the baseline qualification suite.

### What to measure

Record:

- whether the trial completed
- whether the robot recovered from a misstep
- whether recovery required operator intervention
- whether large body tilt remained transient or became unstable
- whether zero-command stop still works after the stressor

### Interpretation

A policy can still count as deployment-credible even if it has a narrow failure
mode, as long as:

- the failure is repeatable
- the failure is well-bounded
- nominal behavior remains trustworthy

The dangerous case is not "fails somewhere." The dangerous case is "fails in a
way we cannot characterize."

## Endurance and Consistency

This stage prevents one-off lucky trials from being mistaken for readiness.

### Repeatability suite

Run at least:

1. repeated launch / enter-velocity / exit trials
2. repeated flat-ground command suite
3. a longer continuous locomotion interval
4. the same suite on more than one surface when possible

### Questions

This stage answers:

- does behavior degrade over time?
- do stop dynamics change as the robot warms up?
- do asymmetries get worse across runs?
- is the launch path reliable enough to trust operationally?

## Metrics To Track

The current runtime logs already expose enough signals to start a useful scorecard.

### Lifecycle metrics

From FSM and startup lines:

- did the controller connect cleanly?
- did it enter `Velocity`?
- did it return to `Passive` normally?
- were there repeated reconnect or mode-change anomalies?

### Command metrics

From `VelocityCmd`:

- max commanded `vx`, `vy`, `wz`
- max filtered `vx`, `vy`, `wz`
- zero-command residual after stick release
- decay time from active command to near zero
- tracking gap between `filtered` command and `lin_vel` where `lin_vel` is trustworthy

### Joint metrics

From `JointDiag raw_action`, `rel_cmd`, `rel_pos`, `err`:

- peak absolute command magnitude per leg
- peak absolute tracking error per leg
- side-aggregated error asymmetry
- persistent left-vs-right error bias
- mirrored-command asymmetry under `+vy` vs `-vy`

### Composure metrics

From `ObsDiag policy_obs`:

- peak base angular velocity
- projected gravity / body tilt envelope
- command-context consistency

### Operational metrics

From operator notes:

- interventions
- slips
- stumbles
- falls
- hesitation before motion onset
- overshoot after release

## Pass / Watch / Fail Classification

Each run should be assigned a simple status.

### Pass

- no fall
- no operator rescue
- command directions are correct
- stop behavior is bounded
- asymmetry, if present, is mild and repeatable
- rough-terrain recovery remains controlled

### Watch

- nominal behavior is good
- but one or more of these appears:
  - mirrored-command asymmetry
  - long stop tail after release
  - one leg or side carries noticeably larger persistent tracking error
  - terrain behavior is good but not yet repeatable

### Fail

- fall
- uncontrolled drift
- runaway command realization
- inability to cleanly stop
- repeated mode / lifecycle instability
- behavior that cannot be explained from logs and operator notes

## Suggested Operator Run Sheet

For each run, capture the following in one note:

- bundle:
- date:
- surface:
- terrain feature:
- battery:
- command caps:
- runtime notes:
- result:
- watchpoints:
- video:

## Log Analysis Workflow

Capture the low-level stream while the active FSM controller is running:

```bash
bash scripts/deploy/run_go2_realtime_monitor.sh <network-interface>
```

Analyze the resulting JSONL stream:

```bash
python scripts/deploy/analyze_go2_realtime_monitor.py \
  --jsonl <monitor-jsonl>

python scripts/deploy/analyze_go2_leg_mirror_pairs.py \
  --jsonl <monitor-jsonl>
```

The analysis surface covers:

- body orientation and angular velocity
- per-joint position, velocity, torque, and temperature
- foot-force/contact balance
- mirrored leg-pair errors and asymmetry
- command and controller timing when available

This does not replace video or operator notes. It makes them easier to compare.

## Canonical Comparison Routine

When comparing two bundles or two runtime revisions:

1. keep the surface the same
2. keep the command suite the same
3. keep the operator the same if possible
4. keep command caps the same
5. collect raw logs for both
6. summarize both with the same script
7. compare:
   - command decay
   - mirrored-command asymmetry
   - side error bias
   - recovery success
   - intervention count

If those are not controlled, the comparison is weak no matter how good the
video looks.

## Recommended Next Step

For the current phase, the practical baseline is:

1. qualify one canonical bundle on flat high-traction ground
2. qualify the same bundle on ramps and rough terrain
3. repeat enough times to establish consistency
4. only then compare future runtime or policy changes against that reference

That gives the project a stable deployment-quality anchor instead of an
accumulating pile of memorable but incomparable runs.
