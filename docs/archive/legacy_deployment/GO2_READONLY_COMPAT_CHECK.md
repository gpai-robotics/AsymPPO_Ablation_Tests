# Go2 Read-Only Compatibility Check

This document records the safe, read-only workflow we used on **May 28, 2026**
to compare two physical Go2 units without entering `Velocity` mode.

The goal is to answer a narrow question:

- are the robots statically similar enough that a learned low-level deploy
  policy should see them as "the same robot"?

This workflow is intentionally conservative:

- no `lowcmd`
- no mode switching from the script
- no walking
- no sport RPC writes

## Why We Added This

The new robot was not showing credible signs of working under the isolated
deploy runtime even after we ruled out several software-side issues:

- stray `vy` / `wz` command contamination
- obvious remote noise
- obvious joint ordering mismatch
- simple target-tracking failure at `Velocity` entry

That made it important to compare the two robots in a way that was:

- safe
- repeatable
- independent of walking behavior

## Scripts

The workflow uses:

- [probe_go2_readonly.py](/home/bhuvan/projects/rma/rma_go2_lab/scripts/deploy/probe_go2_readonly.py)
- [compare_go2_readonly_snapshots.py](/home/bhuvan/projects/rma/rma_go2_lab/scripts/deploy/compare_go2_readonly_snapshots.py)
- [summarize_go2_readonly_snapshot_set.py](/home/bhuvan/projects/rma/rma_go2_lab/scripts/deploy/summarize_go2_readonly_snapshot_set.py)

`probe_go2_readonly.py` records:

- joint position / velocity
- IMU quaternion / gyro / accel
- foot forces
- motor temperatures
- remote state
- sport-mode snapshot if available

`compare_go2_readonly_snapshots.py` prints the raw deltas and now adds a
`Compatibility Flags` section for the largest static-state mismatches.

`summarize_go2_readonly_snapshot_set.py` summarizes repeated captures from one
robot so we can tell whether a suspicious static signature is stable or just
capture-to-capture noise.

## Capture Workflow

Use the same NIC for both robots and capture both:

1. sitting / resting on the ground
2. stable standing pose

Do **not** enter `Velocity`.

Example commands:

```bash
cd /home/bhuvan/projects/rma/rma_go2_lab
python scripts/deploy/probe_go2_readonly.py --net-if enp0s31f6 --duration 5 --print-every 1 --json-out /tmp/go2_old_sit.json
python scripts/deploy/probe_go2_readonly.py --net-if enp0s31f6 --duration 5 --print-every 1 --json-out /tmp/go2_old_stand.json
python scripts/deploy/probe_go2_readonly.py --net-if enp0s31f6 --duration 5 --print-every 1 --json-out /tmp/go2_new_sit.json
python scripts/deploy/probe_go2_readonly.py --net-if enp0s31f6 --duration 5 --print-every 1 --json-out /tmp/go2_new_stand.json
```

Then compare:

```bash
cd /home/bhuvan/projects/rma/rma_go2_lab
python scripts/deploy/compare_go2_readonly_snapshots.py --a /tmp/go2_old_sit.json --b /tmp/go2_new_sit.json
python scripts/deploy/compare_go2_readonly_snapshots.py --a /tmp/go2_old_stand.json --b /tmp/go2_new_stand.json
```

The standing comparison is the most important one because it is closest to the
state just before pressing `Start`.

## Repeated Standing Snapshot Check

If one robot looks suspicious, collect several standing captures from that same
robot and summarize them as a set.

Example:

```bash
cd /home/bhuvan/projects/rma/rma_go2_lab
python scripts/deploy/summarize_go2_readonly_snapshot_set.py --label new_robot_stand /tmp/go2_new_stand_1.json /tmp/go2_new_stand_2.json /tmp/go2_new_stand_3.json
```

Use that to check:

- whether foot-force totals stay consistently high
- whether IMU acceleration / tilt stays consistently offset
- whether motor temperatures are consistently elevated
- whether the robot's joint pose is stable across repeated captures

## Current Heuristics

The comparison script currently flags these ranges:

- joint pose L2:
  - `warn` at `>= 0.10 rad`
  - `high` at `>= 0.18 rad`
- rest joint-velocity L2:
  - `warn` at `>= 0.10 rad/s`
  - `high` at `>= 0.20 rad/s`
- IMU gyro delta L2:
  - `warn` at `>= 0.03`
  - `high` at `>= 0.08`
- IMU accel delta L2:
  - `warn` at `>= 0.30`
  - `high` at `>= 0.60`
- total foot-force delta:
  - `warn` at `>= 40`
  - `high` at `>= 100`
- per-foot force delta:
  - `warn` at `>= 20`
  - `high` at `>= 40`
- mean motor temperature delta:
  - `warn` at `>= 2 C`
  - `high` at `>= 4 C`
- single-joint motor temperature delta:
  - `warn` at `>= 4 C`
  - `high` at `>= 7 C`

These are only triage thresholds. They are not a formal Unitree calibration
spec.

## Findings From May 28, 2026

### Sitting: old vs new

The main results were:

- joint pose difference L2 about `0.143 rad`
- rest joint-velocity difference L2 about `0.152 rad/s`
- IMU gyro close
- IMU accel difference modest at about `0.180`
- foot-force totals very different:
  - old total `52`
  - new total `276`
- motor temperatures consistently higher on the new robot:
  - mean delta about `4.9 C`
  - max single-joint delta `8 C`

Flags:

- `warn`: joint pose differs more than expected
- `warn`: rest joint velocity mismatch is elevated
- `high`: total foot-force differs strongly
- `high`: per-foot force delta is large
- `high`: mean motor temperature differs
- `high`: single-joint motor temperature delta is large

### Standing: old vs new

The standing comparison was more informative.

The main results were:

- joint pose difference L2 about `0.105 rad`
- rest joint-velocity difference L2 about `0.123 rad/s`
- IMU gyro still close
- IMU accel / posture difference large at about `0.753`
- foot-force totals very different:
  - old total `131`
  - new total `297`
- max per-foot force delta `65`
- motor temperatures again higher on the new robot:
  - mean delta about `4.9 C`
  - max single-joint delta `8 C`

Flags:

- `warn`: joint pose differs more than expected
- `warn`: rest joint velocity mismatch is elevated
- `high`: IMU accel / posture mismatch is elevated
- `high`: total foot-force differs strongly
- `high`: per-foot force delta is large
- `high`: mean motor temperature differs
- `high`: single-joint motor temperature delta is large

## Interpretation

The comparison does **not** support a simple "our software completely scrambled
the new robot" explanation.

What looked reasonably close:

- static joint pose
- IMU gyro at rest
- remote inputs

What looked materially different:

- foot-force loading in both sitting and standing
- standing accelerometer / posture signature
- motor temperatures under static load

That means the new robot appears to settle into a meaningfully different loaded
state even before walking.

The most likely conclusion is:

- this is **not** just a command-path issue
- this is **not** obviously a joint-map disaster
- the new robot has a robot-side physical / calibration / load-distribution
  difference that the learned controller is sensitive to

## Practical Use

Use this workflow before more walking tests when:

- one robot works and another does not
- startup behavior differs sharply across units
- you want evidence before chasing controller-side tuning

If the comparison flags large differences in:

- foot force
- standing IMU accel / posture
- motor temperatures

then treat the robot as mechanically or calibration-wise different, even if the
model and software bundle are nominally the same.
