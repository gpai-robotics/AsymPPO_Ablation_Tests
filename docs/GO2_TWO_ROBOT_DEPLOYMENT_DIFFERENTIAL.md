# Go2 Two-Robot Deployment Differential

Use this when the same policy bundle works on one Go2 but destabilizes on another.
The goal is to separate three cases:

- the second robot receives different commands or gains,
- the second robot receives the same commands but tracks them differently,
- the second robot enters policy takeover from a different physical/FSM state.

All commands below are read-only. They subscribe to DDS topics but do not publish
`LowCmd`, switch modes, or start the policy.

## Preconditions

- Use the same machine, repo commit, policy bundle, controller binary, network path, floor, and command profile for both robots.
- Keep batteries in a similar charge band. Low battery can look like weak rear legs or delayed tracking.
- Reboot both robots before a final comparison pass if possible.
- Do not compare one robot in high-level mode against another already released into low-level mode.

## 1. Prepare The Hardware DDS Environment

```bash
cd /home/bhuvan/projects/rma/rma_go2_lab
conda activate go2-hw

python - <<'PY'
import sys
from pathlib import Path
root = Path("reference_repos/sim2real_unitree_sdk2py").resolve()
sys.path.insert(0, str(root))
import cyclonedds
import unitree_sdk2py
print("hardware SDK imports OK")
PY
```

If this fails, fix the hardware Python environment before collecting robot data.

## 2. Static Blueprint Capture

Capture each robot in the same physical state. Do this at least for standing
default stance. If the second robot already looks different in FixStand, also
capture seated/passive and FixStand separately.

Robot A, known-good:

```bash
scripts/deploy/run_go2_readonly_signature_check.sh capture-blueprint go2_a_stand enp0s31f6 8
```

Robot B, failing unit:

```bash
scripts/deploy/run_go2_readonly_signature_check.sh capture-blueprint go2_b_stand enp0s31f6 8
```

Compare the resulting JSON files:

```bash
scripts/deploy/run_go2_readonly_signature_check.sh compare \
  artifacts/go2_readonly_signatures/<timestamp>_go2_a_stand.json \
  artifacts/go2_readonly_signatures/<timestamp>_go2_b_stand.json
```

What to inspect:

- joint `q` offsets while standing,
- IMU quaternion/gravity consistency,
- foot-force distribution,
- motor temperature differences,
- remote state and sport mode state,
- any unexpected field differences in the `blueprint` section.

## 3. Dynamic Policy-Takeover Capture

This is the important capture for the current failure mode. Run the read-only
capture in one terminal while the controller/policy is running in another
terminal.

For Robot A, known-good, start the read-only capture first:

```bash
scripts/deploy/run_go2_readonly_signature_check.sh capture-dynamic-lowcmd \
  go2_a_policy_takeover enp0s31f6 30 0.05
```

Then run the exact same controller/FSM command that normally starts the policy.
Stop after the same window.

Repeat the same sequence for Robot B:

```bash
scripts/deploy/run_go2_readonly_signature_check.sh capture-dynamic-lowcmd \
  go2_b_policy_takeover enp0s31f6 30 0.05
```

This writes:

- `*_summary.json`
- `*_series.jsonl`
- `*_lowcmd_stream.jsonl`

## 4. Compare Dynamic Captures

```bash
scripts/deploy/run_go2_readonly_signature_check.sh compare-dynamic \
  artifacts/go2_readonly_signatures/<a>_go2_a_policy_takeover_series.jsonl \
  artifacts/go2_readonly_signatures/<a>_go2_a_policy_takeover_lowcmd_stream.jsonl \
  artifacts/go2_readonly_signatures/<b>_go2_b_policy_takeover_series.jsonl \
  artifacts/go2_readonly_signatures/<b>_go2_b_policy_takeover_lowcmd_stream.jsonl
```

Interpretation:

- If `q_des`, `kp`, and `kd` match but `q_err`, `tau_est`, or `dq` diverge, the
  issue is likely hardware response, calibration, joint friction, battery/power,
  motor mode, or mechanical condition.
- If `q_des`, `kp`, or `kd` differ, the issue is in runtime config, policy
  bundle path, controller binary, FSM path, or command scaling.
- If the robots differ before Velocity/policy takeover, debug startup stance and
  low-level mode entry before blaming the policy.

## 5. Data To Preserve

For each robot comparison, keep these files together:

- static `capture-blueprint` JSON for standing state,
- dynamic `*_summary.json`,
- dynamic `*_series.jsonl`,
- dynamic `*_lowcmd_stream.jsonl`,
- terminal logs from the controller/FSM terminal,
- battery level and robot serial/identity notes.

Store them under `artifacts/go2_readonly_signatures/` locally. Do not commit
large capture artifacts unless they are intentionally selected as small examples.
