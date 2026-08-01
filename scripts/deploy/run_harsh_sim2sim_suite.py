#!/usr/bin/env python3
"""Compatibility wrapper for the legacy harsh MuJoCo suite entrypoint.

Use ``run_mujoco_ood_suite.py`` for all new work. This wrapper exists only so
older commands keep functioning while routing through the canonical suite
runner.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


RUNNER = Path(__file__).resolve().parent / "run_mujoco_ood_suite.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--control-rate-hz", type=float, default=50.0)
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--trace-steps", type=int, default=0)
    parser.add_argument("--python-exe", default=sys.executable)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print(
        "[INFO] run_harsh_sim2sim_suite.py is legacy. "
        "Forwarding to run_mujoco_ood_suite.py --suite mujoco_nominal_v1"
    )
    cmd = [
        args.python_exe,
        str(RUNNER),
        "--bundle-dir",
        args.bundle_dir,
        "--suite",
        "mujoco_nominal_v1",
        "--control-rate-hz",
        str(args.control_rate_hz),
        "--max-steps",
        str(args.max_steps),
        "--trace-steps",
        str(args.trace_steps),
        "--output-dir",
        args.output_dir,
    ]
    completed = subprocess.run(cmd)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
