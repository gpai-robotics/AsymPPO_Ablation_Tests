#!/usr/bin/env python3
"""Capture and analyze stationary Go2 observations without publishing commands."""

from __future__ import annotations

import argparse
from datetime import datetime
import ipaddress
import os
from pathlib import Path
import re
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CAPTURE_PYTHON = Path(os.environ.get("GO2_HW_PYTHON", "python3"))
DEFAULT_ANALYSIS_PYTHON = Path(os.environ.get("MUJOCO_PYTHON", "python"))
DEFAULT_BUNDLE = (
    REPO_ROOT
    / "rma_go2_lab/policies/exported/go2_blind_rough_asymppo_mjlab_v1_candidate"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--net-if", required=True)
    parser.add_argument("--bundle-dir", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--duration", type=float, default=15.0)
    parser.add_argument(
        "--capture-python",
        type=Path,
        default=DEFAULT_CAPTURE_PYTHON,
        help="Python interpreter containing CycloneDDS and Unitree SDK dependencies.",
    )
    parser.add_argument(
        "--analysis-python",
        type=Path,
        default=DEFAULT_ANALYSIS_PYTHON,
        help="Python interpreter containing PyTorch and NumPy.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "artifacts/hardware_observation_audit",
    )
    return parser.parse_args()


def check_python(python_path: Path, imports: str, label: str) -> None:
    if not python_path.exists():
        raise SystemExit(f"{label} interpreter does not exist: {python_path}")
    result = subprocess.run(
        [str(python_path), "-c", imports],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise SystemExit(f"{label} dependency check failed with {python_path}:\n{detail}")


def interface_ipv4_addresses(interface: str) -> list[str]:
    result = subprocess.run(
        ["ip", "-4", "-o", "addr", "show", "dev", interface],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return []
    addresses = []
    for line in result.stdout.splitlines():
        match = re.search(r"\binet\s+([0-9.]+/[0-9]+)", line)
        if match:
            addresses.append(match.group(1))
    return addresses


def check_network_interface(interface: str) -> None:
    interface_path = Path("/sys/class/net") / interface
    if not interface_path.exists():
        available = sorted(path.name for path in Path("/sys/class/net").iterdir())
        raise SystemExit(
            f"Network interface {interface!r} does not exist. Available interfaces: {available}"
        )
    carrier_path = interface_path / "carrier"
    carrier = carrier_path.read_text().strip() if carrier_path.exists() else "unknown"
    if carrier == "0":
        raise SystemExit(f"Network interface {interface!r} has no physical link carrier.")

    addresses = interface_ipv4_addresses(interface)
    if not addresses:
        raise SystemExit(
            f"Network interface {interface!r} is connected but has no IPv4 address.\n"
            "CycloneDDS cannot bind to it in this state. Restore the Go2-facing static IPv4 "
            "configuration, then rerun the audit.\n"
            "Check with:\n"
            f"  ip -br addr show {interface}\n"
            "Typical Unitree setups place the computer and robot on the same "
            "192.168.123.0/24 subnet; use the address assigned by your existing network profile."
        )

    parsed = [ipaddress.ip_interface(address) for address in addresses]
    print(f"[INFO] Network interface {interface}: IPv4={addresses}, carrier={carrier}")
    if not any(address.ip.is_private for address in parsed):
        print("[WARN] Selected DDS interface has no private IPv4 address.")


def main() -> int:
    args = parse_args()
    capture_python = args.capture_python.resolve()
    analysis_python = args.analysis_python.resolve()
    check_python(capture_python, "import cyclonedds, numpy", "Hardware capture")
    check_python(analysis_python, "import torch, numpy", "Observation analysis")
    check_network_interface(args.net_if)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stream_path = output_dir / f"{timestamp}_stationary_lowstate.jsonl"
    probe_summary_path = output_dir / f"{timestamp}_stationary_probe_summary.json"
    audit_path = output_dir / f"{timestamp}_stationary_observation_audit.json"

    print("[SAFETY] Read-only audit: no LowCmd publisher, mode switch, or policy actuation.")
    print("[SETUP] Keep the robot stationary in a stable standing pose.")
    print(f"[INFO] DDS capture Python: {capture_python}")
    print(f"[INFO] Analysis Python: {analysis_python}")
    capture_cmd = [
        str(capture_python),
        str(REPO_ROOT / "scripts/deploy/probe_go2_readonly.py"),
        "--net-if",
        args.net_if,
        "--duration",
        str(args.duration),
        "--print-every",
        "1.0",
        "--no-sport",
        "--lowstate-stream-jsonl-out",
        str(stream_path),
        "--json-out",
        str(probe_summary_path),
    ]
    subprocess.run(capture_cmd, check=True)

    analyze_cmd = [
        str(analysis_python),
        str(REPO_ROOT / "scripts/deploy/analyze_go2_stationary_observation_audit.py"),
        "--lowstate-jsonl",
        str(stream_path),
        "--bundle-dir",
        str(args.bundle_dir.resolve()),
        "--json-out",
        str(audit_path),
    ]
    subprocess.run(analyze_cmd, check=True)
    print(f"[DONE] Stationary observation audit: {audit_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
