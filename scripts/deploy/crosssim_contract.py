"""Shared loader and fingerprinting for strict Isaac/MuJoCo validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROFILE = REPO_ROOT / "configs" / "validation" / "go2_crosssim_validation_v1.json"


def load_profile(path: str | Path = DEFAULT_PROFILE) -> tuple[Path, dict[str, Any]]:
    profile_path = Path(path).expanduser().resolve()
    return profile_path, json.loads(profile_path.read_text(encoding="utf-8"))


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def matched_scenario_dicts(profile: dict[str, Any]) -> list[dict[str, Any]]:
    matched = profile["matched_scenarios"]
    push = matched["push"]
    scenarios = []
    for name, command in matched["commands"].items():
        item: dict[str, Any] = {"name": name, "command": list(command)}
        if name == "asym_push_left":
            item["wrench_schedule"] = [
                {
                    "start_step": push["start_control_step"],
                    "duration_steps": push["duration_control_steps"],
                    "force_world": list(push["left_force_n"]),
                    "label": "push_left",
                }
            ]
        elif name == "asym_push_right":
            item["wrench_schedule"] = [
                {
                    "start_step": push["start_control_step"],
                    "duration_steps": push["duration_control_steps"],
                    "force_world": list(push["right_force_n"]),
                    "label": "push_right",
                }
            ]
        scenarios.append(item)
    return scenarios


def matched_runtime_contract(profile_path: Path, profile: dict[str, Any]) -> dict[str, Any]:
    scenarios = matched_scenario_dicts(profile)
    return {
        "profile_name": profile["profile_name"],
        "profile_path": str(profile_path),
        "profile_sha256": file_sha256(profile_path),
        "scenario_sha256": canonical_json_sha256(scenarios),
        "timing": profile["timing"],
        "policy": profile["policy"],
        "nominal_robot": profile["nominal_robot"],
        "reset": profile["matched_reset"],
        "hardware_gap": profile["matched_hardware_gap"],
        "scenarios": scenarios,
        "terrain_contract": profile["matched_scenarios"]["terrain_contract"],
        "comparison_gate": profile["comparison_gate"],
    }


def mujoco_scenario_overrides(profile: dict[str, Any]) -> dict[str, Any]:
    gap = profile["matched_hardware_gap"]
    overrides = dict(profile["mujoco_backend"]["runtime_overrides"])
    overrides.pop("actuator_emulation", None)
    overrides.update(
        {
            "obs_delay_steps": gap["observation_delay_control_steps"],
            "action_delay_steps": gap["action_delay_control_steps"],
            "command_delay_steps": gap["command_delay_control_steps"],
            "encoder_bias_range": gap["encoder_bias_rad"],
            "obs_hold_prob": gap["observation_hold_probability"],
        }
    )
    return overrides
