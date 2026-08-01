#!/usr/bin/env python3
"""Visualize MuJoCo terrains used by validation suites.

This is a terrain sanity-check tool. It does not run the policy and it does not
produce validation metrics. Use it to verify that the XML/MJB scenes in the
validation matrix spawn correctly and represent the intended terrain families.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from mujoco_ood_scenarios import scenario_set


DEFAULT_SUITES = [
    "mujoco_nominal_v1",
    "mujoco_disturb_v2_moderate",
    "mujoco_rough_v1",
    "mujoco_rough_v2_hard",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        action="append",
        default=[],
        help=(
            "Validation suite to visualize. Can be passed multiple times. "
            f"Defaults to: {', '.join(DEFAULT_SUITES)}"
        ),
    )
    parser.add_argument(
        "--scenario",
        default="",
        help="Optional scenario name filter. If omitted, unique scenes from selected suites are shown.",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=20.0,
        help="Viewer duration per scene. Use <=0 to keep the scene open until the viewer is closed.",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Print the scene list without opening MuJoCo viewers.",
    )
    return parser.parse_args()


def _load_mujoco():
    try:
        import mujoco
        import mujoco.viewer
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise SystemExit(
            "MuJoCo Python is not importable. Run this inside the MuJoCo environment, "
            "for example the same env used by run_combined_asymppo_model5099_mujoco_validation.sh."
        ) from exc
    return mujoco


def _scene_entries(suites: list[str], scenario_filter: str) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    seen_paths: set[str] = set()
    for suite in suites:
        for scenario in scenario_set(suite):
            if scenario_filter and scenario.name != scenario_filter:
                continue
            payload = scenario.to_json_dict()
            model_path = str(Path(str(payload["model_path"])).resolve())
            key = model_path if not scenario_filter else f"{suite}:{scenario.name}:{model_path}"
            if key in seen_paths:
                continue
            seen_paths.add(key)
            entries.append(
                {
                    "suite": suite,
                    "scenario": scenario.name,
                    "model_path": model_path,
                    "command": str(payload.get("command", "")),
                }
            )
    return entries


def _configure_camera(viewer, model, data) -> None:
    viewer.cam.distance = 8.0
    viewer.cam.azimuth = 135.0
    viewer.cam.elevation = -35.0
    if model.nbody > 1:
        data_body = data.xpos[1]
        viewer.cam.lookat[:] = data_body


def _view_scene(mujoco, entry: dict[str, str], seconds: float) -> None:
    model_path = Path(entry["model_path"])
    if not model_path.exists():
        print(f"[MISSING] {entry['suite']}::{entry['scenario']} -> {model_path}")
        return
    print(f"[VIEW] {entry['suite']}::{entry['scenario']}")
    print(f"[VIEW] model_path={model_path}")
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        _configure_camera(viewer, model, data)
        start = time.perf_counter()
        while viewer.is_running():
            mujoco.mj_forward(model, data)
            viewer.sync()
            if seconds > 0.0 and time.perf_counter() - start >= seconds:
                break
            time.sleep(0.02)


def main() -> int:
    args = parse_args()
    suites = args.suite or DEFAULT_SUITES
    entries = _scene_entries(suites, args.scenario)
    if not entries:
        print("[INFO] No matching scenes.")
        return 1

    print("[INFO] Scenes selected:")
    for idx, entry in enumerate(entries, start=1):
        print(f"{idx:02d}. {entry['suite']}::{entry['scenario']} -> {entry['model_path']}")

    if args.list_only:
        return 0

    mujoco = _load_mujoco()
    for entry in entries:
        _view_scene(mujoco, entry, args.seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
