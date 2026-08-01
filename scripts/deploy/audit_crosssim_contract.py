#!/usr/bin/env python3
"""Audit the frozen Go2 Isaac/MuJoCo cross-simulator validation contract.

Run this with the MuJoCo environment to include compiled MJCF defaults:

    /home/bhuvan/miniconda3/envs/rma-mujoco/bin/python \
      scripts/deploy/audit_crosssim_contract.py --strict
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROFILE = REPO_ROOT / "configs/validation/go2_crosssim_validation_v1.json"
DEFAULT_BUNDLE = REPO_ROOT / "rma_go2_lab/policies/exported/go2_blind_rough_asymppo_mjlab_v1_candidate"
DEFAULT_DEPLOY_CONFIG = DEFAULT_BUNDLE / "go2_blind_rough_asymppo_mjlab_v1_candidate.deploy_config.json"
DEFAULT_MUJOCO_MODEL = REPO_ROOT / "reference_repos/mujoco_menagerie/unitree_go2/scene.xml"
DEFAULT_ISAAC_ASSET = Path("/home/bhuvan/assets/go2/go2.usd")
sys.path.insert(0, str(REPO_ROOT / "scripts/deploy"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--deploy-config", type=Path, default=DEFAULT_DEPLOY_CONFIG)
    parser.add_argument("--mujoco-model", type=Path, default=DEFAULT_MUJOCO_MODEL)
    parser.add_argument("--isaac-asset", type=Path, default=DEFAULT_ISAAC_ASSET)
    parser.add_argument("--isaac-report", type=Path)
    parser.add_argument("--mujoco-report", type=Path)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=REPO_ROOT / "artifacts/diagnostics/go2_crosssim_contract_audit_v1.json",
    )
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _all_close(values: list[float], expected: float, tol: float = 1.0e-8) -> bool:
    return all(abs(float(value) - expected) <= tol for value in values)


def _compiled_mujoco_contract(model_path: Path) -> dict[str, Any]:
    try:
        import mujoco
    except ImportError:
        return {
            "available": False,
            "reason": "Run with the rma-mujoco Python environment to compile and inspect the MJCF model.",
        }

    model = mujoco.MjModel.from_xml_path(str(model_path))

    def object_id(object_type, name: str) -> int:
        return int(mujoco.mj_name2id(model, object_type, name))

    joint_names = [
        "FL_hip_joint",
        "FR_hip_joint",
        "RL_hip_joint",
        "RR_hip_joint",
        "FL_thigh_joint",
        "FR_thigh_joint",
        "RL_thigh_joint",
        "RR_thigh_joint",
        "FL_calf_joint",
        "FR_calf_joint",
        "RL_calf_joint",
        "RR_calf_joint",
    ]
    joint_ids = [object_id(mujoco.mjtObj.mjOBJ_JOINT, name) for name in joint_names]
    dof_ids = [int(model.jnt_dofadr[joint_id]) for joint_id in joint_ids]
    floor_id = object_id(mujoco.mjtObj.mjOBJ_GEOM, "floor")
    foot_ids = {
        name: object_id(mujoco.mjtObj.mjOBJ_GEOM, name)
        for name in ("FL", "FR", "RL", "RR")
    }
    base_id = object_id(mujoco.mjtObj.mjOBJ_BODY, "base")

    return {
        "available": True,
        "mujoco_version": mujoco.__version__,
        "raw_model": {
            "physics_dt_s": float(model.opt.timestep),
            "gravity_m_s2": model.opt.gravity.tolist(),
            "integrator_enum": int(model.opt.integrator),
            "solver_enum": int(model.opt.solver),
            "solver_iterations": int(model.opt.iterations),
            "line_search_iterations": int(model.opt.ls_iterations),
            "friction_cone_enum": int(model.opt.cone),
            "impedance_ratio": float(model.opt.impratio),
            "floor_friction": model.geom_friction[floor_id].tolist(),
            "foot_friction": {
                name: model.geom_friction[geom_id].tolist()
                for name, geom_id in foot_ids.items()
            },
            "joint_passive_damping": model.dof_damping[dof_ids].tolist(),
            "joint_frictionloss": model.dof_frictionloss[dof_ids].tolist(),
            "joint_armature": model.dof_armature[dof_ids].tolist(),
            "base_mass_kg": float(model.body_mass[base_id]),
            "base_inertia": model.body_inertia[base_id].tolist(),
            "body_mass_kg": {
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id): float(model.body_mass[body_id])
                for body_id in range(1, int(model.nbody))
            },
            "joint_ranges_rad": {
                name: model.jnt_range[joint_id].tolist()
                for name, joint_id in zip(joint_names, joint_ids)
            },
        },
    }


def _report_runtime_contract(path: Path | None, backend: str) -> dict[str, Any] | None:
    if path is None:
        return None
    payload = _load_json(path)
    if backend == "mujoco":
        runtime = payload.get("runtime_rehearsal") or payload
        return {
            "path": str(path.resolve()),
            "control_dt": runtime.get("control_dt"),
            "physics_dt": runtime.get("physics_dt"),
            "substeps_per_control": runtime.get("substeps_per_control"),
            "runtime_overrides": runtime.get("runtime_overrides"),
            "model_diagnostics": runtime.get("model_diagnostics"),
        }
    return {
        "path": str(path.resolve()),
        "environment": payload.get("environment"),
        "resolved_runtime_contract": payload.get("resolved_runtime_contract"),
        "push": payload.get("push"),
        "max_steps": payload.get("max_steps"),
        "warmup_steps": payload.get("warmup_steps"),
    }


def main() -> int:
    args = parse_args()
    profile = _load_json(args.profile)
    deploy = _load_json(args.deploy_config)
    checks: list[dict[str, Any]] = []

    def check(name: str, ok: bool, expected: Any, actual: Any, category: str = "exact") -> None:
        checks.append(
            {
                "name": name,
                "category": category,
                "ok": bool(ok),
                "expected": expected,
                "actual": actual,
            }
        )

    timing = profile["timing"]
    policy = profile["policy"]
    nominal = profile["nominal_robot"]
    deploy_control = deploy["control"]
    deploy_policy = deploy["observations"]
    deploy_robot = deploy["robot"]
    deploy_actions = deploy["actions"]

    check("physics_dt", deploy_control["physics_dt"] == timing["physics_dt_s"], timing["physics_dt_s"], deploy_control["physics_dt"])
    check("control_dt", deploy_control["step_dt"] == timing["control_dt_s"], timing["control_dt_s"], deploy_control["step_dt"])
    check(
        "physics_substeps",
        deploy_control["decimation"] == timing["physics_substeps_per_control"],
        timing["physics_substeps_per_control"],
        deploy_control["decimation"],
    )
    check("policy_dim", deploy_policy["policy_dim"] == policy["policy_dim"], policy["policy_dim"], deploy_policy["policy_dim"])
    check(
        "history_length",
        deploy_policy["policy_history_length"] == policy["history_length"],
        policy["history_length"],
        deploy_policy["policy_history_length"],
    )
    check(
        "history_dim",
        deploy_policy["policy_history_dim"] == policy["history_dim"],
        policy["history_dim"],
        deploy_policy["policy_history_dim"],
    )
    check(
        "history_layout",
        deploy_policy.get("history_layout") == policy["history_layout"],
        policy["history_layout"],
        deploy_policy.get("history_layout"),
    )
    check(
        "joint_order",
        deploy_robot["joint_names"] == policy["joint_order"],
        policy["joint_order"],
        deploy_robot["joint_names"],
    )
    observation_order = [term["name"] for term in deploy_policy["policy_order"]]
    check("observation_order", observation_order == policy["observation_order"], policy["observation_order"], observation_order)
    check(
        "action_scale",
        _all_close(deploy_actions["scale"], policy["action_scale"]),
        policy["action_scale"],
        deploy_actions["scale"],
    )
    check(
        "action_offset",
        deploy_actions["offset"] == policy["action_offset"],
        policy["action_offset"],
        deploy_actions["offset"],
    )
    check(
        "joint_stiffness",
        _all_close(deploy_robot["joint_stiffness"], nominal["joint_stiffness_nm_rad"]),
        nominal["joint_stiffness_nm_rad"],
        deploy_robot["joint_stiffness"],
    )
    check(
        "joint_damping",
        _all_close(deploy_robot["joint_damping"], nominal["joint_damping_nm_s_rad"]),
        nominal["joint_damping_nm_s_rad"],
        deploy_robot["joint_damping"],
    )
    check(
        "effort_limit",
        _all_close(deploy_robot["effort_limit"], nominal["effort_limit_nm"]),
        nominal["effort_limit_nm"],
        deploy_robot["effort_limit"],
    )
    check(
        "velocity_limit",
        _all_close(deploy_robot["velocity_limit"], nominal["velocity_limit_rad_s"]),
        nominal["velocity_limit_rad_s"],
        deploy_robot["velocity_limit"],
    )
    check(
        "spawn_height",
        abs(float(deploy_robot["base_init_pos"][2]) - float(nominal["spawn_height_m"])) < 1.0e-8,
        nominal["spawn_height_m"],
        deploy_robot["base_init_pos"][2],
    )

    from mujoco_ood_scenarios import scenario_set

    matched_scenarios = [item.to_json_dict() for item in scenario_set("mujoco_fr_asymmetry_matched_v2")]
    matched_by_name = {item["name"]: item for item in matched_scenarios}
    expected_commands = profile["matched_scenarios"]["commands"]
    expected_gap = profile["matched_hardware_gap"]
    expected_overrides = profile["mujoco_backend"]["runtime_overrides"]
    check(
        "matched_v2_scenario_names",
        set(matched_by_name) == set(expected_commands),
        sorted(expected_commands),
        sorted(matched_by_name),
    )
    for scenario_name, command in expected_commands.items():
        scenario = matched_by_name.get(scenario_name, {})
        check(
            f"matched_v2_command/{scenario_name}",
            scenario.get("command") == command,
            command,
            scenario.get("command"),
        )
        for profile_key, scenario_key in (
            ("observation_delay_control_steps", "obs_delay_steps"),
            ("action_delay_control_steps", "action_delay_steps"),
            ("command_delay_control_steps", "command_delay_steps"),
        ):
            check(
                f"matched_v2_{scenario_key}/{scenario_name}",
                scenario.get(scenario_key) == expected_gap[profile_key],
                expected_gap[profile_key],
                scenario.get(scenario_key),
            )
        for override_key, expected_value in expected_overrides.items():
            if override_key == "actuator_emulation":
                continue
            check(
                f"matched_v2_{override_key}/{scenario_name}",
                scenario.get(override_key) == expected_value,
                expected_value,
                scenario.get(override_key),
            )

    expected_push = profile["matched_scenarios"]["push"]
    for scenario_name, force_key in (("asym_push_left", "left_force_n"), ("asym_push_right", "right_force_n")):
        wrench = matched_by_name.get(scenario_name, {}).get("wrench_schedule", [{}])[0]
        check(
            f"matched_v2_push/{scenario_name}",
            wrench.get("start_step") == expected_push["start_control_step"]
            and wrench.get("duration_steps") == expected_push["duration_control_steps"]
            and wrench.get("force_world") == expected_push[force_key],
            {
                "start_step": expected_push["start_control_step"],
                "duration_steps": expected_push["duration_control_steps"],
                "force_world": expected_push[force_key],
            },
            wrench,
        )

    compiled_mujoco = _compiled_mujoco_contract(args.mujoco_model)
    if compiled_mujoco.get("available"):
        raw = compiled_mujoco["raw_model"]
        check(
            "mujoco_raw_timestep_is_overridden",
            raw["physics_dt_s"] != timing["physics_dt_s"],
            "raw MJCF differs but runtime must override to profile physics_dt",
            raw["physics_dt_s"],
            category="recorded_backend_difference",
        )
        check(
            "mujoco_backend_native_passive_damping_recorded",
            not _all_close(raw["joint_passive_damping"], nominal["passive_joint_damping"]),
            "raw MJCF differs from Isaac and matched validation retains it at scale 1.0",
            raw["joint_passive_damping"],
            category="recorded_backend_difference",
        )
        check(
            "mujoco_backend_native_frictionloss_recorded",
            not _all_close(raw["joint_frictionloss"], nominal["passive_joint_frictionloss"]),
            "raw MJCF differs from Isaac and matched validation retains it at scale 1.0",
            raw["joint_frictionloss"],
            category="recorded_backend_difference",
        )
        check(
            "mujoco_backend_native_armature_recorded",
            not _all_close(raw["joint_armature"], nominal["joint_armature"]),
            "raw MJCF differs from Isaac and matched validation retains it at scale 1.0",
            raw["joint_armature"],
            category="recorded_backend_difference",
        )

    exact_failures = [
        item for item in checks
        if item["category"] == "exact" and not item["ok"]
    ]
    report = {
        "profile": profile,
        "status": "shared_contract_ready" if not exact_failures else "shared_contract_mismatch",
        "important_conclusion": (
            "The shared policy, timing, nominal actuator, reset, command, and disturbance contract can be matched. "
            "The simulators are not and cannot be made physically identical because their solvers, contacts, and "
            "source robot assets differ."
        ),
        "checks": checks,
        "exact_failure_count": len(exact_failures),
        "compiled_mujoco": compiled_mujoco,
        "runtime_reports": {
            "isaac": _report_runtime_contract(args.isaac_report, "isaac"),
            "mujoco": _report_runtime_contract(args.mujoco_report, "mujoco"),
        },
        "artifacts": {
            "profile": {"path": str(args.profile.resolve()), "sha256": _sha256(args.profile)},
            "deploy_config": {"path": str(args.deploy_config.resolve()), "sha256": _sha256(args.deploy_config)},
            "isaac_asset": {"path": str(args.isaac_asset), "sha256": _sha256(args.isaac_asset)},
            "mujoco_scene": {"path": str(args.mujoco_model.resolve()), "sha256": _sha256(args.mujoco_model)},
            "mujoco_robot": {
                "path": str((args.mujoco_model.parent / "go2.xml").resolve()),
                "sha256": _sha256(args.mujoco_model.parent / "go2.xml"),
            },
        },
    }

    text = json.dumps(report, indent=2)
    print(text)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(text + "\n", encoding="utf-8")
    print(f"[INFO] Wrote cross-simulator audit: {args.json_out.resolve()}")
    if args.strict and exact_failures:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
