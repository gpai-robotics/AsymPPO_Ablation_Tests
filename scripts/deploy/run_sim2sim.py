#!/usr/bin/env python3
"""MuJoCo Sim2Sim rehearsal entrypoint for frozen deployment bundles.

This script now acts as a real Sim2Sim preflight gate:

- validate the frozen deployment bundle
- load export metadata
- confirm runtime-contract consistency
- check MuJoCo backend availability
- report what is still missing before a true MuJoCo rollout can happen

The repo does not yet contain a MuJoCo robot runtime bridge, so the most honest
current implementation is a strong preflight contract rather than a fake
"successful" Sim2Sim run.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


DEFAULT_GO2_MODEL_PATH = (
    Path(__file__).resolve().parents[2]
    / "reference_repos"
    / "mujoco_menagerie"
    / "unitree_go2"
    / "scene.xml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", required=True)
    parser.add_argument("--robot", default="go2")
    parser.add_argument("--backend", default="mujoco")
    parser.add_argument("--control-rate-hz", type=float, default=50.0)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument(
        "--history-mode",
        default="runtime",
        help="History update mode for adaptive students.",
    )
    parser.add_argument(
        "--history-ablation",
        default="normal",
        choices=("normal", "zero", "frozen"),
        help="Optional runtime ablation for history-bearing deploy policies.",
    )
    parser.add_argument(
        "--model-path",
        default=str(DEFAULT_GO2_MODEL_PATH),
        help=(
            "MuJoCo model path for the Sim2Sim preflight. Defaults to the "
            "canonical Go2 scene in reference_repos/mujoco_menagerie."
        ),
    )
    parser.add_argument(
        "--json-out",
        default="",
        help="Optional path to save the preflight report as JSON.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero if blockers are found.",
    )
    parser.add_argument(
        "--execute-runtime",
        action="store_true",
        help=(
            "Attempt a real MuJoCo runtime rehearsal using the repo-owned bridge. "
            "If omitted, this script acts as a preflight gate only."
        ),
    )
    parser.add_argument("--command-x", type=float, default=0.5)
    parser.add_argument("--command-y", type=float, default=0.0)
    parser.add_argument("--command-yaw", type=float, default=0.0)
    parser.add_argument(
        "--trace-steps",
        type=int,
        default=25,
        help=(
            "Number of initial control steps to capture in the runtime trace. "
            "Use -1 to capture the full rollout."
        ),
    )
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="Launch the repo-owned MuJoCo runtime with a passive visual viewer.",
    )
    parser.add_argument(
        "--viewer-dt",
        type=float,
        default=0.02,
        help="Viewer sync period in seconds when --viewer is enabled.",
    )
    parser.add_argument(
        "--real-time-factor",
        type=float,
        default=1.0,
        help="Viewer playback speed factor. 1.0 is real time, 0.5 is half speed.",
    )
    parser.add_argument(
        "--latent-clamp-max-abs",
        type=float,
        default=0.0,
        help=(
            "Optional deploy-side clamp on phi(history) latent values for debugging. "
            "0 disables clamping."
        ),
    )
    parser.add_argument(
        "--ground-friction",
        type=float,
        default=0.0,
        help="Optional override for the ground tangential friction coefficient. 0 keeps the scene default.",
    )
    parser.add_argument(
        "--foot-friction",
        type=float,
        default=0.0,
        help="Optional override for the foot tangential friction coefficient. 0 keeps the model default.",
    )
    parser.add_argument(
        "--base-mass-scale",
        type=float,
        default=1.0,
        help="Optional multiplicative scale applied to the base body mass/inertia.",
    )
    parser.add_argument(
        "--motor-strength-scale",
        type=float,
        default=1.0,
        help="Optional multiplicative scale applied to the action-to-target strength.",
    )
    parser.add_argument(
        "--joint-damping-scale",
        type=float,
        default=1.0,
        help="Optional multiplicative scale applied to joint damping in the runtime bridge.",
    )
    parser.add_argument(
        "--passive-joint-damping-scale",
        type=float,
        default=1.0,
        help="Optional multiplicative scale applied to MuJoCo DOF damping on the robot joints.",
    )
    parser.add_argument(
        "--passive-joint-frictionloss-scale",
        type=float,
        default=1.0,
        help="Optional multiplicative scale applied to MuJoCo DOF frictionloss on the robot joints.",
    )
    parser.add_argument(
        "--actuator-model",
        choices=("simple_pd", "isaac_dc_motor"),
        default="simple_pd",
        help="Runtime actuator emulation mode for torque-driven MuJoCo actuators.",
    )
    parser.add_argument(
        "--dc-motor-velocity-limit",
        type=float,
        default=30.0,
        help="Velocity limit used by the Isaac-like DC motor saturation model.",
    )
    parser.add_argument(
        "--teleop-keyboard",
        action="store_true",
        help="Enable keyboard teleoperation for the viewer run.",
    )
    parser.add_argument(
        "--teleop-step-x",
        type=float,
        default=0.1,
        help="Increment for forward/backward keyboard teleop commands.",
    )
    parser.add_argument(
        "--teleop-step-y",
        type=float,
        default=0.05,
        help="Increment for lateral keyboard teleop commands.",
    )
    parser.add_argument(
        "--teleop-step-yaw",
        type=float,
        default=0.15,
        help="Increment for yaw keyboard teleop commands.",
    )
    parser.add_argument(
        "--teleop-limit-x",
        type=float,
        default=1.0,
        help="Absolute clamp for forward keyboard teleop command.",
    )
    parser.add_argument(
        "--teleop-limit-y",
        type=float,
        default=0.4,
        help="Absolute clamp for lateral keyboard teleop command.",
    )
    parser.add_argument(
        "--teleop-limit-yaw",
        type=float,
        default=1.0,
        help="Absolute clamp for yaw keyboard teleop command.",
    )
    parser.add_argument(
        "--scenario-json",
        default="",
        help=(
            "Optional structured scenario JSON describing scene path, runtime overrides, "
            "command schedule, and wrench schedule for MuJoCo OOD testing."
        ),
    )
    parser.add_argument("--seed", type=int, default=0, help="Runtime seed for reset/randomization sampling.")
    parser.add_argument(
        "--reset-pos-xy-jitter",
        type=float,
        default=0.0,
        help="Uniform reset jitter radius applied independently to base x/y position.",
    )
    parser.add_argument(
        "--reset-yaw-jitter-deg",
        type=float,
        default=0.0,
        help="Uniform reset yaw jitter in degrees.",
    )
    parser.add_argument(
        "--reset-joint-pos-jitter",
        type=float,
        default=0.0,
        help="Uniform reset jitter for joint positions around the default pose.",
    )
    parser.add_argument(
        "--reset-joint-vel-jitter",
        type=float,
        default=0.0,
        help="Uniform reset jitter for joint velocities.",
    )
    return parser.parse_args()


def _find_torchscript_artifact(bundle_dir: Path, manifest: dict) -> Path | None:
    for artifact in manifest.get("exported_artifacts", []):
        if artifact.endswith(".torchscript.pt"):
            artifact_path = bundle_dir / artifact
            if artifact_path.exists():
                return artifact_path
    return None


def _find_export_metadata_artifact(bundle_dir: Path, manifest: dict) -> Path | None:
    for artifact in manifest.get("exported_artifacts", []):
        if artifact.endswith(".export_metadata.json"):
            artifact_path = bundle_dir / artifact
            if artifact_path.exists():
                return artifact_path
    return None


def _find_deploy_config_artifact(bundle_dir: Path, manifest: dict) -> Path | None:
    for artifact in manifest.get("exported_artifacts", []):
        if artifact.endswith(".deploy_config.json"):
            artifact_path = bundle_dir / artifact
            if artifact_path.exists():
                return artifact_path
    return None


def _runtime_available(backend: str) -> tuple[bool, str]:
    if backend == "mujoco":
        spec = importlib.util.find_spec("mujoco")
        return spec is not None, "mujoco"
    return False, backend


def _expected_contract_for_policy_kind(policy_kind: str) -> tuple[list[str], str]:
    if policy_kind == "blind_adaptive_student":
        return ["policy", "policy_history"], "per-step history update via phi(history) -> z_hat"
    if policy_kind == "blind_history_policy":
        return ["policy", "policy_history"], ""
    if policy_kind == "blind_fixed_policy":
        return ["policy"], ""
    raise SystemExit(f"Unsupported policy kind for Sim2Sim preflight: {policy_kind}")


def _load_scenario_json(path: str) -> dict[str, object]:
    if not path:
        return {}
    scenario_path = Path(path)
    if not scenario_path.exists():
        raise SystemExit(f"Scenario JSON not found: {scenario_path}")
    return json.loads(scenario_path.read_text())


def main() -> int:
    args = parse_args()
    scenario = _load_scenario_json(args.scenario_json)
    scenario_name = str(scenario.get("name", ""))
    bundle_dir = Path(args.bundle_dir)
    manifest_path = bundle_dir / "bundle_manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"Missing bundle manifest: {manifest_path}")

    manifest = json.loads(manifest_path.read_text())
    metadata_path = _find_export_metadata_artifact(bundle_dir, manifest)
    deploy_config_path = _find_deploy_config_artifact(bundle_dir, manifest)
    metadata = json.loads(metadata_path.read_text()) if metadata_path is not None else {}
    artifact_path = _find_torchscript_artifact(bundle_dir, manifest)
    model_path = Path(str(scenario.get("model_path", args.model_path))) if (scenario.get("model_path") or args.model_path) else None
    scenario_command = scenario.get("command")
    command_x = args.command_x
    command_y = args.command_y
    command_yaw = args.command_yaw
    if isinstance(scenario_command, list) and len(scenario_command) == 3:
        command_x, command_y, command_yaw = map(float, scenario_command)

    ground_friction = float(scenario.get("ground_friction", args.ground_friction))
    foot_friction = float(scenario.get("foot_friction", args.foot_friction))
    base_mass_scale = float(scenario.get("base_mass_scale", args.base_mass_scale))
    motor_strength_scale = float(scenario.get("motor_strength_scale", args.motor_strength_scale))
    joint_damping_scale = float(scenario.get("joint_damping_scale", args.joint_damping_scale))
    passive_joint_damping_scale = float(
        scenario.get("passive_joint_damping_scale", args.passive_joint_damping_scale)
    )
    passive_joint_frictionloss_scale = float(
        scenario.get("passive_joint_frictionloss_scale", args.passive_joint_frictionloss_scale)
    )
    command_schedule = scenario.get("command_schedule", [])
    wrench_schedule = scenario.get("wrench_schedule", [])
    seed = int(scenario.get("seed", args.seed))
    reset_pos_xy_jitter = float(scenario.get("reset_pos_xy_jitter", args.reset_pos_xy_jitter))
    reset_yaw_jitter_deg = float(scenario.get("reset_yaw_jitter_deg", args.reset_yaw_jitter_deg))
    reset_joint_pos_jitter = float(scenario.get("reset_joint_pos_jitter", args.reset_joint_pos_jitter))
    reset_joint_vel_jitter = float(scenario.get("reset_joint_vel_jitter", args.reset_joint_vel_jitter))

    checks: list[dict[str, object]] = []
    blockers: list[str] = []

    def add_check(name: str, ok: bool, detail: str, *, blocking: bool = True) -> None:
        checks.append({"name": name, "ok": ok, "detail": detail, "blocking": blocking})
        if blocking and not ok:
            blockers.append(f"{name}: {detail}")

    policy_kind = manifest.get("policy_kind")
    expected_groups, expected_latent_semantics = _expected_contract_for_policy_kind(policy_kind)
    add_check(
        "policy_kind",
        policy_kind in {"blind_adaptive_student", "blind_history_policy", "blind_fixed_policy"},
        f"found={policy_kind}",
    )
    add_check(
        "deployable_observation_groups",
        manifest.get("deployable_observation_groups") == expected_groups,
        f"found={manifest.get('deployable_observation_groups')}",
    )
    add_check(
        "latent_update_semantics",
        manifest.get("latent_update_semantics", "") == expected_latent_semantics,
        f"found={manifest.get('latent_update_semantics')}",
    )
    add_check(
        "control_rate_match",
        abs(float(manifest.get("control_rate_hz", -1.0)) - float(args.control_rate_hz)) < 1.0e-6,
        f"bundle={manifest.get('control_rate_hz')} requested={args.control_rate_hz}",
    )
    add_check(
        "torchscript_artifact_present",
        artifact_path is not None,
        f"artifact={artifact_path}" if artifact_path else "missing .torchscript.pt export",
    )
    add_check(
        "export_metadata_present",
        metadata_path is not None,
        f"artifact={metadata_path}" if metadata_path else "missing .export_metadata.json sidecar",
    )
    add_check(
        "deploy_config_present",
        deploy_config_path is not None,
        f"artifact={deploy_config_path}" if deploy_config_path else "missing .deploy_config.json sidecar",
    )

    runtime_ok, runtime_name = _runtime_available(args.backend)
    add_check(
        "backend_runtime_available",
        runtime_ok,
        f"backend={runtime_name} available={runtime_ok}",
    )

    add_check(
        "mujoco_model_present",
        model_path is not None and model_path.exists(),
        f"model_path={model_path}" if model_path is not None else "no MuJoCo model path provided",
    )
    add_check(
        "primary_model_selection",
        model_path == DEFAULT_GO2_MODEL_PATH,
        (
            f"model_path={model_path} "
            f"default_primary={DEFAULT_GO2_MODEL_PATH}"
        ),
        blocking=False,
    )

    runtime_contract = metadata.get("runtime_contract", {})
    tensor_contract = metadata.get("tensor_contract", {})
    add_check(
        "tensor_contract_present",
        bool(tensor_contract),
        f"tensor_contract_keys={sorted(tensor_contract.keys()) if tensor_contract else []}",
    )

    bridge_module_ok = importlib.util.find_spec("mujoco_runtime") is not None
    add_check(
        "runtime_bridge_module_present",
        bridge_module_ok,
        "scripts/deploy/mujoco_runtime.py importable" if bridge_module_ok else "mujoco_runtime module not found",
    )

    runtime_rehearsal = None
    status = "ready_for_runtime_bridge" if not blockers else "blocked_preflight"

    if args.execute_runtime and not blockers:
        from mujoco_runtime import BridgeConfig, Go2MujocoDeployBridge

        bridge = Go2MujocoDeployBridge(
            BridgeConfig(
                model_path=model_path,
                policy_artifact_path=artifact_path,
                deploy_config_path=deploy_config_path,
                control_dt=1.0 / args.control_rate_hz,
                command_x=command_x,
                command_y=command_y,
                command_yaw=command_yaw,
                trace_steps=args.trace_steps,
                viewer=args.viewer,
                viewer_dt=args.viewer_dt,
                real_time_factor=args.real_time_factor,
                latent_clamp_max_abs=args.latent_clamp_max_abs,
                policy_kind=policy_kind,
                ground_friction=ground_friction,
                foot_friction=foot_friction,
                base_mass_scale=base_mass_scale,
                motor_strength_scale=motor_strength_scale,
                joint_damping_scale=joint_damping_scale,
                passive_joint_damping_scale=passive_joint_damping_scale,
                passive_joint_frictionloss_scale=passive_joint_frictionloss_scale,
                actuator_model=args.actuator_model,
                dc_motor_velocity_limit=args.dc_motor_velocity_limit,
                teleop_keyboard=args.teleop_keyboard,
                teleop_step_x=args.teleop_step_x,
                teleop_step_y=args.teleop_step_y,
                teleop_step_yaw=args.teleop_step_yaw,
                teleop_limit_x=args.teleop_limit_x,
                teleop_limit_y=args.teleop_limit_y,
                teleop_limit_yaw=args.teleop_limit_yaw,
                scenario_name=scenario_name,
                command_schedule=command_schedule if isinstance(command_schedule, list) else [],
                wrench_schedule=wrench_schedule if isinstance(wrench_schedule, list) else [],
                seed=seed,
                reset_pos_xy_jitter=reset_pos_xy_jitter,
                reset_yaw_jitter_deg=reset_yaw_jitter_deg,
                reset_joint_pos_jitter=reset_joint_pos_jitter,
                reset_joint_vel_jitter=reset_joint_vel_jitter,
                history_ablation=args.history_ablation,
            )
        )
        runtime_rehearsal = bridge.run(args.max_steps)
        status = runtime_rehearsal.get("status", "completed_runtime_rehearsal")

    summary = {
        "bundle_dir": str(bundle_dir),
        "policy_name": manifest.get("policy_name"),
        "policy_kind": manifest.get("policy_kind"),
        "backend": args.backend,
        "robot": args.robot,
        "control_rate_hz": args.control_rate_hz,
        "history_mode": args.history_mode,
        "history_ablation": args.history_ablation,
        "max_steps": args.max_steps,
        "trace_steps": args.trace_steps,
        "latent_clamp_max_abs": args.latent_clamp_max_abs,
        "ground_friction": ground_friction,
        "foot_friction": foot_friction,
        "base_mass_scale": base_mass_scale,
        "motor_strength_scale": motor_strength_scale,
        "joint_damping_scale": joint_damping_scale,
        "passive_joint_damping_scale": passive_joint_damping_scale,
        "passive_joint_frictionloss_scale": passive_joint_frictionloss_scale,
        "actuator_model": args.actuator_model,
        "dc_motor_velocity_limit": args.dc_motor_velocity_limit,
        "command": [command_x, command_y, command_yaw],
        "scenario_json": args.scenario_json or None,
        "scenario_name": scenario_name or None,
        "scenario_command_schedule_len": len(command_schedule) if isinstance(command_schedule, list) else 0,
        "scenario_wrench_schedule_len": len(wrench_schedule) if isinstance(wrench_schedule, list) else 0,
        "seed": seed,
        "reset_pos_xy_jitter": reset_pos_xy_jitter,
        "reset_yaw_jitter_deg": reset_yaw_jitter_deg,
        "reset_joint_pos_jitter": reset_joint_pos_jitter,
        "reset_joint_vel_jitter": reset_joint_vel_jitter,
        "artifact_used": str(artifact_path) if artifact_path is not None else None,
        "export_metadata": str(metadata_path) if metadata_path is not None else None,
        "deploy_config": str(deploy_config_path) if deploy_config_path is not None else None,
        "model_path": str(model_path) if model_path is not None else None,
        "runtime_contract": runtime_contract,
        "tensor_contract": tensor_contract,
        "checks": checks,
        "blockers": blockers,
        "status": status,
        "runtime_rehearsal": runtime_rehearsal,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(summary, indent=2) + "\n")

    if args.strict and blockers:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
