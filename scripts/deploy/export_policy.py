#!/usr/bin/env python3
"""Export a frozen deployment candidate into deployable artifacts.

Supported deployment contracts:

- blind adaptive student:
  - `policy` observation group
  - `policy_history` observation group
  - `phi(history) -> z_hat`
  - `pi(policy, z_hat) -> action`

- blind fixed policy:
  - `policy` observation group
  - direct `policy -> action`

- blind history policy:
  - `policy` observation group
  - `policy_history` observation group
  - direct `(policy, policy_history) -> action`

This keeps each exported module faithful to runtime deployment semantics instead
of exporting the full training-time module with privileged paths.
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path
import sys

GO2_JOINT_NAMES = [
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

GO2_ACTUATOR_NAMES = [
    "FL_hip",
    "FR_hip",
    "RL_hip",
    "RR_hip",
    "FL_thigh",
    "FR_thigh",
    "RL_thigh",
    "RR_thigh",
    "FL_calf",
    "FR_calf",
    "RL_calf",
    "RR_calf",
]

GO2_DEFAULT_JOINT_POS = [
    0.1,
    -0.1,
    0.1,
    -0.1,
    0.8,
    0.8,
    1.0,
    1.0,
    -1.5,
    -1.5,
    -1.5,
    -1.5,
]
GO2_BASE_INIT_POS = [0.0, 0.0, 0.4]
GO2_BASE_INIT_QUAT_WXYZ = [1.0, 0.0, 0.0, 0.0]

GO2_JOINT_STIFFNESS = [25.0] * len(GO2_JOINT_NAMES)
GO2_JOINT_DAMPING = [0.5] * len(GO2_JOINT_NAMES)
GO2_EFFORT_LIMIT = [23.5] * len(GO2_JOINT_NAMES)
GO2_VELOCITY_LIMIT = [30.0] * len(GO2_JOINT_NAMES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-name", required=True, help="Bundle policy name.")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Frozen source checkpoint path used for deployment export.",
    )
    parser.add_argument("--task", required=True, help="Registered training task name.")
    parser.add_argument(
        "--phase",
        required=True,
        help="Training phase lineage, for example blind/B2 or adapt-v3-phase2.",
    )
    parser.add_argument(
        "--bundle-dir",
        required=True,
        help="Output deployment bundle directory under rma_go2_lab/policies/exported/.",
    )
    parser.add_argument(
        "--policy-kind",
        default="blind_adaptive_student",
        choices=["blind_adaptive_student", "blind_fixed_policy", "blind_history_policy"],
        help="Deployment policy contract to export.",
    )
    parser.add_argument(
        "--observation-groups",
        default="policy,policy_history",
        help="Comma-separated deployable observation groups for the exported contract.",
    )
    parser.add_argument(
        "--format",
        action="append",
        dest="formats",
        default=[],
        help="Requested export format. Repeat for multiple formats.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the export request without writing files.",
    )
    parser.add_argument(
        "--policy-history-length",
        type=int,
        default=0,
        help=(
            "Deployment-time history length for history-bearing policies. "
            "Use 0 to auto-infer it from the task env config."
        ),
    )
    parser.add_argument(
        "--command-lin-vel-x",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        default=(0.0, 1.0),
        help="Deployment command clamp for forward velocity.",
    )
    parser.add_argument(
        "--command-lin-vel-y",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        default=(0.0, 0.0),
        help="Deployment command clamp for lateral velocity.",
    )
    parser.add_argument(
        "--command-ang-vel-z",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        default=(0.0, 0.0),
        help="Deployment command clamp for yaw velocity.",
    )
    return parser.parse_args()


def _import_torch():
    try:
        import torch
        import torch.nn as nn
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "PyTorch is not available in this Python environment. "
            "Run this script with the IsaacLab Python environment, for example:\n"
            "env TERM=xterm $ISAACLAB_ROOT/isaaclab.sh -p "
            "$REPO/scripts/deploy/export_policy.py ..."
        ) from exc
    return torch, nn


def _import_task_cfg_loader():
    repo_root = Path(__file__).resolve().parents[2]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
    try:
        from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
        import isaaclab_tasks  # noqa: F401
        import rma_go2_lab  # noqa: F401
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "IsaacLab task config utilities are not available in this Python environment. "
            "Run this script with the IsaacLab Python environment."
        ) from exc
    return load_cfg_from_registry


def _parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _infer_history_length_from_existing_bundle(bundle_dir: Path) -> int:
    deploy_configs = sorted(bundle_dir.glob("*.deploy_config.json"))
    for deploy_config_path in deploy_configs:
        try:
            payload = json.loads(deploy_config_path.read_text())
        except Exception:
            continue
        value = payload.get("observations", {}).get("policy_history_length", 0)
        if isinstance(value, int) and value > 0:
            return value
    return 0


def _infer_history_length_from_task(task: str, policy_kind: str, bundle_dir: Path) -> int:
    if policy_kind == "blind_fixed_policy":
        return 0

    existing_bundle_value = _infer_history_length_from_existing_bundle(bundle_dir)
    if existing_bundle_value > 0:
        return existing_bundle_value

    try:
        load_cfg_from_registry = _import_task_cfg_loader()
        env_cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
    except Exception:
        env_cfg = None

    if env_cfg is not None:
        for attr_name in ("policy_history_length", "adaptation_history_length"):
            value = getattr(env_cfg, attr_name, None)
            if isinstance(value, int) and value > 0:
                return value

        obs_cfg = getattr(env_cfg, "observations", None)
        policy_history_cfg = getattr(obs_cfg, "policy_history", None) if obs_cfg is not None else None
        value = getattr(policy_history_cfg, "history_length", None) if policy_history_cfg is not None else None
        if isinstance(value, int) and value > 0:
            return value

    raise SystemExit(
        f"Could not infer history length for task '{task}' and policy kind '{policy_kind}'. "
        "Pass --policy-history-length explicitly."
    )


def _collect_prefix(state_dict: OrderedDict[str, object], prefix: str) -> OrderedDict[str, object]:
    collected = OrderedDict()
    prefix_with_dot = f"{prefix}."
    for key, value in state_dict.items():
        if key.startswith(prefix_with_dot):
            collected[key[len(prefix_with_dot) :]] = value
    if not collected:
        raise RuntimeError(f"Missing state dict prefix '{prefix_with_dot}'.")
    return collected


def _collect_prefix_if_present(state_dict: OrderedDict[str, object], prefix: str) -> OrderedDict[str, object]:
    collected = OrderedDict()
    prefix_with_dot = f"{prefix}."
    for key, value in state_dict.items():
        if key.startswith(prefix_with_dot):
            collected[key[len(prefix_with_dot) :]] = value
    return collected


def _infer_mlp_dims(module_state: OrderedDict[str, object]) -> tuple[int, list[int], int]:
    linear_indices = sorted({int(key.split(".")[0]) for key in module_state if key.endswith(".weight")})
    weight_shapes = [tuple(module_state[f"{idx}.weight"].shape) for idx in linear_indices]
    input_dim = weight_shapes[0][1]
    hidden_dims = [shape[0] for shape in weight_shapes[:-1]]
    output_dim = weight_shapes[-1][0]
    return input_dim, hidden_dims, output_dim


def _build_mlp(nn, input_dim: int, hidden_dims: list[int], output_dim: int):
    layers = []
    prev_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(nn.ELU())
        prev_dim = hidden_dim
    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)


def _build_history_projection(nn, input_dim: int, output_dim: int):
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.ELU(),
    )


def _infer_conv1d_layers(module_state: OrderedDict[str, object]) -> list[tuple[int, int, int, int, int]]:
    conv_indices = sorted({int(key.split(".")[0]) for key in module_state if key.endswith(".weight")})
    layers = []
    for order_idx, module_idx in enumerate(conv_indices):
        weight = module_state[f"{module_idx}.weight"]
        out_channels, in_channels, kernel_size = weight.shape
        dilation = 2**order_idx
        padding = dilation * (int(kernel_size) - 1) // 2
        layers.append((module_idx, int(in_channels), int(out_channels), int(kernel_size), int(dilation), int(padding)))
    return layers


def _export_torchscript(torch, module, bundle_dir: Path, policy_name: str, policy_obs_dim: int, history_dim: int) -> str:
    output_name = f"{policy_name}.torchscript.pt"
    output_path = bundle_dir / output_name
    dummy_policy = torch.zeros(1, policy_obs_dim, dtype=torch.float32)
    dummy_history = torch.zeros(1, history_dim, dtype=torch.float32)
    traced = torch.jit.trace(module, (dummy_policy, dummy_history))
    traced.save(str(output_path))
    return output_name


def _export_onnx(torch, module, bundle_dir: Path, policy_name: str, policy_obs_dim: int, history_dim: int) -> str:
    output_name = f"{policy_name}.onnx"
    output_path = bundle_dir / output_name
    dummy_policy = torch.zeros(1, policy_obs_dim, dtype=torch.float32)
    dummy_history = torch.zeros(1, history_dim, dtype=torch.float32)
    torch.onnx.export(
        module,
        (dummy_policy, dummy_history),
        str(output_path),
        input_names=["policy_obs", "policy_history"],
        output_names=["action"],
        dynamic_axes={
            "policy_obs": {0: "batch"},
            "policy_history": {0: "batch"},
            "action": {0: "batch"},
        },
        opset_version=17,
    )
    return output_name


def _write_sidecar_metadata(
    bundle_dir: Path,
    policy_name: str,
    checkpoint_path: Path,
    task: str,
    phase: str,
    policy_kind: str,
    deployable_observation_groups: list[str],
    policy_obs_dim: int,
    history_dim: int,
    latent_dim: int,
    action_dim: int,
) -> str:
    output_name = f"{policy_name}.export_metadata.json"
    output_path = bundle_dir / output_name
    payload = {
        "policy_name": policy_name,
        "source_checkpoint": str(checkpoint_path),
        "task": task,
        "phase": phase,
        "runtime_contract": {
            "policy_kind": policy_kind,
            "deployable_observation_groups": deployable_observation_groups,
            "latent_update_semantics": (
                "per-step history update via phi(history) -> z_hat"
                if policy_kind == "blind_adaptive_student"
                else ""
            ),
        },
        "tensor_contract": {
            "policy_obs_dim": policy_obs_dim,
            "policy_history_dim": history_dim,
            "latent_dim": latent_dim,
            "action_dim": action_dim,
            "forward_signature": {
                "inputs": (
                    {
                        "policy_obs": ["batch", policy_obs_dim],
                        "policy_history": ["batch", history_dim],
                    }
                    if history_dim > 0
                    else {
                        "policy_obs": ["batch", policy_obs_dim],
                    }
                ),
                "outputs": {
                    "action": ["batch", action_dim],
                },
            },
        },
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return output_name


def _resolve_go2_default_joint_pos() -> list[float]:
    return GO2_DEFAULT_JOINT_POS.copy()


def _policy_order_for_dim(policy_obs_dim: int) -> list[dict[str, object]]:
    common_terms = [
        {"name": "base_ang_vel", "dim": 3, "history_length": 1, "scale": [1.0, 1.0, 1.0]},
        {"name": "projected_gravity", "dim": 3, "history_length": 1, "scale": [1.0, 1.0, 1.0]},
        {"name": "velocity_commands", "dim": 3, "history_length": 1, "scale": [1.0, 1.0, 1.0]},
        {"name": "joint_pos_rel", "dim": 12, "history_length": 1, "scale": [1.0] * 12},
        {"name": "joint_vel_rel", "dim": 12, "history_length": 1, "scale": [1.0] * 12},
        {"name": "last_action", "dim": 12, "history_length": 1, "scale": [1.0] * 12},
    ]
    if policy_obs_dim == 45:
        return common_terms
    if policy_obs_dim == 48:
        return [{"name": "base_lin_vel", "dim": 3, "history_length": 1, "scale": [1.0, 1.0, 1.0]}] + common_terms
    raise RuntimeError(f"Unsupported Go2 deploy policy observation dimension: {policy_obs_dim}")


def _write_deploy_config(
    bundle_dir: Path,
    policy_name: str,
    policy_history_length: int,
    policy_obs_dim: int,
    command_ranges: dict[str, list[float]],
) -> str:
    output_name = f"{policy_name}.deploy_config.json"
    output_path = bundle_dir / output_name
    policy_order = _policy_order_for_dim(policy_obs_dim)
    payload = {
        "robot": {
            "joint_names": GO2_JOINT_NAMES,
            "actuator_names": GO2_ACTUATOR_NAMES,
            "base_init_pos": GO2_BASE_INIT_POS,
            "base_init_quat_wxyz": GO2_BASE_INIT_QUAT_WXYZ,
            "default_joint_pos": _resolve_go2_default_joint_pos(),
            "joint_stiffness": GO2_JOINT_STIFFNESS,
            "joint_damping": GO2_JOINT_DAMPING,
            "effort_limit": GO2_EFFORT_LIMIT,
            "velocity_limit": GO2_VELOCITY_LIMIT,
        },
        "actions": {
            "type": "JointPositionAction",
            "joint_names": GO2_JOINT_NAMES,
            "joint_ids": list(range(len(GO2_JOINT_NAMES))),
            "scale": [0.25] * len(GO2_JOINT_NAMES),
            "offset": _resolve_go2_default_joint_pos(),
            "clip": [[-100.0, 100.0]] * len(GO2_JOINT_NAMES),
            "use_default_offset": True,
        },
        "observations": {
            "history_layout": "isaaclab_term_major",
            "policy_order": policy_order,
            "policy_dim": policy_obs_dim,
            "policy_history_length": policy_history_length,
            "policy_history_dim": policy_obs_dim * policy_history_length,
            "use_gym_history": False,
        },
        "commands": {
            "base_velocity": {
                "default": [0.5, 0.0, 0.0],
                "ranges": command_ranges,
            }
        },
        "control": {
            "step_dt": 0.02,
            "physics_dt": 0.005,
            "decimation": 4,
        },
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return output_name


def _yaml_scalar(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, str):
        if value == "" or any(ch in value for ch in [":", "#", "[", "]", "{", "}", ",", "'"]):
            return json.dumps(value)
        return value
    return str(value)


def _yaml_dump(value, indent: int = 0) -> list[str]:
    prefix = " " * indent
    if isinstance(value, dict):
        lines: list[str] = []
        for key, item in value.items():
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}{key}:")
                lines.extend(_yaml_dump(item, indent + 2))
            else:
                lines.append(f"{prefix}{key}: {_yaml_scalar(item)}")
        return lines
    if isinstance(value, list):
        lines = []
        for item in value:
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}-")
                lines.extend(_yaml_dump(item, indent + 2))
            else:
                lines.append(f"{prefix}- {_yaml_scalar(item)}")
        return lines
    return [f"{prefix}{_yaml_scalar(value)}"]


def _write_unitree_rl_lab_deploy_yaml(
    bundle_dir: Path,
    policy_name: str,
    policy_history_length: int,
    policy_obs_dim: int,
    command_ranges: dict[str, list[float]],
) -> str:
    output_name = f"{policy_name}.deploy.yaml"
    output_path = bundle_dir / output_name
    policy_order = _policy_order_for_dim(policy_obs_dim)
    payload = {
        "joint_ids_map": list(range(len(GO2_JOINT_NAMES))),
        "step_dt": 0.02,
        "stiffness": GO2_JOINT_STIFFNESS,
        "damping": GO2_JOINT_DAMPING,
        "default_joint_pos": _resolve_go2_default_joint_pos(),
        "commands": {
            "base_velocity": {
                "default": [0.5, 0.0, 0.0],
                "ranges": command_ranges,
            }
        },
        "actions": {
            "JointPositionAction": {
                "joint_names": GO2_JOINT_NAMES,
                "joint_ids": list(range(len(GO2_JOINT_NAMES))),
                "scale": [0.25] * len(GO2_JOINT_NAMES),
                "offset": _resolve_go2_default_joint_pos(),
                "clip": [[-100.0, 100.0]] * len(GO2_JOINT_NAMES),
            }
        },
        "observations": {
            "history_layout": "isaaclab_term_major",
            "policy_order": policy_order,
            "policy_dim": policy_obs_dim,
            "policy_history_length": policy_history_length,
            "policy_history_dim": policy_obs_dim * policy_history_length,
            "policy_kind": "blind_history_policy" if policy_history_length > 0 else "blind_fixed_policy",
        },
    }
    output_path.write_text("\n".join(_yaml_dump(payload)) + "\n")
    return output_name


def _update_bundle_manifest(bundle_dir: Path, artifact_names: list[str]) -> None:
    manifest_path = bundle_dir / "bundle_manifest.json"
    if not manifest_path.exists():
        return

    manifest = json.loads(manifest_path.read_text())
    manifest["exported_artifacts"] = artifact_names
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def _write_export_request(bundle_dir: Path, payload: dict[str, object]) -> None:
    request_path = bundle_dir / "export_request.json"
    request_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> int:
    args = parse_args()
    bundle_dir = Path(args.bundle_dir)
    checkpoint_path = Path(args.checkpoint)
    formats = args.formats or ["torchscript"]
    command_ranges = {
        "lin_vel_x": [float(args.command_lin_vel_x[0]), float(args.command_lin_vel_x[1])],
        "lin_vel_y": [float(args.command_lin_vel_y[0]), float(args.command_lin_vel_y[1])],
        "ang_vel_z": [float(args.command_ang_vel_z[0]), float(args.command_ang_vel_z[1])],
    }
    request_payload = {
        "policy_name": args.policy_name,
        "checkpoint": str(checkpoint_path),
        "task": args.task,
        "phase": args.phase,
        "requested_formats": formats,
        "command_ranges": command_ranges,
    }

    if args.dry_run:
        request_payload["status"] = "planned"
        request_payload["note"] = "Dry run only. No export artifacts written."
        print(json.dumps(request_payload, indent=2, sort_keys=True))
        return 0

    torch, nn = _import_torch()
    resolved_history_length = (
        args.policy_history_length
        if args.policy_history_length > 0
        else _infer_history_length_from_task(args.task, args.policy_kind, bundle_dir)
    )

    class DeployableDynOnlyPolicy(nn.Module):
        def __init__(self, state_dict: OrderedDict[str, object]) -> None:
            super().__init__()
            adaptation_state = _collect_prefix(state_dict, "adaptation_module")
            actor_state = _collect_prefix(state_dict, "actor")

            history_dim, adaptation_hidden_dims, adaptation_output_dim = _infer_mlp_dims(adaptation_state)
            actor_input_dim, actor_hidden_dims, action_dim = _infer_mlp_dims(actor_state)
            latent_dim = adaptation_output_dim
            policy_obs_dim = actor_input_dim - latent_dim
            if policy_obs_dim <= 0:
                raise RuntimeError(
                    "Invalid exported actor input contract. "
                    f"Expected positive policy_obs_dim, got {policy_obs_dim}."
                )

            self.policy_obs_dim = policy_obs_dim
            self.history_dim = history_dim
            self.latent_dim = latent_dim
            self.action_dim = action_dim

            self.adaptation_module = _build_mlp(nn, history_dim, adaptation_hidden_dims, adaptation_output_dim)
            self.actor = _build_mlp(nn, actor_input_dim, actor_hidden_dims, action_dim)
            self.adaptation_module.load_state_dict(adaptation_state, strict=True)
            self.actor.load_state_dict(actor_state, strict=True)

        def forward(self, policy_obs, policy_history):
            latent = self.adaptation_module(policy_history)
            actor_obs = torch.cat([policy_obs, latent], dim=-1)
            return self.actor(actor_obs)

    class DeployableBlindFixedPolicy(nn.Module):
        def __init__(self, state_dict: OrderedDict[str, object]) -> None:
            super().__init__()
            actor_state = _collect_prefix(state_dict, "actor")
            actor_input_dim, actor_hidden_dims, action_dim = _infer_mlp_dims(actor_state)

            self.policy_obs_dim = actor_input_dim
            self.history_dim = 0
            self.latent_dim = 0
            self.action_dim = action_dim
            self.actor = _build_mlp(nn, actor_input_dim, actor_hidden_dims, action_dim)
            self.actor.load_state_dict(actor_state, strict=True)

        def forward(self, policy_obs):
            return self.actor(policy_obs)

    class DeployableBlindHistoryPolicy(nn.Module):
        def __init__(self, state_dict: OrderedDict[str, object], policy_history_length: int) -> None:
            super().__init__()
            temporal_state = _collect_prefix(state_dict, "temporal_encoder")
            projection_state = _collect_prefix(state_dict, "history_projection")
            actor_state = _collect_prefix(state_dict, "actor")

            conv_specs = _infer_conv1d_layers(temporal_state)
            conv_layers: list[nn.Module] = []
            for module_idx, in_channels, out_channels, kernel_size, dilation, padding in conv_specs:
                conv = nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    padding=padding,
                )
                conv.weight.data.copy_(temporal_state[f"{module_idx}.weight"])
                conv.bias.data.copy_(temporal_state[f"{module_idx}.bias"])
                conv_layers.append(conv)
                conv_layers.append(nn.ELU())
            self.temporal_encoder = nn.Sequential(*conv_layers)

            projection_input_dim, projection_hidden_dims, history_feature_dim = _infer_mlp_dims(projection_state)
            actor_input_dim, actor_hidden_dims, action_dim = _infer_mlp_dims(actor_state)
            policy_obs_dim = actor_input_dim - history_feature_dim
            if policy_obs_dim <= 0:
                raise RuntimeError(
                    "Invalid exported history-policy actor input contract. "
                    f"Expected positive policy_obs_dim, got {policy_obs_dim}."
                )

            self.policy_obs_dim = policy_obs_dim
            self.history_dim = policy_obs_dim * int(policy_history_length)
            self.latent_dim = 0
            self.action_dim = action_dim
            self.policy_history_length = int(policy_history_length)
            if projection_hidden_dims:
                raise RuntimeError(
                    "Unexpected blind-history projection architecture during export. "
                    f"Expected a single linear layer + ELU, got hidden dims {projection_hidden_dims}."
                )
            self.history_projection = _build_history_projection(
                nn,
                projection_input_dim,
                history_feature_dim,
            )
            self.actor = _build_mlp(nn, actor_input_dim, actor_hidden_dims, action_dim)
            self.history_projection.load_state_dict(projection_state, strict=True)
            self.actor.load_state_dict(actor_state, strict=True)

        def forward(self, policy_obs, policy_history):
            history = policy_history.view(-1, self.policy_history_length, self.policy_obs_dim).transpose(1, 2)
            temporal = self.temporal_encoder(history)
            pooled = temporal.mean(dim=-1)
            latest = temporal[:, :, -1]
            history_feature = self.history_projection(torch.cat([latest, pooled], dim=-1))
            actor_obs = torch.cat([policy_obs, history_feature], dim=-1)
            return self.actor(actor_obs)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "model_state_dict" not in checkpoint:
        raise SystemExit(f"Checkpoint missing 'model_state_dict': {checkpoint_path}")

    bundle_dir.mkdir(parents=True, exist_ok=True)
    deployable_observation_groups = _parse_csv(args.observation_groups)
    if args.policy_kind == "blind_adaptive_student":
        expected_groups = ["policy", "policy_history"]
        if deployable_observation_groups != expected_groups:
            raise SystemExit(
                f"Adaptive export requires observation groups {expected_groups}, "
                f"got {deployable_observation_groups}."
            )
        module = DeployableDynOnlyPolicy(checkpoint["model_state_dict"]).eval()
    elif args.policy_kind == "blind_history_policy":
        expected_groups = ["policy", "policy_history"]
        if deployable_observation_groups != expected_groups:
            raise SystemExit(
                f"Blind history export requires observation groups {expected_groups}, "
                f"got {deployable_observation_groups}."
            )
        module = DeployableBlindHistoryPolicy(
            checkpoint["model_state_dict"],
            policy_history_length=resolved_history_length,
        ).eval()
    else:
        expected_groups = ["policy"]
        if deployable_observation_groups != expected_groups:
            raise SystemExit(
                f"Blind fixed export requires observation groups {expected_groups}, "
                f"got {deployable_observation_groups}."
            )
        module = DeployableBlindFixedPolicy(checkpoint["model_state_dict"]).eval()

    supported_formats = {"torchscript", "onnx"}
    unsupported = [fmt for fmt in formats if fmt not in supported_formats]
    if unsupported:
        raise SystemExit(
            f"Unsupported export format(s): {', '.join(unsupported)}. "
            f"Supported formats: {', '.join(sorted(supported_formats))}"
        )

    artifact_names: list[str] = []
    if "torchscript" in formats:
        if module.history_dim > 0:
            artifact_names.append(
                _export_torchscript(
                    torch,
                    module,
                    bundle_dir,
                    args.policy_name,
                    module.policy_obs_dim,
                    module.history_dim,
                )
            )
        else:
            output_name = f"{args.policy_name}.torchscript.pt"
            output_path = bundle_dir / output_name
            dummy_policy = torch.zeros(1, module.policy_obs_dim, dtype=torch.float32)
            traced = torch.jit.trace(module, (dummy_policy,))
            traced.save(str(output_path))
            artifact_names.append(output_name)
    if "onnx" in formats:
        if module.history_dim > 0:
            artifact_names.append(
                _export_onnx(
                    torch,
                    module,
                    bundle_dir,
                    args.policy_name,
                    module.policy_obs_dim,
                    module.history_dim,
                )
            )
        else:
            output_name = f"{args.policy_name}.onnx"
            output_path = bundle_dir / output_name
            dummy_policy = torch.zeros(1, module.policy_obs_dim, dtype=torch.float32)
            torch.onnx.export(
                module,
                (dummy_policy,),
                str(output_path),
                input_names=["policy_obs"],
                output_names=["action"],
                dynamic_axes={
                    "policy_obs": {0: "batch"},
                    "action": {0: "batch"},
                },
                opset_version=17,
            )
            artifact_names.append(output_name)

    artifact_names.append(
        _write_sidecar_metadata(
            bundle_dir,
            args.policy_name,
            checkpoint_path,
            args.task,
            args.phase,
            args.policy_kind,
            deployable_observation_groups,
            module.policy_obs_dim,
            module.history_dim,
            module.latent_dim,
            module.action_dim,
        )
    )
    artifact_names.append(
        _write_deploy_config(
            bundle_dir,
            args.policy_name,
            module.history_dim // module.policy_obs_dim if module.history_dim > 0 else 0,
            module.policy_obs_dim,
            command_ranges,
        )
    )
    artifact_names.append(
        _write_unitree_rl_lab_deploy_yaml(
            bundle_dir,
            args.policy_name,
            module.history_dim // module.policy_obs_dim if module.history_dim > 0 else 0,
            module.policy_obs_dim,
            command_ranges,
        )
    )

    _update_bundle_manifest(bundle_dir, artifact_names)

    request_payload["status"] = "completed"
    request_payload["generated_artifacts"] = artifact_names
    request_payload["resolved_policy_history_length"] = (
        module.history_dim // module.policy_obs_dim if module.history_dim > 0 else 0
    )
    request_payload["note"] = f"Export completed for the {args.policy_kind} runtime contract."
    _write_export_request(bundle_dir, request_payload)

    print(f"Exported {args.policy_name} into {bundle_dir}")
    for artifact_name in artifact_names:
        print(f"- {bundle_dir / artifact_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
