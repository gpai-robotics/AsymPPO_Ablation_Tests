#!/usr/bin/env python3
"""Generate golden policy vectors and validate checkpoint/TorchScript parity."""

from __future__ import annotations

import argparse
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import torch
import torch.nn as nn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", required=True)
    parser.add_argument(
        "--output-dir",
        default="artifacts/deployment_validation/golden_inference",
    )
    parser.add_argument("--num-cases", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260611)
    parser.add_argument("--tolerance", type=float, default=1.0e-5)
    parser.add_argument(
        "--skip-cpp",
        action="store_true",
        help="Skip the Unitree C++ ONNX Runtime parity check.",
    )
    return parser.parse_args()


def collect_prefix(state_dict: OrderedDict[str, object], prefix: str) -> OrderedDict[str, object]:
    prefix_with_dot = f"{prefix}."
    result = OrderedDict(
        (key[len(prefix_with_dot) :], value)
        for key, value in state_dict.items()
        if key.startswith(prefix_with_dot)
    )
    if not result:
        raise RuntimeError(f"Checkpoint has no state under prefix '{prefix_with_dot}'.")
    return result


def infer_mlp_dims(module_state: OrderedDict[str, object]) -> tuple[int, list[int], int]:
    indices = sorted({int(key.split(".")[0]) for key in module_state if key.endswith(".weight")})
    shapes = [tuple(module_state[f"{idx}.weight"].shape) for idx in indices]
    return shapes[0][1], [shape[0] for shape in shapes[:-1]], shapes[-1][0]


def build_mlp(input_dim: int, hidden_dims: list[int], output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    previous_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.extend((nn.Linear(previous_dim, hidden_dim), nn.ELU()))
        previous_dim = hidden_dim
    layers.append(nn.Linear(previous_dim, output_dim))
    return nn.Sequential(*layers)


class CheckpointBlindHistoryPolicy(nn.Module):
    """Actor path reconstructed directly from the training checkpoint."""

    def __init__(self, state_dict: OrderedDict[str, object], history_length: int) -> None:
        super().__init__()
        temporal_state = collect_prefix(state_dict, "temporal_encoder")
        projection_state = collect_prefix(state_dict, "history_projection")
        actor_state = collect_prefix(state_dict, "actor")

        conv_indices = sorted(
            {int(key.split(".")[0]) for key in temporal_state if key.endswith(".weight")}
        )
        temporal_layers: list[nn.Module] = []
        for order_idx, module_idx in enumerate(conv_indices):
            weight = temporal_state[f"{module_idx}.weight"]
            out_channels, in_channels, kernel_size = weight.shape
            dilation = 2**order_idx
            padding = dilation * (int(kernel_size) - 1) // 2
            conv = nn.Conv1d(
                int(in_channels),
                int(out_channels),
                int(kernel_size),
                dilation=dilation,
                padding=padding,
            )
            conv.weight.data.copy_(weight)
            conv.bias.data.copy_(temporal_state[f"{module_idx}.bias"])
            temporal_layers.extend((conv, nn.ELU()))
        self.temporal_encoder = nn.Sequential(*temporal_layers)

        projection_input, projection_hidden, history_feature_dim = infer_mlp_dims(projection_state)
        if projection_hidden:
            raise RuntimeError(f"Unexpected history projection hidden layers: {projection_hidden}")
        self.history_projection = nn.Sequential(
            nn.Linear(projection_input, history_feature_dim),
            nn.ELU(),
        )
        self.history_projection.load_state_dict(projection_state, strict=True)

        actor_input, actor_hidden, action_dim = infer_mlp_dims(actor_state)
        self.policy_obs_dim = actor_input - history_feature_dim
        self.history_length = int(history_length)
        self.history_dim = self.policy_obs_dim * self.history_length
        self.action_dim = action_dim
        self.actor = build_mlp(actor_input, actor_hidden, action_dim)
        self.actor.load_state_dict(actor_state, strict=True)

    def forward(self, policy_obs: torch.Tensor, policy_history: torch.Tensor) -> torch.Tensor:
        history = policy_history.view(-1, self.history_length, self.policy_obs_dim)
        temporal = self.temporal_encoder(history.transpose(1, 2))
        history_feature = self.history_projection(
            torch.cat((temporal[:, :, -1], temporal.mean(dim=-1)), dim=-1)
        )
        return self.actor(torch.cat((policy_obs, history_feature), dim=-1))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_rows(path: Path, values: np.ndarray) -> None:
    np.savetxt(path, values, fmt="%.9g")


def generate_inputs(
    rng: np.random.Generator,
    num_cases: int,
    policy_dim: int,
    history_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    policy_obs = rng.normal(0.0, 0.35, size=(num_cases, policy_dim)).astype(np.float32)
    policy_history = rng.normal(0.0, 0.30, size=(num_cases, history_dim)).astype(np.float32)

    # Include exact neutral and deterministic boundary-like cases.
    policy_obs[0] = 0.0
    policy_history[0] = 0.0
    if num_cases > 1:
        policy_obs[1] = np.linspace(-1.0, 1.0, policy_dim, dtype=np.float32)
        policy_history[1] = np.tile(policy_obs[1], history_dim // policy_dim)
    if num_cases > 2:
        policy_obs[2, 3:6] = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        policy_obs[2, 6:9] = np.array([0.5, -0.3, 0.6], dtype=np.float32)
    return policy_obs, policy_history


def run_cpp_parity(
    repo_root: Path,
    output_dir: Path,
    onnx_path: Path,
    obs_path: Path,
    history_path: Path,
    expected_path: Path,
    tolerance: float,
) -> dict[str, object]:
    compiler = shutil.which("g++")
    if compiler is None:
        return {"status": "blocked", "reason": "g++ was not found in PATH."}

    deploy_root = repo_root / "reference_repos/unitree_rl_mjlab/deploy"
    ort_root = deploy_root / "thirdparty/onnxruntime-linux-x64-1.22.0"
    source_path = repo_root / "scripts/deploy/validate_unitree_mjlab_onnx.cpp"
    executable_path = output_dir / "validate_unitree_mjlab_onnx"
    ort_library = ort_root / "lib/libonnxruntime.so.1.22.0"
    required_paths = [
        deploy_root / "include",
        ort_root / "include",
        source_path,
        ort_library,
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        return {"status": "blocked", "reason": "Missing C++ runtime paths.", "missing": missing}

    compile_cmd = [
        compiler,
        "-std=c++17",
        "-O2",
        str(source_path),
        f"-I{deploy_root / 'include'}",
        f"-I{ort_root / 'include'}",
        f"-L{ort_root / 'lib'}",
        f"-Wl,-rpath,{ort_root / 'lib'}",
        "-l:libonnxruntime.so.1.22.0",
        "-o",
        str(executable_path),
    ]
    compile_result = subprocess.run(compile_cmd, capture_output=True, text=True)
    if compile_result.returncode != 0:
        return {
            "status": "fail",
            "stage": "compile",
            "returncode": compile_result.returncode,
            "stdout": compile_result.stdout,
            "stderr": compile_result.stderr,
        }

    run_cmd = [
        str(executable_path),
        str(onnx_path),
        str(obs_path),
        str(history_path),
        str(expected_path),
        str(tolerance),
    ]
    run_result = subprocess.run(run_cmd, capture_output=True, text=True)
    max_abs_error = None
    for line in run_result.stdout.splitlines():
        if line.startswith("Maximum absolute error:"):
            max_abs_error = float(line.split(":", 1)[1].strip())
            break
    return {
        "status": "pass" if run_result.returncode == 0 else "fail",
        "returncode": run_result.returncode,
        "max_abs_error": max_abs_error,
        "tolerance": tolerance,
        "executable": str(executable_path),
        "stdout": run_result.stdout,
        "stderr": run_result.stderr,
    }


def main() -> int:
    args = parse_args()
    bundle_dir = Path(args.bundle_dir).resolve()
    manifest = json.loads((bundle_dir / "bundle_manifest.json").read_text())
    metadata_path = next(bundle_dir.glob("*.export_metadata.json"))
    metadata = json.loads(metadata_path.read_text())
    contract = metadata["tensor_contract"]

    checkpoint_path = Path(manifest["source_checkpoint"])
    torchscript_path = next(bundle_dir.glob("*.torchscript.pt"))
    onnx_path = next(bundle_dir.glob("*.onnx"))
    deploy_config_path = next(bundle_dir.glob("*.deploy_config.json"))
    deploy_config = json.loads(deploy_config_path.read_text())

    policy_dim = int(contract["policy_obs_dim"])
    history_dim = int(contract["policy_history_dim"])
    action_dim = int(contract["action_dim"])
    history_length = int(deploy_config["observations"]["policy_history_length"])
    if history_dim != policy_dim * history_length:
        raise SystemExit("History dimensions are inconsistent in the exported contract.")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_policy = CheckpointBlindHistoryPolicy(
        checkpoint["model_state_dict"],
        history_length=history_length,
    ).eval()
    torchscript_policy = torch.jit.load(str(torchscript_path), map_location="cpu").eval()

    rng = np.random.default_rng(args.seed)
    policy_obs, policy_history = generate_inputs(
        rng,
        args.num_cases,
        policy_dim,
        history_dim,
    )
    obs_tensor = torch.from_numpy(policy_obs)
    history_tensor = torch.from_numpy(policy_history)
    with torch.inference_mode():
        checkpoint_action = checkpoint_policy(obs_tensor, history_tensor).cpu().numpy()
        torchscript_action = torchscript_policy(obs_tensor, history_tensor).cpu().numpy()

    if checkpoint_action.shape != (args.num_cases, action_dim):
        raise SystemExit(f"Unexpected checkpoint action shape: {checkpoint_action.shape}")
    max_abs_error = float(np.max(np.abs(checkpoint_action - torchscript_action)))
    status = "pass" if max_abs_error <= args.tolerance else "fail"

    output_dir = Path(args.output_dir).resolve() / manifest["policy_name"]
    output_dir.mkdir(parents=True, exist_ok=True)
    obs_path = output_dir / "policy_obs.txt"
    history_path = output_dir / "policy_history.txt"
    expected_path = output_dir / "expected_action.txt"
    write_rows(obs_path, policy_obs)
    write_rows(history_path, policy_history)
    write_rows(expected_path, checkpoint_action)

    repo_root = Path(__file__).resolve().parents[2]
    cpp_validation = (
        {"status": "skipped"}
        if args.skip_cpp
        else run_cpp_parity(
            repo_root,
            output_dir,
            onnx_path,
            obs_path,
            history_path,
            expected_path,
            args.tolerance,
        )
    )
    overall_pass = status == "pass" and cpp_validation["status"] in ("pass", "skipped")
    report = {
        "status": "pass" if overall_pass else "fail",
        "policy_name": manifest["policy_name"],
        "seed": args.seed,
        "num_cases": args.num_cases,
        "tolerance": args.tolerance,
        "dimensions": {
            "policy_obs": policy_dim,
            "policy_history": history_dim,
            "action": action_dim,
        },
        "checkpoint_vs_torchscript": {
            "max_abs_error": max_abs_error,
            "pass": status == "pass",
        },
        "artifacts": {
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256(checkpoint_path),
            "torchscript": str(torchscript_path),
            "torchscript_sha256": sha256(torchscript_path),
            "onnx": str(onnx_path),
            "onnx_sha256": sha256(onnx_path),
            "policy_obs": str(obs_path),
            "policy_history": str(history_path),
            "expected_action": str(expected_path),
        },
        "cpp_validation": cpp_validation,
    }
    report_path = output_dir / "golden_inference_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
