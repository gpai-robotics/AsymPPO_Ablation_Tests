"""Scenario manifests for MuJoCo-side OOD and harsh Sim2Sim evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

from materialize_unitree_mujoco_terrain_recipes import ensure_materialized_unitree_terrain_recipes


REPO_ROOT = Path(__file__).resolve().parents[2]
MENAGERIE_SCENE = REPO_ROOT / "reference_repos" / "mujoco_menagerie" / "unitree_go2" / "scene.xml"
CONTINUOUS_CORRIDOR_SCENE = (
    REPO_ROOT
    / "reference_repos"
    / "unitree_mujoco"
    / "unitree_robots"
    / "go2"
    / "scene_eval_continuous_corridor.xml"
)
LOCOMOTION_SCENE = (
    REPO_ROOT
    / "reference_repos"
    / "unitree_mujoco"
    / "unitree_robots"
    / "go2"
    / "scene_eval_locomotion.xml"
)


@dataclass
class MujocoOODScenario:
    name: str
    model_path: str = ""
    command: list[float] = field(default_factory=lambda: [0.5, 0.0, 0.0])
    ground_friction: float | None = None
    foot_friction: float | None = None
    base_mass_scale: float | None = None
    motor_strength_scale: float | None = None
    joint_damping_scale: float | None = None
    passive_joint_damping_scale: float | None = 0.5
    passive_joint_frictionloss_scale: float | None = None
    command_schedule: list[dict[str, object]] = field(default_factory=list)
    wrench_schedule: list[dict[str, object]] = field(default_factory=list)

    def to_json_dict(self) -> dict[str, object]:
        payload = asdict(self)
        if not self.model_path:
            payload["model_path"] = str(MENAGERIE_SCENE)
        return {k: v for k, v in payload.items() if v is not None}


def scenario_set(name: str) -> list[MujocoOODScenario]:
    rough_recipes = ensure_materialized_unitree_terrain_recipes()
    rough_paths = list(rough_recipes.values())

    if name == "mujoco_nominal_v1":
        return [
            MujocoOODScenario("nominal_flat"),
            MujocoOODScenario("low_friction_flat", ground_friction=0.15, foot_friction=0.15),
            MujocoOODScenario("heavy_flat", base_mass_scale=1.35),
            MujocoOODScenario("weak_motor_flat", motor_strength_scale=0.65),
        ]

    if name == "mujoco_disturb_v1":
        return [
            MujocoOODScenario(
                "flat_command_step_up",
                command_schedule=[{"step": 250, "command": [0.8, 0.0, 0.0], "label": "step_up"}],
            ),
            MujocoOODScenario(
                "flat_command_step_down",
                command=[0.8, 0.0, 0.0],
                command_schedule=[{"step": 250, "command": [0.3, 0.0, 0.0], "label": "step_down"}],
            ),
            MujocoOODScenario(
                "flat_yaw_pulse",
                command_schedule=[
                    {"step": 250, "command": [0.5, 0.0, 0.45], "label": "yaw_pulse_on"},
                    {"step": 360, "command": [0.5, 0.0, 0.0], "label": "yaw_pulse_off"},
                ],
            ),
            MujocoOODScenario(
                "flat_lateral_push",
                wrench_schedule=[
                    {"start_step": 220, "duration_steps": 10, "force_world": [0.0, 60.0, 0.0], "label": "lat_push_small"},
                    {"start_step": 320, "duration_steps": 12, "force_world": [0.0, 90.0, 0.0], "label": "lat_push_medium"},
                    {"start_step": 440, "duration_steps": 15, "force_world": [0.0, 120.0, 0.0], "label": "lat_push_strong"},
                ],
            ),
            MujocoOODScenario(
                "flat_yaw_torque_pulse",
                wrench_schedule=[
                    {"start_step": 300, "duration_steps": 12, "torque_world": [0.0, 0.0, 45.0], "label": "yaw_torque"}
                ],
            ),
        ]

    if name == "mujoco_disturb_v2_moderate":
        return [
            MujocoOODScenario(
                "flat_command_step_up_moderate",
                command_schedule=[{"step": 250, "command": [0.7, 0.0, 0.0], "label": "step_up"}],
            ),
            MujocoOODScenario(
                "flat_command_step_down_moderate",
                command=[0.7, 0.0, 0.0],
                command_schedule=[{"step": 250, "command": [0.35, 0.0, 0.0], "label": "step_down"}],
            ),
            MujocoOODScenario(
                "flat_yaw_pulse_moderate",
                command_schedule=[
                    {"step": 250, "command": [0.5, 0.0, 0.25], "label": "yaw_pulse_small"},
                    {"step": 340, "command": [0.5, 0.0, 0.0], "label": "yaw_pulse_off"},
                ],
            ),
            MujocoOODScenario(
                "flat_lateral_push_moderate",
                passive_joint_damping_scale=1.0,
                wrench_schedule=[
                    {"start_step": 220, "duration_steps": 8, "force_world": [0.0, 20.0, 0.0], "label": "lat_push_small"},
                    {"start_step": 320, "duration_steps": 10, "force_world": [0.0, 35.0, 0.0], "label": "lat_push_medium"},
                    {"start_step": 440, "duration_steps": 12, "force_world": [0.0, 50.0, 0.0], "label": "lat_push_large"},
                ],
            ),
            MujocoOODScenario(
                "flat_yaw_torque_pulse_moderate",
                passive_joint_damping_scale=1.0,
                wrench_schedule=[
                    {"start_step": 220, "duration_steps": 8, "torque_world": [0.0, 0.0, 8.0], "label": "yaw_torque_small"},
                    {"start_step": 320, "duration_steps": 10, "torque_world": [0.0, 0.0, 12.0], "label": "yaw_torque_medium"},
                    {"start_step": 440, "duration_steps": 12, "torque_world": [0.0, 0.0, 16.0], "label": "yaw_torque_large"},
                ],
            ),
        ]

    if name == "mujoco_continuous_v1":
        return [
            MujocoOODScenario("continuous_corridor_nominal", model_path=str(CONTINUOUS_CORRIDOR_SCENE)),
            MujocoOODScenario(
                "continuous_corridor_low_friction",
                model_path=str(CONTINUOUS_CORRIDOR_SCENE),
                ground_friction=0.18,
                foot_friction=0.18,
            ),
            MujocoOODScenario(
                "continuous_corridor_yaw_pulse",
                model_path=str(CONTINUOUS_CORRIDOR_SCENE),
                command_schedule=[
                    {"step": 320, "command": [0.5, 0.0, 0.35], "label": "yaw_on"},
                    {"step": 430, "command": [0.5, 0.0, 0.0], "label": "yaw_off"},
                ],
            ),
            MujocoOODScenario(
                "continuous_corridor_lateral_push",
                model_path=str(CONTINUOUS_CORRIDOR_SCENE),
                wrench_schedule=[
                    {"start_step": 320, "duration_steps": 10, "force_world": [0.0, 60.0, 0.0], "label": "lat_push_small"},
                    {"start_step": 470, "duration_steps": 12, "force_world": [0.0, 90.0, 0.0], "label": "lat_push_medium"},
                    {"start_step": 620, "duration_steps": 15, "force_world": [0.0, 120.0, 0.0], "label": "lat_push_strong"},
                ],
            ),
            MujocoOODScenario(
                "continuous_corridor_weak_motor",
                model_path=str(CONTINUOUS_CORRIDOR_SCENE),
                motor_strength_scale=0.7,
            ),
        ]

    if name == "mujoco_rough_v1":
        return [
            MujocoOODScenario(
                "rough_forward_wide_a_nominal",
                model_path=str(rough_recipes["forward_rough_wide_a"]),
            ),
            MujocoOODScenario(
                "rough_forward_wide_b_nominal",
                model_path=str(rough_recipes["forward_rough_wide_b"]),
            ),
            MujocoOODScenario(
                "rough_forward_mixed_c_nominal",
                model_path=str(rough_recipes["forward_rough_mixed_c"]),
            ),
            MujocoOODScenario(
                "rough_forward_with_stairs_d_nominal",
                model_path=str(rough_recipes["forward_rough_with_stairs_d"]),
            ),
            MujocoOODScenario(
                "rough_forward_wide_b_low_friction",
                model_path=str(rough_recipes["forward_rough_wide_b"]),
                ground_friction=0.18,
                foot_friction=0.18,
            ),
            MujocoOODScenario(
                "rough_forward_mixed_c_yaw_pulse",
                model_path=str(rough_recipes["forward_rough_mixed_c"]),
                command_schedule=[
                    {"step": 320, "command": [0.5, 0.0, 0.30], "label": "yaw_on"},
                    {"step": 430, "command": [0.5, 0.0, 0.0], "label": "yaw_off"},
                ],
            ),
            MujocoOODScenario(
                "rough_forward_with_stairs_d_lateral_push",
                model_path=str(rough_recipes["forward_rough_with_stairs_d"]),
                passive_joint_damping_scale=1.0,
                wrench_schedule=[
                    {"start_step": 320, "duration_steps": 8, "force_world": [0.0, 30.0, 0.0], "label": "lat_push_small"},
                    {"start_step": 470, "duration_steps": 10, "force_world": [0.0, 45.0, 0.0], "label": "lat_push_medium"},
                    {"start_step": 620, "duration_steps": 12, "force_world": [0.0, 60.0, 0.0], "label": "lat_push_large"},
                ],
            ),
            MujocoOODScenario(
                "rough_forward_wide_a_weak_motor",
                model_path=str(rough_recipes["forward_rough_wide_a"]),
                motor_strength_scale=0.7,
                joint_damping_scale=0.8,
            ),
        ]

    if name == "mujoco_rough_v2_hard":
        return [
            MujocoOODScenario(
                "technical_e_nominal",
                model_path=str(rough_recipes["forward_technical_e"]),
            ),
            MujocoOODScenario(
                "technical_f_nominal",
                model_path=str(rough_recipes["forward_technical_f"]),
            ),
            MujocoOODScenario(
                "technical_g_nominal",
                model_path=str(rough_recipes["forward_technical_g"]),
            ),
            MujocoOODScenario(
                "technical_h_nominal",
                model_path=str(rough_recipes["forward_technical_h"]),
            ),
            MujocoOODScenario(
                "technical_f_low_friction",
                model_path=str(rough_recipes["forward_technical_f"]),
                ground_friction=0.22,
                foot_friction=0.22,
            ),
            MujocoOODScenario(
                "technical_g_yaw_pulse",
                model_path=str(rough_recipes["forward_technical_g"]),
                command_schedule=[
                    {"step": 300, "command": [0.5, 0.0, 0.25], "label": "yaw_on"},
                    {"step": 410, "command": [0.5, 0.0, 0.0], "label": "yaw_off"},
                ],
            ),
            MujocoOODScenario(
                "technical_h_lateral_push",
                model_path=str(rough_recipes["forward_technical_h"]),
                passive_joint_damping_scale=1.0,
                wrench_schedule=[
                    {"start_step": 280, "duration_steps": 8, "force_world": [0.0, 25.0, 0.0], "label": "lat_push_small"},
                    {"start_step": 430, "duration_steps": 10, "force_world": [0.0, 40.0, 0.0], "label": "lat_push_medium"},
                    {"start_step": 580, "duration_steps": 12, "force_world": [0.0, 55.0, 0.0], "label": "lat_push_large"},
                ],
            ),
            MujocoOODScenario(
                "technical_e_weak_motor",
                model_path=str(rough_recipes["forward_technical_e"]),
                motor_strength_scale=0.78,
                joint_damping_scale=0.85,
            ),
        ]

    if name == "mujoco_hidden_env_v1":
        return [
            MujocoOODScenario("hidden_ultra_low_friction_flat", ground_friction=0.08, foot_friction=0.08),
            MujocoOODScenario("hidden_ultra_high_friction_flat", ground_friction=2.5, foot_friction=2.5),
            MujocoOODScenario("hidden_very_heavy_payload_flat", base_mass_scale=1.5),
            MujocoOODScenario(
                "hidden_very_weak_motor_flat",
                motor_strength_scale=0.6,
                joint_damping_scale=0.7,
            ),
            MujocoOODScenario(
                "hidden_low_friction_heavy_payload_flat",
                ground_friction=0.12,
                foot_friction=0.12,
                base_mass_scale=1.35,
            ),
            MujocoOODScenario(
                "hidden_low_friction_weak_motor_flat",
                ground_friction=0.12,
                foot_friction=0.12,
                motor_strength_scale=0.65,
                joint_damping_scale=0.75,
            ),
            MujocoOODScenario(
                "hidden_heavy_payload_weak_motor_flat",
                base_mass_scale=1.35,
                motor_strength_scale=0.65,
                joint_damping_scale=0.75,
            ),
            MujocoOODScenario(
                "hidden_triple_combo_continuous_corridor",
                model_path=str(rough_paths[1]),
                ground_friction=0.18,
                foot_friction=0.18,
                base_mass_scale=1.3,
                motor_strength_scale=0.72,
                joint_damping_scale=0.8,
            ),
        ]

    if name == "mujoco_limit_v1":
        return [
            *scenario_set("mujoco_nominal_v1"),
            *scenario_set("mujoco_disturb_v1"),
            *scenario_set("mujoco_rough_v1"),
            *scenario_set("mujoco_continuous_v1"),
            *scenario_set("mujoco_hidden_env_v1"),
            MujocoOODScenario("locomotion_scene_nominal", model_path=str(LOCOMOTION_SCENE)),
        ]

    raise ValueError(
        f"Unknown MuJoCo scenario set '{name}'. Expected one of: "
        "mujoco_nominal_v1, mujoco_disturb_v1, mujoco_disturb_v2_moderate, mujoco_rough_v1, mujoco_continuous_v1, "
        "mujoco_hidden_env_v1, mujoco_limit_v1"
    )
