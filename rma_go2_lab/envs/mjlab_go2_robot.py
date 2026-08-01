"""Helpers for mjlab-style Go2 training assets."""

from __future__ import annotations

from copy import deepcopy

from isaaclab.actuators import DCMotorCfg
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG


def make_mjlab_go2_robot_cfg():
    """Return a Go2 robot cfg with joint-type-specific actuator groups."""

    robot_cfg = deepcopy(UNITREE_GO2_CFG)
    robot_cfg.actuators = {
        "hips": DCMotorCfg(
            joint_names_expr=[".*_hip_joint"],
            effort_limit=23.5,
            saturation_effort=23.5,
            velocity_limit=30.0,
            stiffness=20.0,
            damping=1.0,
            friction=0.0,
        ),
        "thighs": DCMotorCfg(
            joint_names_expr=[".*_thigh_joint"],
            effort_limit=23.5,
            saturation_effort=23.5,
            velocity_limit=30.0,
            stiffness=20.0,
            damping=1.0,
            friction=0.0,
        ),
        "calves": DCMotorCfg(
            joint_names_expr=[".*_calf_joint"],
            effort_limit=23.5,
            saturation_effort=23.5,
            velocity_limit=30.0,
            stiffness=40.0,
            damping=2.0,
            friction=0.0,
        ),
    }
    return robot_cfg
