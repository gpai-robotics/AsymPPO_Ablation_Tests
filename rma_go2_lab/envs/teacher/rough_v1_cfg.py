"""Privileged teacher rough-terrain environment, V1.

Teacher V1 keeps the V0 privilege path and warm-start recipe, but adds one
targeted reward intervention to counter the low-base crouch strategy observed
in V0:

- a motion-gated terrain-aware base-height penalty
"""

from __future__ import annotations

import torch

from isaaclab.envs import mdp as base_mdp
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from rma_go2_lab.envs.teacher.rough_cfg import Go2PrivilegedTeacherRoughEnvCfg


def moving_base_height_l2(
    env,
    target_height: float,
    command_name: str = "base_velocity",
    command_threshold: float = 0.15,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
):
    """Penalize base height error only when the commanded motion is non-trivial."""
    command = env.command_manager.get_command(command_name)
    moving_mask = (torch.linalg.norm(command[:, :2], dim=1) > command_threshold).float()
    height_penalty = base_mdp.base_height_l2(
        env,
        target_height=target_height,
        asset_cfg=asset_cfg,
        sensor_cfg=sensor_cfg,
    )
    return moving_mask * height_penalty


@configclass
class Go2PrivilegedTeacherRoughV1EnvCfg(Go2PrivilegedTeacherRoughEnvCfg):
    """Teacher V1: same privilege stack as V0, plus anti-crouch shaping."""

    def __post_init__(self):
        super().__post_init__()

        print("\n========== PRIVILEGED TEACHER ROUGH V1 ==========\n")

        # Teacher V1 keeps V0 fixed except for a single targeted anti-crouch
        # term. The target height is terrain-aware via the height scanner so the
        # reward remains meaningful on rough terrain.
        self.rewards.base_height = RewTerm(
            func=moving_base_height_l2,
            weight=-5.0,
            params={
                "target_height": 0.33,
                "command_name": "base_velocity",
                "command_threshold": 0.15,
                "asset_cfg": SceneEntityCfg("robot"),
                "sensor_cfg": SceneEntityCfg("height_scanner"),
            },
        )
