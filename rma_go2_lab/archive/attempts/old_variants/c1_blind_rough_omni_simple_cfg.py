"""Simple rough-terrain omni branch built from the current C1 line."""

from __future__ import annotations

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from rma_go2_lab.envs.blind.blind_rough_forward_cfg import air_time_variance_penalty
from rma_go2_lab.envs.blind.c1_blind_rough_omni_cfg import Go2C1BlindRoughOmniEnvCfg


@configclass
class Go2C1BlindRoughOmniSimpleEnvCfg(Go2C1BlindRoughOmniEnvCfg):
    """Minimal omni rough branch intended to become the first clean canonical candidate."""

    def __post_init__(self):
        super().__post_init__()

        cmd = self.commands.base_velocity
        cmd.rel_standing_envs = 0.05

        # Give broad omni commands a little more time before declaring the
        # rollout stuck, especially on turning / mixed maneuvers.
        self.terminations.low_progress.params["min_displacement"] = 0.2
        self.terminations.low_progress.params["grace_period_s"] = 4.0

        # Soften "stand still" shaping a touch for the broader command family.
        self.rewards.stand_still_joint_deviation.params["command_threshold"] = 0.2
        self.rewards.stand_still_foot_motion.params["command_threshold"] = 0.2

        # Mild timing-structure reward: encourage less scrapy asymmetric swing durations
        # without prescribing a hard gait pattern.
        self.rewards.air_time_variance = RewTerm(
            func=air_time_variance_penalty,
            weight=-0.05,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
                "command_name": "base_velocity",
                "command_threshold": 0.2,
                "min_recorded_air_time": 0.05,
                "clip_max_air_time": 0.5,
            },
        )

        print("\n========== C1 SIMPLE ROUGH OMNI V1 ==========\n")
