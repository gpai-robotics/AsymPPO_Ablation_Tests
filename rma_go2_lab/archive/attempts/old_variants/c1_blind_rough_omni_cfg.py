"""C1 omnidirectional blind-history env derived from the current StageA line."""

from __future__ import annotations

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass

from rma_go2_lab.envs.blind.c1_blind_rough_teacher_history_cfg import Go2C1BlindRoughTeacherHistoryEnvCfg
from rma_go2_lab.envs.blind import blind_omni_command_curriculums as blind_curriculums


@configclass
class Go2C1BlindRoughOmniEnvCfg(Go2C1BlindRoughTeacherHistoryEnvCfg):
    """First omnidirectional C1 variant with conservative planar/yaw unlock."""

    def __post_init__(self):
        super().__post_init__()

        cmd = self.commands.base_velocity
        cmd.heading_command = False
        cmd.rel_heading_envs = 0.0
        cmd.ranges.lin_vel_x = (-0.1, 0.1)
        cmd.ranges.lin_vel_y = (-0.1, 0.1)
        cmd.ranges.ang_vel_z = (-0.1, 0.1)
        cmd.ranges.heading = None
        cmd.limit_ranges = cmd.ranges.__class__(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-0.4, 0.4),
            ang_vel_z=(-1.0, 1.0),
            heading=None,
        )

        self.curriculum.lin_vel_command_levels = CurrTerm(
            func=blind_curriculums.lin_vel_cmd_levels,
            params={"reward_term_name": "track_lin_vel_xy_exp", "delta": 0.1},
        )
        self.curriculum.ang_vel_command_levels = CurrTerm(
            func=blind_curriculums.ang_vel_cmd_levels,
            params={"reward_term_name": "track_ang_vel_z_exp", "delta": 0.1},
        )

        print("\n========== C1 OMNI V1 STAGEA ==========\n")
