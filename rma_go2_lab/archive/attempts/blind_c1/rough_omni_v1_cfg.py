"""Privileged omni rough-terrain teacher environment, V1."""

from __future__ import annotations

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass

from rma_go2_lab.envs.blind import blind_omni_command_curriculums as blind_curriculums
from rma_go2_lab.envs.teacher.rough_v3_cfg import Go2PrivilegedTeacherRoughV3EnvCfg


@configclass
class Go2PrivilegedTeacherRoughOmniV1EnvCfg(Go2PrivilegedTeacherRoughV3EnvCfg):
    """Privileged rough omni teacher with explicit terrain+dynamics branches."""

    def __post_init__(self):
        super().__post_init__()

        cmd = self.commands.base_velocity
        cmd.heading_command = False
        cmd.rel_heading_envs = 0.0
        cmd.rel_standing_envs = 0.0
        cmd.resampling_time_range = (2.0, 4.0)
        cmd.ranges.lin_vel_x = (-0.2, 0.2)
        cmd.ranges.lin_vel_y = (-0.2, 0.2)
        cmd.ranges.ang_vel_z = (-0.2, 0.2)
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

        print("\n========== PRIVILEGED TEACHER ROUGH OMNI V1 ==========\n")
