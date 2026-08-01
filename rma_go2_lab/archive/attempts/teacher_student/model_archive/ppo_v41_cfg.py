"""Archived PPO config for the stair-focused V4.1 privileged teacher."""

from isaaclab.utils import configclass

from rma_go2_lab.models.teacher.ppo_v4_cfg import Go2PrivilegedTeacherRoughV4PPORunnerCfg


@configclass
class Go2PrivilegedTeacherRoughV41PPORunnerCfg(Go2PrivilegedTeacherRoughV4PPORunnerCfg):
    """Teacher V4.1: V4 supervision with a narrow stair-only terrain intervention."""

    experiment_name = "go2_privileged_teacher_rough_v41_stair_bias"
