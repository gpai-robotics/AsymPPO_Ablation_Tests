"""Archived PPO config for the controlled geometry-bias privileged teacher, V5."""

from isaaclab.utils import configclass

from rma_go2_lab.models.teacher.ppo_v4_cfg import (
    Go2PrivilegedTeacherRoughV4PPORunnerCfg,
    Go2PrivilegedTeacherTerrainAuxAlgorithmCfg,
)


@configclass
class Go2PrivilegedTeacherTerrainPriorityAlgorithmCfg(Go2PrivilegedTeacherTerrainAuxAlgorithmCfg):
    """Keep the same terrain-aux schedule as V4 for a clean intervention."""


@configclass
class Go2PrivilegedTeacherRoughV5PPORunnerCfg(Go2PrivilegedTeacherRoughV4PPORunnerCfg):
    """Teacher V5: V4 supervision plus a modest geometry-biased terrain mix."""

    experiment_name = "go2_privileged_teacher_rough_v5_terrain_priority"

    algorithm = Go2PrivilegedTeacherTerrainPriorityAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.002,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1e-4,
        schedule="adaptive",
        desired_kl=0.01,
        gamma=0.99,
        lam=0.95,
        max_grad_norm=1.0,
        terrain_regression_coef_stage0=0.5,
        terrain_regression_coef_stage1=0.2,
        terrain_regression_coef_stage2=0.05,
        terrain_stage0_end=300,
        terrain_stage1_end=800,
    )
