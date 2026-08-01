"""PPO config for the fail-fast terrain-supervised privileged teacher."""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlPpoAlgorithmCfg

from rma_go2_lab.models.teacher.ppo_cfg import (
    Go2PrivilegedTeacherPolicyCfg,
    Go2PrivilegedTeacherRoughPPORunnerCfg,
)


@configclass
class Go2PrivilegedTeacherTerrainAuxAlgorithmCfg(RslRlPpoAlgorithmCfg):
    class_name: str = "TeacherPPOWithTerrainAux"
    terrain_regression_coef_stage0: float = 0.5
    terrain_regression_coef_stage1: float = 0.2
    terrain_regression_coef_stage2: float = 0.05
    terrain_stage0_end: int = 300
    terrain_stage1_end: int = 800


@configclass
class Go2PrivilegedTeacherRoughV4PPORunnerCfg(Go2PrivilegedTeacherRoughPPORunnerCfg):
    """Teacher V4: V3 plus explicit terrain-target supervision for fail-fast auditing."""

    experiment_name = "go2_privileged_teacher_rough_v4_terrain_aux"

    obs_groups = {
        "policy": ["policy", "dynamics_privileged", "terrain_privileged"],
        "critic": ["policy", "dynamics_privileged", "terrain_privileged"],
    }

    policy = Go2PrivilegedTeacherPolicyCfg(
        init_noise_std=0.35,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        terrain_latent_dim=8,
        terrain_encoder_hidden_dims=[64, 32],
        terrain_target_dim=13,
        terrain_target_hidden_dims=[64],
        privileged_group_name="terrain_privileged",
        warm_start_checkpoint_path="/home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/policies/blind_baseline2_warmstart_final.pt",
    )

    algorithm = Go2PrivilegedTeacherTerrainAuxAlgorithmCfg(
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
