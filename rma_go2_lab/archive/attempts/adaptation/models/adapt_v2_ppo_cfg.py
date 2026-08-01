"""Adapt-V2 PPO config with explicit modular RMA-like decomposition."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg

from rma_go2_lab.models.adaptation.adapt_v1_ppo_cfg import (
    Go2AdaptationStudentV1AlgorithmCfg,
    Go2AdaptationStudentV1PPORunnerCfg,
)


@configclass
class Go2AdaptationStudentV2PolicyCfg(RslRlPpoActorCriticCfg):
    class_name: str = "ModularHistoryAdaptationActorCritic"
    init_noise_std: float = 0.35
    actor_obs_normalization: bool = False
    critic_obs_normalization: bool = False
    actor_hidden_dims: list[int] = [512, 256, 128]
    critic_hidden_dims: list[int] = [512, 256, 128]
    activation: str = "elu"
    latent_dim: int = 128
    adaptation_hidden_dims: list[int] = [256, 128]
    history_group_name: str = "policy_history"
    policy_group_name: str = "policy"
    actor_init_path: str | None = "/home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/policies/blind_baseline2_warmstart_final.pt"


@configclass
class Go2AdaptationStudentV2AlgorithmCfg(Go2AdaptationStudentV1AlgorithmCfg):
    class_name: str = "AdaptationPPOWithV3Latent"


@configclass
class Go2AdaptationStudentV2PPORunnerCfg(Go2AdaptationStudentV1PPORunnerCfg):
    """V2 scaffold: explicit modular split, same latent-training contract as V1."""

    experiment_name = "go2_adaptation_student_history_v2"

    policy = Go2AdaptationStudentV2PolicyCfg()

    algorithm = Go2AdaptationStudentV2AlgorithmCfg(
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
        v3_expert_path="/home/bhuvan/tools/IsaacLab/logs/rsl_rl/go2_privileged_teacher_rough_v3/2026-04-21_15-35-03/model_1999.pt",
        latent_regression_coef=1.0,
        latent_command_threshold=0.1,
        imitation_command_threshold=0.1,
        imitation_coef_stage0=0.05,
        imitation_coef_stage1=0.0,
        imitation_stage0_end=200,
        imitation_stage1_end=500,
    )
