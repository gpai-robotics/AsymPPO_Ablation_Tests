"""PPO variants for the blind rough-terrain baseline."""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


FLAT_EXPERT_CKPT = "/home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/policies/flat1499.pt"
V3_TEACHER_CKPT = "/home/bhuvan/tools/IsaacLab/logs/rsl_rl/go2_privileged_teacher_rough_v3/2026-04-21_15-35-03/model_1999.pt"
V4_TEACHER_MODEL300_CKPT = "/home/bhuvan/tools/IsaacLab/logs/rsl_rl/go2_privileged_teacher_rough_v4_terrain_aux/2026-05-09_10-34-56/model_300.pt"


@configclass
class BlindWarmStartPolicyCfg(RslRlPpoActorCriticCfg):
    class_name: str = "WarmStartActorCritic"
    actor_init_path: str | None = None


@configclass
class BlindHistoryPolicyCfg(RslRlPpoActorCriticCfg):
    class_name: str = "TemporalBlindActorCritic"
    actor_init_path: str | None = None
    history_group_name: str = "policy_history"
    temporal_channels: list[int] = [64, 64]
    temporal_kernel_size: int = 3
    history_feature_dim: int = 64
    history_target_dim: int = 128
    history_target_hidden_dims: list[int] = [128]


@configclass
class BlindImitationAlgorithmCfg(RslRlPpoAlgorithmCfg):
    class_name: str = "BlindPPOWithFlatExpert"
    flat_expert_path: str | None = None
    flat_expert_activation: str = "elu"
    flat_imitation_command_threshold: float = 0.1
    flat_imitation_coef_stage0: float = 0.1
    flat_imitation_coef_stage1: float = 0.03
    flat_imitation_stage0_end: int = 150
    flat_imitation_stage1_end: int = 400


@configclass
class BlindV3TeacherImitationAlgorithmCfg(RslRlPpoAlgorithmCfg):
    class_name: str = "BlindPPOWithV3Teacher"
    v3_expert_path: str | None = None
    latent_command_threshold: float = 0.1
    latent_regression_coef_stage0: float = 0.0
    latent_regression_coef_stage1: float = 0.0
    latent_regression_coef_stage2: float = 0.0
    latent_stage0_end: int = 300
    latent_stage1_end: int = 800
    imitation_command_threshold: float = 0.1
    imitation_coef_stage0: float = 0.2
    imitation_coef_stage1: float = 0.05
    imitation_stage0_end: int = 300
    imitation_stage1_end: int = 800


@configclass
class Go2BlindBaselineScratchPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_envs = None
    num_steps_per_env = 32
    max_iterations = 2000
    save_interval = 20

    experiment_name = "go2_blind_baseline_rough_scratch"

    obs_groups = {
        "policy": ["policy"],
        "critic": ["policy"],
    }

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.35,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
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
    )


@configclass
class Go2BlindBaselineWarmStartPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_envs = None
    num_steps_per_env = 32
    max_iterations = 2000
    save_interval = 20

    experiment_name = "go2_blind_baseline_rough_warmstart"

    obs_groups = {
        "policy": ["policy"],
        "critic": ["policy"],
    }

    policy = BlindWarmStartPolicyCfg(
        init_noise_std=0.35,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        actor_init_path=FLAT_EXPERT_CKPT,
    )

    algorithm = RslRlPpoAlgorithmCfg(
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
    )


@configclass
class Go2BlindBaselineWarmStartImitationPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_envs = None
    num_steps_per_env = 32
    max_iterations = 2000
    save_interval = 20

    experiment_name = "go2_blind_baseline_rough_warmstart_imitation"

    obs_groups = {
        "policy": ["policy"],
        "critic": ["policy"],
    }

    policy = BlindWarmStartPolicyCfg(
        init_noise_std=0.35,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        actor_init_path=FLAT_EXPERT_CKPT,
    )

    algorithm = BlindImitationAlgorithmCfg(
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
        flat_expert_path=FLAT_EXPERT_CKPT,
        flat_expert_activation="elu",
        flat_imitation_command_threshold=0.1,
        flat_imitation_coef_stage0=0.1,
        flat_imitation_coef_stage1=0.03,
        flat_imitation_stage0_end=150,
        flat_imitation_stage1_end=400,
    )


@configclass
class Go2C1EthLikeV1PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_envs = None
    num_steps_per_env = 32
    max_iterations = 2000
    save_interval = 20

    experiment_name = "go2_c1_ethlike_v1"

    obs_groups = {
        "policy": ["policy", "policy_history"],
        "critic": ["policy", "policy_history"],
    }

    policy = BlindHistoryPolicyCfg(
        init_noise_std=0.35,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        actor_init_path=FLAT_EXPERT_CKPT,
        history_group_name="policy_history",
        temporal_channels=[64, 64],
        temporal_kernel_size=3,
        history_feature_dim=64,
        history_target_dim=128,
        history_target_hidden_dims=[128],
    )

    algorithm = BlindV3TeacherImitationAlgorithmCfg(
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
        v3_expert_path=V3_TEACHER_CKPT,
        latent_command_threshold=0.1,
        latent_regression_coef_stage0=0.0,
        latent_regression_coef_stage1=0.0,
        latent_regression_coef_stage2=0.0,
        latent_stage0_end=300,
        latent_stage1_end=800,
        imitation_command_threshold=0.1,
        imitation_coef_stage0=0.2,
        imitation_coef_stage1=0.05,
        imitation_stage0_end=300,
        imitation_stage1_end=800,
    )


@configclass
class Go2C1EthLikeV2PPORunnerCfg(Go2C1EthLikeV1PPORunnerCfg):
    """C1 retrain with explicit history-target supervision from the frozen V3 teacher."""

    experiment_name = "go2_c1_ethlike_v2_temporal_reg"

    policy = BlindHistoryPolicyCfg(
        init_noise_std=0.35,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        actor_init_path=FLAT_EXPERT_CKPT,
        history_group_name="policy_history",
        temporal_channels=[64, 64],
        temporal_kernel_size=3,
        history_feature_dim=64,
        history_target_dim=128,
        history_target_hidden_dims=[128],
    )

    algorithm = BlindV3TeacherImitationAlgorithmCfg(
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
        v3_expert_path=V3_TEACHER_CKPT,
        latent_command_threshold=0.1,
        latent_regression_coef_stage0=0.5,
        latent_regression_coef_stage1=0.2,
        latent_regression_coef_stage2=0.05,
        latent_stage0_end=300,
        latent_stage1_end=800,
        imitation_command_threshold=0.1,
        imitation_coef_stage0=0.1,
        imitation_coef_stage1=0.02,
        imitation_stage0_end=300,
        imitation_stage1_end=800,
    )


@configclass
class Go2C1EthLikeV3PPORunnerCfg(Go2C1EthLikeV1PPORunnerCfg):
    """C1 retrain against the improved V4 terrain-using teacher candidate."""

    experiment_name = "go2_c1_ethlike_v3_v4teacher300"

    policy = BlindHistoryPolicyCfg(
        init_noise_std=0.35,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        actor_init_path=FLAT_EXPERT_CKPT,
        history_group_name="policy_history",
        temporal_channels=[64, 64],
        temporal_kernel_size=3,
        history_feature_dim=64,
        history_target_dim=128,
        history_target_hidden_dims=[128],
    )

    algorithm = BlindV3TeacherImitationAlgorithmCfg(
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
        v3_expert_path=V4_TEACHER_MODEL300_CKPT,
        latent_command_threshold=0.1,
        latent_regression_coef_stage0=0.5,
        latent_regression_coef_stage1=0.2,
        latent_regression_coef_stage2=0.05,
        latent_stage0_end=300,
        latent_stage1_end=800,
        imitation_command_threshold=0.1,
        imitation_coef_stage0=0.1,
        imitation_coef_stage1=0.02,
        imitation_stage0_end=300,
        imitation_stage1_end=800,
    )
