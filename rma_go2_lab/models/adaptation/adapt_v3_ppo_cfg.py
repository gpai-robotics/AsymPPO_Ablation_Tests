"""Adapt-V3 PPO configs.

The active line is now the dynamics-only reboot of Phase 1 Stage A. Historical
terrain-inclusive Phase 2 scaffolds are kept below for reference, but they are
no longer the canonical active path.
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

from rma_go2_lab.models.adaptation.no_adapt_ppo_cfg import Go2AdaptationStudentNoAdaptPPORunnerCfg


@configclass
class Go2AdaptV3PolicyCfg(RslRlPpoActorCriticCfg):
    class_name: str = "RmaV3ActorCritic"
    latent_dim: int = 32
    extrinsics_encoder_hidden_dims: list[int] = [128, 64]
    extrinsics_encoder_mode: str = "mlp"
    extrinsics_identity_init: bool = False
    dynamics_decoder_hidden_dims: list[int] = [64]
    dynamics_decoder_mode: str = "mlp"
    dynamics_decoder_identity_init: bool = False
    adaptation_hidden_dims: list[int] = [256, 128]
    adaptation_encoder_type: str = "mlp"
    temporal_channels: list[int] = [64, 64]
    temporal_kernel_size: int = 3
    history_feature_dim: int = 64
    policy_group_name: str = "policy"
    history_group_name: str = "policy_history"
    terrain_group_name: str | None = "terrain_privileged"
    dynamics_group_name: str = "dynamics_privileged"
    full_init_path: str | None = None
    actor_init_path: str | None = "/home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/policies/blind_baseline2_warmstart_final.pt"
    critic_init_path: str | None = None
    extrinsics_init_path: str | None = None
    actor_only_extrinsics_init_mode: str = "zero"
    actor_only_adaptation_init_mode: str = "zero"


@configclass
class Go2AdaptV3Phase1AlgorithmCfg(RslRlPpoAlgorithmCfg):
    class_name: str = "RmaV3Phase1PPO"
    latent_anchor_coef: float = 3.0
    dynamics_prediction_coef: float = 1.0
    terrain_summary_coef: float = 0.5
    latent_variation_coef: float = 1.0
    latent_std_target: float = 0.05
    latent_pairwise_coef: float = 0.5
    latent_pairwise_max_samples: int = 128
    auxiliary_pretrain_steps: int = 1
    auxiliary_learning_rate: float = 5.0e-4
    auxiliary_warmup_iters: int = 300
    auxiliary_start_factor: float = 0.1
    flat_expert_path: str | None = None
    flat_expert_activation: str = "elu"
    flat_imitation_command_threshold: float = 0.1
    flat_imitation_coef_stage0: float = 0.0
    flat_imitation_coef_stage1: float = 0.0
    flat_imitation_stage0_end: int = 0
    flat_imitation_stage1_end: int = 0


@configclass
class Go2AdaptV3Phase2AlgorithmCfg(RslRlPpoAlgorithmCfg):
    class_name: str = "RmaV3Phase2PPO"
    phase1_reference_path: str | None = None
    surrogate_loss_coef: float = 1.0
    latent_regression_coef: float = 1.0
    latent_l2_coef: float = 0.0
    latent_command_threshold: float = 0.1
    imitation_command_threshold: float = 0.1
    imitation_coef_stage0: float = 0.0
    imitation_coef_stage1: float = 0.0
    imitation_stage0_end: int = 0
    imitation_stage1_end: int = 0
    freeze_critic: bool = False


@configclass
class Go2AdaptV3Phase1PPORunnerCfg(Go2AdaptationStudentNoAdaptPPORunnerCfg):
    """Faithful V3 Phase 1: train mu + pi with privileged extrinsics bottleneck."""

    experiment_name = "go2_adapt_v3_phase1"

    obs_groups = {
        "policy": ["policy", "terrain_privileged", "dynamics_privileged"],
        "critic": ["policy", "terrain_privileged", "dynamics_privileged"],
    }

    policy = Go2AdaptV3PolicyCfg(
        init_noise_std=0.35,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        latent_dim=32,
        extrinsics_encoder_hidden_dims=[128, 64],
        adaptation_hidden_dims=[256, 128],
        policy_group_name="policy",
        history_group_name="policy_history",
        terrain_group_name="terrain_privileged",
        dynamics_group_name="dynamics_privileged",
        actor_init_path=None,
    )

    algorithm = Go2AdaptV3Phase1AlgorithmCfg(
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
        latent_anchor_coef=3.0,
        dynamics_prediction_coef=1.0,
        terrain_summary_coef=0.5,
        latent_variation_coef=1.0,
        latent_std_target=0.05,
        latent_pairwise_coef=0.5,
        latent_pairwise_max_samples=128,
        auxiliary_pretrain_steps=1,
        auxiliary_learning_rate=5.0e-4,
        auxiliary_warmup_iters=300,
        auxiliary_start_factor=0.1,
        flat_expert_path=None,
        flat_imitation_coef_stage0=0.0,
        flat_imitation_coef_stage1=0.0,
        flat_imitation_stage0_end=0,
        flat_imitation_stage1_end=0,
    )


@configclass
class Go2AdaptV3Phase1StageAPPORunnerCfg(Go2AdaptationStudentNoAdaptPPORunnerCfg):
    """Active Stage A: dynamics-only privileged latent with critic-only warm start."""

    experiment_name = "go2_adapt_v3_phase1_stage_a_dyn_only"

    obs_groups = {
        "policy": ["policy", "dynamics_privileged"],
        "critic": ["policy", "dynamics_privileged"],
    }

    policy = Go2AdaptV3PolicyCfg(
        init_noise_std=0.35,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        latent_dim=32,
        extrinsics_encoder_hidden_dims=[128, 64],
        adaptation_hidden_dims=[256, 128],
        policy_group_name="policy",
        history_group_name="policy_history",
        terrain_group_name=None,
        dynamics_group_name="dynamics_privileged",
        actor_init_path="/home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/policies/blind_baseline2_warmstart_final.pt",
        critic_init_path="/home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/policies/blind_baseline2_warmstart_final.pt",
        actor_only_extrinsics_init_mode="small_xavier",
        actor_only_adaptation_init_mode="small_xavier",
    )

    algorithm = Go2AdaptV3Phase1AlgorithmCfg(
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
        latent_anchor_coef=3.0,
        dynamics_prediction_coef=1.0,
        terrain_summary_coef=0.0,
        latent_variation_coef=1.0,
        latent_std_target=0.05,
        latent_pairwise_coef=0.5,
        latent_pairwise_max_samples=128,
        auxiliary_pretrain_steps=1,
        auxiliary_learning_rate=5.0e-4,
        auxiliary_warmup_iters=300,
        auxiliary_start_factor=0.1,
        flat_expert_path="/home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/policies/blind_baseline2_warmstart_final.pt",
        flat_expert_activation="elu",
        flat_imitation_command_threshold=0.1,
        flat_imitation_coef_stage0=0.3,
        flat_imitation_coef_stage1=0.1,
        flat_imitation_stage0_end=200,
        flat_imitation_stage1_end=500,
    )


@configclass
class Go2AdaptV3Phase2PPORunnerCfg(Go2AdaptationStudentNoAdaptPPORunnerCfg):
    """Faithful V3 Phase 2: freeze mu + pi, train phi against z_t = mu(e_t)."""

    experiment_name = "go2_adapt_v3_phase2"

    obs_groups = {
        "policy": ["policy", "policy_history"],
        "critic": ["policy", "policy_history"],
    }

    policy = Go2AdaptV3PolicyCfg(
        init_noise_std=0.35,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        latent_dim=32,
        extrinsics_encoder_hidden_dims=[128, 64],
        adaptation_hidden_dims=[256, 128],
        policy_group_name="policy",
        history_group_name="policy_history",
        terrain_group_name="terrain_privileged",
        dynamics_group_name="dynamics_privileged",
        actor_init_path="/home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/policies/adapt_v3_phase1_stage_a_final.pt",
    )

    algorithm = Go2AdaptV3Phase2AlgorithmCfg(
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
        phase1_reference_path="/home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/policies/adapt_v3_phase1_stage_a_final.pt",
        latent_regression_coef=1.0,
        latent_command_threshold=0.1,
        imitation_command_threshold=0.1,
        imitation_coef_stage0=0.0,
        imitation_coef_stage1=0.0,
        imitation_stage0_end=0,
        imitation_stage1_end=0,
    )


@configclass
class Go2AdaptV3Phase2MixedPPORunnerCfg(Go2AdaptationStudentNoAdaptPPORunnerCfg):
    """Phase 2 mixed continuation from the frozen Phase 2 Stage A bootstrap."""

    experiment_name = "go2_adapt_v3_phase2_mixed"

    obs_groups = {
        "policy": ["policy", "policy_history"],
        "critic": ["policy", "policy_history"],
    }

    policy = Go2AdaptV3PolicyCfg(
        init_noise_std=0.35,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        latent_dim=32,
        extrinsics_encoder_hidden_dims=[128, 64],
        adaptation_hidden_dims=[256, 128],
        policy_group_name="policy",
        history_group_name="policy_history",
        terrain_group_name="terrain_privileged",
        dynamics_group_name="dynamics_privileged",
        full_init_path="/home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/policies/adapt_v3_phase2_stage_a_final.pt",
        actor_init_path=None,
        critic_init_path=None,
    )

    algorithm = Go2AdaptV3Phase2AlgorithmCfg(
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
        phase1_reference_path="/home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/policies/adapt_v3_phase1_stage_a_final.pt",
        latent_regression_coef=1.0,
        latent_command_threshold=0.1,
        imitation_command_threshold=0.1,
        imitation_coef_stage0=0.0,
        imitation_coef_stage1=0.0,
        imitation_stage0_end=0,
        imitation_stage1_end=0,
    )


@configclass
class Go2AdaptV3Phase2StageAPPORunnerCfg(Go2AdaptationStudentNoAdaptPPORunnerCfg):
    """Phase 2 Stage A: train phi first on the stationary Stage A regime.

    This mirrors the Phase 1 lesson: bootstrap the new pathway under a
    survivable locomotion regime first, then reintroduce the harsher switched
    task later.
    """

    experiment_name = "go2_adapt_v3_phase2_stage_a"

    obs_groups = {
        "policy": ["policy", "policy_history"],
        "critic": ["policy", "policy_history"],
    }

    policy = Go2AdaptV3PolicyCfg(
        init_noise_std=0.35,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        latent_dim=32,
        extrinsics_encoder_hidden_dims=[128, 64],
        adaptation_hidden_dims=[256, 128],
        policy_group_name="policy",
        history_group_name="policy_history",
        terrain_group_name="terrain_privileged",
        dynamics_group_name="dynamics_privileged",
        actor_init_path="/home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/policies/adapt_v3_phase1_stage_a_final.pt",
    )

    algorithm = Go2AdaptV3Phase2AlgorithmCfg(
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
        phase1_reference_path="/home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/policies/adapt_v3_phase1_stage_a_final.pt",
        latent_regression_coef=1.0,
        latent_command_threshold=0.1,
        imitation_command_threshold=0.1,
        imitation_coef_stage0=0.3,
        imitation_coef_stage1=0.1,
        imitation_stage0_end=300,
        imitation_stage1_end=800,
    )


@configclass
class Go2AdaptV3DynOnlyPhase2StageAPPORunnerCfg(Go2AdaptationStudentNoAdaptPPORunnerCfg):
    """Dynamics-only Phase 2 Stage A bootstrap from the frozen dynamics-only base."""

    experiment_name = "go2_adapt_v3_phase2_stage_a_dyn_only"

    obs_groups = {
        "policy": ["policy", "policy_history"],
        "critic": ["policy", "policy_history"],
    }

    policy = Go2AdaptV3PolicyCfg(
        init_noise_std=0.35,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        latent_dim=32,
        extrinsics_encoder_hidden_dims=[128, 64],
        adaptation_hidden_dims=[256, 128],
        policy_group_name="policy",
        history_group_name="policy_history",
        terrain_group_name=None,
        dynamics_group_name="dynamics_privileged",
        actor_init_path="/home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/policies/adapt_v3_dyn_only_phase1_stage_a_final.pt",
    )

    algorithm = Go2AdaptV3Phase2AlgorithmCfg(
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
        phase1_reference_path="/home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/policies/adapt_v3_dyn_only_phase1_stage_a_final.pt",
        latent_regression_coef=1.0,
        latent_command_threshold=0.1,
        imitation_command_threshold=0.1,
        imitation_coef_stage0=0.3,
        imitation_coef_stage1=0.1,
        imitation_stage0_end=300,
        imitation_stage1_end=800,
    )


@configclass
class Go2AdaptV3DynOnlyPhase2RecoveryLowProbPPORunnerCfg(Go2AdaptationStudentNoAdaptPPORunnerCfg):
    """Low-switch-probability dyn-only recovery continuation from frozen Stage A.

    This is the cautious recovery lane after discovering that the canonical
    Stage A winner did not actually need within-episode latent motion.
    """

    experiment_name = "go2_adapt_v3_phase2_recovery_low_switch_dyn_only"

    obs_groups = {
        "policy": ["policy", "policy_history"],
        "critic": ["policy", "policy_history"],
    }

    policy = Go2AdaptV3PolicyCfg(
        init_noise_std=0.35,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        latent_dim=32,
        extrinsics_encoder_hidden_dims=[128, 64],
        adaptation_hidden_dims=[256, 128],
        policy_group_name="policy",
        history_group_name="policy_history",
        terrain_group_name=None,
        dynamics_group_name="dynamics_privileged",
        full_init_path=None,
        actor_init_path="/home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/policies/adapt_v3_dyn_only_phase2_stage_a_final.pt",
        critic_init_path=None,
        actor_only_extrinsics_init_mode="zero",
        actor_only_adaptation_init_mode="small_xavier",
    )

    algorithm = Go2AdaptV3Phase2AlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.002,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5e-5,
        schedule="adaptive",
        desired_kl=0.01,
        gamma=0.99,
        lam=0.95,
        max_grad_norm=1.0,
        phase1_reference_path="/home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/policies/adapt_v3_dyn_only_phase1_stage_a_final.pt",
        latent_regression_coef=1.0,
        latent_command_threshold=0.1,
        imitation_command_threshold=0.1,
        imitation_coef_stage0=0.05,
        imitation_coef_stage1=0.0,
        imitation_stage0_end=300,
        imitation_stage1_end=300,
    )


@configclass
class Go2AdaptV3DynOnlyPhase2RecoveryLowProbLatentRegPPORunnerCfg(Go2AdaptationStudentNoAdaptPPORunnerCfg):
    """Low-switch recovery with explicit latent magnitude regularization.

    This is the first training-side fix after confirming that the MuJoCo
    failure mode comes from `phi(history)` latent blow-up under cross-engine
    history shift.
    """

    experiment_name = "go2_adapt_v3_phase2_recovery_low_switch_dyn_only_latent_reg"

    obs_groups = {
        "policy": ["policy", "policy_history"],
        "critic": ["policy", "policy_history"],
    }

    policy = Go2AdaptV3PolicyCfg(
        init_noise_std=0.35,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        latent_dim=32,
        extrinsics_encoder_hidden_dims=[128, 64],
        adaptation_hidden_dims=[256, 128],
        policy_group_name="policy",
        history_group_name="policy_history",
        terrain_group_name=None,
        dynamics_group_name="dynamics_privileged",
        full_init_path=None,
        actor_init_path="/home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/policies/adapt_v3_dyn_only_phase2_stage_a_final.pt",
        critic_init_path=None,
        actor_only_extrinsics_init_mode="zero",
        actor_only_adaptation_init_mode="small_xavier",
    )

    algorithm = Go2AdaptV3Phase2AlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.002,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5e-5,
        schedule="adaptive",
        desired_kl=0.01,
        gamma=0.99,
        lam=0.95,
        max_grad_norm=1.0,
        phase1_reference_path="/home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/policies/adapt_v3_dyn_only_phase1_stage_a_final.pt",
        latent_regression_coef=1.0,
        latent_l2_coef=1.0e-3,
        latent_command_threshold=0.1,
        imitation_command_threshold=0.1,
        imitation_coef_stage0=0.05,
        imitation_coef_stage1=0.0,
        imitation_stage0_end=300,
        imitation_stage1_end=300,
    )


@configclass
class Go2AdaptV3DynOnlyStructuredZ27Phase1StageAPPORunnerCfg(Go2AdaptationStudentNoAdaptPPORunnerCfg):
    """Structured C2 root rebuild: make z align with the 27-D dynamics contract.

    This is the clean first step in the deeper C2 restart: rebuild the root so
    the actor is trained around a smaller, explicitly dynamics-shaped latent
    instead of the older free-form 32-D bottleneck.
    """

    experiment_name = "go2_adapt_v3_phase1_stage_a_dyn_only_structured_z27"

    obs_groups = {
        "policy": ["policy", "dynamics_privileged"],
        "critic": ["policy", "dynamics_privileged"],
    }

    policy = Go2AdaptV3PolicyCfg(
        init_noise_std=0.35,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        latent_dim=27,
        extrinsics_encoder_hidden_dims=[],
        extrinsics_encoder_mode="linear",
        extrinsics_identity_init=True,
        dynamics_decoder_hidden_dims=[],
        dynamics_decoder_mode="linear",
        dynamics_decoder_identity_init=True,
        adaptation_hidden_dims=[256, 128],
        policy_group_name="policy",
        history_group_name="policy_history",
        terrain_group_name=None,
        dynamics_group_name="dynamics_privileged",
        actor_init_path="/home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/policies/blind_baseline2_warmstart_final.pt",
        critic_init_path="/home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/policies/blind_baseline2_warmstart_final.pt",
        actor_only_extrinsics_init_mode="identity",
        actor_only_adaptation_init_mode="small_xavier",
    )

    algorithm = Go2AdaptV3Phase1AlgorithmCfg(
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
        latent_anchor_coef=3.0,
        dynamics_prediction_coef=1.0,
        terrain_summary_coef=0.0,
        latent_variation_coef=1.0,
        latent_std_target=0.05,
        latent_pairwise_coef=0.5,
        latent_pairwise_max_samples=128,
        auxiliary_pretrain_steps=1,
        auxiliary_learning_rate=5.0e-4,
        auxiliary_warmup_iters=300,
        auxiliary_start_factor=0.1,
        flat_expert_path="/home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/policies/blind_baseline2_warmstart_final.pt",
        flat_expert_activation="elu",
        flat_imitation_command_threshold=0.1,
        flat_imitation_coef_stage0=0.3,
        flat_imitation_coef_stage1=0.1,
        flat_imitation_stage0_end=200,
        flat_imitation_stage1_end=500,
    )


@configclass
class Go2AdaptV3DynOnlyStructuredZ27Phase2RecoveryLowProbLatentRegPPORunnerCfg(
    Go2AdaptationStudentNoAdaptPPORunnerCfg
):
    """Structured C2 Phase 2 recovery built on the new 27-D root.

    This tuned version keeps the student closer to the frozen Phase 1 teacher
    for longer, after early tests showed improving latent fit but degrading
    locomotion by ~200 iterations.
    """

    experiment_name = "go2_adapt_v3_phase2_recovery_low_switch_dyn_only_structured_z27"

    obs_groups = {
        "policy": ["policy", "policy_history"],
        "critic": ["policy", "policy_history"],
    }

    policy = Go2AdaptV3PolicyCfg(
        init_noise_std=0.35,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        latent_dim=27,
        extrinsics_encoder_hidden_dims=[],
        extrinsics_encoder_mode="linear",
        extrinsics_identity_init=True,
        dynamics_decoder_hidden_dims=[],
        dynamics_decoder_mode="linear",
        dynamics_decoder_identity_init=True,
        adaptation_hidden_dims=[256, 128],
        policy_group_name="policy",
        history_group_name="policy_history",
        terrain_group_name=None,
        dynamics_group_name="dynamics_privileged",
        full_init_path=None,
        actor_init_path="/home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/policies/adapt_v3_dyn_only_structured_z27_phase1_stage_a_final.pt",
        critic_init_path=None,
        actor_only_extrinsics_init_mode="identity",
        actor_only_adaptation_init_mode="small_xavier",
    )

    algorithm = Go2AdaptV3Phase2AlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.002,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=2.5e-5,
        schedule="adaptive",
        desired_kl=0.01,
        gamma=0.99,
        lam=0.95,
        max_grad_norm=1.0,
        phase1_reference_path="/home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/policies/adapt_v3_dyn_only_structured_z27_phase1_stage_a_final.pt",
        latent_regression_coef=1.0,
        latent_l2_coef=5.0e-3,
        latent_command_threshold=0.1,
        imitation_command_threshold=0.1,
        imitation_coef_stage0=0.20,
        imitation_coef_stage1=0.05,
        imitation_stage0_end=600,
        imitation_stage1_end=1200,
    )


class Go2AdaptV3TerrainLitePhase1StageAPPORunnerCfg(Go2AdaptationStudentNoAdaptPPORunnerCfg):
    """Terrain-lite Phase 1 Stage A from the frozen dynamics-only base."""

    experiment_name = "go2_adapt_v3_phase1_stage_a_terrain_lite"

    obs_groups = {
        "policy": ["policy", "terrain_lite_privileged", "dynamics_privileged"],
        "critic": ["policy", "terrain_lite_privileged", "dynamics_privileged"],
    }

    policy = Go2AdaptV3PolicyCfg(
        init_noise_std=0.35,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        latent_dim=32,
        extrinsics_encoder_hidden_dims=[128, 64],
        adaptation_hidden_dims=[256, 128],
        policy_group_name="policy",
        history_group_name="policy_history",
        terrain_group_name="terrain_lite_privileged",
        dynamics_group_name="dynamics_privileged",
        actor_init_path="/home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/policies/adapt_v3_dyn_only_phase1_stage_a_final.pt",
        extrinsics_init_path="/home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/policies/adapt_v3_dyn_only_phase1_stage_a_final.pt",
    )

    algorithm = Go2AdaptV3Phase1AlgorithmCfg(
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
        latent_anchor_coef=3.0,
        dynamics_prediction_coef=1.0,
        terrain_summary_coef=0.5,
        latent_variation_coef=1.0,
        latent_std_target=0.05,
        latent_pairwise_coef=0.5,
        latent_pairwise_max_samples=128,
        auxiliary_pretrain_steps=1,
        auxiliary_learning_rate=5.0e-4,
        auxiliary_warmup_iters=300,
        auxiliary_start_factor=0.1,
        flat_expert_path="/home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/policies/blind_baseline2_warmstart_final.pt",
        flat_expert_activation="elu",
        flat_imitation_command_threshold=0.1,
        flat_imitation_coef_stage0=0.3,
        flat_imitation_coef_stage1=0.1,
        flat_imitation_stage0_end=200,
        flat_imitation_stage1_end=500,
    )


@configclass
class Go2AdaptV3TerrainLitePhase1PerEpisodePPORunnerCfg(Go2AdaptV3TerrainLitePhase1StageAPPORunnerCfg):
    """Canonical per-episode terrain-lite Phase 1 runner."""

    experiment_name = "go2_adapt_v3_phase1_per_episode_terrain_lite"


@configclass
class Go2AdaptV3TerrainLitePhase2StageAPPORunnerCfg(Go2AdaptationStudentNoAdaptPPORunnerCfg):
    """Terrain-lite Phase 2 Stage A bootstrap from a frozen terrain-lite Phase 1."""

    experiment_name = "go2_adapt_v3_phase2_stage_a_terrain_lite"

    obs_groups = {
        "policy": ["policy", "policy_history"],
        "critic": ["policy", "policy_history"],
    }

    policy = Go2AdaptV3PolicyCfg(
        init_noise_std=0.35,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        latent_dim=32,
        extrinsics_encoder_hidden_dims=[128, 64],
        adaptation_hidden_dims=[256, 128],
        policy_group_name="policy",
        history_group_name="policy_history",
        terrain_group_name="terrain_lite_privileged",
        dynamics_group_name="dynamics_privileged",
        actor_init_path="/home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/policies/adapt_v3_terrain_lite_phase1_stage_a_final.pt",
    )

    algorithm = Go2AdaptV3Phase2AlgorithmCfg(
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
        phase1_reference_path="/home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/policies/adapt_v3_terrain_lite_phase1_stage_a_final.pt",
        latent_regression_coef=1.0,
        latent_command_threshold=0.1,
        imitation_command_threshold=0.1,
        imitation_coef_stage0=0.3,
        imitation_coef_stage1=0.1,
        imitation_stage0_end=300,
        imitation_stage1_end=800,
    )
