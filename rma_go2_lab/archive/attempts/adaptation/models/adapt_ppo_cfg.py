"""Future adaptation-student PPO config scaffold."""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

from rma_go2_lab.models.adaptation.no_adapt_ppo_cfg import Go2AdaptationStudentNoAdaptPPORunnerCfg


@configclass
class Go2AdaptationStudentPolicyCfg(RslRlPpoActorCriticCfg):
    class_name: str = "HistoryEncoderStudentActorCritic"
    latent_dim: int = 8
    history_encoder_hidden_dims: list[int] = [256, 128]
    history_group_name: str = "policy_history"
    actor_init_path: str | None = "/home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/policies/blind_baseline2_warmstart_final.pt"


@configclass
class Go2AdaptationStudentImitationAlgorithmCfg(RslRlPpoAlgorithmCfg):
    class_name: str = "AdaptationPPOWithV3Expert"
    v3_expert_path: str | None = None
    imitation_command_threshold: float = 0.1
    imitation_coef_stage0: float = 0.2
    imitation_coef_stage1: float = 0.05
    imitation_stage0_end: int = 300
    imitation_stage1_end: int = 800


@configclass
class Go2AdaptationStudentPPORunnerCfg(Go2AdaptationStudentNoAdaptPPORunnerCfg):
    """Future trainable adaptation-student scaffold.

    This config is intentionally PPO-shaped already, but it should not be
    treated as the final adaptation recipe until the repo adds:

    - explicit supervision / imitation choice
    - adaptation-specific diagnostics
    - frozen-expert interaction path
    """

    experiment_name = "go2_adaptation_student_history_v0"

    obs_groups = {
        "policy": ["policy", "policy_history"],
        "critic": ["policy", "policy_history"],
    }

    policy = Go2AdaptationStudentPolicyCfg(
        init_noise_std=0.35,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        latent_dim=8,
        history_encoder_hidden_dims=[256, 128],
        history_group_name="policy_history",
        actor_init_path="/home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/policies/blind_baseline2_warmstart_final.pt",
    )

    algorithm = Go2AdaptationStudentImitationAlgorithmCfg(
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
        imitation_command_threshold=0.1,
        imitation_coef_stage0=0.2,
        imitation_coef_stage1=0.05,
        imitation_stage0_end=300,
        imitation_stage1_end=800,
    )
