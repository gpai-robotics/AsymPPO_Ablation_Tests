"""PPO config for the first privileged teacher."""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)

B2_CKPT = "/home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/policies/blind_baseline2_warmstart_final.pt"


@configclass
class Go2PrivilegedTeacherPolicyCfg(RslRlPpoActorCriticCfg):
    class_name: str = "TerrainEncoderActorCritic"
    terrain_latent_dim: int = 32
    terrain_encoder_hidden_dims: list[int] = [128, 64]
    terrain_target_dim: int | None = None
    terrain_target_hidden_dims: list[int] | None = None
    privileged_group_name: str = "privileged"
    warm_start_checkpoint_path: str | None = None


@configclass
class Go2PrivilegedTeacherRoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Teacher V0 PPO recipe.

    This intentionally stays close to the frozen blind PPO ladder so the first
    teacher comparison isolates observation privilege rather than optimizer
    retuning.
    """

    num_envs = None
    num_steps_per_env = 32
    max_iterations = 2000
    save_interval = 20

    experiment_name = "go2_privileged_teacher_rough_v0"

    obs_groups = {
        "policy": ["policy", "privileged"],
        "critic": ["policy", "privileged"],
    }

    policy = Go2PrivilegedTeacherPolicyCfg(
        init_noise_std=0.35,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        terrain_latent_dim=32,
        terrain_encoder_hidden_dims=[128, 64],
        privileged_group_name="privileged",
        warm_start_checkpoint_path=B2_CKPT,
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
