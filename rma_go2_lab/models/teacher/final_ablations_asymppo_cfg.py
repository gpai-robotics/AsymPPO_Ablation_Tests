"""Runner config for the combined AsymPPO rough stage."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

import rsl_rl.runners.on_policy_runner as _rsl_on_policy_runner
from rma_go2_lab.models.ablations.history_actor_critic import TemporalBlindActorCritic
from rma_go2_lab.models.ablations.policy_cfg import AsymPpoHistoryPolicyCfg
from rma_go2_lab.models.teacher.combined_stage_checkpoints import resolve_stage_checkpoint


_rsl_on_policy_runner.TemporalBlindActorCritic = TemporalBlindActorCritic


def make_combined_asymppo_policy(actor_init_path: str | None = None) -> AsymPpoHistoryPolicyCfg:
    """Build the shared combined AsymPPO temporal policy config."""

    return AsymPpoHistoryPolicyCfg(
        init_noise_std=0.35,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        actor_init_path=actor_init_path,
        history_group_name="policy_history",
        temporal_channels=[64, 64],
        temporal_kernel_size=3,
        history_feature_dim=64,
        history_target_dim=128,
        history_target_hidden_dims=[128],
    )


@configclass
class Go2BlindRoughMjlabCombinedRoughRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Stage 2: warm-start from the branch-specific flat prior and train rough."""

    num_envs = None
    num_steps_per_env = 32
    max_iterations = 3000
    save_interval = 50
    experiment_name = "go2_asymppo_ablations"

    obs_groups = {
        "policy": ["policy", "policy_history"],
        "critic": ["policy", "policy_history", "critic_privileged", "dynamics_privileged", "terrain_privileged"],
    }

    policy = make_combined_asymppo_policy(
        actor_init_path=resolve_stage_checkpoint(
            ("COMBINED_FLAT_PRIOR_CKPT", "GO2_COMBINED_FLAT_PRIOR_CKPT"),
            "rough-stage flat-prior",
        )
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
