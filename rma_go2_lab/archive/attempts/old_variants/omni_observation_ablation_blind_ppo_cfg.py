"""Archived rough omni observation-ablation runner configs."""

from pathlib import Path

from isaaclab.utils import configclass

from rma_go2_lab.models.blind.variants_ppo_cfg import BlindHistoryPolicyCfg, Go2C1EthLikeV3PPORunnerCfg


OMNI_FLAT_CONTACT_EXPERT_CKPT = "/home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/policies/flat_omni_contact_v1.pt"
OMNI_FLAT_CONTACT_PHASE_EXPERT_CKPT = (
    "/home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/policies/flat_omni_contact_phase_v1.pt"
)


def _resolve_c1_omni_v2_warmstart() -> str | None:
    if Path(OMNI_FLAT_CONTACT_EXPERT_CKPT).exists():
        return OMNI_FLAT_CONTACT_EXPERT_CKPT
    return None


def _resolve_c1_omni_v3_warmstart() -> str | None:
    if Path(OMNI_FLAT_CONTACT_PHASE_EXPERT_CKPT).exists():
        return OMNI_FLAT_CONTACT_PHASE_EXPERT_CKPT
    return None


@configclass
class Go2C1OmniV2PPORunnerCfg(Go2C1EthLikeV3PPORunnerCfg):
    """Archived rough omni contact-only branch."""

    experiment_name = "go2_c1_omni_v2_stagea"

    policy = BlindHistoryPolicyCfg(
        init_noise_std=0.35,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        actor_init_path=_resolve_c1_omni_v2_warmstart(),
        history_group_name="policy_history",
        temporal_channels=[64, 64],
        temporal_kernel_size=3,
        history_feature_dim=64,
        history_target_dim=128,
        history_target_hidden_dims=[128],
    )


@configclass
class Go2C1OmniV3PPORunnerCfg(Go2C1EthLikeV3PPORunnerCfg):
    """Archived rough omni contact+phase branch."""

    experiment_name = "go2_c1_omni_v3_stagea"

    policy = BlindHistoryPolicyCfg(
        init_noise_std=0.35,
        actor_obs_normalization=False,base_contact_frac
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        actor_init_path=_resolve_c1_omni_v3_warmstart(),
        history_group_name="policy_history",
        temporal_channels=[64, 64],
        temporal_kernel_size=3,
        history_feature_dim=64,
        history_target_dim=128,
        history_target_hidden_dims=[128],
    )
