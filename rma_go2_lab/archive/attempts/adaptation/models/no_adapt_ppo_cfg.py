"""Initial no-adaptation student PPO config.

This is the first concrete adaptation-phase baseline:

- deployable proprio-only observations
- no latent adapter
- same policy family intended for the future student branch

It exists to measure the no-adaptation gap explicitly before adding any
history-based latent inference.
"""

from isaaclab.utils import configclass

from rma_go2_lab.models.blind.blind_rough_runner_cfg import (
    BlindWarmStartPolicyCfg,
    Go2BlindRoughForwardWarmStartPPORunnerCfg,
)


@configclass
class Go2AdaptationStudentNoAdaptPPORunnerCfg(Go2BlindRoughForwardWarmStartPPORunnerCfg):
    """Phase-B student baseline without an adaptation latent."""

    experiment_name = "go2_adaptation_student_no_adapt_v0"

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
        actor_init_path="/home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/policies/blind_baseline2_warmstart_final.pt",
    )
