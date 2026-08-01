"""PPO config for the second privileged teacher iteration."""

from isaaclab.utils import configclass

from rma_go2_lab.models.teacher.ppo_cfg import (
    Go2PrivilegedTeacherPolicyCfg,
    Go2PrivilegedTeacherRoughPPORunnerCfg,
)


@configclass
class Go2PrivilegedTeacherRoughV1PPORunnerCfg(Go2PrivilegedTeacherRoughPPORunnerCfg):
    """Teacher V1 PPO recipe.

    Identical to V0 except for experiment naming. The intended controlled
    variable is the V1 anti-crouch reward term in the environment config.
    """

    experiment_name = "go2_privileged_teacher_rough_v1"

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
        warm_start_checkpoint_path="/home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/policies/blind_baseline2_warmstart_final.pt",
    )
