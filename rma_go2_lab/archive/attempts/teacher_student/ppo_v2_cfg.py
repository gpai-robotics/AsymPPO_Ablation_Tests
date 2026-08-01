"""PPO config for the compressed-encoder privileged teacher iteration."""

from isaaclab.utils import configclass

from rma_go2_lab.models.teacher.ppo_cfg import (
    Go2PrivilegedTeacherPolicyCfg,
    Go2PrivilegedTeacherRoughPPORunnerCfg,
)


@configclass
class Go2PrivilegedTeacherRoughV2PPORunnerCfg(Go2PrivilegedTeacherRoughPPORunnerCfg):
    """Teacher V2 PPO recipe.

    V2 keeps the V1 anti-crouch environment and warm-start recipe, but
    compresses the terrain privilege harder to test whether a smaller latent
    leads to a more task-relevant privileged code.
    """

    experiment_name = "go2_privileged_teacher_rough_v2"

    policy = Go2PrivilegedTeacherPolicyCfg(
        init_noise_std=0.35,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        terrain_latent_dim=8,
        terrain_encoder_hidden_dims=[64, 32],
        privileged_group_name="privileged",
        warm_start_checkpoint_path="/home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/policies/blind_baseline2_warmstart_final.pt",
    )
