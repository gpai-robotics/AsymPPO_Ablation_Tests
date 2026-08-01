"""PPO config for the terrain-plus-dynamics privileged teacher.

This config defines the intended observation contract, not a guarantee that a
frozen checkpoint uses every exposed privileged source.

The current `model_1999` dependency audits show:
- clear dependence on `dynamics_privileged`
- no measurable dependence on `terrain_privileged` on the audited forward
  probes
"""

from isaaclab.utils import configclass

from rma_go2_lab.models.teacher.ppo_cfg import (
    Go2PrivilegedTeacherPolicyCfg,
    Go2PrivilegedTeacherRoughPPORunnerCfg,
)


@configclass
class Go2PrivilegedTeacherRoughV3PPORunnerCfg(Go2PrivilegedTeacherRoughPPORunnerCfg):
    """Teacher V3 PPO recipe.

    V3 keeps the V2 compressed terrain encoder and warm-start recipe, but adds
    an explicit raw dynamics-privilege branch for hidden simulator factors.

    The frozen V3 checkpoint should currently be described as:
    - architecturally terrain + dynamics privileged
    - audit-validated as dynamics-privileged
    - not yet audit-validated as terrain-using
    """

    experiment_name = "go2_privileged_teacher_rough_v3"

    obs_groups = {
        "policy": ["policy", "dynamics_privileged", "terrain_privileged"],
        "critic": ["policy", "dynamics_privileged", "terrain_privileged"],
    }

    policy = Go2PrivilegedTeacherPolicyCfg(
        init_noise_std=0.35,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        terrain_latent_dim=8,
        terrain_encoder_hidden_dims=[64, 32],
        privileged_group_name="terrain_privileged",
        warm_start_checkpoint_path="/home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/policies/blind_baseline2_warmstart_final.pt",
    )
