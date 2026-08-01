"""Archived PPO config for a direct-scan privileged teacher, closer to IsaacLab stock usage.

Teacher V6 intentionally follows the tried-and-tested IsaacLab pattern more
closely:

- no custom terrain encoder side branch
- raw terrain scan is concatenated directly into the policy observations
- hidden dynamics privilege is still available as a separate input group

This gives us a simpler root branch to compare against the more custom V3/V4/V5
teacher lines.
"""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg

from rma_go2_lab.models.teacher.ppo_cfg import Go2PrivilegedTeacherRoughPPORunnerCfg


@configclass
class Go2PrivilegedTeacherRoughV6PPORunnerCfg(Go2PrivilegedTeacherRoughPPORunnerCfg):
    """Teacher V6: direct raw terrain scan + dynamics privilege into a plain policy."""

    experiment_name = "go2_privileged_teacher_rough_v6_directscan"

    obs_groups = {
        "policy": ["policy", "terrain_privileged", "dynamics_privileged"],
        "critic": ["policy", "terrain_privileged", "dynamics_privileged"],
    }

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.35,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
