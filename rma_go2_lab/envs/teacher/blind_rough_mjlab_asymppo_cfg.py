"""MJLAB-contract asymmetric PPO rough blind policy config.

This branch is not an RMA teacher/student setup. The actor is deployable and
blind, while the critic is privileged during asymmetric PPO training.
"""

from __future__ import annotations

from isaaclab.managers import EventTermCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from rma_go2_lab.envs.asymppo.rough_omni_cfg import Go2AsymPpoRoughOmniEnvCfg
from rma_go2_lab.envs.mjlab_contract import MjlabCriticPrivilegedObsCfg, apply_mjlab_policy_contract


@configclass
class Go2BlindRoughMjlabAsymPpoEnvCfg(Go2AsymPpoRoughOmniEnvCfg):
    """C1-style rough omni baseline with the deploy-honest MJLAB actor contract."""

    mjlab_use_gait_phase: bool = False

    def __post_init__(self):
        super().__post_init__()

        # Keep the same robot/actuator model as the frozen flat prior. The
        # mjlab actuator prior is a separate ablation and needs its own flat.
        apply_mjlab_policy_contract(
            self.observations.policy,
            include_gait_phase=self.mjlab_use_gait_phase,
        )
        apply_mjlab_policy_contract(
            self.observations.policy_history,
            include_gait_phase=self.mjlab_use_gait_phase,
        )
        self.observations.critic_privileged = MjlabCriticPrivilegedObsCfg()

        # Keep the proven C1 dynamics envelope for the clean AsymPPO diagnosis.
        # The wider kp/kd and COM ablations both suppressed terrain progression.
        self.events.motor_strength.params["stiffness_distribution_params"] = (0.6, 1.4)
        self.events.motor_strength.params["damping_distribution_params"] = (0.6, 1.4)

        # Keep the recovery and COM disturbance pressure from the previous run,
        # but remove the wide-gain ablation so we isolate gain randomization.
        self.events.push_robot = EventTermCfg(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(6.0, 10.0),
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "velocity_range": {
                    "x": (-0.35, 0.35),
                    "y": (-0.35, 0.35),
                    "yaw": (-0.4, 0.4),
                },
            },
        )
        self.events.base_com = EventTermCfg(
            func=mdp.randomize_rigid_body_com,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="base"),
                "com_range": {
                    "x": (-0.03, 0.03),
                    "y": (-0.03, 0.03),
                    "z": (-0.01, 0.01),
                },
            },
        )

        print("\n========== GO2 BLIND ROUGH MJLAB ASYMMETRIC PPO V1 ==========\n")


# Backwards-compatible alias for older local scripts/checkpoints that imported
# the previous name before this branch was clarified as asymmetric PPO.
Go2RoughMjlabTeacherEnvCfg = Go2BlindRoughMjlabAsymPpoEnvCfg
