"""Mjlab-contract blind student aligned with the C1 rough omni recipe."""

from __future__ import annotations

from isaaclab.utils import configclass

from rma_go2_lab.envs.blind.c1_blind_rough_omni_usable_cfg import Go2C1BlindRoughOmniUsableEnvCfg
from rma_go2_lab.envs.mjlab_contract import MjlabCriticPrivilegedObsCfg, apply_mjlab_policy_contract


@configclass
class Go2BlindRoughMjlabStudentEnvCfg(Go2C1BlindRoughOmniUsableEnvCfg):
    """C1-style blind history student with the no-base-lin-vel actor contract."""

    mjlab_use_gait_phase: bool = False

    def __post_init__(self):
        super().__post_init__()

        apply_mjlab_policy_contract(
            self.observations.policy,
            include_gait_phase=self.mjlab_use_gait_phase,
        )
        apply_mjlab_policy_contract(
            self.observations.policy_history,
            include_gait_phase=self.mjlab_use_gait_phase,
        )
        self.observations.critic_privileged = MjlabCriticPrivilegedObsCfg()

        print("\n========== RMA GO2 BLIND MJLAB STUDENT V1 ==========\n")

