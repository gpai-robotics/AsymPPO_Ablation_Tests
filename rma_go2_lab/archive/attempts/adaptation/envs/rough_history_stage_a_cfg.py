"""Dynamics-only Stage A history env for active Adapt-V3 bring-up."""

from __future__ import annotations

import copy

from isaaclab.utils import configclass

from rma_go2_lab.envs.adaptation.rough_cfg import Go2AdaptationStudentRoughEnvCfg
from rma_go2_lab.envs.teacher.rough_v3_cfg import (
    TeacherDynamicsPrivilegedObsCfg,
    TrackedRandomizeRigidBodyMass,
    TrackedRandomizeRigidBodyMaterial,
)


@configclass
class Go2AdaptationStudentHistoryStageARoughEnvCfg(Go2AdaptationStudentRoughEnvCfg):
    """Active Adapt-V3 Stage A with proprio history and dynamics-only privilege."""

    adaptation_history_length: int = 20

    def __post_init__(self):
        super().__post_init__()
        self.events.physics_material.func = TrackedRandomizeRigidBodyMaterial
        if self.events.add_base_mass is not None:
            self.events.add_base_mass.func = TrackedRandomizeRigidBodyMass

        self.observations.policy_history = copy.deepcopy(self.observations.policy)
        self.observations.policy_history.history_length = self.adaptation_history_length
        self.observations.policy_history.flatten_history_dim = True
        self.observations.policy_history.enable_corruption = False
        self.observations.policy_history.concatenate_terms = True
        self.observations.dynamics_privileged = TeacherDynamicsPrivilegedObsCfg()
        self.adaptation_switch_episode_prob = 0.0
        print("\n========== ADAPTATION STUDENT HISTORY ROUGH STAGE A (DYNAMICS ONLY) ==========\n")
