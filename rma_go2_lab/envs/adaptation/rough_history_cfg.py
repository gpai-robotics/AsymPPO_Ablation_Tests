"""Future adaptation-student env with explicit proprio history observations.

This env also exposes privileged teacher-only observation groups so the future
history student can learn against the frozen `V3` expert without making those
signals deployable.
"""

from __future__ import annotations

import copy

from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils import configclass

from rma_go2_lab.envs.adaptation.rough_cfg import Go2AdaptationStudentRoughEnvCfg
from rma_go2_lab.envs.teacher.rough_cfg import TeacherPrivilegedObsCfg
from rma_go2_lab.envs.teacher.rough_v3_cfg import (
    TeacherDynamicsPrivilegedObsCfg,
    TrackedRandomizeRigidBodyMass,
    TrackedRandomizeRigidBodyMaterial,
)


@configclass
class Go2AdaptationStudentHistoryRoughEnvCfg(Go2AdaptationStudentRoughEnvCfg):
    """Phase-C scaffold with flattened proprio history.

    This keeps the deployable current observation group unchanged while exposing
    a second observation group containing short proprio history for the future
    history encoder.
    """

    adaptation_history_length: int = 20

    def __post_init__(self):
        super().__post_init__()

        print("\n========== ADAPTATION STUDENT HISTORY ROUGH ==========\n")

        self.scene.height_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base",
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
            ray_alignment="yaw",
            pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        self.events.physics_material.func = TrackedRandomizeRigidBodyMaterial
        if self.events.add_base_mass is not None:
            self.events.add_base_mass.func = TrackedRandomizeRigidBodyMass

        self.observations.policy_history = copy.deepcopy(self.observations.policy)
        self.observations.policy_history.history_length = self.adaptation_history_length
        self.observations.policy_history.flatten_history_dim = True
        self.observations.policy_history.enable_corruption = False
        self.observations.policy_history.concatenate_terms = True
        self.observations.terrain_privileged = TeacherPrivilegedObsCfg()
        self.observations.dynamics_privileged = TeacherDynamicsPrivilegedObsCfg()
