"""Terrain-lite Stage A history env for Adapt-V3 bring-up."""

from __future__ import annotations

import copy

from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils import configclass

from rma_go2_lab.envs.adaptation.rough_cfg import Go2AdaptationStudentRoughEnvCfg
from rma_go2_lab.envs.teacher.rough_v3_cfg import (
    TeacherDynamicsPrivilegedObsCfg,
    TrackedRandomizeRigidBodyMass,
    TrackedRandomizeRigidBodyMaterial,
)
from rma_go2_lab.envs.teacher.terrain_lite import TeacherTerrainLitePrivilegedObsCfg


@configclass
class Go2AdaptationStudentHistoryStageATerrainLiteRoughEnvCfg(Go2AdaptationStudentRoughEnvCfg):
    """Stage A with proprio history, dynamics privilege, and compact terrain labels."""

    adaptation_history_length: int = 20

    def __post_init__(self):
        super().__post_init__()

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
        self.observations.dynamics_privileged = TeacherDynamicsPrivilegedObsCfg()
        self.observations.terrain_lite_privileged = TeacherTerrainLitePrivilegedObsCfg()
        self.adaptation_switch_episode_prob = 0.0
        print("\n========== ADAPTATION STUDENT HISTORY ROUGH STAGE A (TERRAIN LITE) ==========\n")
