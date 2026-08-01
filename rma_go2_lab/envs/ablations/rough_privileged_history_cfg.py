"""C1 ETH-like blind history env with teacher-only privileged groups."""

from __future__ import annotations

from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils import configclass

from rma_go2_lab.envs.ablations.rough_history_base_cfg import Go2AsymPpoHistoryBaseEnvCfg
from rma_go2_lab.envs.teacher.rough_cfg import TeacherPrivilegedObsCfg
from rma_go2_lab.envs.teacher.rough_v3_cfg import (
    TeacherDynamicsPrivilegedObsCfg,
    TrackedRandomizeRigidBodyMass,
    TrackedRandomizeRigidBodyMaterial,
)


@configclass
class Go2AsymPpoPrivilegedHistoryEnvCfg(Go2AsymPpoHistoryBaseEnvCfg):
    """Blind history student env with teacher-only terrain+dynamics privilege."""

    policy_history_length: int = 50

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

        # Keep the deployable student observations blind while exposing terrain
        # only through the teacher-side privileged group.
        self.observations.policy.height_scan = None
        self.events.physics_material.func = TrackedRandomizeRigidBodyMaterial
        if self.events.add_base_mass is not None:
            self.events.add_base_mass.func = TrackedRandomizeRigidBodyMass

        self.observations.terrain_privileged = TeacherPrivilegedObsCfg()
        self.observations.dynamics_privileged = TeacherDynamicsPrivilegedObsCfg()

        print("\n========== C1 ETH-LIKE V1 BLIND HISTORY ==========\n")
