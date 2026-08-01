"""Privileged teacher rough-terrain environment.

Teacher V0 is the smallest fair extension beyond the frozen blind ladder:

- same rough terrain family
- same command regime
- same rewards and terminations
- actor/critic receive an additional privileged local terrain height scan
"""

from __future__ import annotations

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils import configclass

from rma_go2_lab.envs.blind.rough_cfg import Go2BlindBaselineRoughEnvCfg


@configclass
class TeacherPrivilegedObsCfg(ObsGroup):
    """Privileged observations available only to the teacher path."""

    height_scan = ObsTerm(
        func=mdp.height_scan,
        params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        clip=(-1.0, 1.0),
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class Go2PrivilegedTeacherRoughEnvCfg(Go2BlindBaselineRoughEnvCfg):
    """Frozen-blind-matched rough environment with terrain privilege enabled."""

    def __post_init__(self):
        super().__post_init__()

        print("\n========== PRIVILEGED TEACHER ROUGH V0 ==========\n")

        # Restore the terrain scanner that the blind ladder intentionally
        # disabled. This is the only new information channel in Teacher V0.
        self.scene.height_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base",
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
            ray_alignment="yaw",
            pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        # Keep the nominal policy group proprioceptive so the privilege remains
        # explicit and separately mappable in the runner config.
        self.observations.policy.height_scan = None
        self.observations.privileged = TeacherPrivilegedObsCfg()
