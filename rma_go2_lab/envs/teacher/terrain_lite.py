"""Compact privileged terrain descriptors derived from IsaacLab height scans."""

from __future__ import annotations

import torch

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from rma_go2_lab.models.teacher.terrain_targets import terrain_lite_from_scan


def terrain_lite_scan(
    env,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"),
) -> torch.Tensor:
    """Compress the 187-point height scan into gait-relevant terrain cues.

    The scanner remains the privileged simulator-only source. This observation
    exposes compact features that should be easier for a blind history encoder
    to infer than an arbitrary learned embedding of the full local map.
    """

    import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

    scan = mdp.height_scan(env, sensor_cfg=sensor_cfg)
    return terrain_lite_from_scan(scan)


@configclass
class TeacherTerrainLitePrivilegedObsCfg(ObsGroup):
    """Student-inferable terrain privilege derived from the local height scan."""

    terrain_lite = ObsTerm(
        func=terrain_lite_scan,
        params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        clip=(-1.0, 1.0),
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True
