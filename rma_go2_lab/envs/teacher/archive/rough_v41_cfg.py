"""Archived privileged teacher rough-terrain environment, V4.1.

Teacher V4.1 is the smallest stair-focused follow-up to stable V4:

- keep the V4/V3 hidden-dynamics recipe intact
- keep the same startup difficulty and curriculum entry point
- reintroduce only a modest amount of stair terrain
- leave boxes disabled so the intervention isolates the stair bottleneck
"""

from __future__ import annotations

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.utils import configclass

from rma_go2_lab.envs.teacher.rough_v3_cfg import Go2PrivilegedTeacherRoughV3EnvCfg


@configclass
class Go2PrivilegedTeacherRoughV41EnvCfg(Go2PrivilegedTeacherRoughV3EnvCfg):
    """Teacher V4.1: stable V4 recipe plus a modest stair-only terrain bias."""

    def __post_init__(self):
        super().__post_init__()

        print("\n========== PRIVILEGED TEACHER ROUGH V4.1 ==========\n")

        # Match the stable blind/V4 startup regime and only change the terrain
        # family mix so this branch isolates the stair bottleneck.
        self.scene.terrain.max_init_terrain_level = 2
        self.curriculum.terrain_levels.func = mdp.terrain_levels_vel

        terrain_gen = self.scene.terrain.terrain_generator
        terrain_gen.sub_terrains["pyramid_stairs"].proportion = 0.05
        terrain_gen.sub_terrains["pyramid_stairs_inv"].proportion = 0.05
        terrain_gen.sub_terrains["boxes"].proportion = 0.0
        terrain_gen.sub_terrains["random_rough"].proportion = 0.2
        terrain_gen.sub_terrains["hf_pyramid_slope"].proportion = 0.1
        terrain_gen.sub_terrains["hf_pyramid_slope_inv"].proportion = 0.1
