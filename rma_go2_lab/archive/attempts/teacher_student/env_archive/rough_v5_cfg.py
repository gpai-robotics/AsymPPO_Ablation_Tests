"""Archived privileged teacher rough-terrain environment, V5.

Teacher V5 is a controlled follow-up to V4:

- keep the V4 dynamics randomization and startup difficulty intact
- introduce only a modest geometry bias toward boxes and stairs
- avoid a full domain shift so warm-start remains stable and comparable
"""

from __future__ import annotations

from isaaclab.managers import EventTermCfg, SceneEntityCfg
from isaaclab.utils import configclass
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from rma_go2_lab.envs.teacher.rough_v3_cfg import Go2PrivilegedTeacherRoughV3EnvCfg


@configclass
class Go2PrivilegedTeacherRoughV5EnvCfg(Go2PrivilegedTeacherRoughV3EnvCfg):
    """Teacher V5: V4-compatible regime plus a mild geometry emphasis."""

    def __post_init__(self):
        super().__post_init__()

        print("\n========== PRIVILEGED TEACHER ROUGH V5 ==========\n")

        # Keep the same startup difficulty as V4 so the warm-started blind
        # policy is not shocked by a sudden terrain-depth jump.
        self.scene.terrain.max_init_terrain_level = 2
        self.curriculum.terrain_levels.func = mdp.terrain_levels_vel

        terrain_gen = self.scene.terrain.terrain_generator
        terrain_gen.sub_terrains["random_rough"].proportion = 0.15
        terrain_gen.sub_terrains["boxes"].proportion = 0.10
        terrain_gen.sub_terrains["pyramid_stairs"].proportion = 0.05
        terrain_gen.sub_terrains["pyramid_stairs_inv"].proportion = 0.05
        terrain_gen.sub_terrains["hf_pyramid_slope"].proportion = 0.10
        terrain_gen.sub_terrains["hf_pyramid_slope_inv"].proportion = 0.10

        # Keep V4/V3 hidden-dynamics randomization unchanged so this branch
        # isolates geometry-mix pressure rather than rewriting the full recipe.
