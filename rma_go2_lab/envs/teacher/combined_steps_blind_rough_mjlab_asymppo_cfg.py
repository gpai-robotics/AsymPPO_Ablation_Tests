"""Combined AsymPPO stair fine-tune stage.

Stage 3 of the combined branch fine-tunes the rough-stage policy on stairs-only
terrain.

Do not use this task name for the frozen validated hardware candidate.  That
remains ``Go2-Blind-Rough-MJLAB-AsymPPO-V1``.
"""

from __future__ import annotations

from isaaclab.utils import configclass

from rma_go2_lab.envs.teacher.combined_rough_blind_mjlab_asymppo_cfg import (
    Go2BlindRoughMjlabCombinedRoughEnvCfg,
)


@configclass
class Go2BlindRoughMjlabCombinedStepsEnvCfg(Go2BlindRoughMjlabCombinedRoughEnvCfg):
    """Stage 3: rough policy fine-tuned on stairs and inverted stairs."""

    def __post_init__(self):
        super().__post_init__()

        terrain_gen = self.scene.terrain.terrain_generator
        self.scene.terrain.max_init_terrain_level = 1
        terrain_gen.sub_terrains["random_rough"].proportion = 0.0
        terrain_gen.sub_terrains["hf_pyramid_slope"].proportion = 0.0
        terrain_gen.sub_terrains["hf_pyramid_slope_inv"].proportion = 0.0
        terrain_gen.sub_terrains["pyramid_stairs"].proportion = 0.5
        terrain_gen.sub_terrains["pyramid_stairs_inv"].proportion = 0.5
        terrain_gen.sub_terrains["boxes"].proportion = 0.0

        for terrain_name in ("pyramid_stairs", "pyramid_stairs_inv"):
            terrain_cfg = terrain_gen.sub_terrains[terrain_name]
            terrain_cfg.step_height_range = (0.12, 0.12)
            terrain_cfg.step_width = 0.30
            terrain_cfg.platform_width = 3.0
        

        self.rewards.feet_air_time.weight = 0.5
        self.rewards.lin_vel_z_l2.weight = -0.5
        self.rewards.stable_progress.weight = 0.5
        self.rewards.adaptive_swing_recovery.weight = 0.25

        print("\n========== GO2 COMBINED ASYMPPO STEPS V1 ==========\n")
