"""Canonical per-episode terrain-lite history env for active Adapt-V3."""

from __future__ import annotations

from isaaclab.utils import configclass

from rma_go2_lab.envs.adaptation.rough_history_stage_a_terrain_lite_cfg import (
    Go2AdaptationStudentHistoryStageATerrainLiteRoughEnvCfg,
)


@configclass
class Go2AdaptationStudentHistoryPerEpisodeTerrainLiteRoughEnvCfg(
    Go2AdaptationStudentHistoryStageATerrainLiteRoughEnvCfg
):
    """Canonical terrain-aware student env under per-episode domain randomization."""

    def __post_init__(self):
        super().__post_init__()
        self.adaptation_switch_episode_prob = 0.0
        print("\n========== ADAPT-V3 TERRAIN-LITE STUDENT (PER-EPISODE DR) ==========\n")
