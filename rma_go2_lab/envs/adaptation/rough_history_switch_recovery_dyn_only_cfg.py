"""Low-switch-probability dyn-only history env for Adapt-V3 recovery work."""

from __future__ import annotations

from isaaclab.utils import configclass

from rma_go2_lab.envs.adaptation.rough_history_stage_a_cfg import (
    Go2AdaptationStudentHistoryStageARoughEnvCfg,
)


@configclass
class Go2AdaptationStudentHistoryDynOnlyRecoverySwitchLowProbEnvCfg(
    Go2AdaptationStudentHistoryStageARoughEnvCfg
):
    """Cautious dyn-only recovery env with rare within-episode hidden switches.

    The goal is to reintroduce genuine adaptation pressure without jumping
    straight back to the more aggressive switch probabilities that previously
    degraded locomotion.
    """

    def __post_init__(self):
        super().__post_init__()
        self.adaptation_switch_episode_prob = 0.05
        self.adaptation_switch_step = 500
        print("\n========== ADAPT-V3 DYN-ONLY RECOVERY (LOW SWITCH PROB) ==========\n")
