"""Blind rough-terrain env with flattened proprioceptive history."""

from __future__ import annotations

import copy

from isaaclab.utils import configclass

from rma_go2_lab.envs.blind.rough_cfg import Go2BlindBaselineRoughEnvCfg


@configclass
class Go2BlindBaselineHistoryRoughEnvCfg(Go2BlindBaselineRoughEnvCfg):
    """Blind rough env augmented with deployable proprioceptive history."""

    policy_history_length: int = 20

    def __post_init__(self):
        super().__post_init__()

        self.observations.policy_history = copy.deepcopy(self.observations.policy)
        self.observations.policy_history.history_length = self.policy_history_length
        self.observations.policy_history.flatten_history_dim = True
        self.observations.policy_history.enable_corruption = False
        self.observations.policy_history.concatenate_terms = True

        print("\n========== BLIND BASELINE HISTORY ROUGH ==========\n")
