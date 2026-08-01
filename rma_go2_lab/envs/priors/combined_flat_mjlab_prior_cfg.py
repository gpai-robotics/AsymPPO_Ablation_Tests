"""Flat prior for the combined AsymPPO branch."""

from __future__ import annotations

from isaaclab.utils import configclass

from rma_go2_lab.envs.priors.flat_mjlab_prior_cfg import Go2FlatMjlabPriorEnvCfg


@configclass
class Go2CombinedFlatMjlabPriorEnvCfg(Go2FlatMjlabPriorEnvCfg):
    """Stage 1 flat prior under the deploy-honest MJLAB actor contract."""

    def __post_init__(self):
        super().__post_init__()
        print("\n========== GO2 COMBINED FLAT MJLAB PRIOR V1 ==========\n")
