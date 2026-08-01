"""Archived flat omni prior augmented with explicit per-foot contact observations."""

from __future__ import annotations

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from rma_go2_lab.envs.contact_observations import foot_contact_state
from rma_go2_lab.envs.priors.flat_omni_v1_cfg import Go2FlatOmniV1EnvCfg


@configclass
class Go2FlatOmniContactV1EnvCfg(Go2FlatOmniV1EnvCfg):
    """Archived flat omni prior with explicit per-foot contact state in the policy obs."""

    def __post_init__(self):
        super().__post_init__()

        self.observations.policy.foot_contact_state = ObsTerm(
            func=foot_contact_state,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
                "threshold": 1.0,
            },
        )

        print("\n========== RMA GO2 FLAT OMNI CONTACT PRIOR V1 ==========\n")
