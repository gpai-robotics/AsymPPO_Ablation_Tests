"""Archived C1 omni v2 with explicit per-foot contact observations."""

from __future__ import annotations

import copy

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from rma_go2_lab.envs.blind.c1_omni_v1_cfg import Go2C1OmniV1EnvCfg
from rma_go2_lab.envs.contact_observations import foot_contact_state


@configclass
class Go2C1OmniV2EnvCfg(Go2C1OmniV1EnvCfg):
    """Archived second C1 omni branch with per-foot contact in policy and history obs."""

    def __post_init__(self):
        super().__post_init__()

        contact_term = ObsTerm(
            func=foot_contact_state,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
                "threshold": 1.0,
            },
        )
        self.observations.policy.foot_contact_state = contact_term
        self.observations.policy_history.foot_contact_state = copy.deepcopy(contact_term)

        print("\n========== C1 OMNI V2 STAGEA (FOOT CONTACT) ==========\n")
