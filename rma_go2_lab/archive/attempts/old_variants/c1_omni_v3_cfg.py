"""Archived C1 omni v3 with explicit per-foot contact and timing-phase observations."""

from __future__ import annotations

import copy

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from rma_go2_lab.envs.blind.c1_omni_v1_cfg import Go2C1OmniV1EnvCfg
from rma_go2_lab.envs.contact_observations import foot_contact_state, foot_phase_features


@configclass
class Go2C1OmniV3EnvCfg(Go2C1OmniV1EnvCfg):
    """Archived third C1 omni branch with contact-state and phase-like timing features."""

    def __post_init__(self):
        super().__post_init__()

        contact_term = ObsTerm(
            func=foot_contact_state,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
                "threshold": 1.0,
            },
        )
        phase_term = ObsTerm(
            func=foot_phase_features,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
                "stance_scale_s": 0.30,
                "swing_scale_s": 0.20,
            },
        )

        self.observations.policy.foot_contact_state = contact_term
        self.observations.policy.foot_phase_features = phase_term
        self.observations.policy_history.foot_contact_state = copy.deepcopy(contact_term)
        self.observations.policy_history.foot_phase_features = copy.deepcopy(phase_term)

        print("\n========== C1 OMNI V3 STAGEA (FOOT CONTACT + PHASE) ==========\n")
