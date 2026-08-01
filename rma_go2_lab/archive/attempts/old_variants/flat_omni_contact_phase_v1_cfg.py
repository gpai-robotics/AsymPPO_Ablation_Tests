"""Archived flat omni prior with explicit per-foot contact and timing-phase observations."""

from __future__ import annotations

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from rma_go2_lab.envs.contact_observations import foot_contact_state, foot_phase_features
from rma_go2_lab.envs.priors.flat_omni_v1_cfg import Go2FlatOmniV1EnvCfg


@configclass
class Go2FlatOmniContactPhaseV1EnvCfg(Go2FlatOmniV1EnvCfg):
    """Archived flat omni prior with contact state plus per-foot timing-phase features."""

    def __post_init__(self):
        super().__post_init__()

        self.observations.policy.foot_contact_state = ObsTerm(
            func=foot_contact_state,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
                "threshold": 1.0,
            },
        )
        self.observations.policy.foot_phase_features = ObsTerm(
            func=foot_phase_features,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
                "stance_scale_s": 0.30,
                "swing_scale_s": 0.20,
            },
        )

        print("\n========== RMA GO2 FLAT OMNI CONTACT+PHASE PRIOR V1 ==========\n")
