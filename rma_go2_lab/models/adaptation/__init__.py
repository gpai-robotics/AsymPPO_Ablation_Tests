"""Adaptation-phase model configs."""

from .actor_critic import HistoryEncoderStudentActorCritic
from .frozen_adapt_v3_phase1 import FrozenAdaptV3Phase1
from .modular_actor_critic import ModularHistoryAdaptationActorCritic
from .ppo_rma_v3_phase1 import RmaV3Phase1PPO
from .ppo_rma_v3_phase2 import RmaV3Phase2PPO
from .frozen_v3_expert import FrozenV3Expert
from .ppo_with_v3_expert import AdaptationPPOWithV3Expert
from .ppo_with_v3_latent import AdaptationPPOWithV3Latent
from .rma_v3_actor_critic import RmaV3ActorCritic

__all__ = [
    "HistoryEncoderStudentActorCritic",
    "FrozenAdaptV3Phase1",
    "ModularHistoryAdaptationActorCritic",
    "RmaV3Phase1PPO",
    "RmaV3Phase2PPO",
    "FrozenV3Expert",
    "AdaptationPPOWithV3Expert",
    "AdaptationPPOWithV3Latent",
    "RmaV3ActorCritic",
]
