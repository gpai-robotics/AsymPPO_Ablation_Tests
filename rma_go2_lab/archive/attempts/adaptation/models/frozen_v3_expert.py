from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict

from rma_go2_lab.models.teacher.actor_critic import TerrainEncoderActorCritic


class FrozenV3Expert(nn.Module):
    """Read-only privileged teacher rebuilt from a frozen V3/V4-style checkpoint."""

    def __init__(self, checkpoint_path: str, device: str = "cpu") -> None:
        super().__init__()

        dummy_obs = TensorDict(
            {
                "policy": torch.zeros(1, 48),
                "dynamics_privileged": torch.zeros(1, 27),
                "terrain_privileged": torch.zeros(1, 187),
            },
            batch_size=[1],
        )
        obs_groups = {
            "policy": ["policy", "dynamics_privileged", "terrain_privileged"],
            "critic": ["policy", "dynamics_privileged", "terrain_privileged"],
        }
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint["model_state_dict"]

        has_terrain_target_head = any(key.startswith("terrain_target_head.") for key in state_dict.keys())
        self.policy = TerrainEncoderActorCritic(
            obs=dummy_obs,
            obs_groups=obs_groups,
            num_actions=12,
            init_noise_std=0.35,
            actor_obs_normalization=False,
            critic_obs_normalization=False,
            actor_hidden_dims=[512, 256, 128],
            critic_hidden_dims=[512, 256, 128],
            activation="elu",
            terrain_latent_dim=8,
            terrain_encoder_hidden_dims=[64, 32],
            terrain_target_dim=13 if has_terrain_target_head else None,
            terrain_target_hidden_dims=[64] if has_terrain_target_head else None,
            privileged_group_name="terrain_privileged",
        )

        self.policy.load_state_dict(state_dict, strict=True)
        self.policy.eval()
        for param in self.policy.parameters():
            param.requires_grad_(False)

        # Adapt-V1 targets a live internal teacher feature instead of the
        # terrain encoder output. The saved V3 checkpoint's terrain encoder is
        # fully zeroed, so its latent is not a useful supervision target.
        self.latent_target_name = "teacher_actor_penultimate"
        self.latent_target_dim = 128

    def forward(self, obs: TensorDict) -> torch.Tensor:
        return self.policy.act_inference(obs)

    def encode_terrain_latent(self, obs: TensorDict) -> torch.Tensor:
        """Return the frozen teacher's compact terrain latent."""
        return self.policy.terrain_encoder(obs[self.policy.privileged_group_name])

    def encode_actor_penultimate(self, obs: TensorDict) -> torch.Tensor:
        """Return the frozen teacher's penultimate actor feature."""
        x = self.policy.get_actor_obs(obs)
        x = self.policy.actor_obs_normalizer(x)
        actor_layers = list(self.policy.actor.children())
        for layer in actor_layers[:-1]:
            x = layer(x)
        return x

    def get_latent_target(self, obs: TensorDict) -> torch.Tensor:
        """Return the canonical explicit latent target for Adapt-V1.

        Current choice:
        - 128-dim penultimate actor feature from the frozen teacher

        This stays on a path that the teacher actually uses to produce actions,
        which makes it safer than supervising against a dead encoder output.
        """
        return self.encode_actor_penultimate(obs)
