from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict

from rma_go2_lab.models.adaptation.rma_v3_actor_critic import RmaV3ActorCritic


class FrozenAdaptV3Phase1(nn.Module):
    """Read-only Phase 1 V3 reference for Phase 2 latent supervision."""

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cpu",
        terrain_group_name: str | None = "terrain_privileged",
        dynamics_group_name: str = "dynamics_privileged",
        terrain_dim: int | None = None,
    ) -> None:
        super().__init__()

        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint["model_state_dict"]

        latent_dim = int(state_dict["extrinsics_encoder.0.weight"].shape[0])
        extrinsics_first_in = int(state_dict["extrinsics_encoder.0.weight"].shape[1])
        extrinsics_hidden_dims = self._infer_hidden_dims(state_dict, "extrinsics_encoder.")
        extrinsics_encoder_mode = "linear" if len(extrinsics_hidden_dims) == 0 else "mlp"
        dynamics_decoder_hidden_dims = self._infer_hidden_dims(state_dict, "dynamics_decoder.")
        dynamics_decoder_mode = "linear" if len(dynamics_decoder_hidden_dims) == 0 else "mlp"

        inferred_terrain_dim = 0 if terrain_group_name is None else max(0, extrinsics_first_in - 27)
        resolved_terrain_dim = terrain_dim if terrain_dim is not None else inferred_terrain_dim

        dummy_obs_dict = {
            "policy": torch.zeros(1, 48),
            "policy_history": torch.zeros(1, 960),
            dynamics_group_name: torch.zeros(1, 27),
        }
        if terrain_group_name is not None:
            dummy_obs_dict[terrain_group_name] = torch.zeros(1, resolved_terrain_dim)
        dummy_obs = TensorDict(dummy_obs_dict, batch_size=[1])

        policy_groups = ["policy"]
        if terrain_group_name is not None:
            policy_groups.append(terrain_group_name)
        policy_groups.append(dynamics_group_name)
        obs_groups = {
            "policy": policy_groups,
            "critic": list(policy_groups),
        }
        self.policy = RmaV3ActorCritic(
            obs=dummy_obs,
            obs_groups=obs_groups,
            num_actions=12,
            init_noise_std=0.35,
            actor_obs_normalization=False,
            critic_obs_normalization=False,
            actor_hidden_dims=[512, 256, 128],
            critic_hidden_dims=[512, 256, 128],
            activation="elu",
            latent_dim=latent_dim,
            extrinsics_encoder_hidden_dims=extrinsics_hidden_dims,
            extrinsics_encoder_mode=extrinsics_encoder_mode,
            adaptation_hidden_dims=[256, 128],
            dynamics_decoder_hidden_dims=dynamics_decoder_hidden_dims,
            dynamics_decoder_mode=dynamics_decoder_mode,
            policy_group_name="policy",
            history_group_name="policy_history",
            terrain_group_name=terrain_group_name,
            dynamics_group_name=dynamics_group_name,
        )

        self.policy.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.policy.eval()
        for param in self.policy.parameters():
            param.requires_grad_(False)

    @staticmethod
    def _infer_hidden_dims(state_dict: dict[str, torch.Tensor], prefix: str) -> list[int]:
        linear_indices: list[int] = []
        for key in state_dict.keys():
            if key.startswith(prefix) and key.endswith(".weight"):
                try:
                    linear_indices.append(int(key[len(prefix) :].split(".")[0]))
                except ValueError:
                    continue
        if not linear_indices:
            return []
        linear_indices = sorted(set(linear_indices))
        hidden_dims: list[int] = []
        for idx in linear_indices[:-1]:
            weight = state_dict[f"{prefix}{idx}.weight"]
            hidden_dims.append(int(weight.shape[0]))
        return hidden_dims

    def forward(self, obs: TensorDict) -> torch.Tensor:
        return self.policy.act_inference(obs)

    def encode_extrinsics_latent(self, obs: TensorDict) -> torch.Tensor:
        return self.policy.encode_extrinsics_latent(obs)
