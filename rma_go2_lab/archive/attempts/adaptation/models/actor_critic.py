from __future__ import annotations

from collections import OrderedDict
from typing import Any

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal

from rsl_rl.modules import ActorCritic
from rsl_rl.networks import EmpiricalNormalization, MLP


class HistoryEncoderStudentActorCritic(ActorCritic):
    """Future adaptation student with a learned history latent.

    This scaffold is intentionally simple:

    - current deployable proprio group remains explicit as `policy`
    - a second observation group carries flattened observation history
    - a small encoder maps history -> latent `z_hat_t`
    - actor/critic consume `policy + z_hat_t`

    The class is ready for future training once the repo decides the final
    supervision path (pure PPO, imitation, latent regression, or hybrid).
    """

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: tuple[int] | list[int] = (256, 256, 256),
        critic_hidden_dims: tuple[int] | list[int] = (256, 256, 256),
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        state_dependent_std: bool = False,
        latent_dim: int = 8,
        history_encoder_hidden_dims: tuple[int] | list[int] = (256, 128),
        history_group_name: str = "policy_history",
        actor_init_path: str | None = None,
        **kwargs: dict[str, Any],
    ) -> None:
        nn.Module.__init__(self)
        if kwargs:
            print(
                "HistoryEncoderStudentActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs])
            )

        self.obs_groups = obs_groups
        self.state_dependent_std = state_dependent_std
        self.noise_std_type = noise_std_type
        self.history_group_name = history_group_name

        self._actor_has_history = history_group_name in obs_groups["policy"]
        self._critic_has_history = history_group_name in obs_groups["critic"]
        if not (self._actor_has_history or self._critic_has_history):
            raise ValueError(
                f"HistoryEncoderStudentActorCritic expects observation group '{history_group_name}' "
                "to appear in either policy or critic mappings."
            )

        history_obs = obs[history_group_name]
        assert len(history_obs.shape) == 2, "History observations must be flattened per environment."
        history_dim = history_obs.shape[-1]

        self.history_encoder = MLP(history_dim, latent_dim, history_encoder_hidden_dims, activation)
        print(f"History encoder: {self.history_encoder}")

        actor_non_history_dim = 0
        for obs_group in obs_groups["policy"]:
            assert len(obs[obs_group].shape) == 2, "The ActorCritic module only supports 1D observations."
            if obs_group != history_group_name:
                actor_non_history_dim += obs[obs_group].shape[-1]

        critic_non_history_dim = 0
        for obs_group in obs_groups["critic"]:
            assert len(obs[obs_group].shape) == 2, "The ActorCritic module only supports 1D observations."
            if obs_group != history_group_name:
                critic_non_history_dim += obs[obs_group].shape[-1]

        num_actor_obs = actor_non_history_dim + (latent_dim if self._actor_has_history else 0)
        num_critic_obs = critic_non_history_dim + (latent_dim if self._critic_has_history else 0)

        if self.state_dependent_std:
            self.actor = MLP(num_actor_obs, [2, num_actions], actor_hidden_dims, activation)
        else:
            self.actor = MLP(num_actor_obs, num_actions, actor_hidden_dims, activation)
        print(f"Actor MLP: {self.actor}")

        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs)
        else:
            self.actor_obs_normalizer = nn.Identity()

        self.critic = MLP(num_critic_obs, 1, critic_hidden_dims, activation)
        print(f"Critic MLP: {self.critic}")

        self.critic_obs_normalization = critic_obs_normalization
        if critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(num_critic_obs)
        else:
            self.critic_obs_normalizer = nn.Identity()

        if self.state_dependent_std:
            torch.nn.init.zeros_(self.actor[-2].weight[num_actions:])
            if self.noise_std_type == "scalar":
                torch.nn.init.constant_(self.actor[-2].bias[num_actions:], init_noise_std)
            elif self.noise_std_type == "log":
                torch.nn.init.constant_(
                    self.actor[-2].bias[num_actions:], torch.log(torch.tensor(init_noise_std + 1e-7))
                )
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        else:
            if self.noise_std_type == "scalar":
                self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
            elif self.noise_std_type == "log":
                self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        self.distribution = None
        Normal.set_default_validate_args(False)

        if actor_init_path:
            self.load_actor_only(actor_init_path)

    def _encode_obs_groups(self, obs: TensorDict, group_names: list[str]) -> torch.Tensor:
        obs_list = []
        for group_name in group_names:
            if group_name == self.history_group_name:
                obs_list.append(self.encode_history_latent(obs))
            else:
                obs_list.append(obs[group_name])
        return torch.cat(obs_list, dim=-1)

    def encode_history_latent(self, obs: TensorDict) -> torch.Tensor:
        """Return the student adaptation latent z_hat_t from history."""
        return self.history_encoder(obs[self.history_group_name])

    def get_actor_obs(self, obs: TensorDict) -> torch.Tensor:
        return self._encode_obs_groups(obs, self.obs_groups["policy"])

    def get_critic_obs(self, obs: TensorDict) -> torch.Tensor:
        return self._encode_obs_groups(obs, self.obs_groups["critic"])

    def load_actor_only(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint["model_state_dict"]

        actor_state = OrderedDict()
        for key, value in state_dict.items():
            if key.startswith("actor."):
                actor_state[key[len("actor.") :]] = value

        if not actor_state:
            raise RuntimeError(f"No actor weights found in checkpoint: {checkpoint_path}")

        self._small_init_module(self.history_encoder)
        self._partial_copy_mlp(self.actor, actor_state)
        print(
            f"[INFO] Warm-started adaptation student actor from {checkpoint_path}. "
            "History encoder keeps small non-zero initialization so it remains trainable."
        )

    @staticmethod
    def _small_init_module(module: nn.Module, gain: float = 0.1) -> None:
        for submodule in module.modules():
            if isinstance(submodule, nn.Linear):
                nn.init.xavier_uniform_(submodule.weight, gain=gain)
                if submodule.bias is not None:
                    nn.init.zeros_(submodule.bias)

    @staticmethod
    def _copy_if_present(target_tensor: torch.Tensor, source_state: dict[str, torch.Tensor], key: str) -> bool:
        if key not in source_state:
            return False
        source_tensor = source_state[key]
        if target_tensor.shape == source_tensor.shape:
            with torch.no_grad():
                target_tensor.copy_(source_tensor)
            return True
        return False

    def _partial_copy_mlp(self, module: nn.Module, source_state: dict[str, torch.Tensor]) -> None:
        first_weight_key = "0.weight"
        first_bias_key = "0.bias"
        if first_weight_key in source_state:
            source_weight = source_state[first_weight_key]
            target_weight = module[0].weight
            with torch.no_grad():
                nn.init.xavier_uniform_(target_weight, gain=0.1)
                shared_in = min(source_weight.shape[1], target_weight.shape[1])
                shared_out = min(source_weight.shape[0], target_weight.shape[0])
                target_weight[:shared_out, :shared_in].copy_(source_weight[:shared_out, :shared_in])
        if first_bias_key in source_state:
            source_bias = source_state[first_bias_key]
            target_bias = module[0].bias
            shared = min(source_bias.shape[0], target_bias.shape[0])
            with torch.no_grad():
                if target_bias is not None:
                    target_bias.zero_()
                target_bias[:shared].copy_(source_bias[:shared])

        for idx, layer in enumerate(module):
            if idx == 0 or not isinstance(layer, nn.Linear):
                continue
            self._copy_if_present(layer.weight, source_state, f"{idx}.weight")
            self._copy_if_present(layer.bias, source_state, f"{idx}.bias")
