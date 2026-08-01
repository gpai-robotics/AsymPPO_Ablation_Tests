from __future__ import annotations

from collections import OrderedDict
from typing import Any

import torch
import torch.nn as nn
from torch.distributions import Normal

try:
    from tensordict import TensorDict
except ModuleNotFoundError:
    TensorDict = dict[str, torch.Tensor]  # type: ignore[misc,assignment]

from rsl_rl.modules import ActorCritic
from rsl_rl.networks import EmpiricalNormalization, MLP


class RmaV3ActorCritic(ActorCritic):
    """Faithful V3 actor-critic with explicit mu / pi / phi decomposition.

    The important contract is:

    - mu:
        e_t -> z_t
    - phi:
        history -> z_hat_t
    - pi:
        x_t, a_{t-1}, z -> action

    This same module is used for both Phase 1 and Phase 2 by switching which
    *real* environment groups appear in the actor/critic observation mapping:

    - Phase 1:
        ``policy + terrain_privileged + dynamics_privileged``
        becomes ``policy + mu(e_t)``
    - Phase 2:
        ``policy + policy_history``
        becomes ``policy + phi(history)``
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
        latent_dim: int = 32,
        extrinsics_encoder_hidden_dims: tuple[int] | list[int] = (128, 64),
        extrinsics_encoder_mode: str = "mlp",
        extrinsics_identity_init: bool = False,
        dynamics_decoder_hidden_dims: tuple[int] | list[int] = (64,),
        dynamics_decoder_mode: str = "mlp",
        dynamics_decoder_identity_init: bool = False,
        adaptation_hidden_dims: tuple[int] | list[int] = (256, 128),
        adaptation_bottleneck_dim: int | None = None,
        adaptation_decoder_hidden_dims: tuple[int] | list[int] = (),
        adaptation_residual_mode: bool = False,
        adaptation_residual_scale: float = 1.0,
        adaptation_encoder_type: str = "mlp",
        temporal_channels: tuple[int] | list[int] = (64, 64),
        temporal_kernel_size: int = 3,
        history_feature_dim: int = 64,
        policy_group_name: str = "policy",
        history_group_name: str = "policy_history",
        terrain_group_name: str | None = "terrain_privileged",
        dynamics_group_name: str = "dynamics_privileged",
        full_init_path: str | None = None,
        actor_init_path: str | None = None,
        critic_init_path: str | None = None,
        extrinsics_init_path: str | None = None,
        actor_only_extrinsics_init_mode: str = "zero",
        actor_only_adaptation_init_mode: str = "zero",
        **kwargs: dict[str, Any],
    ) -> None:
        nn.Module.__init__(self)
        if kwargs:
            print(
                "RmaV3ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs])
            )

        self.obs_groups = obs_groups
        self.state_dependent_std = state_dependent_std
        self.noise_std_type = noise_std_type
        self.policy_group_name = policy_group_name
        self.history_group_name = history_group_name
        self.terrain_group_name = terrain_group_name
        self.dynamics_group_name = dynamics_group_name
        self.latent_dim = latent_dim
        self.adaptation_encoder_type = adaptation_encoder_type
        self.adaptation_bottleneck_dim = adaptation_bottleneck_dim
        self.adaptation_residual_mode = adaptation_residual_mode
        self.adaptation_residual_scale = float(adaptation_residual_scale)
        self.actor_only_extrinsics_init_mode = actor_only_extrinsics_init_mode
        self.actor_only_adaptation_init_mode = actor_only_adaptation_init_mode
        self.latent_mode = "normal"
        self.extrinsics_group_names = tuple(
            group_name for group_name in (terrain_group_name, dynamics_group_name) if group_name is not None
        )

        if policy_group_name not in obs.keys():
            raise ValueError(f"RmaV3ActorCritic requires observation group '{policy_group_name}'.")
        if history_group_name not in obs.keys():
            raise ValueError(f"RmaV3ActorCritic requires observation group '{history_group_name}'.")
        if terrain_group_name is not None and terrain_group_name not in obs.keys():
            raise ValueError(f"RmaV3ActorCritic requires observation group '{terrain_group_name}'.")
        if dynamics_group_name not in obs.keys():
            raise ValueError(f"RmaV3ActorCritic requires observation group '{dynamics_group_name}'.")

        history_obs = obs[history_group_name]
        assert len(history_obs.shape) == 2, "History observations must be flattened per environment."
        history_dim = history_obs.shape[-1]

        current_policy_obs = obs[policy_group_name]
        assert len(current_policy_obs.shape) == 2, "Current policy observations must be flattened per environment."
        self.current_policy_dim = current_policy_obs.shape[-1]
        if history_dim % self.current_policy_dim != 0:
            raise ValueError(
                f"History dim {history_dim} is not divisible by current policy dim {self.current_policy_dim}."
            )
        self.history_dim = history_dim
        self.history_length = history_dim // self.current_policy_dim

        terrain_dim = obs[terrain_group_name].shape[-1] if terrain_group_name is not None else 0
        dynamics_dim = obs[dynamics_group_name].shape[-1]
        extrinsics_dim = terrain_dim + dynamics_dim
        self.terrain_dim = terrain_dim
        self.dynamics_dim = dynamics_dim
        self.extrinsics_dim = extrinsics_dim
        self.extrinsics_summary_dim = dynamics_dim + (4 if terrain_group_name is not None else 0)

        if extrinsics_encoder_mode == "mlp":
            self.extrinsics_encoder = MLP(extrinsics_dim, latent_dim, extrinsics_encoder_hidden_dims, activation)
        elif extrinsics_encoder_mode == "linear":
            self.extrinsics_encoder = nn.Sequential(nn.Linear(extrinsics_dim, latent_dim))
        else:
            raise ValueError(
                f"Unknown extrinsics encoder mode: {extrinsics_encoder_mode}. "
                "Expected one of {'mlp', 'linear'}."
            )
        print(f"Extrinsics encoder (mu): {self.extrinsics_encoder}")

        if dynamics_decoder_mode == "mlp":
            self.dynamics_decoder = MLP(latent_dim, dynamics_dim, dynamics_decoder_hidden_dims, activation)
        elif dynamics_decoder_mode == "linear":
            self.dynamics_decoder = nn.Sequential(nn.Linear(latent_dim, dynamics_dim))
        else:
            raise ValueError(
                f"Unknown dynamics decoder mode: {dynamics_decoder_mode}. "
                "Expected one of {'mlp', 'linear'}."
            )
        print(f"Dynamics decoder: {self.dynamics_decoder}")

        if extrinsics_identity_init:
            self._init_identity_linear_stack(self.extrinsics_encoder, name="extrinsics encoder")
        if dynamics_decoder_identity_init:
            self._init_identity_linear_stack(self.dynamics_decoder, name="dynamics decoder")

        self.terrain_summary_decoder = None
        if terrain_group_name is not None:
            self.terrain_summary_decoder = MLP(latent_dim, 4, [32], activation)
            print(f"Terrain summary decoder: {self.terrain_summary_decoder}")

        self.temporal_encoder, self.history_projection, adaptation_input_dim = self._build_adaptation_preprocessor(
            adaptation_encoder_type=adaptation_encoder_type,
            history_dim=history_dim,
            history_feature_dim=history_feature_dim,
            temporal_channels=temporal_channels,
            temporal_kernel_size=temporal_kernel_size,
            prefix="Adaptation",
        )

        adaptation_output_dim = latent_dim if adaptation_bottleneck_dim is None else int(adaptation_bottleneck_dim)
        self.adaptation_module = MLP(adaptation_input_dim, adaptation_output_dim, adaptation_hidden_dims, activation)
        print(f"Adaptation module (phi): {self.adaptation_module}")
        self.adaptation_decoder: nn.Module | None = None
        if adaptation_bottleneck_dim is not None:
            self.adaptation_decoder = MLP(
                int(adaptation_bottleneck_dim),
                latent_dim,
                adaptation_decoder_hidden_dims,
                activation,
            )
            print(f"Adaptation decoder (phi-post): {self.adaptation_decoder}")

        self.base_temporal_encoder: nn.Module | None = None
        self.base_history_projection: nn.Module | None = None
        self.base_adaptation_module: nn.Module | None = None
        self.base_adaptation_decoder: nn.Module | None = None
        if self.adaptation_residual_mode:
            (
                self.base_temporal_encoder,
                self.base_history_projection,
                base_adaptation_input_dim,
            ) = self._build_adaptation_preprocessor(
                adaptation_encoder_type=adaptation_encoder_type,
                history_dim=history_dim,
                history_feature_dim=history_feature_dim,
                temporal_channels=temporal_channels,
                temporal_kernel_size=temporal_kernel_size,
                prefix="Frozen base adaptation",
            )
            self.base_adaptation_module = MLP(
                base_adaptation_input_dim,
                latent_dim,
                adaptation_hidden_dims,
                activation,
            )
            print(f"Frozen base adaptation module (phi-base): {self.base_adaptation_module}")

        actor_obs_dim = self._count_obs_dim(obs_groups["policy"], obs)
        critic_obs_dim = self._count_obs_dim(obs_groups["critic"], obs)

        if self.state_dependent_std:
            self.actor = MLP(actor_obs_dim, [2, num_actions], actor_hidden_dims, activation)
        else:
            self.actor = MLP(actor_obs_dim, num_actions, actor_hidden_dims, activation)
        print(f"Base policy actor (pi): {self.actor}")

        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(actor_obs_dim)
        else:
            self.actor_obs_normalizer = nn.Identity()

        self.critic = MLP(critic_obs_dim, 1, critic_hidden_dims, activation)
        print(f"Critic MLP: {self.critic}")

        self.critic_obs_normalization = critic_obs_normalization
        if critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(critic_obs_dim)
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

        if full_init_path:
            self.load_full_checkpoint(full_init_path)
        elif actor_init_path:
            self.load_actor_only(actor_init_path)
        elif critic_init_path:
            self.load_critic_only(critic_init_path)
        if extrinsics_init_path:
            self.load_extrinsics_encoder_partial(extrinsics_init_path)

    def _count_obs_dim(self, group_names: list[str], obs: TensorDict) -> int:
        total = 0
        extrinsics_added = False
        for group_name in group_names:
            if group_name == self.history_group_name:
                total += self.latent_dim
            elif group_name in self.extrinsics_group_names:
                if not extrinsics_added:
                    total += self.latent_dim
                    extrinsics_added = True
            else:
                assert len(obs[group_name].shape) == 2, "The ActorCritic module only supports 1D observations."
                total += obs[group_name].shape[-1]
        return total

    def _encode_obs_groups(self, obs: TensorDict, group_names: list[str]) -> torch.Tensor:
        obs_list = []
        extrinsics_added = False
        for group_name in group_names:
            if group_name == self.history_group_name:
                obs_list.append(self.encode_history_latent(obs))
            elif group_name in self.extrinsics_group_names:
                if not extrinsics_added:
                    obs_list.append(self.encode_extrinsics_latent(obs))
                    extrinsics_added = True
            else:
                obs_list.append(obs[group_name])
        return torch.cat(obs_list, dim=-1)

    def _get_extrinsics_input(self, obs: TensorDict) -> torch.Tensor:
        pieces = []
        if self.terrain_group_name is not None:
            terrain_obs = obs[self.terrain_group_name]
            if self.latent_mode == "no_terrain":
                terrain_obs = torch.zeros_like(terrain_obs)
            elif self.latent_mode == "shuffled_terrain":
                terrain_obs = terrain_obs.roll(shifts=1, dims=0)
            pieces.append(terrain_obs)
        dynamics_obs = obs[self.dynamics_group_name]
        if self.latent_mode == "no_dynamics":
            dynamics_obs = torch.zeros_like(dynamics_obs)
        elif self.latent_mode == "shuffled_dynamics":
            dynamics_obs = dynamics_obs.roll(shifts=1, dims=0)
        pieces.append(dynamics_obs)
        return torch.cat(pieces, dim=-1)

    def _apply_latent_mode_to_latent(self, latent: torch.Tensor) -> torch.Tensor:
        if self.latent_mode == "normal":
            return latent
        if self.latent_mode == "zero":
            return torch.zeros_like(latent)
        if self.latent_mode == "shuffled":
            return latent.roll(shifts=1, dims=0)
        if self.latent_mode == "random":
            return torch.randn_like(latent)
        return latent

    def _apply_latent_mode_to_history(self, history_obs: torch.Tensor) -> torch.Tensor:
        if self.latent_mode == "no_history":
            return torch.zeros_like(history_obs)
        if self.latent_mode != "shuffled_time":
            return history_obs
        history = history_obs.view(-1, self.history_length, self.current_policy_dim)
        history = history.roll(shifts=1, dims=1)
        return history.view(-1, self.history_dim)

    def _build_adaptation_preprocessor(
        self,
        adaptation_encoder_type: str,
        history_dim: int,
        history_feature_dim: int,
        temporal_channels: tuple[int] | list[int],
        temporal_kernel_size: int,
        prefix: str,
    ) -> tuple[nn.Module | None, nn.Module | None, int]:
        if adaptation_encoder_type == "mlp":
            return None, None, history_dim
        if adaptation_encoder_type == "tcn":
            conv_layers: list[nn.Module] = []
            in_channels = self.current_policy_dim
            kernel_size = int(temporal_kernel_size)
            for layer_idx, out_channels in enumerate(temporal_channels):
                dilation = 2**layer_idx
                padding = dilation * (kernel_size - 1) // 2
                conv_layers.append(
                    nn.Conv1d(
                        in_channels=in_channels,
                        out_channels=int(out_channels),
                        kernel_size=kernel_size,
                        padding=padding,
                        dilation=dilation,
                    )
                )
                conv_layers.append(nn.ELU())
                in_channels = int(out_channels)
            temporal_encoder = nn.Sequential(*conv_layers)
            history_projection = nn.Sequential(
                nn.Linear(in_channels * 2, int(history_feature_dim)),
                nn.ELU(),
            )
            print(f"{prefix} temporal encoder (phi-pre): {temporal_encoder}")
            print(f"{prefix} history projection (phi-pre): {history_projection}")
            return temporal_encoder, history_projection, int(history_feature_dim)
        raise ValueError(
            f"Unknown adaptation encoder type: {adaptation_encoder_type}. "
            "Expected one of {'mlp', 'tcn'}."
        )

    def _encode_history_features(
        self,
        history_obs: torch.Tensor,
        temporal_encoder: nn.Module | None,
        history_projection: nn.Module | None,
    ) -> torch.Tensor:
        if self.adaptation_encoder_type == "tcn":
            if temporal_encoder is None or history_projection is None:
                raise RuntimeError("TCN adaptation encoder requested but temporal modules are missing.")
            history = history_obs.view(-1, self.history_length, self.current_policy_dim).transpose(1, 2)
            temporal = temporal_encoder(history)
            pooled = temporal.mean(dim=-1)
            latest = temporal[:, :, -1]
            return history_projection(torch.cat([latest, pooled], dim=-1))
        return history_obs

    def _adaptation_latent_from_features(
        self,
        features: torch.Tensor,
        module: nn.Module,
        decoder: nn.Module | None,
    ) -> torch.Tensor:
        latent = module(features)
        if decoder is not None:
            latent = decoder(latent)
        return latent

    def _base_history_latent(self, history_obs: torch.Tensor) -> torch.Tensor:
        if not self.adaptation_residual_mode:
            return torch.zeros((history_obs.shape[0], self.latent_dim), device=history_obs.device, dtype=history_obs.dtype)
        if self.base_adaptation_module is None:
            raise RuntimeError("Residual adaptation mode requires a frozen base adaptation module.")
        features = self._encode_history_features(
            history_obs,
            self.base_temporal_encoder,
            self.base_history_projection,
        )
        return self._adaptation_latent_from_features(
            features,
            self.base_adaptation_module,
            self.base_adaptation_decoder,
        )

    def encode_extrinsics_latent(self, obs: TensorDict) -> torch.Tensor:
        """Teacher-side path: mu(e_t) -> z_t."""
        latent = self.extrinsics_encoder(self._get_extrinsics_input(obs))
        return self._apply_latent_mode_to_latent(latent)

    def decode_extrinsics_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """Auxiliary Phase 1 path: z_t -> reconstructed e_t."""
        dynamics = self.decode_dynamics_latent(latent)
        if self.terrain_summary_decoder is None:
            return dynamics
        terrain_summary = self.decode_terrain_summary_latent(latent)
        return torch.cat([terrain_summary, dynamics], dim=-1)

    def reconstruct_extrinsics(self, obs: TensorDict) -> torch.Tensor:
        return self.decode_extrinsics_latent(self.encode_extrinsics_latent(obs))

    def decode_dynamics_latent(self, latent: torch.Tensor) -> torch.Tensor:
        return self.dynamics_decoder(latent)

    def decode_terrain_summary_latent(self, latent: torch.Tensor) -> torch.Tensor:
        if self.terrain_summary_decoder is None:
            raise RuntimeError("Terrain summary decoder is disabled for dynamics-only Adapt-V3.")
        return self.terrain_summary_decoder(latent)

    @staticmethod
    def terrain_summary(terrain_obs: torch.Tensor) -> torch.Tensor:
        if terrain_obs.shape[-1] <= 16:
            return terrain_obs[:, :4]
        return torch.stack(
            [
                terrain_obs.mean(dim=-1),
                terrain_obs.std(dim=-1),
                terrain_obs.min(dim=-1).values,
                terrain_obs.max(dim=-1).values,
            ],
            dim=-1,
        )

    def extrinsics_summary(self, obs: TensorDict) -> torch.Tensor:
        if self.terrain_group_name is None:
            return obs[self.dynamics_group_name]
        return torch.cat(
            [
                self.terrain_summary(obs[self.terrain_group_name]),
                obs[self.dynamics_group_name],
            ],
            dim=-1,
        )

    def adapt_from_history(self, history_obs: torch.Tensor) -> torch.Tensor:
        """Deployable path: phi(history) -> z_hat_t."""
        history_obs = self._apply_latent_mode_to_history(history_obs)
        base_latent = self._base_history_latent(history_obs)
        features = self._encode_history_features(
            history_obs,
            self.temporal_encoder,
            self.history_projection,
        )
        latent = self._adaptation_latent_from_features(
            features,
            self.adaptation_module,
            self.adaptation_decoder,
        )
        if self.adaptation_residual_mode:
            latent = base_latent + self.adaptation_residual_scale * latent
        return self._apply_latent_mode_to_latent(latent)

    def encode_history_bottleneck(self, obs: TensorDict) -> torch.Tensor:
        """Return the compact history code before optional decoding to the actor latent."""
        history_obs = self._apply_latent_mode_to_history(obs[self.history_group_name])
        features = self._encode_history_features(
            history_obs,
            self.temporal_encoder,
            self.history_projection,
        )
        return self.adaptation_module(features)

    def predict_history_dynamics(self, obs: TensorDict) -> torch.Tensor:
        """Deployable helper for history -> hidden dynamics prediction.

        When the adaptation path targets the latent directly, this returns the
        dynamics decoded from the predicted latent. When the path targets
        dynamics explicitly, this returns the predicted dynamics tensor itself.
        """

        history_obs = self._apply_latent_mode_to_history(obs[self.history_group_name])
        base_latent = self._base_history_latent(history_obs)
        features = self._encode_history_features(
            history_obs,
            self.temporal_encoder,
            self.history_projection,
        )
        adaptation_out = self._adaptation_latent_from_features(
            features,
            self.adaptation_module,
            self.adaptation_decoder,
        )
        if self.adaptation_residual_mode:
            adaptation_out = base_latent + self.adaptation_residual_scale * adaptation_out
        return self.decode_dynamics_latent(adaptation_out)

    def encode_history_latent(self, obs: TensorDict) -> torch.Tensor:
        return self.adapt_from_history(obs[self.history_group_name])

    def load_residual_base_from_checkpoint(self, checkpoint_path: str) -> None:
        """Copy a full history->latent adapter into the frozen residual base path."""
        if not self.adaptation_residual_mode:
            return
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint["model_state_dict"]
        mapping = [
            ("temporal_encoder", self.temporal_encoder, "base_temporal_encoder", self.base_temporal_encoder),
            ("history_projection", self.history_projection, "base_history_projection", self.base_history_projection),
            ("adaptation_module", self.adaptation_module, "base_adaptation_module", self.base_adaptation_module),
            ("adaptation_decoder", self.adaptation_decoder, "base_adaptation_decoder", self.base_adaptation_decoder),
        ]
        copied_any = False
        for source_prefix, source_module, target_prefix, target_module in mapping:
            if target_module is None:
                continue
            source_state = {
                key[len(source_prefix) + 1 :]: value
                for key, value in state_dict.items()
                if key.startswith(f"{source_prefix}.")
            }
            if not source_state:
                continue
            target_state = target_module.state_dict()
            compatible_state = {
                key: value
                for key, value in source_state.items()
                if key in target_state and target_state[key].shape == value.shape
            }
            target_module.load_state_dict(compatible_state, strict=False)
            copied_any = copied_any or bool(compatible_state)
        if copied_any:
            self.freeze_residual_base()
            print(f"[INFO] Warm-started frozen residual base adaptation path from {checkpoint_path}.")
        else:
            print(f"[WARN] No compatible history-adaptation weights found for frozen residual base in {checkpoint_path}.")

    def freeze_residual_base(self) -> None:
        modules = [
            self.base_temporal_encoder,
            self.base_history_projection,
            self.base_adaptation_module,
            self.base_adaptation_decoder,
        ]
        for module in modules:
            if module is None:
                continue
            module.eval()
            for param in module.parameters():
                param.requires_grad_(False)

    def zero_trainable_adaptation_path(self) -> None:
        modules = [
            self.temporal_encoder,
            self.history_projection,
            self.adaptation_module,
            self.adaptation_decoder,
        ]
        for module in modules:
            if module is None:
                continue
            self._zero_module(module)

    def get_actor_obs(self, obs: TensorDict) -> torch.Tensor:
        return self._encode_obs_groups(obs, self.obs_groups["policy"])

    def get_critic_obs(self, obs: TensorDict) -> torch.Tensor:
        return self._encode_obs_groups(obs, self.obs_groups["critic"])

    def act_with_latent(self, current_policy_obs: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        """Explicit deployment helper: pi(x_t, a_{t-1}, z) -> action mean."""
        actor_obs = torch.cat([current_policy_obs, latent], dim=-1)
        actor_obs = self.actor_obs_normalizer(actor_obs)
        if self.state_dependent_std:
            mean, _ = self.actor(actor_obs)
            return mean
        return self.actor(actor_obs)

    def load_actor_only(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint["model_state_dict"]

        actor_state = OrderedDict()
        critic_state = OrderedDict()
        for key, value in state_dict.items():
            if key.startswith("actor."):
                actor_state[key[len("actor.") :]] = value
            if key.startswith("critic."):
                critic_state[key[len("critic.") :]] = value

        if not actor_state:
            raise RuntimeError(f"No actor weights found in checkpoint: {checkpoint_path}")

        self._init_module_for_actor_only(self.extrinsics_encoder, self.actor_only_extrinsics_init_mode)
        self._init_adaptation_path_for_actor_only()
        self._partial_copy_mlp(self.actor, actor_state)
        if critic_state:
            self._partial_copy_mlp(self.critic, critic_state)
        print(
            f"[INFO] Warm-started V3 actor/critic trunk from {checkpoint_path}. "
            f"Extrinsics encoder init='{self.actor_only_extrinsics_init_mode}', "
            f"adaptation encoder init='{self.actor_only_adaptation_init_mode}'."
        )

    def load_full_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.load_state_dict(checkpoint["model_state_dict"], strict=True)
        print(f"[INFO] Loaded full V3 policy checkpoint from {checkpoint_path}.")

    def load_critic_only(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint["model_state_dict"]

        critic_state = OrderedDict()
        for key, value in state_dict.items():
            if key.startswith("critic."):
                critic_state[key[len("critic.") :]] = value

        if not critic_state:
            raise RuntimeError(f"No critic weights found in checkpoint: {checkpoint_path}")

        self._partial_copy_mlp(self.critic, critic_state)
        print(
            f"[INFO] Warm-started V3 critic only from {checkpoint_path}. "
            "Actor, extrinsics encoder, and adaptation module start from fresh initialization."
        )

    def load_extrinsics_encoder_partial(self, checkpoint_path: str) -> None:
        """Warm-start mu from a checkpoint with compatible downstream layers.

        This is mainly for terrain-lite: copy the frozen dynamics-only encoder
        into the dynamics columns of the larger terrain+dynamics input while
        leaving the new terrain columns near-zero so the branch starts close to
        the proven dynamics-only behavior.
        """

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint["model_state_dict"]
        first_weight_key = "extrinsics_encoder.0.weight"
        first_bias_key = "extrinsics_encoder.0.bias"
        source_first_weight = state_dict.get(first_weight_key)

        if source_first_weight is None:
            raise RuntimeError(f"No extrinsics encoder weights found in checkpoint: {checkpoint_path}")

        target_first = self.extrinsics_encoder[0]
        if not isinstance(target_first, nn.Linear):
            raise RuntimeError("Expected extrinsics_encoder[0] to be a Linear layer.")

        source_input_dim = source_first_weight.shape[1]
        target_input_dim = target_first.weight.shape[1]
        if source_input_dim > target_input_dim:
            raise RuntimeError(
                "Cannot warm-start extrinsics encoder from a larger input space: "
                f"source={source_input_dim}, target={target_input_dim}"
            )

        with torch.no_grad():
            target_first.weight.zero_()
            dynamics_offset = max(0, target_input_dim - source_input_dim)
            target_first.weight[:, dynamics_offset : dynamics_offset + source_input_dim].copy_(source_first_weight)
            if first_bias_key in state_dict:
                target_first.bias.copy_(state_dict[first_bias_key])

        for idx, layer in enumerate(self.extrinsics_encoder):
            if idx == 0 or not isinstance(layer, nn.Linear):
                continue
            self._copy_if_present(layer.weight, state_dict, f"extrinsics_encoder.{idx}.weight")
            self._copy_if_present(layer.bias, state_dict, f"extrinsics_encoder.{idx}.bias")

        for idx, layer in enumerate(self.dynamics_decoder):
            if not isinstance(layer, nn.Linear):
                continue
            self._copy_if_present(layer.weight, state_dict, f"dynamics_decoder.{idx}.weight")
            self._copy_if_present(layer.bias, state_dict, f"dynamics_decoder.{idx}.bias")

        print(
            f"[INFO] Warm-started V3 extrinsics encoder from {checkpoint_path}; "
            f"copied {source_input_dim} source inputs into target inputs "
            f"[{dynamics_offset}:{dynamics_offset + source_input_dim})."
        )

    @staticmethod
    def _zero_module(module: nn.Module) -> None:
        for param in module.parameters():
            nn.init.zeros_(param)

    @classmethod
    def _init_module_for_actor_only(cls, module: nn.Module, mode: str) -> None:
        if mode == "identity":
            cls._init_identity_linear_stack(module, name="actor-only module")
            return
        if mode == "zero":
            cls._zero_module(module)
            return
        if mode in {"xavier", "small_xavier"}:
            gain = 1.0 if mode == "xavier" else 0.1
            for submodule in module.modules():
                if isinstance(submodule, (nn.Linear, nn.Conv1d)):
                    nn.init.xavier_uniform_(submodule.weight, gain=gain)
                    if submodule.bias is not None:
                        nn.init.zeros_(submodule.bias)
            return
        raise ValueError(
            f"Unknown actor-only encoder init mode: {mode}. "
            "Expected one of {'zero', 'xavier', 'small_xavier', 'identity'}."
        )

    @staticmethod
    def _init_identity_linear_stack(module: nn.Module, name: str) -> None:
        linear_layers = [submodule for submodule in module.modules() if isinstance(submodule, nn.Linear)]
        if len(linear_layers) != 1:
            raise ValueError(f"Identity init for {name} expects exactly one Linear layer.")
        linear = linear_layers[0]
        if linear.weight.shape[0] != linear.weight.shape[1]:
            raise ValueError(
                f"Identity init for {name} requires a square Linear layer, got {tuple(linear.weight.shape)}."
            )
        with torch.no_grad():
            linear.weight.zero_()
            linear.weight.copy_(torch.eye(linear.weight.shape[0], device=linear.weight.device, dtype=linear.weight.dtype))
            if linear.bias is not None:
                linear.bias.zero_()

    def _init_adaptation_path_for_actor_only(self) -> None:
        if self.temporal_encoder is not None:
            self._init_module_for_actor_only(self.temporal_encoder, self.actor_only_adaptation_init_mode)
        if self.history_projection is not None:
            self._init_module_for_actor_only(self.history_projection, self.actor_only_adaptation_init_mode)
        self._init_module_for_actor_only(self.adaptation_module, self.actor_only_adaptation_init_mode)
        if self.adaptation_decoder is not None:
            self._init_module_for_actor_only(self.adaptation_decoder, self.actor_only_adaptation_init_mode)
        if self.adaptation_residual_mode:
            self.freeze_residual_base()

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
                # Preserve the copied blind-policy slice, but do not leave the
                # new latent-input columns dead. A zero latent slice lets the
                # policy completely ignore z at initialization, which in turn
                # starves mu of gradient. Small random latent columns keep the
                # warm start stable while allowing latent sensitivity to emerge.
                nn.init.xavier_uniform_(target_weight)
                shared_in = min(source_weight.shape[1], target_weight.shape[1])
                shared_out = min(source_weight.shape[0], target_weight.shape[0])
                target_weight[:shared_out, :shared_in].copy_(source_weight[:shared_out, :shared_in])
        if first_bias_key in source_state:
            source_bias = source_state[first_bias_key]
            target_bias = module[0].bias
            shared = min(source_bias.shape[0], target_bias.shape[0])
            with torch.no_grad():
                target_bias.zero_()
                target_bias[:shared].copy_(source_bias[:shared])

        for idx, layer in enumerate(module):
            if idx == 0 or not isinstance(layer, nn.Linear):
                continue
            self._copy_if_present(layer.weight, source_state, f"{idx}.weight")
            self._copy_if_present(layer.bias, source_state, f"{idx}.bias")
