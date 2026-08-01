from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.algorithms import PPO
from rma_go2_lab.models.blind.frozen_flat_expert import FrozenFlatExpert


class RmaV3Phase1PPO(PPO):
    """Minimal Phase 1 PPO wrapper for the faithful V3 branch.

    The learning objective is still standard PPO; the important change is the
    policy architecture, which now routes privileged factors only through
    ``mu(e_t) -> z_t`` before the actor.
    """

    def __init__(
        self,
        policy,
        *args,
        latent_anchor_coef: float = 10.0,
        dynamics_prediction_coef: float = 5.0,
        terrain_summary_coef: float = 1.0,
        latent_variation_coef: float = 1.0,
        latent_std_target: float = 0.05,
        latent_pairwise_coef: float = 2.0,
        latent_pairwise_max_samples: int = 128,
        auxiliary_pretrain_steps: int = 2,
        auxiliary_learning_rate: float = 5.0e-4,
        auxiliary_warmup_iters: int = 300,
        auxiliary_start_factor: float = 0.1,
        flat_expert_path: str | None = None,
        flat_expert_activation: str = "elu",
        flat_imitation_command_threshold: float = 0.1,
        flat_imitation_coef_stage0: float = 0.0,
        flat_imitation_coef_stage1: float = 0.0,
        flat_imitation_stage0_end: int = 0,
        flat_imitation_stage1_end: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(policy, *args, **kwargs)
        self.latent_anchor_coef = float(latent_anchor_coef)
        self.dynamics_prediction_coef = float(dynamics_prediction_coef)
        self.terrain_summary_coef = float(terrain_summary_coef)
        self.latent_variation_coef = float(latent_variation_coef)
        self.latent_std_target = float(latent_std_target)
        self.latent_pairwise_coef = float(latent_pairwise_coef)
        self.latent_pairwise_max_samples = int(latent_pairwise_max_samples)
        self.auxiliary_pretrain_steps = int(auxiliary_pretrain_steps)
        self.auxiliary_warmup_iters = max(1, int(auxiliary_warmup_iters))
        self.auxiliary_start_factor = float(auxiliary_start_factor)
        self.flat_imitation_command_threshold = float(flat_imitation_command_threshold)
        self.flat_imitation_coef_stage0 = float(flat_imitation_coef_stage0)
        self.flat_imitation_coef_stage1 = float(flat_imitation_coef_stage1)
        self.flat_imitation_stage0_end = int(flat_imitation_stage0_end)
        self.flat_imitation_stage1_end = int(flat_imitation_stage1_end)
        self.update_counter = 0
        aux_params = (
            list(self.policy.extrinsics_encoder.parameters())
            + list(self.policy.dynamics_decoder.parameters())
        )
        if self.policy.terrain_summary_decoder is not None:
            aux_params += list(self.policy.terrain_summary_decoder.parameters())
        self.auxiliary_optimizer = torch.optim.Adam(aux_params, lr=float(auxiliary_learning_rate))
        self.prediction_loss_fn = nn.MSELoss(reduction="mean")
        self.imitation_loss_fn = nn.MSELoss(reduction="none")
        self.flat_expert = None
        if flat_expert_path:
            self.flat_expert = FrozenFlatExpert(
                checkpoint_path=flat_expert_path,
                activation=flat_expert_activation,
                device=self.device,
            ).to(self.device)

    def _auxiliary_scale(self) -> float:
        progress = min(1.0, self.update_counter / float(self.auxiliary_warmup_iters))
        return self.auxiliary_start_factor + (1.0 - self.auxiliary_start_factor) * progress

    def _current_imitation_coef(self) -> float:
        if self.update_counter < self.flat_imitation_stage0_end:
            return self.flat_imitation_coef_stage0
        if self.update_counter < self.flat_imitation_stage1_end:
            return self.flat_imitation_coef_stage1
        return 0.0

    def _imitation_mask(self, obs_batch) -> torch.Tensor | None:
        if self.flat_expert is None or "policy" not in obs_batch.keys():
            return None
        command = obs_batch["policy"][:, 9:12]
        return (torch.linalg.norm(command, dim=-1) > self.flat_imitation_command_threshold).float()

    @staticmethod
    def _pairwise_sqdist(x: torch.Tensor) -> torch.Tensor:
        # cdist avoids materializing an N x N x D tensor.
        return torch.cdist(x, x, p=2).pow(2) / x.shape[-1]

    def _pairwise_structure_loss(self, latent: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if latent.shape[0] < 2:
            return torch.tensor(0.0, device=latent.device)
        if latent.shape[0] > self.latent_pairwise_max_samples:
            perm = torch.randperm(latent.shape[0], device=latent.device)[: self.latent_pairwise_max_samples]
            latent = latent[perm]
            target = target[perm]
        latent_dist = self._pairwise_sqdist(latent)
        target_dist = self._pairwise_sqdist(target)
        # Normalize to keep the loss scale stable across batches.
        latent_dist = latent_dist / (latent_dist.mean().detach() + 1e-6)
        target_dist = target_dist / (target_dist.mean().detach() + 1e-6)
        eye = torch.eye(latent_dist.shape[0], device=latent.device, dtype=torch.bool)
        return ((latent_dist[~eye] - target_dist[~eye]) ** 2).mean()

    @staticmethod
    def _normalize_features(x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, keepdim=True).clamp_min(1.0e-4)
        return (x - mean) / std

    def _compute_auxiliary_losses(
        self, reference_obs_batch
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        latent = self.policy.encode_extrinsics_latent(reference_obs_batch)
        target_dynamics = reference_obs_batch[self.policy.dynamics_group_name]
        predicted_dynamics = self.policy.decode_dynamics_latent(latent)
        target_summary = self.policy.extrinsics_summary(reference_obs_batch)
        normalized_target_summary = self._normalize_features(target_summary)
        latent_anchor = latent[:, : self.policy.extrinsics_summary_dim]
        latent_anchor_loss = self.prediction_loss_fn(latent_anchor, normalized_target_summary)

        dynamics_prediction_loss = self.prediction_loss_fn(predicted_dynamics, target_dynamics)
        terrain_summary_loss = torch.tensor(0.0, device=self.device)
        if self.policy.terrain_group_name is not None and self.policy.terrain_summary_decoder is not None:
            target_terrain_summary = self.policy.terrain_summary(reference_obs_batch[self.policy.terrain_group_name])
            predicted_terrain_summary = self.policy.decode_terrain_summary_latent(latent)
            terrain_summary_loss = self.prediction_loss_fn(predicted_terrain_summary, target_terrain_summary)
        latent_batch_std = latent.std(dim=0).mean()
        latent_variation_bonus = torch.relu(torch.tensor(self.latent_std_target, device=self.device) - latent_batch_std)
        latent_pairwise_loss = self._pairwise_structure_loss(latent_anchor, normalized_target_summary)
        return (
            latent_anchor_loss,
            dynamics_prediction_loss,
            terrain_summary_loss,
            latent_batch_std,
            latent_variation_bonus,
            latent_pairwise_loss,
        )

    def update(self) -> dict[str, float]:
        self.update_counter += 1
        aux_scale = self._auxiliary_scale()
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        mean_latent_anchor_loss = 0
        mean_dynamics_prediction_loss = 0
        mean_terrain_summary_loss = 0
        mean_latent_batch_std = 0
        mean_latent_variation_bonus = 0
        mean_latent_pairwise_loss = 0
        mean_flat_imitation_loss = 0
        mean_flat_imitation_active_frac = 0
        mean_flat_imitation_coef = 0
        mean_rnd_loss = 0 if self.rnd else None
        mean_symmetry_loss = 0 if self.symmetry else None

        if self.policy.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        required_groups = {"policy", self.policy.dynamics_group_name}
        if self.policy.terrain_group_name is not None:
            required_groups.add(self.policy.terrain_group_name)

        for (
            obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hidden_states_batch,
            masks_batch,
        ) in generator:
            num_aug = 1
            original_batch_size = obs_batch.batch_size[0]

            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            if self.symmetry and self.symmetry["use_data_augmentation"]:
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                obs_batch, actions_batch = data_augmentation_func(
                    obs=obs_batch,
                    actions=actions_batch,
                    env=self.symmetry["_env"],
                )
                num_aug = int(obs_batch.batch_size[0] / original_batch_size)
                old_actions_log_prob_batch = old_actions_log_prob_batch.repeat(num_aug, 1)
                target_values_batch = target_values_batch.repeat(num_aug, 1)
                advantages_batch = advantages_batch.repeat(num_aug, 1)
                returns_batch = returns_batch.repeat(num_aug, 1)

            latent_anchor_loss = torch.tensor(0.0, device=self.device)
            dynamics_prediction_loss = torch.tensor(0.0, device=self.device)
            terrain_summary_loss = torch.tensor(0.0, device=self.device)
            latent_batch_std = torch.tensor(0.0, device=self.device)
            latent_variation_bonus = torch.tensor(0.0, device=self.device)
            latent_pairwise_loss = torch.tensor(0.0, device=self.device)

            if required_groups.issubset(set(obs_batch.keys())):
                reference_obs_batch = obs_batch[:original_batch_size]
                for _ in range(self.auxiliary_pretrain_steps):
                    self.auxiliary_optimizer.zero_grad()
                    (
                        aux_anchor_loss,
                        aux_dyn_loss,
                        aux_terrain_loss,
                        aux_latent_std,
                        aux_variation_bonus,
                        aux_pairwise_loss,
                    ) = self._compute_auxiliary_losses(reference_obs_batch)
                    aux_loss = (
                        aux_scale
                        * (
                            self.latent_anchor_coef * aux_anchor_loss
                            + self.dynamics_prediction_coef * aux_dyn_loss
                            + self.terrain_summary_coef * aux_terrain_loss
                            + self.latent_variation_coef * aux_variation_bonus
                            + self.latent_pairwise_coef * aux_pairwise_loss
                        )
                    )
                    aux_loss.backward()
                    nn.utils.clip_grad_norm_(
                        list(self.policy.extrinsics_encoder.parameters())
                        + list(self.policy.dynamics_decoder.parameters())
                        + (
                            list(self.policy.terrain_summary_decoder.parameters())
                            if self.policy.terrain_summary_decoder is not None
                            else []
                        ),
                        self.max_grad_norm,
                    )
                    self.auxiliary_optimizer.step()

                (
                    latent_anchor_loss,
                    dynamics_prediction_loss,
                    terrain_summary_loss,
                    latent_batch_std,
                    latent_variation_bonus,
                    latent_pairwise_loss,
                ) = self._compute_auxiliary_losses(reference_obs_batch)

            self.policy.act(obs_batch, masks=masks_batch, hidden_state=hidden_states_batch[0])
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            value_batch = self.policy.evaluate(obs_batch, masks=masks_batch, hidden_state=hidden_states_batch[1])
            mu_batch = self.policy.action_mean[:original_batch_size]
            sigma_batch = self.policy.action_std[:original_batch_size]
            entropy_batch = self.policy.entropy[:original_batch_size]

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()
            loss = loss + aux_scale * self.latent_anchor_coef * latent_anchor_loss
            loss = loss + aux_scale * self.dynamics_prediction_coef * dynamics_prediction_loss
            loss = loss + aux_scale * self.terrain_summary_coef * terrain_summary_loss
            loss = loss + aux_scale * self.latent_variation_coef * latent_variation_bonus
            loss = loss + aux_scale * self.latent_pairwise_coef * latent_pairwise_loss

            imitation_loss = torch.tensor(0.0, device=self.device)
            imitation_active_frac = torch.tensor(0.0, device=self.device)
            imitation_coef = self._current_imitation_coef()
            if self.flat_expert is not None and imitation_coef > 0.0:
                expert_obs_batch = obs_batch[:original_batch_size]
                mask = self._imitation_mask(expert_obs_batch)
                if mask is not None:
                    imitation_active_frac = mask.mean()
                    if torch.count_nonzero(mask) > 0:
                        flat_actions = self.flat_expert(expert_obs_batch["policy"]).detach()
                        per_sample = self.imitation_loss_fn(mu_batch, flat_actions).sum(dim=-1)
                        imitation_loss = (per_sample * mask).sum() / (mask.sum() + 1e-6)
                        loss = loss + imitation_coef * imitation_loss

            if self.symmetry:
                if not self.symmetry["use_data_augmentation"]:
                    data_augmentation_func = self.symmetry["data_augmentation_func"]
                    obs_batch, _ = data_augmentation_func(obs=obs_batch, actions=None, env=self.symmetry["_env"])
                    num_aug = int(obs_batch.shape[0] / original_batch_size)
                mean_actions_batch = self.policy.act_inference(obs_batch.detach().clone())
                action_mean_orig = mean_actions_batch[:original_batch_size]
                _, actions_mean_symm_batch = data_augmentation_func(
                    obs=None, actions=action_mean_orig, env=self.symmetry["_env"]
                )
                mse_loss = torch.nn.MSELoss()
                symmetry_loss = mse_loss(
                    mean_actions_batch[original_batch_size:], actions_mean_symm_batch.detach()[original_batch_size:]
                )
                if self.symmetry["use_mirror_loss"]:
                    loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
                else:
                    symmetry_loss = symmetry_loss.detach()

            if self.rnd:
                with torch.no_grad():
                    rnd_state_batch = self.rnd.get_rnd_state(obs_batch[:original_batch_size])
                    rnd_state_batch = self.rnd.state_normalizer(rnd_state_batch)
                predicted_embedding = self.rnd.predictor(rnd_state_batch)
                target_embedding = self.rnd.target(rnd_state_batch).detach()
                mseloss = torch.nn.MSELoss()
                rnd_loss = mseloss(predicted_embedding, target_embedding)

            self.optimizer.zero_grad()
            loss.backward()
            if self.rnd:
                self.rnd_optimizer.zero_grad()
                rnd_loss.backward()

            if self.is_multi_gpu:
                self.reduce_parameters()

            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
            if self.rnd_optimizer:
                self.rnd_optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            mean_latent_anchor_loss += latent_anchor_loss.item()
            mean_dynamics_prediction_loss += dynamics_prediction_loss.item()
            mean_terrain_summary_loss += terrain_summary_loss.item()
            mean_latent_batch_std += latent_batch_std.item()
            mean_latent_variation_bonus += latent_variation_bonus.item()
            mean_latent_pairwise_loss += latent_pairwise_loss.item()
            mean_flat_imitation_loss += imitation_loss.item()
            mean_flat_imitation_active_frac += imitation_active_frac.item()
            mean_flat_imitation_coef += imitation_coef
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_latent_anchor_loss /= num_updates
        mean_dynamics_prediction_loss /= num_updates
        mean_terrain_summary_loss /= num_updates
        mean_latent_batch_std /= num_updates
        mean_latent_variation_bonus /= num_updates
        mean_latent_pairwise_loss /= num_updates
        mean_flat_imitation_loss /= num_updates
        mean_flat_imitation_active_frac /= num_updates
        mean_flat_imitation_coef /= num_updates
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates

        self.storage.clear()

        loss_dict = {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "latent_anchor": mean_latent_anchor_loss,
            "dynamics_prediction": mean_dynamics_prediction_loss,
            "terrain_summary_prediction": mean_terrain_summary_loss,
            "latent_batch_std": mean_latent_batch_std,
            "latent_variation_bonus": mean_latent_variation_bonus,
            "latent_pairwise": mean_latent_pairwise_loss,
            "auxiliary_scale": aux_scale,
            "flat_imitation": mean_flat_imitation_loss,
            "flat_imitation_active_frac": mean_flat_imitation_active_frac,
            "flat_imitation_coef": mean_flat_imitation_coef,
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss

        return loss_dict
