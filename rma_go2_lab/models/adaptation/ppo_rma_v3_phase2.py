from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.algorithms import PPO

from rma_go2_lab.models.adaptation.frozen_adapt_v3_phase1 import FrozenAdaptV3Phase1


class RmaV3Phase2PPO(PPO):
    """Phase 2 PPO with latent supervision against frozen V3 Phase 1.

    This is the faithful RMA training phase where:

    - ``mu`` and ``pi`` come from a frozen Phase 1 checkpoint
    - ``phi`` predicts ``z_hat`` from history
    - the actor acts using ``pi(x_t, a_{t-1}, z_hat_t)``
    - optimization uses PPO plus latent regression to ``z_t = mu(e_t)``
    """

    def __init__(
        self,
        policy,
        *args,
        phase1_reference_path: str | None = None,
        surrogate_loss_coef: float = 1.0,
        latent_regression_coef: float = 1.0,
        latent_l2_coef: float = 0.0,
        latent_command_threshold: float = 0.1,
        imitation_command_threshold: float = 0.1,
        imitation_coef_stage0: float = 0.0,
        imitation_coef_stage1: float = 0.0,
        imitation_stage0_end: int = 0,
        imitation_stage1_end: int = 0,
        freeze_critic: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(policy, *args, **kwargs)
        self.surrogate_loss_coef = float(surrogate_loss_coef)
        self.latent_regression_coef = float(latent_regression_coef)
        self.latent_l2_coef = float(latent_l2_coef)
        self.latent_command_threshold = float(latent_command_threshold)
        self.imitation_command_threshold = float(imitation_command_threshold)
        self.imitation_coef_stage0 = float(imitation_coef_stage0)
        self.imitation_coef_stage1 = float(imitation_coef_stage1)
        self.imitation_stage0_end = int(imitation_stage0_end)
        self.imitation_stage1_end = int(imitation_stage1_end)
        self.freeze_critic = bool(freeze_critic)
        self._update_counter = 0
        self.phase1_reference = None
        self.latent_loss_fn = nn.MSELoss(reduction="none")
        self.imitation_loss_fn = nn.MSELoss(reduction="none")

        if phase1_reference_path:
            self.phase1_reference = FrozenAdaptV3Phase1(
                checkpoint_path=phase1_reference_path,
                device=self.device,
                terrain_group_name=self.policy.terrain_group_name,
                dynamics_group_name=self.policy.dynamics_group_name,
                terrain_dim=self.policy.terrain_dim,
            ).to(self.device)

        self._freeze_phase2_modules()
        trainable_params = [param for param in self.policy.parameters() if param.requires_grad]
        if not trainable_params:
            raise RuntimeError("Adapt-V3 Phase2 has no trainable parameters after freezing policy modules.")
        self.optimizer = optim.Adam(trainable_params, lr=self.learning_rate)

    def _freeze_phase2_modules(self) -> None:
        """Freeze the privileged base and keep only the adaptation path trainable.

        Phase 2 is meant to learn ``phi(history) -> z_hat`` against a frozen
        Stage A privileged base. The value function remains trainable so PPO can
        still fit the rollout distribution seen under the evolving adaptation
        module.
        """

        modules_to_freeze = [
            self.policy.actor,
            self.policy.extrinsics_encoder,
            self.policy.dynamics_decoder,
        ]
        if self.freeze_critic:
            modules_to_freeze.append(self.policy.critic)
        if self.policy.terrain_summary_decoder is not None:
            modules_to_freeze.append(self.policy.terrain_summary_decoder)
        for module in modules_to_freeze:
            module.eval()
            for param in module.parameters():
                param.requires_grad_(False)

        if hasattr(self.policy, "std"):
            self.policy.std.requires_grad_(False)
        if hasattr(self.policy, "log_std"):
            self.policy.log_std.requires_grad_(False)

    def _current_imitation_coef(self) -> float:
        if self._update_counter < self.imitation_stage0_end:
            return self.imitation_coef_stage0
        if self._update_counter < self.imitation_stage1_end:
            return self.imitation_coef_stage1
        return 0.0

    def _command_mask(self, obs_batch, threshold: float) -> torch.Tensor | None:
        if "policy" not in obs_batch.keys():
            return None
        command = obs_batch["policy"][:, 9:12]
        return (torch.linalg.norm(command, dim=-1) > threshold).float()

    def update(self) -> dict[str, float]:
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        mean_latent_regression_loss = 0
        mean_latent_active_frac = 0
        mean_student_latent_batch_std = 0
        mean_teacher_latent_batch_std = 0
        mean_student_latent_l2 = 0
        mean_student_latent_max_abs = 0
        mean_teacher_imitation_loss = 0
        mean_teacher_imitation_active_frac = 0
        mean_teacher_imitation_coef = 0
        mean_latent_cosine = 0
        mean_rnd_loss = 0 if self.rnd else None
        mean_symmetry_loss = 0 if self.symmetry else None

        if self.policy.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

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

            loss = (
                self.surrogate_loss_coef * surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_batch.mean()
            )

            latent_regression_loss = torch.tensor(0.0, device=self.device)
            latent_active_frac = torch.tensor(0.0, device=self.device)
            student_latent_batch_std = torch.tensor(0.0, device=self.device)
            teacher_latent_batch_std = torch.tensor(0.0, device=self.device)
            student_latent_l2 = torch.tensor(0.0, device=self.device)
            student_latent_max_abs = torch.tensor(0.0, device=self.device)
            latent_cosine = torch.tensor(0.0, device=self.device)
            teacher_imitation_loss = torch.tensor(0.0, device=self.device)
            teacher_imitation_active_frac = torch.tensor(0.0, device=self.device)
            teacher_imitation_coef = self._current_imitation_coef()

            required_groups = {"policy", "policy_history", self.policy.dynamics_group_name}
            if self.policy.terrain_group_name is not None:
                required_groups.add(self.policy.terrain_group_name)
            if self.phase1_reference is not None and required_groups.issubset(set(obs_batch.keys())):
                reference_obs_batch = obs_batch[:original_batch_size]
                latent_mask = self._command_mask(reference_obs_batch, self.latent_command_threshold)
                if latent_mask is not None:
                    latent_active_frac = latent_mask.mean()
                    if torch.count_nonzero(latent_mask) > 0:
                        student_latent = self.policy.encode_history_latent(reference_obs_batch)
                        teacher_latent = self.phase1_reference.encode_extrinsics_latent(reference_obs_batch).detach()
                        if student_latent.shape != teacher_latent.shape:
                            raise RuntimeError(
                                "Adapt-V3 Phase2 latent shape mismatch: "
                                f"student={tuple(student_latent.shape)} teacher={tuple(teacher_latent.shape)}"
                            )
                        student_latent_batch_std = student_latent.std(dim=0).mean()
                        teacher_latent_batch_std = teacher_latent.std(dim=0).mean()
                        per_sample_latent_l2 = student_latent.pow(2).mean(dim=-1)
                        student_latent_l2 = (per_sample_latent_l2 * latent_mask).sum() / (latent_mask.sum() + 1e-6)
                        per_sample_latent_max_abs = student_latent.abs().amax(dim=-1)
                        student_latent_max_abs = (per_sample_latent_max_abs * latent_mask).sum() / (
                            latent_mask.sum() + 1e-6
                        )
                        per_sample_latent = self.latent_loss_fn(student_latent, teacher_latent).mean(dim=-1)
                        latent_regression_loss = (per_sample_latent * latent_mask).sum() / (latent_mask.sum() + 1e-6)
                        loss = loss + self.latent_regression_coef * latent_regression_loss
                        if self.latent_l2_coef > 0.0:
                            loss = loss + self.latent_l2_coef * student_latent_l2

                        cosine = nn.functional.cosine_similarity(student_latent, teacher_latent, dim=-1)
                        latent_cosine = (cosine * latent_mask).sum() / (latent_mask.sum() + 1e-6)

                if teacher_imitation_coef > 0.0:
                    imitation_mask = self._command_mask(reference_obs_batch, self.imitation_command_threshold)
                    if imitation_mask is not None:
                        teacher_imitation_active_frac = imitation_mask.mean()
                        if torch.count_nonzero(imitation_mask) > 0:
                            teacher_actions = self.phase1_reference(reference_obs_batch).detach()
                            per_sample = self.imitation_loss_fn(mu_batch, teacher_actions).sum(dim=-1)
                            teacher_imitation_loss = (per_sample * imitation_mask).sum() / (imitation_mask.sum() + 1e-6)
                            loss = loss + teacher_imitation_coef * teacher_imitation_loss

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
            mean_latent_regression_loss += latent_regression_loss.item()
            mean_latent_active_frac += latent_active_frac.item()
            mean_student_latent_batch_std += student_latent_batch_std.item()
            mean_teacher_latent_batch_std += teacher_latent_batch_std.item()
            mean_student_latent_l2 += student_latent_l2.item()
            mean_student_latent_max_abs += student_latent_max_abs.item()
            mean_latent_cosine += latent_cosine.item()
            mean_teacher_imitation_loss += teacher_imitation_loss.item()
            mean_teacher_imitation_active_frac += teacher_imitation_active_frac.item()
            mean_teacher_imitation_coef += teacher_imitation_coef
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_latent_regression_loss /= num_updates
        mean_latent_active_frac /= num_updates
        mean_student_latent_batch_std /= num_updates
        mean_teacher_latent_batch_std /= num_updates
        mean_student_latent_l2 /= num_updates
        mean_student_latent_max_abs /= num_updates
        mean_latent_cosine /= num_updates
        mean_teacher_imitation_loss /= num_updates
        mean_teacher_imitation_active_frac /= num_updates
        mean_teacher_imitation_coef /= num_updates
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates

        self.storage.clear()
        self._update_counter += 1

        loss_dict = {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "latent_regression": mean_latent_regression_loss,
            "latent_active_frac": mean_latent_active_frac,
            "student_latent_batch_std": mean_student_latent_batch_std,
            "teacher_latent_batch_std": mean_teacher_latent_batch_std,
            "student_latent_l2": mean_student_latent_l2,
            "student_latent_max_abs": mean_student_latent_max_abs,
            "latent_cosine": mean_latent_cosine,
            "teacher_imitation": mean_teacher_imitation_loss,
            "teacher_imitation_active_frac": mean_teacher_imitation_active_frac,
            "teacher_imitation_coef": mean_teacher_imitation_coef,
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss

        return loss_dict
