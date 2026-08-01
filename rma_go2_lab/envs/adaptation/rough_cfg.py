"""Deployable student rough-terrain environment for the adaptation phase.

Phase B starts from a deployable proprio-only interface, but unlike the frozen
blind baseline it injects controlled within-episode switches during training.

The intent is to create an honest no-adaptation baseline in a regime where
single-policy averaging should be less sufficient:

- no terrain height scan
- no direct dynamics privilege
- same rough-terrain family as the blind and teacher phases
- one sampled mid-episode hidden-dynamics switch per environment
"""

from __future__ import annotations

import torch

import isaaclab.utils.math as math_utils
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils import configclass

from rma_go2_lab.envs.blind.rough_cfg import Go2BlindBaselineRoughEnvCfg


@configclass
class Go2AdaptationStudentRoughEnvCfg(Go2BlindBaselineRoughEnvCfg):
    """Phase-B deployable student env scaffold with training-time switches."""

    adaptation_switch_step: int = 500
    adaptation_switch_episode_prob: float = 1.0
    adaptation_enable_friction_switch: bool = True
    adaptation_enable_mass_switch: bool = True
    adaptation_enable_motor_switch: bool = True

    def __post_init__(self):
        super().__post_init__()

        print("\n========== ADAPTATION STUDENT ROUGH ==========\n")


class Go2AdaptationStudentRoughEnv(ManagerBasedRLEnv):
    """RL env with one sampled hidden-dynamics switch per episode.

    The switch is intentionally not exposed through observations. This keeps the
    no-adaptation baseline deployable while making the hidden factor changes
    load-bearing for later history-latent work.
    """

    cfg: Go2AdaptationStudentRoughEnvCfg

    def __init__(self, cfg: Go2AdaptationStudentRoughEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)

        num_envs = cfg.scene.num_envs
        device = self.device

        self._switch_step = int(cfg.adaptation_switch_step)
        self._switch_episode_prob = float(cfg.adaptation_switch_episode_prob)
        self._switch_episode_step_buf = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._switch_applied = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self._switch_enabled = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self._switch_apply_count = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._switch_scenario = torch.zeros(num_envs, dtype=torch.long, device=device)

        self._switch_static_friction = torch.full((num_envs,), float("nan"), device=device)
        self._switch_dynamic_friction = torch.full((num_envs,), float("nan"), device=device)
        self._switch_mass_offset = torch.full((num_envs,), float("nan"), device=device)
        self._switch_motor_stiffness_scale = torch.full((num_envs,), float("nan"), device=device)
        self._switch_motor_damping_scale = torch.full((num_envs,), float("nan"), device=device)

        # The base class reset path runs before these switch buffers exist, so
        # the very first live episode needs an explicit target sample here.
        self._sample_switch_targets(torch.arange(num_envs, device=device, dtype=torch.long))

    def step(self, action: torch.Tensor):
        # Track true episode age locally instead of relying on IsaacLab's
        # `episode_length_buf`, which RSL-RL intentionally randomizes at the
        # start of training for rollout decorrelation.
        self._switch_episode_step_buf += 1

        obs, reward, terminated, truncated, extras = super().step(action)

        if self._switch_step >= 0:
            pending_env_ids = (
                (~self._switch_applied)
                & self._switch_enabled
                & (self._switch_episode_step_buf >= self._switch_step)
            ).nonzero(
                as_tuple=False
            ).squeeze(-1)
            if len(pending_env_ids) > 0:
                self._validate_switch_targets(pending_env_ids)
                self._apply_switches(pending_env_ids)
                self._switch_applied[pending_env_ids] = True
                self._switch_apply_count[pending_env_ids] += 1
                if torch.any(self._switch_apply_count[pending_env_ids] > 1):
                    raise RuntimeError("Adaptation switch applied more than once in a single episode.")

        return obs, reward, terminated, truncated, extras

    def _reset_idx(self, env_ids):
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if len(env_ids) == 0:
            return

        # Log episode-bounded switch statistics before the reset clears the
        # bookkeeping. The primary metric is "how many completed episodes lived
        # long enough to reach the switch horizon *and* were switch-enabled?".
        #
        # This is more robust than relying only on `_switch_applied`, because
        # it measures the episode-level fact we care about directly from the
        # pre-reset buffers.
        episode_lengths = self._switch_episode_step_buf[env_ids]
        switch_enabled = self._switch_enabled[env_ids]
        if self._switch_step >= 0:
            switch_horizon_reached = switch_enabled & (episode_lengths >= self._switch_step)
        else:
            switch_horizon_reached = torch.zeros_like(switch_enabled)

        switch_reached_frac = switch_horizon_reached.float().mean().item()
        switch_applied_frac = self._switch_applied[env_ids].float().mean().item()

        # This should never happen. If it does, the switch bookkeeping and the
        # episode-length view of the world disagree, so fail loudly.
        impossible_env_ids = env_ids[self._switch_applied[env_ids] & ~switch_horizon_reached]
        if len(impossible_env_ids) > 0:
            bad_env_ids = impossible_env_ids.tolist()
            bad_lengths = episode_lengths[self._switch_applied[env_ids] & ~switch_horizon_reached].tolist()
            raise RuntimeError(
                "Switch bookkeeping inconsistency detected: some episodes are marked as switched "
                "even though they did not live long enough to reach the configured switch horizon. "
                f"switch_step={self._switch_step}, env_ids={bad_env_ids}, episode_lengths={bad_lengths}"
            )

        super()._reset_idx(env_ids)

        self.extras.setdefault("log", {})
        self.extras["log"]["adaptation_switch_reached_frac"] = switch_reached_frac
        self.extras["log"]["adaptation_switch_applied_frac"] = switch_applied_frac
        self._switch_episode_step_buf[env_ids] = 0
        self._switch_applied[env_ids] = False
        self._switch_enabled[env_ids] = False
        self._switch_apply_count[env_ids] = 0
        self._sample_switch_targets(env_ids)

    def _validate_switch_targets(self, env_ids: torch.Tensor) -> None:
        friction_valid = (~torch.isnan(self._switch_static_friction[env_ids])) & (
            ~torch.isnan(self._switch_dynamic_friction[env_ids])
        )
        mass_valid = ~torch.isnan(self._switch_mass_offset[env_ids])
        motor_valid = (~torch.isnan(self._switch_motor_stiffness_scale[env_ids])) & (
            ~torch.isnan(self._switch_motor_damping_scale[env_ids])
        )

        valid_target_count = friction_valid.long() + mass_valid.long() + motor_valid.long()
        if torch.any(valid_target_count != 1):
            bad_env_ids = env_ids[valid_target_count != 1].tolist()
            raise RuntimeError(
                "Each pending adaptation env must have exactly one valid switch target. "
                f"Broken env ids: {bad_env_ids}"
            )

    def _sample_switch_targets(self, env_ids) -> None:
        scenario_names = []
        if self.cfg.adaptation_enable_friction_switch:
            scenario_names.append("ultra_low_friction")
        if self.cfg.adaptation_enable_mass_switch:
            scenario_names.append("very_heavy")
        if self.cfg.adaptation_enable_motor_switch:
            scenario_names.append("very_weak_motor")
        if not scenario_names:
            raise RuntimeError("Adaptation env has no enabled switch scenarios.")

        choices = torch.randint(len(scenario_names), (len(env_ids),), device=self.device)
        enabled_mask = torch.rand(len(env_ids), device=self.device) < self._switch_episode_prob

        self._switch_static_friction[env_ids] = torch.nan
        self._switch_dynamic_friction[env_ids] = torch.nan
        self._switch_mass_offset[env_ids] = torch.nan
        self._switch_motor_stiffness_scale[env_ids] = torch.nan
        self._switch_motor_damping_scale[env_ids] = torch.nan
        self._switch_enabled[env_ids] = enabled_mask
        self._switch_scenario[env_ids] = -1

        for local_idx, env_id in enumerate(env_ids):
            if not bool(enabled_mask[local_idx].item()):
                continue
            scenario = scenario_names[int(choices[local_idx].item())]
            if scenario == "ultra_low_friction":
                self._switch_scenario[env_id] = 0
                self._switch_static_friction[env_id] = 0.1
                self._switch_dynamic_friction[env_id] = 0.1
            elif scenario == "very_heavy":
                self._switch_scenario[env_id] = 1
                self._switch_mass_offset[env_id] = 4.0
            elif scenario == "very_weak_motor":
                self._switch_scenario[env_id] = 2
                self._switch_motor_stiffness_scale[env_id] = 0.7
                self._switch_motor_damping_scale[env_id] = 0.7
            else:
                raise ValueError(f"Unknown adaptation switch scenario: {scenario}")

    def _trigger_event_term(self, term_name: str, env_ids: torch.Tensor, param_overrides: dict) -> None:
        term_cfg = self.event_manager.get_term_cfg(term_name)
        params = dict(term_cfg.params)
        params.update(param_overrides)
        term_cfg.func(self, env_ids, **params)

    def _resample_material_buckets(
        self, static_range: tuple[float, float], dynamic_range: tuple[float, float]
    ) -> None:
        term_cfg = self.event_manager.get_term_cfg("physics_material")
        term = term_cfg.func
        if not hasattr(term, "material_buckets"):
            return
        restitution_range = term_cfg.params.get("restitution_range", (0.0, 0.0))
        num_buckets = int(term_cfg.params.get("num_buckets", 1))
        ranges = torch.tensor(
            [static_range, dynamic_range, restitution_range],
            device="cpu",
            dtype=torch.float32,
        )
        term.material_buckets = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (num_buckets, 3), device="cpu")
        if term_cfg.params.get("make_consistent", False):
            term.material_buckets[:, 1] = torch.min(term.material_buckets[:, 0], term.material_buckets[:, 1])

    def _apply_switches(self, env_ids: torch.Tensor) -> None:
        friction_env_ids = env_ids[~torch.isnan(self._switch_static_friction[env_ids])]
        if len(friction_env_ids) > 0:
            friction_pairs = torch.stack(
                [self._switch_static_friction[friction_env_ids], self._switch_dynamic_friction[friction_env_ids]], dim=-1
            )
            unique_pairs = torch.unique(friction_pairs, dim=0)
            for pair in unique_pairs:
                static_value = float(pair[0].item())
                dynamic_value = float(pair[1].item())
                pair_env_ids = friction_env_ids[
                    (self._switch_static_friction[friction_env_ids] == static_value)
                    & (self._switch_dynamic_friction[friction_env_ids] == dynamic_value)
                ]
                self._resample_material_buckets((static_value, static_value), (dynamic_value, dynamic_value))
                self._trigger_event_term(
                    "physics_material",
                    pair_env_ids,
                    {
                        "static_friction_range": (static_value, static_value),
                        "dynamic_friction_range": (dynamic_value, dynamic_value),
                    },
                )

        mass_env_ids = env_ids[~torch.isnan(self._switch_mass_offset[env_ids])]
        if len(mass_env_ids) > 0:
            unique_mass_values = torch.unique(self._switch_mass_offset[mass_env_ids])
            for mass_value_tensor in unique_mass_values:
                mass_value = float(mass_value_tensor.item())
                value_env_ids = mass_env_ids[self._switch_mass_offset[mass_env_ids] == mass_value]
                self._trigger_event_term(
                    "add_base_mass",
                    value_env_ids,
                    {"mass_distribution_params": (mass_value, mass_value)},
                )

        motor_env_ids = env_ids[~torch.isnan(self._switch_motor_stiffness_scale[env_ids])]
        if len(motor_env_ids) > 0:
            motor_pairs = torch.stack(
                [
                    self._switch_motor_stiffness_scale[motor_env_ids],
                    self._switch_motor_damping_scale[motor_env_ids],
                ],
                dim=-1,
            )
            unique_pairs = torch.unique(motor_pairs, dim=0)
            for pair in unique_pairs:
                stiffness_value = float(pair[0].item())
                damping_value = float(pair[1].item())
                pair_env_ids = motor_env_ids[
                    (self._switch_motor_stiffness_scale[motor_env_ids] == stiffness_value)
                    & (self._switch_motor_damping_scale[motor_env_ids] == damping_value)
                ]
                self._trigger_event_term(
                    "motor_strength",
                    pair_env_ids,
                    {
                        "stiffness_distribution_params": (stiffness_value, stiffness_value),
                        "damping_distribution_params": (damping_value, damping_value),
                    },
                )
