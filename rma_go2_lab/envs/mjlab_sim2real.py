"""Sim2real helpers for the mjlab-style branch.

This module keeps the branch-local sim2real logic out of upstream IsaacLab:

- encoder bias:
  - observations see biased joint positions
  - position actions subtract the bias before writing physical targets
- observation delay:
  - selected policy observations pass through a stateful delay modifier
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.buffers.delay_buffer import DelayBuffer
from isaaclab.utils.modifiers import ModifierCfg

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")
_ENCODER_BIAS_STORE_ATTR = "_mjlab_encoder_bias_store"
_OBSERVATION_DELAY_STATES: dict[tuple[str, str, tuple[int, ...]], dict[str, torch.Tensor | DelayBuffer | int]] = {}


def _resolve_env_ids(
    env,
    env_ids: torch.Tensor | Sequence[int] | slice | None,
) -> torch.Tensor:
    """Return env ids as an index tensor on the env device."""

    if env_ids is None or isinstance(env_ids, slice):
        return torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    if isinstance(env_ids, torch.Tensor):
        return env_ids.to(device=env.device, dtype=torch.long)
    return torch.as_tensor(env_ids, device=env.device, dtype=torch.long)


def _resolve_joint_ids(asset, joint_ids) -> torch.Tensor:
    """Resolve joint ids into an index tensor."""

    device = asset.data.joint_pos.device
    if isinstance(joint_ids, slice):
        return torch.arange(asset.num_joints, device=device, dtype=torch.long)
    if isinstance(joint_ids, torch.Tensor):
        return joint_ids.to(device=device, dtype=torch.long)
    return torch.as_tensor(joint_ids, device=device, dtype=torch.long)


def _get_encoder_bias_store(env) -> dict[str, torch.Tensor]:
    """Create or return the per-asset encoder bias store on the env."""

    store = getattr(env, _ENCODER_BIAS_STORE_ATTR, None)
    if store is None:
        store = {}
        setattr(env, _ENCODER_BIAS_STORE_ATTR, store)
    return store


def ensure_encoder_bias(
    env,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Return the full per-joint encoder bias tensor for an asset."""

    asset = env.scene[asset_cfg.name]
    store = _get_encoder_bias_store(env)
    bias = store.get(asset_cfg.name)
    expected_shape = (env.num_envs, asset.num_joints)
    if bias is None or tuple(bias.shape) != expected_shape or bias.device != env.device:
        bias = torch.zeros(expected_shape, device=env.device)
        store[asset_cfg.name] = bias
    return bias


def get_encoder_bias(
    env,
    asset_name: str = "robot",
    joint_ids=slice(None),
) -> torch.Tensor:
    """Return encoder bias for the requested joints."""

    full_bias = ensure_encoder_bias(env, SceneEntityCfg(asset_name))
    asset = env.scene[asset_name]
    resolved_joint_ids = _resolve_joint_ids(asset, joint_ids)
    return full_bias[:, resolved_joint_ids]


def randomize_encoder_bias(
    env,
    env_ids: torch.Tensor | Sequence[int] | slice | None,
    bias_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
    """Randomize persistent joint encoder bias for selected environments."""

    asset = env.scene[asset_cfg.name]
    full_bias = ensure_encoder_bias(env, asset_cfg)
    resolved_env_ids = _resolve_env_ids(env, env_ids)
    resolved_joint_ids = _resolve_joint_ids(asset, asset_cfg.joint_ids)
    samples = torch.empty(
        (len(resolved_env_ids), len(resolved_joint_ids)),
        device=env.device,
    ).uniform_(bias_range[0], bias_range[1])
    full_bias[resolved_env_ids[:, None], resolved_joint_ids] = samples


def joint_pos_rel_biased(
    env,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Relative joint position observation with persistent encoder bias."""

    asset = env.scene[asset_cfg.name]
    resolved_joint_ids = _resolve_joint_ids(asset, asset_cfg.joint_ids)
    joint_pos_rel = asset.data.joint_pos[:, resolved_joint_ids] - asset.data.default_joint_pos[:, resolved_joint_ids]
    return joint_pos_rel + get_encoder_bias(env, asset_name=asset_cfg.name, joint_ids=resolved_joint_ids)


def joint_pos_biased(
    env,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Alias matching the repo's observation-term naming."""

    return joint_pos_rel_biased(env, asset_cfg=asset_cfg)


class EncoderBiasJointPositionAction(JointPositionAction):
    """Joint-position action that compensates persistent encoder bias."""

    def apply_actions(self):
        target = self.processed_actions - get_encoder_bias(
            self._env,
            asset_name=self.cfg.asset_name,
            joint_ids=self._joint_ids,
        )
        self._asset.set_joint_position_target(target, joint_ids=self._joint_ids)


def _get_delay_state(
    data: torch.Tensor,
    *,
    term_key: str,
    min_lag: int,
    max_lag: int,
    update_period: int,
    per_env_phase: bool,
) -> dict[str, torch.Tensor | DelayBuffer | int]:
    state_key = (term_key, str(data.device), tuple(data.shape))
    state = _OBSERVATION_DELAY_STATES.get(state_key)
    if state is None:
        num_envs = data.shape[0]
        phase_offsets = (
            torch.arange(num_envs, device=data.device, dtype=torch.long) % max(update_period, 1)
            if update_period > 0 and per_env_phase
            else torch.zeros(num_envs, device=data.device, dtype=torch.long)
        )
        lags = torch.full((num_envs,), min_lag, device=data.device, dtype=torch.long)
        state = {
            "buffer": DelayBuffer(history_length=max_lag, batch_size=num_envs, device=str(data.device)),
            "phase_offsets": phase_offsets,
            "lags": lags,
            "step_index": 0,
        }
        _OBSERVATION_DELAY_STATES[state_key] = state
    return state


def observation_delay(
    data: torch.Tensor,
    *,
    term_key: str,
    min_lag: int,
    max_lag: int,
    per_env: bool = True,
    hold_prob: float = 0.0,
    update_period: int = 0,
    per_env_phase: bool = True,
) -> torch.Tensor:
    """Stateful stochastic observation delay as a function-style modifier."""

    if max_lag <= 0:
        return data

    state = _get_delay_state(
        data,
        term_key=term_key,
        min_lag=min_lag,
        max_lag=max_lag,
        update_period=update_period,
        per_env_phase=per_env_phase,
    )
    buffer = state["buffer"]
    phase_offsets = state["phase_offsets"]
    lags = state["lags"]
    step_index = int(state["step_index"])
    num_envs = data.shape[0]

    if update_period <= 0:
        resample_mask = torch.ones(num_envs, device=data.device, dtype=torch.bool)
    elif per_env_phase:
        resample_mask = ((step_index + phase_offsets) % update_period) == 0
    else:
        should_resample = (step_index % update_period) == 0
        resample_mask = torch.full((num_envs,), should_resample, device=data.device, dtype=torch.bool)

    if hold_prob > 0.0:
        hold_mask = torch.rand(num_envs, device=data.device) < hold_prob
        resample_mask &= ~hold_mask

    env_ids = torch.nonzero(resample_mask, as_tuple=False).squeeze(-1)
    if len(env_ids) > 0:
        if max_lag <= min_lag:
            new_lags = torch.full((len(env_ids),), min_lag, device=data.device, dtype=torch.long)
        elif per_env:
            new_lags = torch.randint(
                low=min_lag,
                high=max_lag + 1,
                size=(len(env_ids),),
                device=data.device,
                dtype=torch.long,
            )
        else:
            new_lag = torch.randint(
                low=min_lag,
                high=max_lag + 1,
                size=(1,),
                device=data.device,
                dtype=torch.long,
            )
            new_lags = new_lag.expand(len(env_ids))
        lags[env_ids] = new_lags
        buffer.set_time_lag(lags)

    delayed = buffer.compute(data)
    state["step_index"] = step_index + 1
    return delayed


def stepwise_value(
    env,
    env_ids,
    data,
    milestones: Sequence[int],
    values: Sequence,
):
    """Pick a staged value based on the global training step.

    The returned value is used directly by curriculum terms that rewrite
    manager configs at runtime.
    """

    stage = 0
    for threshold in milestones:
        if env.common_step_counter >= threshold:
            stage += 1
    stage = min(stage, len(values) - 1)
    return values[stage]


def make_observation_delay_modifier_cfg(
    *,
    min_lag: int,
    max_lag: int,
    per_env: bool = True,
    hold_prob: float = 0.0,
    update_period: int = 0,
    per_env_phase: bool = True,
) -> ModifierCfg:
    """Create a delay modifier config."""

    return ModifierCfg(
        func=observation_delay,
        params={
            "term_key": "unset",
            "min_lag": min_lag,
            "max_lag": max_lag,
            "per_env": per_env,
            "hold_prob": hold_prob,
            "update_period": update_period,
            "per_env_phase": per_env_phase,
        },
    )


def append_observation_delay(
    obs_term,
    *,
    term_key: str,
    min_lag: int,
    max_lag: int,
    per_env: bool = True,
    hold_prob: float = 0.0,
    update_period: int = 0,
    per_env_phase: bool = True,
) -> None:
    """Append a delay modifier to an observation term if delay is enabled."""

    if max_lag <= 0:
        return
    modifier = make_observation_delay_modifier_cfg(
        min_lag=min_lag,
        max_lag=max_lag,
        per_env=per_env,
        hold_prob=hold_prob,
        update_period=update_period,
        per_env_phase=per_env_phase,
    )
    modifier.params["term_key"] = term_key
    if obs_term.modifiers is None:
        obs_term.modifiers = [modifier]
    else:
        obs_term.modifiers.append(modifier)


def apply_policy_sensor_delay(
    policy_obs,
    *,
    min_lag: int,
    max_lag: int,
    per_env: bool = True,
    hold_prob: float = 0.0,
    update_period: int = 0,
    per_env_phase: bool = True,
) -> None:
    """Apply delay to policy sensor-like terms, but not commands/actions."""

    delayed_term_candidates = (
        ("base_ang_vel",),
        ("projected_gravity",),
        ("joint_pos_rel", "joint_pos"),
        ("joint_vel_rel", "joint_vel"),
    )
    for candidate_names in delayed_term_candidates:
        term = None
        for term_name in candidate_names:
            term = getattr(policy_obs, term_name, None)
            if term is not None:
                break
        if term is None:
            continue
        append_observation_delay(
            term,
            term_key=term_name,
            min_lag=min_lag,
            max_lag=max_lag,
            per_env=per_env,
            hold_prob=hold_prob,
            update_period=update_period,
            per_env_phase=per_env_phase,
        )


def mjlab_phase_from_step(step_count: int, phase_step_thresholds: tuple[int, ...]) -> int:
    """Map a global step count to a compact phase id.

    Args:
        step_count: Global env step counter.
        phase_step_thresholds: Strictly increasing step thresholds that mark
            the beginning of later phases. For example, `(8e6, 20e6, 40e6)`
            yields four phases: `0, 1, 2, 3`.
    """

    phase = 0
    for threshold in phase_step_thresholds:
        if step_count < threshold:
            break
        phase += 1
    return phase


def mjlab_phase_progress(step_count: int, phase_step_thresholds: tuple[int, ...]) -> float:
    """Return a normalized progress scalar for the current phase ladder."""

    if not phase_step_thresholds:
        return 1.0
    last_threshold = float(phase_step_thresholds[-1])
    if last_threshold <= 0.0:
        return 1.0
    return min(1.0, float(step_count) / last_threshold)


def terrain_levels_vel_by_type(
    env,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    reward_term_name: str = "track_lin_vel_xy_exp",
    reward_threshold_scale: float = 0.8,
) -> dict[str, float]:
    """Apply the standard terrain curriculum and expose per-terrain-type stats.

    Each terrain family is advanced independently. Families that are already
    tracking well are allowed to climb the difficulty ladder, while families
    that are underperforming are held back so they get more training on their
    current difficulty band.
    """

    terrain = env.scene.terrain
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command("base_velocity")
    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    terrain_names = list(terrain.cfg.terrain_generator.sub_terrains.keys())
    type_ids = terrain.terrain_types[env_ids]

    stats: dict[str, float] = {}
    for terrain_id, terrain_name in enumerate(terrain_names):
        type_mask = type_ids == terrain_id
        if not torch.any(type_mask):
            stats[terrain_name] = float("nan")
            continue

        type_env_ids = env_ids[type_mask]
        distance = torch.norm(asset.data.root_pos_w[type_env_ids, :2] - terrain.env_origins[type_env_ids, :2], dim=1)
        move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
        move_down = distance < torch.norm(command[type_env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
        move_down *= ~move_up

        type_reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][type_env_ids]) / env.max_episode_length_s
        reward_threshold = reward_term.weight * reward_threshold_scale
        if type_reward < reward_threshold:
            move_up = torch.zeros_like(move_up)

        terrain.update_env_origins(type_env_ids, move_up, move_down)

        stats[terrain_name] = float(torch.mean(terrain.terrain_levels[type_env_ids].float()).item())
        stats[f"{terrain_name}/reward"] = float(type_reward.item())
        stats[f"{terrain_name}/eligible_up"] = float(move_up.float().mean().item())
    return stats
