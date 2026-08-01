"""Reusable contact-based observation helpers for locomotion pipelines."""

from __future__ import annotations

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor


def foot_contact_state(
    env,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_foot"),
    threshold: float = 1.0,
) -> torch.Tensor:
    """Binary per-foot contact state derived from the configured contact sensor.

    The returned tensor has shape ``(num_envs, num_feet)`` and is ordered by the
    sensor body's resolved foot indices. We use the max force over the contact
    sensor history window to suppress one-step flicker.
    """

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1).values
    return (forces > threshold).float()


def foot_phase_features(
    env,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_foot"),
    stance_scale_s: float = 0.30,
    swing_scale_s: float = 0.20,
) -> torch.Tensor:
    """Per-foot timing progress features derived from contact/air timers.

    Returns a tensor of shape ``(num_envs, 2 * num_feet)`` with:

    - normalized stance progress for each foot
    - normalized swing progress for each foot

    This is intentionally not a global gait clock. It is a local phase-like
    descriptor built from the contact sensor's tracked timing state, which makes
    it easy to pair with the explicit binary contact observation.
    """

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    current_contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    current_air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]

    if current_contact_time is None or current_air_time is None:
        raise RuntimeError(
            "foot_phase_features requires the contact sensor to track air/contact time. "
            "Enable track_air_time=True on the configured contact sensor."
        )

    stance_progress = torch.clamp(current_contact_time / max(stance_scale_s, 1.0e-6), 0.0, 1.0)
    swing_progress = torch.clamp(current_air_time / max(swing_scale_s, 1.0e-6), 0.0, 1.0)
    return torch.cat((stance_progress, swing_progress), dim=-1)
