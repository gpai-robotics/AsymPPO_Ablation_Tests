"""History layout helpers shared by deployment runtimes.

IsaacLab applies group-level history to each observation term independently,
flattens each term's complete history, and then concatenates terms. Given an
internal frame buffer shaped ``[history, policy_dim]``, deployment must
therefore emit term-major rather than directly flattening the frame buffer.
"""

from __future__ import annotations

from typing import Any

import numpy as np


ISAACLAB_TERM_MAJOR = "isaaclab_term_major"
TIME_MAJOR = "time_major"
SUPPORTED_HISTORY_LAYOUTS = (ISAACLAB_TERM_MAJOR, TIME_MAJOR)


def resolve_history_layout(observations_cfg: dict[str, Any]) -> str:
    """Resolve the frozen history layout, defaulting legacy bundles to IsaacLab."""
    layout = str(observations_cfg.get("history_layout", ISAACLAB_TERM_MAJOR))
    if layout not in SUPPORTED_HISTORY_LAYOUTS:
        raise ValueError(
            f"Unsupported policy history layout {layout!r}; expected one of {SUPPORTED_HISTORY_LAYOUTS}."
        )
    return layout


def flatten_policy_history(
    frame_history: np.ndarray,
    policy_order: list[dict[str, Any]],
    *,
    layout: str,
) -> np.ndarray:
    """Flatten ``[history, policy_dim]`` using the training-time layout."""
    history = np.asarray(frame_history, dtype=np.float32)
    if history.ndim != 2:
        raise ValueError(f"Expected 2D frame history, got shape {history.shape}.")
    if layout == TIME_MAJOR:
        return history.reshape(-1).copy()
    if layout != ISAACLAB_TERM_MAJOR:
        raise ValueError(f"Unsupported policy history layout: {layout!r}")

    cursor = 0
    term_histories: list[np.ndarray] = []
    for term in policy_order:
        dim = int(term["dim"])
        term_histories.append(history[:, cursor : cursor + dim].reshape(-1))
        cursor += dim
    if cursor != history.shape[1]:
        raise ValueError(
            f"Policy observation terms sum to {cursor}, but history frames have dimension {history.shape[1]}."
        )
    return np.concatenate(term_histories, dtype=np.float32)
