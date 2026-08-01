"""Resolve optional external repositories used by deployment tools.

Large third-party repositories are intentionally not committed into this
repository. Resolution order is:

1. Explicit environment variable.
2. Legacy ``<repo>/reference_repos/<name>`` location.
3. Shared workstation ``<workspace>/RefRepo/<name>`` location.
"""

from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SHARED_REFERENCE_ROOT = REPO_ROOT.parents[1] / "RefRepo"


def resolve_reference_repo(name: str, env_var: str) -> Path:
    override = os.environ.get(env_var)
    if override:
        return Path(override).expanduser().resolve()

    legacy = REPO_ROOT / "reference_repos" / name
    if legacy.exists():
        return legacy.resolve()

    shared = SHARED_REFERENCE_ROOT / name
    if shared.exists():
        return shared.resolve()

    return legacy


UNITREE_RL_MJLAB_ROOT = resolve_reference_repo("unitree_rl_mjlab", "RMA_UNITREE_RL_MJLAB_ROOT")
MUJOCO_MENAGERIE_ROOT = resolve_reference_repo("mujoco_menagerie", "RMA_MUJOCO_MENAGERIE_ROOT")
UNITREE_MUJOCO_ROOT = resolve_reference_repo("unitree_mujoco", "RMA_UNITREE_MUJOCO_ROOT")
MJLAB_ROOT = resolve_reference_repo("mjlab", "RMA_MJLAB_ROOT")
SDK2PY_ROOT = resolve_reference_repo("sim2real_unitree_sdk2py", "RMA_UNITREE_SDK2PY_ROOT")
