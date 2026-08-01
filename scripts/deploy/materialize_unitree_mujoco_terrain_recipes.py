#!/usr/bin/env python3
"""Materialize forward-facing MuJoCo terrain recipes using Unitree's terrain tool.

This script treats the upstream ``terrain_generator.py`` as the authoritative
terrain-construction API and builds a small family of repo-owned terrain
recipes on top of it. The recipes are intentionally:

- forward-facing relative to the default Go2 spawn
- wide enough to avoid the "narrow corridor" feel
- mixed across discrete rough blocks and heightfield sections
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import xml.etree.ElementTree as ET


REPO_ROOT = Path(__file__).resolve().parents[2]
TERRAIN_TOOL_DIR = REPO_ROOT / "reference_repos" / "unitree_mujoco" / "terrain_tool"
TERRAIN_TOOL_PATH = TERRAIN_TOOL_DIR / "terrain_generator.py"
GO2_SCENE_DIR = REPO_ROOT / "reference_repos" / "unitree_mujoco" / "unitree_robots" / "go2"


RECIPES: dict[str, dict[str, object]] = {
    "forward_rough_wide_a": {
        "boxes": [
            {"position": [0.95, -1.65, -0.02], "size": [0.50, 0.50, 0.04]},
            {"position": [0.95, 1.65, -0.02], "size": [0.50, 0.50, 0.04]},
            {"position": [3.15, -1.75, 0.05], "size": [0.40, 0.40, 0.10], "euler": [0.0, 0.0, 0.18]},
            {"position": [3.55, 1.72, 0.06], "size": [0.46, 0.36, 0.12], "euler": [0.0, 0.0, -0.22]},
        ],
        "rough": [
            {
                "init_pos": [0.55, -1.60, 0.0],
                "nums": [9, 10],
                "box_size": [0.30, 0.28, 0.14],
                "separation": [0.26, 0.28],
                "box_size_rand": [0.08, 0.08, 0.10],
                "box_euler_rand": [0.18, 0.18, 0.14],
                "separation_rand": [0.04, 0.04],
            },
            {
                "init_pos": [2.80, -1.20, 0.0],
                "nums": [4, 7],
                "box_size": [0.22, 0.26, 0.10],
                "separation": [0.34, 0.34],
                "box_size_rand": [0.04, 0.04, 0.05],
                "box_euler_rand": [0.10, 0.10, 0.12],
                "separation_rand": [0.03, 0.03],
            }
        ],
        "perlin": [
            {
                "position": [4.45, 0.0, 0.0],
                "size": [2.9, 2.9],
                "height_scale": 0.12,
                "negative_height": 0.12,
                "smooth": 62.0,
            }
        ],
        "suspend_stairs": [
            {
                "init_pos": [6.55, -0.85, 0.0],
                "yaw": 0.0,
                "width": 0.22,
                "height": 0.08,
                "length": 0.95,
                "gap": 0.03,
                "stair_nums": 5,
            }
        ],
        "image": [
            {
                "position": [8.05, 0.35, 0.0],
                "euler": [0.0, 0.0, 0.10],
                "size": [2.9, 2.4],
                "height_scale": 0.02,
                "negative_height": 0.10,
            }
        ],
        "geometries": [
            {"position": [6.10, 1.18, 0.14], "size": [0.42, 0.42, 0.28], "geo_type": "cylinder"},
            {"position": [7.15, 1.55, 0.18], "size": [0.34, 0.34, 0.36], "geo_type": "sphere"},
        ],
    },
    "forward_rough_wide_b": {
        "boxes": [
            {"position": [1.25, -1.75, 0.05], "size": [0.60, 0.45, 0.10], "euler": [0.0, 0.0, 0.22]},
            {"position": [1.20, 1.75, 0.05], "size": [0.60, 0.45, 0.10], "euler": [0.0, 0.0, -0.22]},
            {"position": [4.10, 0.00, -0.01], "size": [0.75, 2.80, 0.02]},
        ],
        "rough": [
            {
                "init_pos": [0.55, -1.70, 0.0],
                "nums": [8, 11],
                "box_size": [0.30, 0.34, 0.18],
                "separation": [0.26, 0.26],
                "box_size_rand": [0.09, 0.09, 0.11],
                "box_euler_rand": [0.20, 0.20, 0.16],
                "separation_rand": [0.05, 0.04],
            },
            {
                "init_pos": [5.15, -1.05, 0.0],
                "nums": [3, 7],
                "box_size": [0.24, 0.24, 0.12],
                "separation": [0.38, 0.34],
                "box_size_rand": [0.04, 0.04, 0.05],
                "box_euler_rand": [0.12, 0.12, 0.12],
                "separation_rand": [0.03, 0.03],
            }
        ],
        "perlin": [
            {
                "position": [3.80, 0.00, 0.0],
                "size": [2.4, 3.4],
                "height_scale": 0.14,
                "negative_height": 0.14,
                "smooth": 48.0,
            }
        ],
        "stairs": [
            {
                "init_pos": [6.25, -0.95, 0.0],
                "yaw": 0.08,
                "width": 0.20,
                "height": 0.06,
                "length": 1.35,
                "stair_nums": 6,
            }
        ],
        "image": [
            {
                "position": [8.60, 0.55, 0.0],
                "euler": [0.0, 0.0, -0.14],
                "size": [3.1, 2.8],
                "height_scale": 0.02,
                "negative_height": 0.10,
            }
        ],
        "geometries": [
            {"position": [7.05, 1.20, 0.20], "size": [0.45, 0.20, 0.40], "geo_type": "capsule", "euler": [0.0, 1.57, 0.35]},
            {"position": [8.10, -1.25, 0.16], "size": [0.42, 0.30, 0.32], "geo_type": "ellipsoid"},
        ],
    },
    "forward_rough_mixed_c": {
        "boxes": [
            {"position": [0.95, 0.00, -0.01], "size": [0.90, 3.20, 0.02]},
            {"position": [2.55, -1.55, 0.08], "size": [0.50, 0.50, 0.16], "euler": [0.0, 0.0, 0.30]},
            {"position": [2.95, 1.55, 0.08], "size": [0.50, 0.50, 0.16], "euler": [0.0, 0.0, -0.30]},
        ],
        "rough": [
            {
                "init_pos": [1.05, -1.35, 0.0],
                "nums": [5, 8],
                "box_size": [0.26, 0.28, 0.12],
                "separation": [0.34, 0.30],
                "box_size_rand": [0.06, 0.06, 0.08],
                "box_euler_rand": [0.14, 0.14, 0.14],
                "separation_rand": [0.04, 0.05],
            },
            {
                "init_pos": [4.10, -1.05, 0.0],
                "nums": [4, 7],
                "box_size": [0.22, 0.24, 0.09],
                "separation": [0.36, 0.34],
                "box_size_rand": [0.05, 0.05, 0.05],
                "box_euler_rand": [0.10, 0.10, 0.10],
                "separation_rand": [0.03, 0.03],
            },
        ],
        "perlin": [
            {
                "position": [3.85, 1.20, 0.0],
                "size": [1.9, 1.8],
                "height_scale": 0.08,
                "negative_height": 0.12,
                "smooth": 88.0,
            },
            {
                "position": [5.95, -1.10, 0.0],
                "size": [2.2, 1.8],
                "height_scale": 0.13,
                "negative_height": 0.12,
                "smooth": 56.0,
            }
        ],
        "suspend_stairs": [
            {
                "init_pos": [6.30, 0.95, 0.0],
                "yaw": -0.12,
                "width": 0.20,
                "height": 0.07,
                "length": 1.00,
                "gap": 0.025,
                "stair_nums": 5,
            }
        ],
        "image": [
            {
                "position": [8.35, 0.15, 0.0],
                "euler": [0.0, 0.0, -0.10],
                "size": [2.8, 3.0],
                "height_scale": 0.02,
                "negative_height": 0.10,
            }
        ],
        "geometries": [
            {"position": [5.55, 1.55, 0.18], "size": [0.32, 0.32, 0.36], "geo_type": "sphere"},
            {"position": [7.65, -1.35, 0.18], "size": [0.40, 0.28, 0.36], "geo_type": "cylinder", "euler": [0.0, 0.0, 0.22]},
        ],
    },
    "forward_rough_with_stairs_d": {
        "boxes": [
            {"position": [0.75, -1.55, -0.01], "size": [0.45, 0.55, 0.02]},
            {"position": [0.75, 1.55, -0.01], "size": [0.45, 0.55, 0.02]},
            {"position": [5.35, 1.55, 0.08], "size": [0.62, 0.42, 0.16], "euler": [0.0, 0.0, -0.18]},
        ],
        "rough": [
            {
                "init_pos": [0.55, -1.45, 0.0],
                "nums": [6, 8],
                "box_size": [0.30, 0.28, 0.14],
                "separation": [0.30, 0.30],
                "box_size_rand": [0.06, 0.06, 0.08],
                "box_euler_rand": [0.14, 0.14, 0.12],
                "separation_rand": [0.04, 0.04],
            },
            {
                "init_pos": [3.20, -1.20, 0.0],
                "nums": [3, 6],
                "box_size": [0.24, 0.24, 0.11],
                "separation": [0.38, 0.34],
                "box_size_rand": [0.04, 0.04, 0.05],
                "box_euler_rand": [0.10, 0.10, 0.10],
                "separation_rand": [0.03, 0.03],
            }
        ],
        "perlin": [
            {
                "position": [4.35, -0.15, 0.0],
                "size": [2.6, 2.3],
                "height_scale": 0.12,
                "negative_height": 0.12,
                "smooth": 68.0,
            }
        ],
        "stairs": [
            {
                "init_pos": [6.55, -0.55, 0.0],
                "yaw": 0.05,
                "width": 0.18,
                "height": 0.05,
                "length": 2.2,
                "stair_nums": 6,
            }
        ],
        "suspend_stairs": [
            {
                "init_pos": [6.55, 1.05, 0.0],
                "yaw": -0.08,
                "width": 0.18,
                "height": 0.08,
                "length": 1.10,
                "gap": 0.03,
                "stair_nums": 5,
            }
        ],
        "image": [
            {
                "position": [9.10, 0.10, 0.0],
                "euler": [0.0, 0.0, 0.08],
                "size": [2.7, 2.3],
                "height_scale": 0.02,
                "negative_height": 0.10,
            }
        ],
        "geometries": [
            {"position": [8.15, -1.25, 0.16], "size": [0.42, 0.42, 0.30], "geo_type": "cylinder"},
            {"position": [8.55, 1.35, 0.18], "size": [0.44, 0.24, 0.34], "geo_type": "capsule", "euler": [0.0, 1.57, -0.22]},
        ],
    },
    "forward_technical_e": {
        "boxes": [
            {"position": [0.95, 0.00, -0.01], "size": [0.80, 3.30, 0.02]},
            {"position": [2.20, -1.55, 0.10], "size": [0.44, 0.44, 0.20], "euler": [0.0, 0.0, 0.28]},
            {"position": [2.55, 1.50, 0.10], "size": [0.44, 0.44, 0.20], "euler": [0.0, 0.0, -0.25]},
            {"position": [7.20, 0.00, -0.01], "size": [0.55, 2.60, 0.02]},
        ],
        "rough": [
            {
                "init_pos": [0.75, -1.45, 0.0],
                "nums": [7, 9],
                "box_size": [0.26, 0.24, 0.18],
                "separation": [0.28, 0.26],
                "box_size_rand": [0.08, 0.08, 0.12],
                "box_euler_rand": [0.22, 0.22, 0.18],
                "separation_rand": [0.04, 0.04],
            },
            {
                "init_pos": [4.10, -1.00, 0.0],
                "nums": [4, 6],
                "box_size": [0.20, 0.22, 0.14],
                "separation": [0.34, 0.34],
                "box_size_rand": [0.04, 0.04, 0.06],
                "box_euler_rand": [0.12, 0.12, 0.14],
                "separation_rand": [0.03, 0.03],
            },
        ],
        "perlin": [
            {
                "position": [4.35, 0.85, 0.0],
                "size": [2.1, 1.8],
                "height_scale": 0.16,
                "negative_height": 0.14,
                "smooth": 42.0,
            },
            {
                "position": [5.55, -1.05, 0.0],
                "size": [1.8, 1.8],
                "height_scale": 0.14,
                "negative_height": 0.14,
                "smooth": 38.0,
            },
        ],
        "stairs": [
            {
                "init_pos": [6.35, -0.65, 0.0],
                "yaw": 0.10,
                "width": 0.17,
                "height": 0.06,
                "length": 1.20,
                "stair_nums": 7,
            }
        ],
        "suspend_stairs": [
            {
                "init_pos": [6.40, 0.95, 0.0],
                "yaw": -0.10,
                "width": 0.18,
                "height": 0.09,
                "length": 0.95,
                "gap": 0.025,
                "stair_nums": 6,
            }
        ],
        "image": [
            {
                "position": [8.55, 0.10, 0.0],
                "euler": [0.0, 0.0, 0.14],
                "size": [3.0, 2.5],
                "height_scale": 0.03,
                "negative_height": 0.12,
            }
        ],
        "geometries": [
            {"position": [3.85, 1.55, 0.20], "size": [0.40, 0.40, 0.38], "geo_type": "cylinder"},
            {"position": [7.95, -1.30, 0.18], "size": [0.46, 0.22, 0.36], "geo_type": "capsule", "euler": [0.0, 1.57, 0.26]},
        ],
    },
    "forward_technical_f": {
        "boxes": [
            {"position": [1.05, -1.75, 0.04], "size": [0.55, 0.48, 0.08], "euler": [0.0, 0.0, 0.25]},
            {"position": [1.05, 1.75, 0.04], "size": [0.55, 0.48, 0.08], "euler": [0.0, 0.0, -0.25]},
            {"position": [4.30, 0.00, 0.05], "size": [0.45, 3.10, 0.10]},
            {"position": [8.10, 0.00, -0.01], "size": [0.48, 2.50, 0.02]},
        ],
        "rough": [
            {
                "init_pos": [0.70, -1.65, 0.0],
                "nums": [6, 10],
                "box_size": [0.24, 0.28, 0.20],
                "separation": [0.26, 0.24],
                "box_size_rand": [0.08, 0.08, 0.12],
                "box_euler_rand": [0.24, 0.24, 0.20],
                "separation_rand": [0.04, 0.04],
            }
        ],
        "perlin": [
            {
                "position": [3.20, 0.00, 0.0],
                "size": [1.8, 3.3],
                "height_scale": 0.18,
                "negative_height": 0.16,
                "smooth": 32.0,
            }
        ],
        "stairs": [
            {
                "init_pos": [5.95, -0.85, 0.0],
                "yaw": 0.14,
                "width": 0.16,
                "height": 0.07,
                "length": 1.00,
                "stair_nums": 8,
            }
        ],
        "suspend_stairs": [
            {
                "init_pos": [6.10, 0.95, 0.0],
                "yaw": -0.14,
                "width": 0.16,
                "height": 0.10,
                "length": 0.92,
                "gap": 0.035,
                "stair_nums": 7,
            }
        ],
        "image": [
            {
                "position": [8.85, 0.15, 0.0],
                "euler": [0.0, 0.0, -0.12],
                "size": [3.2, 2.7],
                "height_scale": 0.035,
                "negative_height": 0.13,
            }
        ],
        "geometries": [
            {"position": [5.25, 1.35, 0.18], "size": [0.42, 0.28, 0.34], "geo_type": "ellipsoid"},
            {"position": [7.35, 1.30, 0.22], "size": [0.48, 0.22, 0.42], "geo_type": "capsule", "euler": [0.0, 1.57, -0.30]},
        ],
    },
    "forward_technical_g": {
        "boxes": [
            {"position": [1.10, 0.00, -0.01], "size": [0.65, 3.20, 0.02]},
            {"position": [2.80, -1.15, 0.10], "size": [0.42, 0.42, 0.20]},
            {"position": [2.80, 1.15, 0.10], "size": [0.42, 0.42, 0.20]},
            {"position": [6.80, -1.55, 0.07], "size": [0.55, 0.45, 0.14], "euler": [0.0, 0.0, 0.22]},
            {"position": [7.15, 1.55, 0.07], "size": [0.55, 0.45, 0.14], "euler": [0.0, 0.0, -0.22]},
        ],
        "rough": [
            {
                "init_pos": [1.10, -1.10, 0.0],
                "nums": [5, 7],
                "box_size": [0.22, 0.22, 0.16],
                "separation": [0.33, 0.32],
                "box_size_rand": [0.06, 0.06, 0.10],
                "box_euler_rand": [0.18, 0.18, 0.18],
                "separation_rand": [0.04, 0.04],
            },
            {
                "init_pos": [4.75, -1.55, 0.0],
                "nums": [4, 10],
                "box_size": [0.18, 0.18, 0.11],
                "separation": [0.22, 0.24],
                "box_size_rand": [0.04, 0.04, 0.06],
                "box_euler_rand": [0.10, 0.10, 0.12],
                "separation_rand": [0.02, 0.02],
            },
        ],
        "perlin": [
            {
                "position": [4.10, 1.15, 0.0],
                "size": [2.2, 1.7],
                "height_scale": 0.15,
                "negative_height": 0.15,
                "smooth": 40.0,
            }
        ],
        "suspend_stairs": [
            {
                "init_pos": [6.00, 0.10, 0.0],
                "yaw": 0.0,
                "width": 0.15,
                "height": 0.10,
                "length": 0.90,
                "gap": 0.04,
                "stair_nums": 8,
            }
        ],
        "image": [
            {
                "position": [8.60, 0.00, 0.0],
                "euler": [0.0, 0.0, 0.18],
                "size": [2.7, 3.0],
                "height_scale": 0.03,
                "negative_height": 0.12,
            }
        ],
        "geometries": [
            {"position": [5.95, -0.95, 0.18], "size": [0.34, 0.34, 0.34], "geo_type": "sphere"},
            {"position": [8.05, 1.00, 0.20], "size": [0.42, 0.42, 0.28], "geo_type": "cylinder"},
        ],
    },
    "forward_technical_h": {
        "boxes": [
            {"position": [0.95, -1.40, -0.01], "size": [0.50, 0.52, 0.02]},
            {"position": [0.95, 1.40, -0.01], "size": [0.50, 0.52, 0.02]},
            {"position": [3.70, 0.00, 0.06], "size": [0.52, 2.90, 0.12]},
            {"position": [8.55, 0.00, 0.05], "size": [0.42, 2.20, 0.10]},
        ],
        "rough": [
            {
                "init_pos": [0.85, -1.35, 0.0],
                "nums": [6, 8],
                "box_size": [0.22, 0.24, 0.16],
                "separation": [0.28, 0.28],
                "box_size_rand": [0.07, 0.07, 0.11],
                "box_euler_rand": [0.20, 0.20, 0.18],
                "separation_rand": [0.04, 0.04],
            }
        ],
        "perlin": [
            {
                "position": [5.05, -1.05, 0.0],
                "size": [1.8, 1.7],
                "height_scale": 0.17,
                "negative_height": 0.16,
                "smooth": 34.0,
            },
            {
                "position": [5.05, 1.05, 0.0],
                "size": [1.8, 1.7],
                "height_scale": 0.17,
                "negative_height": 0.16,
                "smooth": 34.0,
            }
        ],
        "stairs": [
            {
                "init_pos": [6.55, -1.00, 0.0],
                "yaw": 0.16,
                "width": 0.15,
                "height": 0.07,
                "length": 0.95,
                "stair_nums": 8,
            }
        ],
        "suspend_stairs": [
            {
                "init_pos": [6.55, 1.00, 0.0],
                "yaw": -0.16,
                "width": 0.15,
                "height": 0.10,
                "length": 0.88,
                "gap": 0.04,
                "stair_nums": 8,
            }
        ],
        "image": [
            {
                "position": [9.00, 0.00, 0.0],
                "euler": [0.0, 0.0, 0.0],
                "size": [2.6, 2.8],
                "height_scale": 0.04,
                "negative_height": 0.14,
            }
        ],
        "geometries": [
            {"position": [7.55, 0.00, 0.22], "size": [0.48, 0.22, 0.44], "geo_type": "capsule", "euler": [0.0, 1.57, 0.0]},
            {"position": [7.95, -1.25, 0.18], "size": [0.40, 0.26, 0.34], "geo_type": "ellipsoid"},
            {"position": [7.95, 1.25, 0.18], "size": [0.40, 0.26, 0.34], "geo_type": "ellipsoid"},
        ],
    },
}


def _load_terrain_tool_module():
    spec = importlib.util.spec_from_file_location("unitree_mujoco_terrain_generator", TERRAIN_TOOL_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load terrain tool module from {TERRAIN_TOOL_PATH}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Unitree terrain_tool dependencies are missing. "
            "Install them in the MuJoCo env, e.g. `pip install noise opencv-python numpy`, "
            "then rerun terrain recipe materialization."
        ) from exc
    return module


def _apply_recipe(module, tg, recipe_name: str, recipe_cfg: dict[str, object]) -> None:
    for box_cfg in recipe_cfg.get("boxes", []):
        tg.AddBox(
            position=box_cfg["position"],
            euler=box_cfg.get("euler", [0.0, 0.0, 0.0]),
            size=box_cfg["size"],
        )

    for geom_cfg in recipe_cfg.get("geometries", []):
        tg.AddGeometry(
            position=geom_cfg["position"],
            euler=geom_cfg.get("euler", [0.0, 0.0, 0.0]),
            size=geom_cfg["size"],
            geo_type=geom_cfg.get("geo_type", "box"),
        )

    for rough_cfg in recipe_cfg.get("rough", []):
        tg.AddRoughGround(
            init_pos=rough_cfg["init_pos"],
            euler=rough_cfg.get("euler", [0.0, 0.0, 0.0]),
            nums=rough_cfg["nums"],
            box_size=rough_cfg["box_size"],
            box_euler=rough_cfg.get("box_euler", [0.0, 0.0, 0.0]),
            separation=rough_cfg["separation"],
            box_size_rand=rough_cfg.get("box_size_rand", [0.05, 0.05, 0.05]),
            box_euler_rand=rough_cfg.get("box_euler_rand", [0.2, 0.2, 0.2]),
            separation_rand=rough_cfg.get("separation_rand", [0.05, 0.05]),
        )

    for idx, perlin_cfg in enumerate(recipe_cfg.get("perlin", [])):
        tg.AddPerlinHeighField(
            position=perlin_cfg["position"],
            euler=perlin_cfg.get("euler", [0.0, 0.0, 0.0]),
            size=perlin_cfg["size"],
            height_scale=perlin_cfg["height_scale"],
            negative_height=perlin_cfg.get("negative_height", 0.12),
            image_width=perlin_cfg.get("image_width", 128),
            img_height=perlin_cfg.get("img_height", 128),
            smooth=perlin_cfg["smooth"],
            perlin_octaves=perlin_cfg.get("perlin_octaves", 6),
            perlin_persistence=perlin_cfg.get("perlin_persistence", 0.5),
            perlin_lacunarity=perlin_cfg.get("perlin_lacunarity", 2.0),
            output_hfield_image=f"{recipe_name}_height_field_{idx}.png",
        )

    for stair_cfg in recipe_cfg.get("stairs", []):
        tg.AddStairs(
            init_pos=stair_cfg["init_pos"],
            yaw=stair_cfg.get("yaw", 0.0),
            width=stair_cfg.get("width", 0.2),
            height=stair_cfg.get("height", 0.15),
            length=stair_cfg.get("length", 1.5),
            stair_nums=stair_cfg.get("stair_nums", 10),
        )

    for suspend_cfg in recipe_cfg.get("suspend_stairs", []):
        tg.AddSuspendStairs(
            init_pos=suspend_cfg["init_pos"],
            yaw=suspend_cfg.get("yaw", 0.0),
            width=suspend_cfg.get("width", 0.2),
            height=suspend_cfg.get("height", 0.15),
            length=suspend_cfg.get("length", 1.5),
            gap=suspend_cfg.get("gap", 0.1),
            stair_nums=suspend_cfg.get("stair_nums", 10),
        )

    for idx, image_cfg in enumerate(recipe_cfg.get("image", [])):
        tg.AddHeighFieldFromImage(
            position=image_cfg["position"],
            euler=image_cfg.get("euler", [0.0, 0.0, 0.0]),
            size=image_cfg["size"],
            height_scale=image_cfg.get("height_scale", 0.02),
            negative_height=image_cfg.get("negative_height", 0.1),
            input_img=str(TERRAIN_TOOL_DIR / "unitree_robot.jpeg"),
            output_hfield_image=f"{recipe_name}_image_hfield_{idx}.png",
            image_scale=image_cfg.get("image_scale", [1.0, 1.0]),
            invert_gray=image_cfg.get("invert_gray", False),
        )


def _write_recipe(module, recipe_name: str, recipe_cfg: dict[str, object]) -> Path:
    output_scene = GO2_SCENE_DIR / f"scene_eval_{recipe_name}.xml"
    old_input = module.INPUT_SCENE_PATH
    old_output = module.OUTPUT_SCENE_PATH
    old_cwd = Path.cwd()
    try:
        os.chdir(TERRAIN_TOOL_DIR)
        module.INPUT_SCENE_PATH = str(TERRAIN_TOOL_DIR / "scene.xml")
        module.OUTPUT_SCENE_PATH = str(output_scene)
        tg = module.TerrainGenerator()
        _apply_recipe(module, tg, recipe_name, recipe_cfg)
        tg.Save()
    finally:
        os.chdir(old_cwd)
        module.INPUT_SCENE_PATH = old_input
        module.OUTPUT_SCENE_PATH = old_output
    _uniquify_hfield_names(output_scene)
    return output_scene


def _uniquify_hfield_names(scene_path: Path) -> None:
    tree = ET.parse(scene_path)
    root = tree.getroot()
    asset = root.find("asset")
    worldbody = root.find("worldbody")
    if asset is None or worldbody is None:
        return

    rename_map: dict[str, str] = {}
    perlin_idx = 0
    image_idx = 0
    for hfield in asset.findall("hfield"):
        old_name = hfield.attrib.get("name")
        if not old_name:
            continue
        file_name = hfield.attrib.get("file", "")
        if "image_hfield" in file_name:
            new_name = f"image_hfield_{image_idx}"
            image_idx += 1
        else:
            new_name = f"perlin_hfield_{perlin_idx}"
            perlin_idx += 1
        rename_map.setdefault(old_name, new_name)
        hfield.attrib["name"] = new_name

    used_counts: dict[str, int] = {}
    for geom in worldbody.findall("geom"):
        hfield_name = geom.attrib.get("hfield")
        if not hfield_name:
            continue
        if hfield_name not in rename_map:
            continue
        idx = used_counts.get(hfield_name, 0)
        if "image_hfield" in hfield_name:
            geom.attrib["hfield"] = f"image_hfield_{idx}"
        else:
            geom.attrib["hfield"] = f"perlin_hfield_{idx}"
        used_counts[hfield_name] = idx + 1

    tree.write(scene_path)


def ensure_materialized_unitree_terrain_recipes() -> dict[str, Path]:
    module = _load_terrain_tool_module()
    outputs: dict[str, Path] = {}
    for recipe_name, recipe_cfg in RECIPES.items():
        outputs[recipe_name] = _write_recipe(module, recipe_name, recipe_cfg)
    return outputs


if __name__ == "__main__":
    for recipe_name, path in ensure_materialized_unitree_terrain_recipes().items():
        print(recipe_name, path)
